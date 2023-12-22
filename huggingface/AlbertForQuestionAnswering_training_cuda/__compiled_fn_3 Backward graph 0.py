from __future__ import annotations



def forward(self, primals_4: "f32[128]", primals_16: "f32[4096]", primals_22: "f32[4096]", primals_28: "i64[1, 512]", expand: "i64[1, 512]", slice_2: "i64[1, 512]", mul_1: "f32[1, 512, 128]", view: "f32[512, 128]", view_2: "f32[512, 4096]", view_18: "f32[512, 4096]", mul_3: "f32[1, 512, 4096]", view_20: "f32[512, 4096]", addmm_5: "f32[512, 16384]", tanh: "f32[1, 512, 16384]", view_22: "f32[512, 16384]", mul_9: "f32[1, 512, 4096]", view_24: "f32[512, 4096]", view_40: "f32[512, 4096]", mul_11: "f32[1, 512, 4096]", view_42: "f32[512, 4096]", addmm_11: "f32[512, 16384]", tanh_1: "f32[1, 512, 16384]", view_44: "f32[512, 16384]", mul_17: "f32[1, 512, 4096]", view_46: "f32[512, 4096]", view_62: "f32[512, 4096]", mul_19: "f32[1, 512, 4096]", view_64: "f32[512, 4096]", addmm_17: "f32[512, 16384]", tanh_2: "f32[1, 512, 16384]", view_66: "f32[512, 16384]", mul_25: "f32[1, 512, 4096]", view_68: "f32[512, 4096]", view_84: "f32[512, 4096]", mul_27: "f32[1, 512, 4096]", view_86: "f32[512, 4096]", addmm_23: "f32[512, 16384]", tanh_3: "f32[1, 512, 16384]", view_88: "f32[512, 16384]", mul_33: "f32[1, 512, 4096]", view_90: "f32[512, 4096]", view_106: "f32[512, 4096]", mul_35: "f32[1, 512, 4096]", view_108: "f32[512, 4096]", addmm_29: "f32[512, 16384]", tanh_4: "f32[1, 512, 16384]", view_110: "f32[512, 16384]", mul_41: "f32[1, 512, 4096]", view_112: "f32[512, 4096]", view_128: "f32[512, 4096]", mul_43: "f32[1, 512, 4096]", view_130: "f32[512, 4096]", addmm_35: "f32[512, 16384]", tanh_5: "f32[1, 512, 16384]", view_132: "f32[512, 16384]", mul_49: "f32[1, 512, 4096]", view_134: "f32[512, 4096]", view_150: "f32[512, 4096]", mul_51: "f32[1, 512, 4096]", view_152: "f32[512, 4096]", addmm_41: "f32[512, 16384]", tanh_6: "f32[1, 512, 16384]", view_154: "f32[512, 16384]", mul_57: "f32[1, 512, 4096]", view_156: "f32[512, 4096]", view_172: "f32[512, 4096]", mul_59: "f32[1, 512, 4096]", view_174: "f32[512, 4096]", addmm_47: "f32[512, 16384]", tanh_7: "f32[1, 512, 16384]", view_176: "f32[512, 16384]", mul_65: "f32[1, 512, 4096]", view_178: "f32[512, 4096]", view_194: "f32[512, 4096]", mul_67: "f32[1, 512, 4096]", view_196: "f32[512, 4096]", addmm_53: "f32[512, 16384]", tanh_8: "f32[1, 512, 16384]", view_198: "f32[512, 16384]", mul_73: "f32[1, 512, 4096]", view_200: "f32[512, 4096]", view_216: "f32[512, 4096]", mul_75: "f32[1, 512, 4096]", view_218: "f32[512, 4096]", addmm_59: "f32[512, 16384]", tanh_9: "f32[1, 512, 16384]", view_220: "f32[512, 16384]", mul_81: "f32[1, 512, 4096]", view_222: "f32[512, 4096]", view_238: "f32[512, 4096]", mul_83: "f32[1, 512, 4096]", view_240: "f32[512, 4096]", addmm_65: "f32[512, 16384]", tanh_10: "f32[1, 512, 16384]", view_242: "f32[512, 16384]", mul_89: "f32[1, 512, 4096]", view_244: "f32[512, 4096]", view_260: "f32[512, 4096]", mul_91: "f32[1, 512, 4096]", view_262: "f32[512, 4096]", addmm_71: "f32[512, 16384]", tanh_11: "f32[1, 512, 16384]", view_264: "f32[512, 16384]", mul_97: "f32[1, 512, 4096]", view_266: "f32[512, 4096]", sub_39: "f32[1, 512]", ne: "b8[1]", sub_41: "f32[1, 512]", ne_3: "b8[1]", ne_6: "b8[1, 1]", where_4: "i64[1, 1]", ne_8: "b8[1, 1]", where_6: "i64[1, 1]", permute_134: "f32[2, 4096]", div_30: "f32[1, 512, 1]", permute_138: "f32[4096, 16384]", permute_142: "f32[16384, 4096]", div_31: "f32[1, 512, 1]", permute_146: "f32[4096, 4096]", permute_151: "f32[64, 512, 512]", permute_152: "f32[64, 64, 512]", alias_29: "f32[1, 64, 512, 512]", permute_153: "f32[64, 64, 512]", permute_154: "f32[64, 512, 64]", permute_159: "f32[4096, 4096]", permute_163: "f32[4096, 4096]", permute_167: "f32[4096, 4096]", div_33: "f32[1, 512, 1]", div_34: "f32[1, 512, 1]", permute_184: "f32[64, 512, 512]", permute_185: "f32[64, 64, 512]", alias_31: "f32[1, 64, 512, 512]", permute_186: "f32[64, 64, 512]", permute_187: "f32[64, 512, 64]", div_36: "f32[1, 512, 1]", div_37: "f32[1, 512, 1]", permute_217: "f32[64, 512, 512]", permute_218: "f32[64, 64, 512]", alias_33: "f32[1, 64, 512, 512]", permute_219: "f32[64, 64, 512]", permute_220: "f32[64, 512, 64]", div_39: "f32[1, 512, 1]", div_40: "f32[1, 512, 1]", permute_250: "f32[64, 512, 512]", permute_251: "f32[64, 64, 512]", alias_35: "f32[1, 64, 512, 512]", permute_252: "f32[64, 64, 512]", permute_253: "f32[64, 512, 64]", div_42: "f32[1, 512, 1]", div_43: "f32[1, 512, 1]", permute_283: "f32[64, 512, 512]", permute_284: "f32[64, 64, 512]", alias_37: "f32[1, 64, 512, 512]", permute_285: "f32[64, 64, 512]", permute_286: "f32[64, 512, 64]", div_45: "f32[1, 512, 1]", div_46: "f32[1, 512, 1]", permute_316: "f32[64, 512, 512]", permute_317: "f32[64, 64, 512]", alias_39: "f32[1, 64, 512, 512]", permute_318: "f32[64, 64, 512]", permute_319: "f32[64, 512, 64]", div_48: "f32[1, 512, 1]", div_49: "f32[1, 512, 1]", permute_349: "f32[64, 512, 512]", permute_350: "f32[64, 64, 512]", alias_41: "f32[1, 64, 512, 512]", permute_351: "f32[64, 64, 512]", permute_352: "f32[64, 512, 64]", div_51: "f32[1, 512, 1]", div_52: "f32[1, 512, 1]", permute_382: "f32[64, 512, 512]", permute_383: "f32[64, 64, 512]", alias_43: "f32[1, 64, 512, 512]", permute_384: "f32[64, 64, 512]", permute_385: "f32[64, 512, 64]", div_54: "f32[1, 512, 1]", div_55: "f32[1, 512, 1]", permute_415: "f32[64, 512, 512]", permute_416: "f32[64, 64, 512]", alias_45: "f32[1, 64, 512, 512]", permute_417: "f32[64, 64, 512]", permute_418: "f32[64, 512, 64]", div_57: "f32[1, 512, 1]", div_58: "f32[1, 512, 1]", permute_448: "f32[64, 512, 512]", permute_449: "f32[64, 64, 512]", alias_47: "f32[1, 64, 512, 512]", permute_450: "f32[64, 64, 512]", permute_451: "f32[64, 512, 64]", div_60: "f32[1, 512, 1]", div_61: "f32[1, 512, 1]", permute_481: "f32[64, 512, 512]", permute_482: "f32[64, 64, 512]", alias_49: "f32[1, 64, 512, 512]", permute_483: "f32[64, 64, 512]", permute_484: "f32[64, 512, 64]", div_63: "f32[1, 512, 1]", div_64: "f32[1, 512, 1]", permute_514: "f32[64, 512, 512]", permute_515: "f32[64, 64, 512]", alias_51: "f32[1, 64, 512, 512]", permute_516: "f32[64, 64, 512]", permute_517: "f32[64, 512, 64]", permute_534: "f32[4096, 128]", div_66: "f32[1, 512, 1]", tangents_1: "f32[]", tangents_2: "f32[1, 512]", tangents_3: "f32[1, 512]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_21: "f32[1, 512, 16384]" = torch.ops.aten.view.default(addmm_5, [1, 512, 16384]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_5: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(view_21, 0.5)
    alias_1: "f32[1, 512, 16384]" = torch.ops.aten.alias.default(tanh)
    add_9: "f32[1, 512, 16384]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_43: "f32[1, 512, 16384]" = torch.ops.aten.view.default(addmm_11, [1, 512, 16384]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_13: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(view_43, 0.5)
    alias_3: "f32[1, 512, 16384]" = torch.ops.aten.alias.default(tanh_1)
    add_18: "f32[1, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_1, 1.0);  tanh_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_65: "f32[1, 512, 16384]" = torch.ops.aten.view.default(addmm_17, [1, 512, 16384]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_21: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(view_65, 0.5)
    alias_5: "f32[1, 512, 16384]" = torch.ops.aten.alias.default(tanh_2)
    add_27: "f32[1, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_2, 1.0);  tanh_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_87: "f32[1, 512, 16384]" = torch.ops.aten.view.default(addmm_23, [1, 512, 16384]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_29: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(view_87, 0.5)
    alias_7: "f32[1, 512, 16384]" = torch.ops.aten.alias.default(tanh_3)
    add_36: "f32[1, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_3, 1.0);  tanh_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_109: "f32[1, 512, 16384]" = torch.ops.aten.view.default(addmm_29, [1, 512, 16384]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_37: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(view_109, 0.5)
    alias_9: "f32[1, 512, 16384]" = torch.ops.aten.alias.default(tanh_4)
    add_45: "f32[1, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_4, 1.0);  tanh_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_131: "f32[1, 512, 16384]" = torch.ops.aten.view.default(addmm_35, [1, 512, 16384]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_45: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(view_131, 0.5)
    alias_11: "f32[1, 512, 16384]" = torch.ops.aten.alias.default(tanh_5)
    add_54: "f32[1, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_5, 1.0);  tanh_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_153: "f32[1, 512, 16384]" = torch.ops.aten.view.default(addmm_41, [1, 512, 16384]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_53: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(view_153, 0.5)
    alias_13: "f32[1, 512, 16384]" = torch.ops.aten.alias.default(tanh_6)
    add_63: "f32[1, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_6, 1.0);  tanh_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_175: "f32[1, 512, 16384]" = torch.ops.aten.view.default(addmm_47, [1, 512, 16384]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_61: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(view_175, 0.5)
    alias_15: "f32[1, 512, 16384]" = torch.ops.aten.alias.default(tanh_7)
    add_72: "f32[1, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_7, 1.0);  tanh_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_197: "f32[1, 512, 16384]" = torch.ops.aten.view.default(addmm_53, [1, 512, 16384]);  addmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_69: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(view_197, 0.5)
    alias_17: "f32[1, 512, 16384]" = torch.ops.aten.alias.default(tanh_8)
    add_81: "f32[1, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_8, 1.0);  tanh_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_219: "f32[1, 512, 16384]" = torch.ops.aten.view.default(addmm_59, [1, 512, 16384]);  addmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_77: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(view_219, 0.5)
    alias_19: "f32[1, 512, 16384]" = torch.ops.aten.alias.default(tanh_9)
    add_90: "f32[1, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_9, 1.0);  tanh_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_241: "f32[1, 512, 16384]" = torch.ops.aten.view.default(addmm_65, [1, 512, 16384]);  addmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_85: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(view_241, 0.5)
    alias_21: "f32[1, 512, 16384]" = torch.ops.aten.alias.default(tanh_10)
    add_99: "f32[1, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_10, 1.0);  tanh_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_263: "f32[1, 512, 16384]" = torch.ops.aten.view.default(addmm_71, [1, 512, 16384]);  addmm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_93: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(view_263, 0.5)
    alias_23: "f32[1, 512, 16384]" = torch.ops.aten.alias.default(tanh_11)
    add_108: "f32[1, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_11, 1.0);  tanh_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:1288, code: start_loss = loss_fct(start_logits, start_positions)
    alias_24: "f32[1, 512]" = torch.ops.aten.alias.default(sub_39);  sub_39 = None
    full_default_2: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sum_14: "i64[]" = torch.ops.aten.sum.default(ne);  ne = None
    convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_14, torch.float32);  sum_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:1289, code: end_loss = loss_fct(end_logits, end_positions)
    alias_25: "f32[1, 512]" = torch.ops.aten.alias.default(sub_41);  sub_41 = None
    sum_17: "i64[]" = torch.ops.aten.sum.default(ne_3);  ne_3 = None
    convert_element_type_1: "f32[]" = torch.ops.prims.convert_element_type.default(sum_17, torch.float32);  sum_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:1290, code: total_loss = (start_loss + end_loss) / 2
    div_27: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, 2);  tangents_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:1289, code: end_loss = loss_fct(end_logits, end_positions)
    div_28: "f32[]" = torch.ops.aten.div.Tensor(div_27, convert_element_type_1);  convert_element_type_1 = None
    full_default_6: "f32[1, 512]" = torch.ops.aten.full.default([1, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    scatter: "f32[1, 512]" = torch.ops.aten.scatter.value(full_default_6, 1, where_4, -1.0);  where_4 = None
    where_5: "f32[1, 1]" = torch.ops.aten.where.self(ne_6, div_28, full_default_2);  ne_6 = div_28 = None
    mul_99: "f32[1, 512]" = torch.ops.aten.mul.Tensor(scatter, where_5);  scatter = where_5 = None
    alias_26: "f32[1, 512]" = torch.ops.aten.alias.default(alias_25);  alias_25 = None
    exp_14: "f32[1, 512]" = torch.ops.aten.exp.default(alias_26);  alias_26 = None
    sum_19: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(mul_99, [1], True)
    mul_100: "f32[1, 512]" = torch.ops.aten.mul.Tensor(exp_14, sum_19);  exp_14 = sum_19 = None
    sub_42: "f32[1, 512]" = torch.ops.aten.sub.Tensor(mul_99, mul_100);  mul_99 = mul_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:1289, code: end_loss = loss_fct(end_logits, end_positions)
    add_113: "f32[1, 512]" = torch.ops.aten.add.Tensor(tangents_3, sub_42);  tangents_3 = sub_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:1288, code: start_loss = loss_fct(start_logits, start_positions)
    div_29: "f32[]" = torch.ops.aten.div.Tensor(div_27, convert_element_type);  div_27 = convert_element_type = None
    scatter_1: "f32[1, 512]" = torch.ops.aten.scatter.value(full_default_6, 1, where_6, -1.0);  full_default_6 = where_6 = None
    where_7: "f32[1, 1]" = torch.ops.aten.where.self(ne_8, div_29, full_default_2);  ne_8 = div_29 = None
    mul_101: "f32[1, 512]" = torch.ops.aten.mul.Tensor(scatter_1, where_7);  scatter_1 = where_7 = None
    alias_27: "f32[1, 512]" = torch.ops.aten.alias.default(alias_24);  alias_24 = None
    exp_15: "f32[1, 512]" = torch.ops.aten.exp.default(alias_27);  alias_27 = None
    sum_20: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(mul_101, [1], True)
    mul_102: "f32[1, 512]" = torch.ops.aten.mul.Tensor(exp_15, sum_20);  exp_15 = sum_20 = None
    sub_43: "f32[1, 512]" = torch.ops.aten.sub.Tensor(mul_101, mul_102);  mul_101 = mul_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:1288, code: start_loss = loss_fct(start_logits, start_positions)
    add_114: "f32[1, 512]" = torch.ops.aten.add.Tensor(tangents_2, sub_43);  tangents_2 = sub_43 = None
    
    # No stacktrace found for following nodes
    unsqueeze_6: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(add_113, 2);  add_113 = None
    unsqueeze_7: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(add_114, 2);  add_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:1271, code: start_logits, end_logits = logits.split(1, dim=-1)
    cat: "f32[1, 512, 2]" = torch.ops.aten.cat.default([unsqueeze_7, unsqueeze_6], 2);  unsqueeze_7 = unsqueeze_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:1270, code: logits: torch.Tensor = self.qa_outputs(sequence_output)
    view_268: "f32[512, 2]" = torch.ops.aten.view.default(cat, [512, 2]);  cat = None
    mm: "f32[512, 4096]" = torch.ops.aten.mm.default(view_268, permute_134);  permute_134 = None
    permute_135: "f32[2, 512]" = torch.ops.aten.permute.default(view_268, [1, 0])
    mm_1: "f32[2, 4096]" = torch.ops.aten.mm.default(permute_135, view_266);  permute_135 = view_266 = None
    permute_136: "f32[4096, 2]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_21: "f32[1, 2]" = torch.ops.aten.sum.dim_IntList(view_268, [0], True);  view_268 = None
    view_269: "f32[2]" = torch.ops.aten.view.default(sum_21, [2]);  sum_21 = None
    permute_137: "f32[2, 4096]" = torch.ops.aten.permute.default(permute_136, [1, 0]);  permute_136 = None
    view_270: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm, [1, 512, 4096]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    mul_104: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_270, primals_22)
    mul_105: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_104, 4096)
    sum_22: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_104, [2], True)
    mul_106: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_104, mul_97);  mul_104 = None
    sum_23: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_106, [2], True);  mul_106 = None
    mul_107: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_97, sum_23);  sum_23 = None
    sub_45: "f32[1, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_105, sum_22);  mul_105 = sum_22 = None
    sub_46: "f32[1, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_45, mul_107);  sub_45 = mul_107 = None
    mul_108: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(div_30, sub_46);  div_30 = sub_46 = None
    mul_109: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_270, mul_97);  mul_97 = None
    sum_24: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_109, [0, 1]);  mul_109 = None
    sum_25: "f32[4096]" = torch.ops.aten.sum.dim_IntList(view_270, [0, 1]);  view_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_271: "f32[512, 4096]" = torch.ops.aten.view.default(mul_108, [512, 4096])
    mm_2: "f32[512, 16384]" = torch.ops.aten.mm.default(view_271, permute_138)
    permute_139: "f32[4096, 512]" = torch.ops.aten.permute.default(view_271, [1, 0])
    mm_3: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_139, view_264);  permute_139 = view_264 = None
    permute_140: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
    sum_26: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_271, [0], True);  view_271 = None
    view_272: "f32[4096]" = torch.ops.aten.view.default(sum_26, [4096]);  sum_26 = None
    permute_141: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_140, [1, 0]);  permute_140 = None
    view_273: "f32[1, 512, 16384]" = torch.ops.aten.view.default(mm_2, [1, 512, 16384]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_110: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(view_273, mul_93);  mul_93 = None
    mul_111: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(view_273, add_108);  view_273 = add_108 = None
    alias_28: "f32[1, 512, 16384]" = torch.ops.aten.alias.default(alias_23);  alias_23 = None
    mul_112: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(alias_28, alias_28);  alias_28 = None
    sub_47: "f32[1, 512, 16384]" = torch.ops.aten.sub.Tensor(1, mul_112);  mul_112 = None
    mul_113: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_110, sub_47);  mul_110 = sub_47 = None
    mul_114: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_113, 0.7978845608028654);  mul_113 = None
    mul_115: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_114, 0.044715)
    pow_13: "f32[1, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_263, 2.0);  view_263 = None
    mul_116: "f32[1, 512, 16384]" = torch.ops.aten.mul.Scalar(pow_13, 3.0);  pow_13 = None
    mul_117: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_115, mul_116);  mul_115 = mul_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_115: "f32[1, 512, 16384]" = torch.ops.aten.add.Tensor(mul_114, mul_117);  mul_114 = mul_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_118: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_111, 0.5);  mul_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_116: "f32[1, 512, 16384]" = torch.ops.aten.add.Tensor(add_115, mul_118);  add_115 = mul_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_274: "f32[512, 16384]" = torch.ops.aten.view.default(add_116, [512, 16384]);  add_116 = None
    mm_4: "f32[512, 4096]" = torch.ops.aten.mm.default(view_274, permute_142)
    permute_143: "f32[16384, 512]" = torch.ops.aten.permute.default(view_274, [1, 0])
    mm_5: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_143, view_262);  permute_143 = view_262 = None
    permute_144: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
    sum_27: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_274, [0], True);  view_274 = None
    view_275: "f32[16384]" = torch.ops.aten.view.default(sum_27, [16384]);  sum_27 = None
    permute_145: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_144, [1, 0]);  permute_144 = None
    view_276: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_4, [1, 512, 4096]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_117: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_108, view_276);  mul_108 = view_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    mul_120: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_117, primals_16)
    mul_121: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_120, 4096)
    sum_28: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_120, [2], True)
    mul_122: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_120, mul_91);  mul_120 = None
    sum_29: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_122, [2], True);  mul_122 = None
    mul_123: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_91, sum_29);  sum_29 = None
    sub_49: "f32[1, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_121, sum_28);  mul_121 = sum_28 = None
    sub_50: "f32[1, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_49, mul_123);  sub_49 = mul_123 = None
    mul_124: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(div_31, sub_50);  div_31 = sub_50 = None
    mul_125: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_117, mul_91);  mul_91 = None
    sum_30: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_125, [0, 1]);  mul_125 = None
    sum_31: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_117, [0, 1]);  add_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_277: "f32[512, 4096]" = torch.ops.aten.view.default(mul_124, [512, 4096])
    mm_6: "f32[512, 4096]" = torch.ops.aten.mm.default(view_277, permute_146)
    permute_147: "f32[4096, 512]" = torch.ops.aten.permute.default(view_277, [1, 0])
    mm_7: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_147, view_260);  permute_147 = view_260 = None
    permute_148: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
    sum_32: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_277, [0], True);  view_277 = None
    view_278: "f32[4096]" = torch.ops.aten.view.default(sum_32, [4096]);  sum_32 = None
    permute_149: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_148, [1, 0]);  permute_148 = None
    view_279: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_6, [1, 512, 4096]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    view_280: "f32[1, 512, 64, 64]" = torch.ops.aten.view.default(view_279, [1, 512, 64, 64]);  view_279 = None
    permute_150: "f32[1, 64, 512, 64]" = torch.ops.aten.permute.default(view_280, [0, 2, 1, 3]);  view_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_281: "f32[64, 512, 64]" = torch.ops.aten.view.default(permute_150, [64, 512, 64]);  permute_150 = None
    bmm_24: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(permute_151, view_281);  permute_151 = None
    bmm_25: "f32[64, 512, 512]" = torch.ops.aten.bmm.default(view_281, permute_152);  view_281 = permute_152 = None
    view_282: "f32[1, 64, 512, 64]" = torch.ops.aten.view.default(bmm_24, [1, 64, 512, 64]);  bmm_24 = None
    view_283: "f32[1, 64, 512, 512]" = torch.ops.aten.view.default(bmm_25, [1, 64, 512, 512]);  bmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    mul_126: "f32[1, 64, 512, 512]" = torch.ops.aten.mul.Tensor(view_283, alias_29);  view_283 = None
    sum_33: "f32[1, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_126, [-1], True)
    mul_127: "f32[1, 64, 512, 512]" = torch.ops.aten.mul.Tensor(alias_29, sum_33);  alias_29 = sum_33 = None
    sub_51: "f32[1, 64, 512, 512]" = torch.ops.aten.sub.Tensor(mul_126, mul_127);  mul_126 = mul_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:332, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_32: "f32[1, 64, 512, 512]" = torch.ops.aten.div.Tensor(sub_51, 8.0);  sub_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_284: "f32[64, 512, 512]" = torch.ops.aten.view.default(div_32, [64, 512, 512]);  div_32 = None
    bmm_26: "f32[64, 64, 512]" = torch.ops.aten.bmm.default(permute_153, view_284);  permute_153 = None
    bmm_27: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(view_284, permute_154);  view_284 = permute_154 = None
    view_285: "f32[1, 64, 64, 512]" = torch.ops.aten.view.default(bmm_26, [1, 64, 64, 512]);  bmm_26 = None
    view_286: "f32[1, 64, 512, 64]" = torch.ops.aten.view.default(bmm_27, [1, 64, 512, 64]);  bmm_27 = None
    permute_155: "f32[1, 64, 512, 64]" = torch.ops.aten.permute.default(view_285, [0, 1, 3, 2]);  view_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_156: "f32[1, 512, 64, 64]" = torch.ops.aten.permute.default(view_282, [0, 2, 1, 3]);  view_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_39: "f32[1, 512, 64, 64]" = torch.ops.aten.clone.default(permute_156, memory_format = torch.contiguous_format);  permute_156 = None
    view_287: "f32[1, 512, 4096]" = torch.ops.aten.view.default(clone_39, [1, 512, 4096]);  clone_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_157: "f32[1, 512, 64, 64]" = torch.ops.aten.permute.default(permute_155, [0, 2, 1, 3]);  permute_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_288: "f32[1, 512, 4096]" = torch.ops.aten.view.default(permute_157, [1, 512, 4096]);  permute_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_158: "f32[1, 512, 64, 64]" = torch.ops.aten.permute.default(view_286, [0, 2, 1, 3]);  view_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_40: "f32[1, 512, 64, 64]" = torch.ops.aten.clone.default(permute_158, memory_format = torch.contiguous_format);  permute_158 = None
    view_289: "f32[1, 512, 4096]" = torch.ops.aten.view.default(clone_40, [1, 512, 4096]);  clone_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_290: "f32[512, 4096]" = torch.ops.aten.view.default(view_287, [512, 4096]);  view_287 = None
    mm_8: "f32[512, 4096]" = torch.ops.aten.mm.default(view_290, permute_159)
    permute_160: "f32[4096, 512]" = torch.ops.aten.permute.default(view_290, [1, 0])
    mm_9: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_160, view_244);  permute_160 = None
    permute_161: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
    sum_34: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_290, [0], True);  view_290 = None
    view_291: "f32[4096]" = torch.ops.aten.view.default(sum_34, [4096]);  sum_34 = None
    permute_162: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_161, [1, 0]);  permute_161 = None
    view_292: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_8, [1, 512, 4096]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_118: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_124, view_292);  mul_124 = view_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_293: "f32[512, 4096]" = torch.ops.aten.view.default(view_288, [512, 4096]);  view_288 = None
    mm_10: "f32[512, 4096]" = torch.ops.aten.mm.default(view_293, permute_163)
    permute_164: "f32[4096, 512]" = torch.ops.aten.permute.default(view_293, [1, 0])
    mm_11: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_164, view_244);  permute_164 = None
    permute_165: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_11, [1, 0]);  mm_11 = None
    sum_35: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_293, [0], True);  view_293 = None
    view_294: "f32[4096]" = torch.ops.aten.view.default(sum_35, [4096]);  sum_35 = None
    permute_166: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_165, [1, 0]);  permute_165 = None
    view_295: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_10, [1, 512, 4096]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_119: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_118, view_295);  add_118 = view_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_296: "f32[512, 4096]" = torch.ops.aten.view.default(view_289, [512, 4096]);  view_289 = None
    mm_12: "f32[512, 4096]" = torch.ops.aten.mm.default(view_296, permute_167)
    permute_168: "f32[4096, 512]" = torch.ops.aten.permute.default(view_296, [1, 0])
    mm_13: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_168, view_244);  permute_168 = view_244 = None
    permute_169: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_13, [1, 0]);  mm_13 = None
    sum_36: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_296, [0], True);  view_296 = None
    view_297: "f32[4096]" = torch.ops.aten.view.default(sum_36, [4096]);  sum_36 = None
    permute_170: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_169, [1, 0]);  permute_169 = None
    view_298: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_12, [1, 512, 4096]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_120: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_119, view_298);  add_119 = view_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    mul_129: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_120, primals_22)
    mul_130: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_129, 4096)
    sum_37: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_129, [2], True)
    mul_131: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_129, mul_89);  mul_129 = None
    sum_38: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_131, [2], True);  mul_131 = None
    mul_132: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_89, sum_38);  sum_38 = None
    sub_53: "f32[1, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_130, sum_37);  mul_130 = sum_37 = None
    sub_54: "f32[1, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_53, mul_132);  sub_53 = mul_132 = None
    mul_133: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(div_33, sub_54);  div_33 = sub_54 = None
    mul_134: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_120, mul_89);  mul_89 = None
    sum_39: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_134, [0, 1]);  mul_134 = None
    sum_40: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_120, [0, 1]);  add_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_121: "f32[4096]" = torch.ops.aten.add.Tensor(sum_24, sum_39);  sum_24 = sum_39 = None
    add_122: "f32[4096]" = torch.ops.aten.add.Tensor(sum_25, sum_40);  sum_25 = sum_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_299: "f32[512, 4096]" = torch.ops.aten.view.default(mul_133, [512, 4096])
    mm_14: "f32[512, 16384]" = torch.ops.aten.mm.default(view_299, permute_138)
    permute_172: "f32[4096, 512]" = torch.ops.aten.permute.default(view_299, [1, 0])
    mm_15: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_172, view_242);  permute_172 = view_242 = None
    permute_173: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_15, [1, 0]);  mm_15 = None
    sum_41: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_299, [0], True);  view_299 = None
    view_300: "f32[4096]" = torch.ops.aten.view.default(sum_41, [4096]);  sum_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_123: "f32[4096]" = torch.ops.aten.add.Tensor(view_272, view_300);  view_272 = view_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    permute_174: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_173, [1, 0]);  permute_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_124: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(permute_141, permute_174);  permute_141 = permute_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_301: "f32[1, 512, 16384]" = torch.ops.aten.view.default(mm_14, [1, 512, 16384]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_135: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(view_301, mul_85);  mul_85 = None
    mul_136: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(view_301, add_99);  view_301 = add_99 = None
    alias_30: "f32[1, 512, 16384]" = torch.ops.aten.alias.default(alias_21);  alias_21 = None
    mul_137: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(alias_30, alias_30);  alias_30 = None
    sub_55: "f32[1, 512, 16384]" = torch.ops.aten.sub.Tensor(1, mul_137);  mul_137 = None
    mul_138: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_135, sub_55);  mul_135 = sub_55 = None
    mul_139: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_138, 0.7978845608028654);  mul_138 = None
    mul_140: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_139, 0.044715)
    pow_14: "f32[1, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_241, 2.0);  view_241 = None
    mul_141: "f32[1, 512, 16384]" = torch.ops.aten.mul.Scalar(pow_14, 3.0);  pow_14 = None
    mul_142: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_140, mul_141);  mul_140 = mul_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_125: "f32[1, 512, 16384]" = torch.ops.aten.add.Tensor(mul_139, mul_142);  mul_139 = mul_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_143: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_136, 0.5);  mul_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_126: "f32[1, 512, 16384]" = torch.ops.aten.add.Tensor(add_125, mul_143);  add_125 = mul_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_302: "f32[512, 16384]" = torch.ops.aten.view.default(add_126, [512, 16384]);  add_126 = None
    mm_16: "f32[512, 4096]" = torch.ops.aten.mm.default(view_302, permute_142)
    permute_176: "f32[16384, 512]" = torch.ops.aten.permute.default(view_302, [1, 0])
    mm_17: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_176, view_240);  permute_176 = view_240 = None
    permute_177: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_17, [1, 0]);  mm_17 = None
    sum_42: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_302, [0], True);  view_302 = None
    view_303: "f32[16384]" = torch.ops.aten.view.default(sum_42, [16384]);  sum_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_127: "f32[16384]" = torch.ops.aten.add.Tensor(view_275, view_303);  view_275 = view_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    permute_178: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_177, [1, 0]);  permute_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_128: "f32[16384, 4096]" = torch.ops.aten.add.Tensor(permute_145, permute_178);  permute_145 = permute_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_304: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_16, [1, 512, 4096]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_129: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_133, view_304);  mul_133 = view_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    mul_145: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_129, primals_16)
    mul_146: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_145, 4096)
    sum_43: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_145, [2], True)
    mul_147: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_145, mul_83);  mul_145 = None
    sum_44: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_147, [2], True);  mul_147 = None
    mul_148: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_83, sum_44);  sum_44 = None
    sub_57: "f32[1, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_146, sum_43);  mul_146 = sum_43 = None
    sub_58: "f32[1, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_57, mul_148);  sub_57 = mul_148 = None
    mul_149: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(div_34, sub_58);  div_34 = sub_58 = None
    mul_150: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_129, mul_83);  mul_83 = None
    sum_45: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_150, [0, 1]);  mul_150 = None
    sum_46: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_129, [0, 1]);  add_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_130: "f32[4096]" = torch.ops.aten.add.Tensor(sum_30, sum_45);  sum_30 = sum_45 = None
    add_131: "f32[4096]" = torch.ops.aten.add.Tensor(sum_31, sum_46);  sum_31 = sum_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_305: "f32[512, 4096]" = torch.ops.aten.view.default(mul_149, [512, 4096])
    mm_18: "f32[512, 4096]" = torch.ops.aten.mm.default(view_305, permute_146)
    permute_180: "f32[4096, 512]" = torch.ops.aten.permute.default(view_305, [1, 0])
    mm_19: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_180, view_238);  permute_180 = view_238 = None
    permute_181: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_19, [1, 0]);  mm_19 = None
    sum_47: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_305, [0], True);  view_305 = None
    view_306: "f32[4096]" = torch.ops.aten.view.default(sum_47, [4096]);  sum_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_132: "f32[4096]" = torch.ops.aten.add.Tensor(view_278, view_306);  view_278 = view_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    permute_182: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_181, [1, 0]);  permute_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_133: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(permute_149, permute_182);  permute_149 = permute_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_307: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_18, [1, 512, 4096]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    view_308: "f32[1, 512, 64, 64]" = torch.ops.aten.view.default(view_307, [1, 512, 64, 64]);  view_307 = None
    permute_183: "f32[1, 64, 512, 64]" = torch.ops.aten.permute.default(view_308, [0, 2, 1, 3]);  view_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_309: "f32[64, 512, 64]" = torch.ops.aten.view.default(permute_183, [64, 512, 64]);  permute_183 = None
    bmm_28: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(permute_184, view_309);  permute_184 = None
    bmm_29: "f32[64, 512, 512]" = torch.ops.aten.bmm.default(view_309, permute_185);  view_309 = permute_185 = None
    view_310: "f32[1, 64, 512, 64]" = torch.ops.aten.view.default(bmm_28, [1, 64, 512, 64]);  bmm_28 = None
    view_311: "f32[1, 64, 512, 512]" = torch.ops.aten.view.default(bmm_29, [1, 64, 512, 512]);  bmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    mul_151: "f32[1, 64, 512, 512]" = torch.ops.aten.mul.Tensor(view_311, alias_31);  view_311 = None
    sum_48: "f32[1, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_151, [-1], True)
    mul_152: "f32[1, 64, 512, 512]" = torch.ops.aten.mul.Tensor(alias_31, sum_48);  alias_31 = sum_48 = None
    sub_59: "f32[1, 64, 512, 512]" = torch.ops.aten.sub.Tensor(mul_151, mul_152);  mul_151 = mul_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:332, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_35: "f32[1, 64, 512, 512]" = torch.ops.aten.div.Tensor(sub_59, 8.0);  sub_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_312: "f32[64, 512, 512]" = torch.ops.aten.view.default(div_35, [64, 512, 512]);  div_35 = None
    bmm_30: "f32[64, 64, 512]" = torch.ops.aten.bmm.default(permute_186, view_312);  permute_186 = None
    bmm_31: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(view_312, permute_187);  view_312 = permute_187 = None
    view_313: "f32[1, 64, 64, 512]" = torch.ops.aten.view.default(bmm_30, [1, 64, 64, 512]);  bmm_30 = None
    view_314: "f32[1, 64, 512, 64]" = torch.ops.aten.view.default(bmm_31, [1, 64, 512, 64]);  bmm_31 = None
    permute_188: "f32[1, 64, 512, 64]" = torch.ops.aten.permute.default(view_313, [0, 1, 3, 2]);  view_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_189: "f32[1, 512, 64, 64]" = torch.ops.aten.permute.default(view_310, [0, 2, 1, 3]);  view_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_41: "f32[1, 512, 64, 64]" = torch.ops.aten.clone.default(permute_189, memory_format = torch.contiguous_format);  permute_189 = None
    view_315: "f32[1, 512, 4096]" = torch.ops.aten.view.default(clone_41, [1, 512, 4096]);  clone_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_190: "f32[1, 512, 64, 64]" = torch.ops.aten.permute.default(permute_188, [0, 2, 1, 3]);  permute_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_316: "f32[1, 512, 4096]" = torch.ops.aten.view.default(permute_190, [1, 512, 4096]);  permute_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_191: "f32[1, 512, 64, 64]" = torch.ops.aten.permute.default(view_314, [0, 2, 1, 3]);  view_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_42: "f32[1, 512, 64, 64]" = torch.ops.aten.clone.default(permute_191, memory_format = torch.contiguous_format);  permute_191 = None
    view_317: "f32[1, 512, 4096]" = torch.ops.aten.view.default(clone_42, [1, 512, 4096]);  clone_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_318: "f32[512, 4096]" = torch.ops.aten.view.default(view_315, [512, 4096]);  view_315 = None
    mm_20: "f32[512, 4096]" = torch.ops.aten.mm.default(view_318, permute_159)
    permute_193: "f32[4096, 512]" = torch.ops.aten.permute.default(view_318, [1, 0])
    mm_21: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_193, view_222);  permute_193 = None
    permute_194: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_21, [1, 0]);  mm_21 = None
    sum_49: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_318, [0], True);  view_318 = None
    view_319: "f32[4096]" = torch.ops.aten.view.default(sum_49, [4096]);  sum_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_134: "f32[4096]" = torch.ops.aten.add.Tensor(view_291, view_319);  view_291 = view_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    permute_195: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_194, [1, 0]);  permute_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_135: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(permute_162, permute_195);  permute_162 = permute_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_320: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_20, [1, 512, 4096]);  mm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_136: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_149, view_320);  mul_149 = view_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_321: "f32[512, 4096]" = torch.ops.aten.view.default(view_316, [512, 4096]);  view_316 = None
    mm_22: "f32[512, 4096]" = torch.ops.aten.mm.default(view_321, permute_163)
    permute_197: "f32[4096, 512]" = torch.ops.aten.permute.default(view_321, [1, 0])
    mm_23: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_197, view_222);  permute_197 = None
    permute_198: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_23, [1, 0]);  mm_23 = None
    sum_50: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_321, [0], True);  view_321 = None
    view_322: "f32[4096]" = torch.ops.aten.view.default(sum_50, [4096]);  sum_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_137: "f32[4096]" = torch.ops.aten.add.Tensor(view_294, view_322);  view_294 = view_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    permute_199: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_198, [1, 0]);  permute_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_138: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(permute_166, permute_199);  permute_166 = permute_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_323: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_22, [1, 512, 4096]);  mm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_139: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_136, view_323);  add_136 = view_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_324: "f32[512, 4096]" = torch.ops.aten.view.default(view_317, [512, 4096]);  view_317 = None
    mm_24: "f32[512, 4096]" = torch.ops.aten.mm.default(view_324, permute_167)
    permute_201: "f32[4096, 512]" = torch.ops.aten.permute.default(view_324, [1, 0])
    mm_25: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_201, view_222);  permute_201 = view_222 = None
    permute_202: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_25, [1, 0]);  mm_25 = None
    sum_51: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_324, [0], True);  view_324 = None
    view_325: "f32[4096]" = torch.ops.aten.view.default(sum_51, [4096]);  sum_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_140: "f32[4096]" = torch.ops.aten.add.Tensor(view_297, view_325);  view_297 = view_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    permute_203: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_202, [1, 0]);  permute_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_141: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(permute_170, permute_203);  permute_170 = permute_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_326: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_24, [1, 512, 4096]);  mm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_142: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_139, view_326);  add_139 = view_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    mul_154: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_142, primals_22)
    mul_155: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_154, 4096)
    sum_52: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_154, [2], True)
    mul_156: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_154, mul_81);  mul_154 = None
    sum_53: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_156, [2], True);  mul_156 = None
    mul_157: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_81, sum_53);  sum_53 = None
    sub_61: "f32[1, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_155, sum_52);  mul_155 = sum_52 = None
    sub_62: "f32[1, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_61, mul_157);  sub_61 = mul_157 = None
    mul_158: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(div_36, sub_62);  div_36 = sub_62 = None
    mul_159: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_142, mul_81);  mul_81 = None
    sum_54: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_159, [0, 1]);  mul_159 = None
    sum_55: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_142, [0, 1]);  add_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_143: "f32[4096]" = torch.ops.aten.add.Tensor(add_121, sum_54);  add_121 = sum_54 = None
    add_144: "f32[4096]" = torch.ops.aten.add.Tensor(add_122, sum_55);  add_122 = sum_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_327: "f32[512, 4096]" = torch.ops.aten.view.default(mul_158, [512, 4096])
    mm_26: "f32[512, 16384]" = torch.ops.aten.mm.default(view_327, permute_138)
    permute_205: "f32[4096, 512]" = torch.ops.aten.permute.default(view_327, [1, 0])
    mm_27: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_205, view_220);  permute_205 = view_220 = None
    permute_206: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_27, [1, 0]);  mm_27 = None
    sum_56: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_327, [0], True);  view_327 = None
    view_328: "f32[4096]" = torch.ops.aten.view.default(sum_56, [4096]);  sum_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_145: "f32[4096]" = torch.ops.aten.add.Tensor(add_123, view_328);  add_123 = view_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    permute_207: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_206, [1, 0]);  permute_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_146: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_124, permute_207);  add_124 = permute_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_329: "f32[1, 512, 16384]" = torch.ops.aten.view.default(mm_26, [1, 512, 16384]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_160: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(view_329, mul_77);  mul_77 = None
    mul_161: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(view_329, add_90);  view_329 = add_90 = None
    alias_32: "f32[1, 512, 16384]" = torch.ops.aten.alias.default(alias_19);  alias_19 = None
    mul_162: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(alias_32, alias_32);  alias_32 = None
    sub_63: "f32[1, 512, 16384]" = torch.ops.aten.sub.Tensor(1, mul_162);  mul_162 = None
    mul_163: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_160, sub_63);  mul_160 = sub_63 = None
    mul_164: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_163, 0.7978845608028654);  mul_163 = None
    mul_165: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_164, 0.044715)
    pow_15: "f32[1, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_219, 2.0);  view_219 = None
    mul_166: "f32[1, 512, 16384]" = torch.ops.aten.mul.Scalar(pow_15, 3.0);  pow_15 = None
    mul_167: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_165, mul_166);  mul_165 = mul_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_147: "f32[1, 512, 16384]" = torch.ops.aten.add.Tensor(mul_164, mul_167);  mul_164 = mul_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_168: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_161, 0.5);  mul_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_148: "f32[1, 512, 16384]" = torch.ops.aten.add.Tensor(add_147, mul_168);  add_147 = mul_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_330: "f32[512, 16384]" = torch.ops.aten.view.default(add_148, [512, 16384]);  add_148 = None
    mm_28: "f32[512, 4096]" = torch.ops.aten.mm.default(view_330, permute_142)
    permute_209: "f32[16384, 512]" = torch.ops.aten.permute.default(view_330, [1, 0])
    mm_29: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_209, view_218);  permute_209 = view_218 = None
    permute_210: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_29, [1, 0]);  mm_29 = None
    sum_57: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_330, [0], True);  view_330 = None
    view_331: "f32[16384]" = torch.ops.aten.view.default(sum_57, [16384]);  sum_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_149: "f32[16384]" = torch.ops.aten.add.Tensor(add_127, view_331);  add_127 = view_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    permute_211: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_210, [1, 0]);  permute_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_150: "f32[16384, 4096]" = torch.ops.aten.add.Tensor(add_128, permute_211);  add_128 = permute_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_332: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_28, [1, 512, 4096]);  mm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_151: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_158, view_332);  mul_158 = view_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    mul_170: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_151, primals_16)
    mul_171: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_170, 4096)
    sum_58: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_170, [2], True)
    mul_172: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_170, mul_75);  mul_170 = None
    sum_59: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_172, [2], True);  mul_172 = None
    mul_173: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_75, sum_59);  sum_59 = None
    sub_65: "f32[1, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_171, sum_58);  mul_171 = sum_58 = None
    sub_66: "f32[1, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_65, mul_173);  sub_65 = mul_173 = None
    mul_174: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(div_37, sub_66);  div_37 = sub_66 = None
    mul_175: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_151, mul_75);  mul_75 = None
    sum_60: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_175, [0, 1]);  mul_175 = None
    sum_61: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_151, [0, 1]);  add_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_152: "f32[4096]" = torch.ops.aten.add.Tensor(add_130, sum_60);  add_130 = sum_60 = None
    add_153: "f32[4096]" = torch.ops.aten.add.Tensor(add_131, sum_61);  add_131 = sum_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_333: "f32[512, 4096]" = torch.ops.aten.view.default(mul_174, [512, 4096])
    mm_30: "f32[512, 4096]" = torch.ops.aten.mm.default(view_333, permute_146)
    permute_213: "f32[4096, 512]" = torch.ops.aten.permute.default(view_333, [1, 0])
    mm_31: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_213, view_216);  permute_213 = view_216 = None
    permute_214: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_31, [1, 0]);  mm_31 = None
    sum_62: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_333, [0], True);  view_333 = None
    view_334: "f32[4096]" = torch.ops.aten.view.default(sum_62, [4096]);  sum_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_154: "f32[4096]" = torch.ops.aten.add.Tensor(add_132, view_334);  add_132 = view_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    permute_215: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_214, [1, 0]);  permute_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_155: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_133, permute_215);  add_133 = permute_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_335: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_30, [1, 512, 4096]);  mm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    view_336: "f32[1, 512, 64, 64]" = torch.ops.aten.view.default(view_335, [1, 512, 64, 64]);  view_335 = None
    permute_216: "f32[1, 64, 512, 64]" = torch.ops.aten.permute.default(view_336, [0, 2, 1, 3]);  view_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_337: "f32[64, 512, 64]" = torch.ops.aten.view.default(permute_216, [64, 512, 64]);  permute_216 = None
    bmm_32: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(permute_217, view_337);  permute_217 = None
    bmm_33: "f32[64, 512, 512]" = torch.ops.aten.bmm.default(view_337, permute_218);  view_337 = permute_218 = None
    view_338: "f32[1, 64, 512, 64]" = torch.ops.aten.view.default(bmm_32, [1, 64, 512, 64]);  bmm_32 = None
    view_339: "f32[1, 64, 512, 512]" = torch.ops.aten.view.default(bmm_33, [1, 64, 512, 512]);  bmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    mul_176: "f32[1, 64, 512, 512]" = torch.ops.aten.mul.Tensor(view_339, alias_33);  view_339 = None
    sum_63: "f32[1, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_176, [-1], True)
    mul_177: "f32[1, 64, 512, 512]" = torch.ops.aten.mul.Tensor(alias_33, sum_63);  alias_33 = sum_63 = None
    sub_67: "f32[1, 64, 512, 512]" = torch.ops.aten.sub.Tensor(mul_176, mul_177);  mul_176 = mul_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:332, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_38: "f32[1, 64, 512, 512]" = torch.ops.aten.div.Tensor(sub_67, 8.0);  sub_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_340: "f32[64, 512, 512]" = torch.ops.aten.view.default(div_38, [64, 512, 512]);  div_38 = None
    bmm_34: "f32[64, 64, 512]" = torch.ops.aten.bmm.default(permute_219, view_340);  permute_219 = None
    bmm_35: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(view_340, permute_220);  view_340 = permute_220 = None
    view_341: "f32[1, 64, 64, 512]" = torch.ops.aten.view.default(bmm_34, [1, 64, 64, 512]);  bmm_34 = None
    view_342: "f32[1, 64, 512, 64]" = torch.ops.aten.view.default(bmm_35, [1, 64, 512, 64]);  bmm_35 = None
    permute_221: "f32[1, 64, 512, 64]" = torch.ops.aten.permute.default(view_341, [0, 1, 3, 2]);  view_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_222: "f32[1, 512, 64, 64]" = torch.ops.aten.permute.default(view_338, [0, 2, 1, 3]);  view_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_43: "f32[1, 512, 64, 64]" = torch.ops.aten.clone.default(permute_222, memory_format = torch.contiguous_format);  permute_222 = None
    view_343: "f32[1, 512, 4096]" = torch.ops.aten.view.default(clone_43, [1, 512, 4096]);  clone_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_223: "f32[1, 512, 64, 64]" = torch.ops.aten.permute.default(permute_221, [0, 2, 1, 3]);  permute_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_344: "f32[1, 512, 4096]" = torch.ops.aten.view.default(permute_223, [1, 512, 4096]);  permute_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_224: "f32[1, 512, 64, 64]" = torch.ops.aten.permute.default(view_342, [0, 2, 1, 3]);  view_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_44: "f32[1, 512, 64, 64]" = torch.ops.aten.clone.default(permute_224, memory_format = torch.contiguous_format);  permute_224 = None
    view_345: "f32[1, 512, 4096]" = torch.ops.aten.view.default(clone_44, [1, 512, 4096]);  clone_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_346: "f32[512, 4096]" = torch.ops.aten.view.default(view_343, [512, 4096]);  view_343 = None
    mm_32: "f32[512, 4096]" = torch.ops.aten.mm.default(view_346, permute_159)
    permute_226: "f32[4096, 512]" = torch.ops.aten.permute.default(view_346, [1, 0])
    mm_33: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_226, view_200);  permute_226 = None
    permute_227: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_33, [1, 0]);  mm_33 = None
    sum_64: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_346, [0], True);  view_346 = None
    view_347: "f32[4096]" = torch.ops.aten.view.default(sum_64, [4096]);  sum_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_156: "f32[4096]" = torch.ops.aten.add.Tensor(add_134, view_347);  add_134 = view_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    permute_228: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_227, [1, 0]);  permute_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_157: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_135, permute_228);  add_135 = permute_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_348: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_32, [1, 512, 4096]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_158: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_174, view_348);  mul_174 = view_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_349: "f32[512, 4096]" = torch.ops.aten.view.default(view_344, [512, 4096]);  view_344 = None
    mm_34: "f32[512, 4096]" = torch.ops.aten.mm.default(view_349, permute_163)
    permute_230: "f32[4096, 512]" = torch.ops.aten.permute.default(view_349, [1, 0])
    mm_35: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_230, view_200);  permute_230 = None
    permute_231: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_35, [1, 0]);  mm_35 = None
    sum_65: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_349, [0], True);  view_349 = None
    view_350: "f32[4096]" = torch.ops.aten.view.default(sum_65, [4096]);  sum_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_159: "f32[4096]" = torch.ops.aten.add.Tensor(add_137, view_350);  add_137 = view_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    permute_232: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_231, [1, 0]);  permute_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_160: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_138, permute_232);  add_138 = permute_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_351: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_34, [1, 512, 4096]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_161: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_158, view_351);  add_158 = view_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_352: "f32[512, 4096]" = torch.ops.aten.view.default(view_345, [512, 4096]);  view_345 = None
    mm_36: "f32[512, 4096]" = torch.ops.aten.mm.default(view_352, permute_167)
    permute_234: "f32[4096, 512]" = torch.ops.aten.permute.default(view_352, [1, 0])
    mm_37: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_234, view_200);  permute_234 = view_200 = None
    permute_235: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_37, [1, 0]);  mm_37 = None
    sum_66: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_352, [0], True);  view_352 = None
    view_353: "f32[4096]" = torch.ops.aten.view.default(sum_66, [4096]);  sum_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_162: "f32[4096]" = torch.ops.aten.add.Tensor(add_140, view_353);  add_140 = view_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    permute_236: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_235, [1, 0]);  permute_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_163: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_141, permute_236);  add_141 = permute_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_354: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_36, [1, 512, 4096]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_164: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_161, view_354);  add_161 = view_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    mul_179: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_164, primals_22)
    mul_180: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_179, 4096)
    sum_67: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_179, [2], True)
    mul_181: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_179, mul_73);  mul_179 = None
    sum_68: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_181, [2], True);  mul_181 = None
    mul_182: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_73, sum_68);  sum_68 = None
    sub_69: "f32[1, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_180, sum_67);  mul_180 = sum_67 = None
    sub_70: "f32[1, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_69, mul_182);  sub_69 = mul_182 = None
    mul_183: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(div_39, sub_70);  div_39 = sub_70 = None
    mul_184: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_164, mul_73);  mul_73 = None
    sum_69: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_184, [0, 1]);  mul_184 = None
    sum_70: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_164, [0, 1]);  add_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_165: "f32[4096]" = torch.ops.aten.add.Tensor(add_143, sum_69);  add_143 = sum_69 = None
    add_166: "f32[4096]" = torch.ops.aten.add.Tensor(add_144, sum_70);  add_144 = sum_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_355: "f32[512, 4096]" = torch.ops.aten.view.default(mul_183, [512, 4096])
    mm_38: "f32[512, 16384]" = torch.ops.aten.mm.default(view_355, permute_138)
    permute_238: "f32[4096, 512]" = torch.ops.aten.permute.default(view_355, [1, 0])
    mm_39: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_238, view_198);  permute_238 = view_198 = None
    permute_239: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_39, [1, 0]);  mm_39 = None
    sum_71: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_355, [0], True);  view_355 = None
    view_356: "f32[4096]" = torch.ops.aten.view.default(sum_71, [4096]);  sum_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_167: "f32[4096]" = torch.ops.aten.add.Tensor(add_145, view_356);  add_145 = view_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    permute_240: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_239, [1, 0]);  permute_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_168: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_146, permute_240);  add_146 = permute_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_357: "f32[1, 512, 16384]" = torch.ops.aten.view.default(mm_38, [1, 512, 16384]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_185: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(view_357, mul_69);  mul_69 = None
    mul_186: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(view_357, add_81);  view_357 = add_81 = None
    alias_34: "f32[1, 512, 16384]" = torch.ops.aten.alias.default(alias_17);  alias_17 = None
    mul_187: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(alias_34, alias_34);  alias_34 = None
    sub_71: "f32[1, 512, 16384]" = torch.ops.aten.sub.Tensor(1, mul_187);  mul_187 = None
    mul_188: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_185, sub_71);  mul_185 = sub_71 = None
    mul_189: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_188, 0.7978845608028654);  mul_188 = None
    mul_190: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_189, 0.044715)
    pow_16: "f32[1, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_197, 2.0);  view_197 = None
    mul_191: "f32[1, 512, 16384]" = torch.ops.aten.mul.Scalar(pow_16, 3.0);  pow_16 = None
    mul_192: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_190, mul_191);  mul_190 = mul_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_169: "f32[1, 512, 16384]" = torch.ops.aten.add.Tensor(mul_189, mul_192);  mul_189 = mul_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_193: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_186, 0.5);  mul_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_170: "f32[1, 512, 16384]" = torch.ops.aten.add.Tensor(add_169, mul_193);  add_169 = mul_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_358: "f32[512, 16384]" = torch.ops.aten.view.default(add_170, [512, 16384]);  add_170 = None
    mm_40: "f32[512, 4096]" = torch.ops.aten.mm.default(view_358, permute_142)
    permute_242: "f32[16384, 512]" = torch.ops.aten.permute.default(view_358, [1, 0])
    mm_41: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_242, view_196);  permute_242 = view_196 = None
    permute_243: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_41, [1, 0]);  mm_41 = None
    sum_72: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_358, [0], True);  view_358 = None
    view_359: "f32[16384]" = torch.ops.aten.view.default(sum_72, [16384]);  sum_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_171: "f32[16384]" = torch.ops.aten.add.Tensor(add_149, view_359);  add_149 = view_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    permute_244: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_243, [1, 0]);  permute_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_172: "f32[16384, 4096]" = torch.ops.aten.add.Tensor(add_150, permute_244);  add_150 = permute_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_360: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_40, [1, 512, 4096]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_173: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_183, view_360);  mul_183 = view_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    mul_195: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_173, primals_16)
    mul_196: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_195, 4096)
    sum_73: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_195, [2], True)
    mul_197: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_195, mul_67);  mul_195 = None
    sum_74: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_197, [2], True);  mul_197 = None
    mul_198: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_67, sum_74);  sum_74 = None
    sub_73: "f32[1, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_196, sum_73);  mul_196 = sum_73 = None
    sub_74: "f32[1, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_73, mul_198);  sub_73 = mul_198 = None
    mul_199: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(div_40, sub_74);  div_40 = sub_74 = None
    mul_200: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_173, mul_67);  mul_67 = None
    sum_75: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_200, [0, 1]);  mul_200 = None
    sum_76: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_173, [0, 1]);  add_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_174: "f32[4096]" = torch.ops.aten.add.Tensor(add_152, sum_75);  add_152 = sum_75 = None
    add_175: "f32[4096]" = torch.ops.aten.add.Tensor(add_153, sum_76);  add_153 = sum_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_361: "f32[512, 4096]" = torch.ops.aten.view.default(mul_199, [512, 4096])
    mm_42: "f32[512, 4096]" = torch.ops.aten.mm.default(view_361, permute_146)
    permute_246: "f32[4096, 512]" = torch.ops.aten.permute.default(view_361, [1, 0])
    mm_43: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_246, view_194);  permute_246 = view_194 = None
    permute_247: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_43, [1, 0]);  mm_43 = None
    sum_77: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_361, [0], True);  view_361 = None
    view_362: "f32[4096]" = torch.ops.aten.view.default(sum_77, [4096]);  sum_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_176: "f32[4096]" = torch.ops.aten.add.Tensor(add_154, view_362);  add_154 = view_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    permute_248: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_247, [1, 0]);  permute_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_177: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_155, permute_248);  add_155 = permute_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_363: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_42, [1, 512, 4096]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    view_364: "f32[1, 512, 64, 64]" = torch.ops.aten.view.default(view_363, [1, 512, 64, 64]);  view_363 = None
    permute_249: "f32[1, 64, 512, 64]" = torch.ops.aten.permute.default(view_364, [0, 2, 1, 3]);  view_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_365: "f32[64, 512, 64]" = torch.ops.aten.view.default(permute_249, [64, 512, 64]);  permute_249 = None
    bmm_36: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(permute_250, view_365);  permute_250 = None
    bmm_37: "f32[64, 512, 512]" = torch.ops.aten.bmm.default(view_365, permute_251);  view_365 = permute_251 = None
    view_366: "f32[1, 64, 512, 64]" = torch.ops.aten.view.default(bmm_36, [1, 64, 512, 64]);  bmm_36 = None
    view_367: "f32[1, 64, 512, 512]" = torch.ops.aten.view.default(bmm_37, [1, 64, 512, 512]);  bmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    mul_201: "f32[1, 64, 512, 512]" = torch.ops.aten.mul.Tensor(view_367, alias_35);  view_367 = None
    sum_78: "f32[1, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_201, [-1], True)
    mul_202: "f32[1, 64, 512, 512]" = torch.ops.aten.mul.Tensor(alias_35, sum_78);  alias_35 = sum_78 = None
    sub_75: "f32[1, 64, 512, 512]" = torch.ops.aten.sub.Tensor(mul_201, mul_202);  mul_201 = mul_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:332, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_41: "f32[1, 64, 512, 512]" = torch.ops.aten.div.Tensor(sub_75, 8.0);  sub_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_368: "f32[64, 512, 512]" = torch.ops.aten.view.default(div_41, [64, 512, 512]);  div_41 = None
    bmm_38: "f32[64, 64, 512]" = torch.ops.aten.bmm.default(permute_252, view_368);  permute_252 = None
    bmm_39: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(view_368, permute_253);  view_368 = permute_253 = None
    view_369: "f32[1, 64, 64, 512]" = torch.ops.aten.view.default(bmm_38, [1, 64, 64, 512]);  bmm_38 = None
    view_370: "f32[1, 64, 512, 64]" = torch.ops.aten.view.default(bmm_39, [1, 64, 512, 64]);  bmm_39 = None
    permute_254: "f32[1, 64, 512, 64]" = torch.ops.aten.permute.default(view_369, [0, 1, 3, 2]);  view_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_255: "f32[1, 512, 64, 64]" = torch.ops.aten.permute.default(view_366, [0, 2, 1, 3]);  view_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_45: "f32[1, 512, 64, 64]" = torch.ops.aten.clone.default(permute_255, memory_format = torch.contiguous_format);  permute_255 = None
    view_371: "f32[1, 512, 4096]" = torch.ops.aten.view.default(clone_45, [1, 512, 4096]);  clone_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_256: "f32[1, 512, 64, 64]" = torch.ops.aten.permute.default(permute_254, [0, 2, 1, 3]);  permute_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_372: "f32[1, 512, 4096]" = torch.ops.aten.view.default(permute_256, [1, 512, 4096]);  permute_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_257: "f32[1, 512, 64, 64]" = torch.ops.aten.permute.default(view_370, [0, 2, 1, 3]);  view_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_46: "f32[1, 512, 64, 64]" = torch.ops.aten.clone.default(permute_257, memory_format = torch.contiguous_format);  permute_257 = None
    view_373: "f32[1, 512, 4096]" = torch.ops.aten.view.default(clone_46, [1, 512, 4096]);  clone_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_374: "f32[512, 4096]" = torch.ops.aten.view.default(view_371, [512, 4096]);  view_371 = None
    mm_44: "f32[512, 4096]" = torch.ops.aten.mm.default(view_374, permute_159)
    permute_259: "f32[4096, 512]" = torch.ops.aten.permute.default(view_374, [1, 0])
    mm_45: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_259, view_178);  permute_259 = None
    permute_260: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_45, [1, 0]);  mm_45 = None
    sum_79: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_374, [0], True);  view_374 = None
    view_375: "f32[4096]" = torch.ops.aten.view.default(sum_79, [4096]);  sum_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_178: "f32[4096]" = torch.ops.aten.add.Tensor(add_156, view_375);  add_156 = view_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    permute_261: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_260, [1, 0]);  permute_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_179: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_157, permute_261);  add_157 = permute_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_376: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_44, [1, 512, 4096]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_180: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_199, view_376);  mul_199 = view_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_377: "f32[512, 4096]" = torch.ops.aten.view.default(view_372, [512, 4096]);  view_372 = None
    mm_46: "f32[512, 4096]" = torch.ops.aten.mm.default(view_377, permute_163)
    permute_263: "f32[4096, 512]" = torch.ops.aten.permute.default(view_377, [1, 0])
    mm_47: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_263, view_178);  permute_263 = None
    permute_264: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_47, [1, 0]);  mm_47 = None
    sum_80: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_377, [0], True);  view_377 = None
    view_378: "f32[4096]" = torch.ops.aten.view.default(sum_80, [4096]);  sum_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_181: "f32[4096]" = torch.ops.aten.add.Tensor(add_159, view_378);  add_159 = view_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    permute_265: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_264, [1, 0]);  permute_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_182: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_160, permute_265);  add_160 = permute_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_379: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_46, [1, 512, 4096]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_183: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_180, view_379);  add_180 = view_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_380: "f32[512, 4096]" = torch.ops.aten.view.default(view_373, [512, 4096]);  view_373 = None
    mm_48: "f32[512, 4096]" = torch.ops.aten.mm.default(view_380, permute_167)
    permute_267: "f32[4096, 512]" = torch.ops.aten.permute.default(view_380, [1, 0])
    mm_49: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_267, view_178);  permute_267 = view_178 = None
    permute_268: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_49, [1, 0]);  mm_49 = None
    sum_81: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_380, [0], True);  view_380 = None
    view_381: "f32[4096]" = torch.ops.aten.view.default(sum_81, [4096]);  sum_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_184: "f32[4096]" = torch.ops.aten.add.Tensor(add_162, view_381);  add_162 = view_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    permute_269: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_268, [1, 0]);  permute_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_185: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_163, permute_269);  add_163 = permute_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_382: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_48, [1, 512, 4096]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_186: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_183, view_382);  add_183 = view_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    mul_204: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_186, primals_22)
    mul_205: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_204, 4096)
    sum_82: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_204, [2], True)
    mul_206: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_204, mul_65);  mul_204 = None
    sum_83: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_206, [2], True);  mul_206 = None
    mul_207: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_65, sum_83);  sum_83 = None
    sub_77: "f32[1, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_205, sum_82);  mul_205 = sum_82 = None
    sub_78: "f32[1, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_77, mul_207);  sub_77 = mul_207 = None
    mul_208: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(div_42, sub_78);  div_42 = sub_78 = None
    mul_209: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_186, mul_65);  mul_65 = None
    sum_84: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_209, [0, 1]);  mul_209 = None
    sum_85: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_186, [0, 1]);  add_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_187: "f32[4096]" = torch.ops.aten.add.Tensor(add_165, sum_84);  add_165 = sum_84 = None
    add_188: "f32[4096]" = torch.ops.aten.add.Tensor(add_166, sum_85);  add_166 = sum_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_383: "f32[512, 4096]" = torch.ops.aten.view.default(mul_208, [512, 4096])
    mm_50: "f32[512, 16384]" = torch.ops.aten.mm.default(view_383, permute_138)
    permute_271: "f32[4096, 512]" = torch.ops.aten.permute.default(view_383, [1, 0])
    mm_51: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_271, view_176);  permute_271 = view_176 = None
    permute_272: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_51, [1, 0]);  mm_51 = None
    sum_86: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_383, [0], True);  view_383 = None
    view_384: "f32[4096]" = torch.ops.aten.view.default(sum_86, [4096]);  sum_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_189: "f32[4096]" = torch.ops.aten.add.Tensor(add_167, view_384);  add_167 = view_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    permute_273: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_272, [1, 0]);  permute_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_190: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_168, permute_273);  add_168 = permute_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_385: "f32[1, 512, 16384]" = torch.ops.aten.view.default(mm_50, [1, 512, 16384]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_210: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(view_385, mul_61);  mul_61 = None
    mul_211: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(view_385, add_72);  view_385 = add_72 = None
    alias_36: "f32[1, 512, 16384]" = torch.ops.aten.alias.default(alias_15);  alias_15 = None
    mul_212: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(alias_36, alias_36);  alias_36 = None
    sub_79: "f32[1, 512, 16384]" = torch.ops.aten.sub.Tensor(1, mul_212);  mul_212 = None
    mul_213: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_210, sub_79);  mul_210 = sub_79 = None
    mul_214: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_213, 0.7978845608028654);  mul_213 = None
    mul_215: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_214, 0.044715)
    pow_17: "f32[1, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_175, 2.0);  view_175 = None
    mul_216: "f32[1, 512, 16384]" = torch.ops.aten.mul.Scalar(pow_17, 3.0);  pow_17 = None
    mul_217: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_215, mul_216);  mul_215 = mul_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_191: "f32[1, 512, 16384]" = torch.ops.aten.add.Tensor(mul_214, mul_217);  mul_214 = mul_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_218: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_211, 0.5);  mul_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_192: "f32[1, 512, 16384]" = torch.ops.aten.add.Tensor(add_191, mul_218);  add_191 = mul_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_386: "f32[512, 16384]" = torch.ops.aten.view.default(add_192, [512, 16384]);  add_192 = None
    mm_52: "f32[512, 4096]" = torch.ops.aten.mm.default(view_386, permute_142)
    permute_275: "f32[16384, 512]" = torch.ops.aten.permute.default(view_386, [1, 0])
    mm_53: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_275, view_174);  permute_275 = view_174 = None
    permute_276: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_53, [1, 0]);  mm_53 = None
    sum_87: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_386, [0], True);  view_386 = None
    view_387: "f32[16384]" = torch.ops.aten.view.default(sum_87, [16384]);  sum_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_193: "f32[16384]" = torch.ops.aten.add.Tensor(add_171, view_387);  add_171 = view_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    permute_277: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_276, [1, 0]);  permute_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_194: "f32[16384, 4096]" = torch.ops.aten.add.Tensor(add_172, permute_277);  add_172 = permute_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_388: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_52, [1, 512, 4096]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_195: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_208, view_388);  mul_208 = view_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    mul_220: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_195, primals_16)
    mul_221: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_220, 4096)
    sum_88: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_220, [2], True)
    mul_222: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_220, mul_59);  mul_220 = None
    sum_89: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_222, [2], True);  mul_222 = None
    mul_223: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_59, sum_89);  sum_89 = None
    sub_81: "f32[1, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_221, sum_88);  mul_221 = sum_88 = None
    sub_82: "f32[1, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_81, mul_223);  sub_81 = mul_223 = None
    mul_224: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(div_43, sub_82);  div_43 = sub_82 = None
    mul_225: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_195, mul_59);  mul_59 = None
    sum_90: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_225, [0, 1]);  mul_225 = None
    sum_91: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_195, [0, 1]);  add_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_196: "f32[4096]" = torch.ops.aten.add.Tensor(add_174, sum_90);  add_174 = sum_90 = None
    add_197: "f32[4096]" = torch.ops.aten.add.Tensor(add_175, sum_91);  add_175 = sum_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_389: "f32[512, 4096]" = torch.ops.aten.view.default(mul_224, [512, 4096])
    mm_54: "f32[512, 4096]" = torch.ops.aten.mm.default(view_389, permute_146)
    permute_279: "f32[4096, 512]" = torch.ops.aten.permute.default(view_389, [1, 0])
    mm_55: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_279, view_172);  permute_279 = view_172 = None
    permute_280: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_55, [1, 0]);  mm_55 = None
    sum_92: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_389, [0], True);  view_389 = None
    view_390: "f32[4096]" = torch.ops.aten.view.default(sum_92, [4096]);  sum_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_198: "f32[4096]" = torch.ops.aten.add.Tensor(add_176, view_390);  add_176 = view_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    permute_281: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_280, [1, 0]);  permute_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_199: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_177, permute_281);  add_177 = permute_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_391: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_54, [1, 512, 4096]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    view_392: "f32[1, 512, 64, 64]" = torch.ops.aten.view.default(view_391, [1, 512, 64, 64]);  view_391 = None
    permute_282: "f32[1, 64, 512, 64]" = torch.ops.aten.permute.default(view_392, [0, 2, 1, 3]);  view_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_393: "f32[64, 512, 64]" = torch.ops.aten.view.default(permute_282, [64, 512, 64]);  permute_282 = None
    bmm_40: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(permute_283, view_393);  permute_283 = None
    bmm_41: "f32[64, 512, 512]" = torch.ops.aten.bmm.default(view_393, permute_284);  view_393 = permute_284 = None
    view_394: "f32[1, 64, 512, 64]" = torch.ops.aten.view.default(bmm_40, [1, 64, 512, 64]);  bmm_40 = None
    view_395: "f32[1, 64, 512, 512]" = torch.ops.aten.view.default(bmm_41, [1, 64, 512, 512]);  bmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    mul_226: "f32[1, 64, 512, 512]" = torch.ops.aten.mul.Tensor(view_395, alias_37);  view_395 = None
    sum_93: "f32[1, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_226, [-1], True)
    mul_227: "f32[1, 64, 512, 512]" = torch.ops.aten.mul.Tensor(alias_37, sum_93);  alias_37 = sum_93 = None
    sub_83: "f32[1, 64, 512, 512]" = torch.ops.aten.sub.Tensor(mul_226, mul_227);  mul_226 = mul_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:332, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_44: "f32[1, 64, 512, 512]" = torch.ops.aten.div.Tensor(sub_83, 8.0);  sub_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_396: "f32[64, 512, 512]" = torch.ops.aten.view.default(div_44, [64, 512, 512]);  div_44 = None
    bmm_42: "f32[64, 64, 512]" = torch.ops.aten.bmm.default(permute_285, view_396);  permute_285 = None
    bmm_43: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(view_396, permute_286);  view_396 = permute_286 = None
    view_397: "f32[1, 64, 64, 512]" = torch.ops.aten.view.default(bmm_42, [1, 64, 64, 512]);  bmm_42 = None
    view_398: "f32[1, 64, 512, 64]" = torch.ops.aten.view.default(bmm_43, [1, 64, 512, 64]);  bmm_43 = None
    permute_287: "f32[1, 64, 512, 64]" = torch.ops.aten.permute.default(view_397, [0, 1, 3, 2]);  view_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_288: "f32[1, 512, 64, 64]" = torch.ops.aten.permute.default(view_394, [0, 2, 1, 3]);  view_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_47: "f32[1, 512, 64, 64]" = torch.ops.aten.clone.default(permute_288, memory_format = torch.contiguous_format);  permute_288 = None
    view_399: "f32[1, 512, 4096]" = torch.ops.aten.view.default(clone_47, [1, 512, 4096]);  clone_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_289: "f32[1, 512, 64, 64]" = torch.ops.aten.permute.default(permute_287, [0, 2, 1, 3]);  permute_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_400: "f32[1, 512, 4096]" = torch.ops.aten.view.default(permute_289, [1, 512, 4096]);  permute_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_290: "f32[1, 512, 64, 64]" = torch.ops.aten.permute.default(view_398, [0, 2, 1, 3]);  view_398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_48: "f32[1, 512, 64, 64]" = torch.ops.aten.clone.default(permute_290, memory_format = torch.contiguous_format);  permute_290 = None
    view_401: "f32[1, 512, 4096]" = torch.ops.aten.view.default(clone_48, [1, 512, 4096]);  clone_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_402: "f32[512, 4096]" = torch.ops.aten.view.default(view_399, [512, 4096]);  view_399 = None
    mm_56: "f32[512, 4096]" = torch.ops.aten.mm.default(view_402, permute_159)
    permute_292: "f32[4096, 512]" = torch.ops.aten.permute.default(view_402, [1, 0])
    mm_57: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_292, view_156);  permute_292 = None
    permute_293: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_57, [1, 0]);  mm_57 = None
    sum_94: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_402, [0], True);  view_402 = None
    view_403: "f32[4096]" = torch.ops.aten.view.default(sum_94, [4096]);  sum_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_200: "f32[4096]" = torch.ops.aten.add.Tensor(add_178, view_403);  add_178 = view_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    permute_294: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_293, [1, 0]);  permute_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_201: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_179, permute_294);  add_179 = permute_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_404: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_56, [1, 512, 4096]);  mm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_202: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_224, view_404);  mul_224 = view_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_405: "f32[512, 4096]" = torch.ops.aten.view.default(view_400, [512, 4096]);  view_400 = None
    mm_58: "f32[512, 4096]" = torch.ops.aten.mm.default(view_405, permute_163)
    permute_296: "f32[4096, 512]" = torch.ops.aten.permute.default(view_405, [1, 0])
    mm_59: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_296, view_156);  permute_296 = None
    permute_297: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_59, [1, 0]);  mm_59 = None
    sum_95: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_405, [0], True);  view_405 = None
    view_406: "f32[4096]" = torch.ops.aten.view.default(sum_95, [4096]);  sum_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_203: "f32[4096]" = torch.ops.aten.add.Tensor(add_181, view_406);  add_181 = view_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    permute_298: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_297, [1, 0]);  permute_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_204: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_182, permute_298);  add_182 = permute_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_407: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_58, [1, 512, 4096]);  mm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_205: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_202, view_407);  add_202 = view_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_408: "f32[512, 4096]" = torch.ops.aten.view.default(view_401, [512, 4096]);  view_401 = None
    mm_60: "f32[512, 4096]" = torch.ops.aten.mm.default(view_408, permute_167)
    permute_300: "f32[4096, 512]" = torch.ops.aten.permute.default(view_408, [1, 0])
    mm_61: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_300, view_156);  permute_300 = view_156 = None
    permute_301: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_61, [1, 0]);  mm_61 = None
    sum_96: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_408, [0], True);  view_408 = None
    view_409: "f32[4096]" = torch.ops.aten.view.default(sum_96, [4096]);  sum_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_206: "f32[4096]" = torch.ops.aten.add.Tensor(add_184, view_409);  add_184 = view_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    permute_302: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_301, [1, 0]);  permute_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_207: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_185, permute_302);  add_185 = permute_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_410: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_60, [1, 512, 4096]);  mm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_208: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_205, view_410);  add_205 = view_410 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    mul_229: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_208, primals_22)
    mul_230: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_229, 4096)
    sum_97: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_229, [2], True)
    mul_231: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_229, mul_57);  mul_229 = None
    sum_98: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_231, [2], True);  mul_231 = None
    mul_232: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_57, sum_98);  sum_98 = None
    sub_85: "f32[1, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_230, sum_97);  mul_230 = sum_97 = None
    sub_86: "f32[1, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_85, mul_232);  sub_85 = mul_232 = None
    mul_233: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(div_45, sub_86);  div_45 = sub_86 = None
    mul_234: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_208, mul_57);  mul_57 = None
    sum_99: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_234, [0, 1]);  mul_234 = None
    sum_100: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_208, [0, 1]);  add_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_209: "f32[4096]" = torch.ops.aten.add.Tensor(add_187, sum_99);  add_187 = sum_99 = None
    add_210: "f32[4096]" = torch.ops.aten.add.Tensor(add_188, sum_100);  add_188 = sum_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_411: "f32[512, 4096]" = torch.ops.aten.view.default(mul_233, [512, 4096])
    mm_62: "f32[512, 16384]" = torch.ops.aten.mm.default(view_411, permute_138)
    permute_304: "f32[4096, 512]" = torch.ops.aten.permute.default(view_411, [1, 0])
    mm_63: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_304, view_154);  permute_304 = view_154 = None
    permute_305: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_63, [1, 0]);  mm_63 = None
    sum_101: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_411, [0], True);  view_411 = None
    view_412: "f32[4096]" = torch.ops.aten.view.default(sum_101, [4096]);  sum_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_211: "f32[4096]" = torch.ops.aten.add.Tensor(add_189, view_412);  add_189 = view_412 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    permute_306: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_305, [1, 0]);  permute_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_212: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_190, permute_306);  add_190 = permute_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_413: "f32[1, 512, 16384]" = torch.ops.aten.view.default(mm_62, [1, 512, 16384]);  mm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_235: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(view_413, mul_53);  mul_53 = None
    mul_236: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(view_413, add_63);  view_413 = add_63 = None
    alias_38: "f32[1, 512, 16384]" = torch.ops.aten.alias.default(alias_13);  alias_13 = None
    mul_237: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(alias_38, alias_38);  alias_38 = None
    sub_87: "f32[1, 512, 16384]" = torch.ops.aten.sub.Tensor(1, mul_237);  mul_237 = None
    mul_238: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_235, sub_87);  mul_235 = sub_87 = None
    mul_239: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_238, 0.7978845608028654);  mul_238 = None
    mul_240: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_239, 0.044715)
    pow_18: "f32[1, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_153, 2.0);  view_153 = None
    mul_241: "f32[1, 512, 16384]" = torch.ops.aten.mul.Scalar(pow_18, 3.0);  pow_18 = None
    mul_242: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_240, mul_241);  mul_240 = mul_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_213: "f32[1, 512, 16384]" = torch.ops.aten.add.Tensor(mul_239, mul_242);  mul_239 = mul_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_243: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_236, 0.5);  mul_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_214: "f32[1, 512, 16384]" = torch.ops.aten.add.Tensor(add_213, mul_243);  add_213 = mul_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_414: "f32[512, 16384]" = torch.ops.aten.view.default(add_214, [512, 16384]);  add_214 = None
    mm_64: "f32[512, 4096]" = torch.ops.aten.mm.default(view_414, permute_142)
    permute_308: "f32[16384, 512]" = torch.ops.aten.permute.default(view_414, [1, 0])
    mm_65: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_308, view_152);  permute_308 = view_152 = None
    permute_309: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_65, [1, 0]);  mm_65 = None
    sum_102: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_414, [0], True);  view_414 = None
    view_415: "f32[16384]" = torch.ops.aten.view.default(sum_102, [16384]);  sum_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_215: "f32[16384]" = torch.ops.aten.add.Tensor(add_193, view_415);  add_193 = view_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    permute_310: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_309, [1, 0]);  permute_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_216: "f32[16384, 4096]" = torch.ops.aten.add.Tensor(add_194, permute_310);  add_194 = permute_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_416: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_64, [1, 512, 4096]);  mm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_217: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_233, view_416);  mul_233 = view_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    mul_245: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_217, primals_16)
    mul_246: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_245, 4096)
    sum_103: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_245, [2], True)
    mul_247: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_245, mul_51);  mul_245 = None
    sum_104: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_247, [2], True);  mul_247 = None
    mul_248: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_51, sum_104);  sum_104 = None
    sub_89: "f32[1, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_246, sum_103);  mul_246 = sum_103 = None
    sub_90: "f32[1, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_89, mul_248);  sub_89 = mul_248 = None
    mul_249: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(div_46, sub_90);  div_46 = sub_90 = None
    mul_250: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_217, mul_51);  mul_51 = None
    sum_105: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_250, [0, 1]);  mul_250 = None
    sum_106: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_217, [0, 1]);  add_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_218: "f32[4096]" = torch.ops.aten.add.Tensor(add_196, sum_105);  add_196 = sum_105 = None
    add_219: "f32[4096]" = torch.ops.aten.add.Tensor(add_197, sum_106);  add_197 = sum_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_417: "f32[512, 4096]" = torch.ops.aten.view.default(mul_249, [512, 4096])
    mm_66: "f32[512, 4096]" = torch.ops.aten.mm.default(view_417, permute_146)
    permute_312: "f32[4096, 512]" = torch.ops.aten.permute.default(view_417, [1, 0])
    mm_67: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_312, view_150);  permute_312 = view_150 = None
    permute_313: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_67, [1, 0]);  mm_67 = None
    sum_107: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_417, [0], True);  view_417 = None
    view_418: "f32[4096]" = torch.ops.aten.view.default(sum_107, [4096]);  sum_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_220: "f32[4096]" = torch.ops.aten.add.Tensor(add_198, view_418);  add_198 = view_418 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    permute_314: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_313, [1, 0]);  permute_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_221: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_199, permute_314);  add_199 = permute_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_419: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_66, [1, 512, 4096]);  mm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    view_420: "f32[1, 512, 64, 64]" = torch.ops.aten.view.default(view_419, [1, 512, 64, 64]);  view_419 = None
    permute_315: "f32[1, 64, 512, 64]" = torch.ops.aten.permute.default(view_420, [0, 2, 1, 3]);  view_420 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_421: "f32[64, 512, 64]" = torch.ops.aten.view.default(permute_315, [64, 512, 64]);  permute_315 = None
    bmm_44: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(permute_316, view_421);  permute_316 = None
    bmm_45: "f32[64, 512, 512]" = torch.ops.aten.bmm.default(view_421, permute_317);  view_421 = permute_317 = None
    view_422: "f32[1, 64, 512, 64]" = torch.ops.aten.view.default(bmm_44, [1, 64, 512, 64]);  bmm_44 = None
    view_423: "f32[1, 64, 512, 512]" = torch.ops.aten.view.default(bmm_45, [1, 64, 512, 512]);  bmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    mul_251: "f32[1, 64, 512, 512]" = torch.ops.aten.mul.Tensor(view_423, alias_39);  view_423 = None
    sum_108: "f32[1, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_251, [-1], True)
    mul_252: "f32[1, 64, 512, 512]" = torch.ops.aten.mul.Tensor(alias_39, sum_108);  alias_39 = sum_108 = None
    sub_91: "f32[1, 64, 512, 512]" = torch.ops.aten.sub.Tensor(mul_251, mul_252);  mul_251 = mul_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:332, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_47: "f32[1, 64, 512, 512]" = torch.ops.aten.div.Tensor(sub_91, 8.0);  sub_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_424: "f32[64, 512, 512]" = torch.ops.aten.view.default(div_47, [64, 512, 512]);  div_47 = None
    bmm_46: "f32[64, 64, 512]" = torch.ops.aten.bmm.default(permute_318, view_424);  permute_318 = None
    bmm_47: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(view_424, permute_319);  view_424 = permute_319 = None
    view_425: "f32[1, 64, 64, 512]" = torch.ops.aten.view.default(bmm_46, [1, 64, 64, 512]);  bmm_46 = None
    view_426: "f32[1, 64, 512, 64]" = torch.ops.aten.view.default(bmm_47, [1, 64, 512, 64]);  bmm_47 = None
    permute_320: "f32[1, 64, 512, 64]" = torch.ops.aten.permute.default(view_425, [0, 1, 3, 2]);  view_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_321: "f32[1, 512, 64, 64]" = torch.ops.aten.permute.default(view_422, [0, 2, 1, 3]);  view_422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_49: "f32[1, 512, 64, 64]" = torch.ops.aten.clone.default(permute_321, memory_format = torch.contiguous_format);  permute_321 = None
    view_427: "f32[1, 512, 4096]" = torch.ops.aten.view.default(clone_49, [1, 512, 4096]);  clone_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_322: "f32[1, 512, 64, 64]" = torch.ops.aten.permute.default(permute_320, [0, 2, 1, 3]);  permute_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_428: "f32[1, 512, 4096]" = torch.ops.aten.view.default(permute_322, [1, 512, 4096]);  permute_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_323: "f32[1, 512, 64, 64]" = torch.ops.aten.permute.default(view_426, [0, 2, 1, 3]);  view_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_50: "f32[1, 512, 64, 64]" = torch.ops.aten.clone.default(permute_323, memory_format = torch.contiguous_format);  permute_323 = None
    view_429: "f32[1, 512, 4096]" = torch.ops.aten.view.default(clone_50, [1, 512, 4096]);  clone_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_430: "f32[512, 4096]" = torch.ops.aten.view.default(view_427, [512, 4096]);  view_427 = None
    mm_68: "f32[512, 4096]" = torch.ops.aten.mm.default(view_430, permute_159)
    permute_325: "f32[4096, 512]" = torch.ops.aten.permute.default(view_430, [1, 0])
    mm_69: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_325, view_134);  permute_325 = None
    permute_326: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_69, [1, 0]);  mm_69 = None
    sum_109: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_430, [0], True);  view_430 = None
    view_431: "f32[4096]" = torch.ops.aten.view.default(sum_109, [4096]);  sum_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_222: "f32[4096]" = torch.ops.aten.add.Tensor(add_200, view_431);  add_200 = view_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    permute_327: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_326, [1, 0]);  permute_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_223: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_201, permute_327);  add_201 = permute_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_432: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_68, [1, 512, 4096]);  mm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_224: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_249, view_432);  mul_249 = view_432 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_433: "f32[512, 4096]" = torch.ops.aten.view.default(view_428, [512, 4096]);  view_428 = None
    mm_70: "f32[512, 4096]" = torch.ops.aten.mm.default(view_433, permute_163)
    permute_329: "f32[4096, 512]" = torch.ops.aten.permute.default(view_433, [1, 0])
    mm_71: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_329, view_134);  permute_329 = None
    permute_330: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_71, [1, 0]);  mm_71 = None
    sum_110: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_433, [0], True);  view_433 = None
    view_434: "f32[4096]" = torch.ops.aten.view.default(sum_110, [4096]);  sum_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_225: "f32[4096]" = torch.ops.aten.add.Tensor(add_203, view_434);  add_203 = view_434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    permute_331: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_330, [1, 0]);  permute_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_226: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_204, permute_331);  add_204 = permute_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_435: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_70, [1, 512, 4096]);  mm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_227: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_224, view_435);  add_224 = view_435 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_436: "f32[512, 4096]" = torch.ops.aten.view.default(view_429, [512, 4096]);  view_429 = None
    mm_72: "f32[512, 4096]" = torch.ops.aten.mm.default(view_436, permute_167)
    permute_333: "f32[4096, 512]" = torch.ops.aten.permute.default(view_436, [1, 0])
    mm_73: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_333, view_134);  permute_333 = view_134 = None
    permute_334: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_73, [1, 0]);  mm_73 = None
    sum_111: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_436, [0], True);  view_436 = None
    view_437: "f32[4096]" = torch.ops.aten.view.default(sum_111, [4096]);  sum_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_228: "f32[4096]" = torch.ops.aten.add.Tensor(add_206, view_437);  add_206 = view_437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    permute_335: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_334, [1, 0]);  permute_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_229: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_207, permute_335);  add_207 = permute_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_438: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_72, [1, 512, 4096]);  mm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_230: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_227, view_438);  add_227 = view_438 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    mul_254: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_230, primals_22)
    mul_255: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_254, 4096)
    sum_112: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_254, [2], True)
    mul_256: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_254, mul_49);  mul_254 = None
    sum_113: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_256, [2], True);  mul_256 = None
    mul_257: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_49, sum_113);  sum_113 = None
    sub_93: "f32[1, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_255, sum_112);  mul_255 = sum_112 = None
    sub_94: "f32[1, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_93, mul_257);  sub_93 = mul_257 = None
    mul_258: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(div_48, sub_94);  div_48 = sub_94 = None
    mul_259: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_230, mul_49);  mul_49 = None
    sum_114: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_259, [0, 1]);  mul_259 = None
    sum_115: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_230, [0, 1]);  add_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_231: "f32[4096]" = torch.ops.aten.add.Tensor(add_209, sum_114);  add_209 = sum_114 = None
    add_232: "f32[4096]" = torch.ops.aten.add.Tensor(add_210, sum_115);  add_210 = sum_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_439: "f32[512, 4096]" = torch.ops.aten.view.default(mul_258, [512, 4096])
    mm_74: "f32[512, 16384]" = torch.ops.aten.mm.default(view_439, permute_138)
    permute_337: "f32[4096, 512]" = torch.ops.aten.permute.default(view_439, [1, 0])
    mm_75: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_337, view_132);  permute_337 = view_132 = None
    permute_338: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_75, [1, 0]);  mm_75 = None
    sum_116: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_439, [0], True);  view_439 = None
    view_440: "f32[4096]" = torch.ops.aten.view.default(sum_116, [4096]);  sum_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_233: "f32[4096]" = torch.ops.aten.add.Tensor(add_211, view_440);  add_211 = view_440 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    permute_339: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_338, [1, 0]);  permute_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_234: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_212, permute_339);  add_212 = permute_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_441: "f32[1, 512, 16384]" = torch.ops.aten.view.default(mm_74, [1, 512, 16384]);  mm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_260: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(view_441, mul_45);  mul_45 = None
    mul_261: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(view_441, add_54);  view_441 = add_54 = None
    alias_40: "f32[1, 512, 16384]" = torch.ops.aten.alias.default(alias_11);  alias_11 = None
    mul_262: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(alias_40, alias_40);  alias_40 = None
    sub_95: "f32[1, 512, 16384]" = torch.ops.aten.sub.Tensor(1, mul_262);  mul_262 = None
    mul_263: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_260, sub_95);  mul_260 = sub_95 = None
    mul_264: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_263, 0.7978845608028654);  mul_263 = None
    mul_265: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_264, 0.044715)
    pow_19: "f32[1, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_131, 2.0);  view_131 = None
    mul_266: "f32[1, 512, 16384]" = torch.ops.aten.mul.Scalar(pow_19, 3.0);  pow_19 = None
    mul_267: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_265, mul_266);  mul_265 = mul_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_235: "f32[1, 512, 16384]" = torch.ops.aten.add.Tensor(mul_264, mul_267);  mul_264 = mul_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_268: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_261, 0.5);  mul_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_236: "f32[1, 512, 16384]" = torch.ops.aten.add.Tensor(add_235, mul_268);  add_235 = mul_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_442: "f32[512, 16384]" = torch.ops.aten.view.default(add_236, [512, 16384]);  add_236 = None
    mm_76: "f32[512, 4096]" = torch.ops.aten.mm.default(view_442, permute_142)
    permute_341: "f32[16384, 512]" = torch.ops.aten.permute.default(view_442, [1, 0])
    mm_77: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_341, view_130);  permute_341 = view_130 = None
    permute_342: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_77, [1, 0]);  mm_77 = None
    sum_117: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_442, [0], True);  view_442 = None
    view_443: "f32[16384]" = torch.ops.aten.view.default(sum_117, [16384]);  sum_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_237: "f32[16384]" = torch.ops.aten.add.Tensor(add_215, view_443);  add_215 = view_443 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    permute_343: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_342, [1, 0]);  permute_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_238: "f32[16384, 4096]" = torch.ops.aten.add.Tensor(add_216, permute_343);  add_216 = permute_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_444: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_76, [1, 512, 4096]);  mm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_239: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_258, view_444);  mul_258 = view_444 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    mul_270: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_239, primals_16)
    mul_271: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_270, 4096)
    sum_118: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_270, [2], True)
    mul_272: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_270, mul_43);  mul_270 = None
    sum_119: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_272, [2], True);  mul_272 = None
    mul_273: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_43, sum_119);  sum_119 = None
    sub_97: "f32[1, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_271, sum_118);  mul_271 = sum_118 = None
    sub_98: "f32[1, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_97, mul_273);  sub_97 = mul_273 = None
    mul_274: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(div_49, sub_98);  div_49 = sub_98 = None
    mul_275: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_239, mul_43);  mul_43 = None
    sum_120: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_275, [0, 1]);  mul_275 = None
    sum_121: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_239, [0, 1]);  add_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_240: "f32[4096]" = torch.ops.aten.add.Tensor(add_218, sum_120);  add_218 = sum_120 = None
    add_241: "f32[4096]" = torch.ops.aten.add.Tensor(add_219, sum_121);  add_219 = sum_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_445: "f32[512, 4096]" = torch.ops.aten.view.default(mul_274, [512, 4096])
    mm_78: "f32[512, 4096]" = torch.ops.aten.mm.default(view_445, permute_146)
    permute_345: "f32[4096, 512]" = torch.ops.aten.permute.default(view_445, [1, 0])
    mm_79: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_345, view_128);  permute_345 = view_128 = None
    permute_346: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_79, [1, 0]);  mm_79 = None
    sum_122: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_445, [0], True);  view_445 = None
    view_446: "f32[4096]" = torch.ops.aten.view.default(sum_122, [4096]);  sum_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_242: "f32[4096]" = torch.ops.aten.add.Tensor(add_220, view_446);  add_220 = view_446 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    permute_347: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_346, [1, 0]);  permute_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_243: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_221, permute_347);  add_221 = permute_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_447: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_78, [1, 512, 4096]);  mm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    view_448: "f32[1, 512, 64, 64]" = torch.ops.aten.view.default(view_447, [1, 512, 64, 64]);  view_447 = None
    permute_348: "f32[1, 64, 512, 64]" = torch.ops.aten.permute.default(view_448, [0, 2, 1, 3]);  view_448 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_449: "f32[64, 512, 64]" = torch.ops.aten.view.default(permute_348, [64, 512, 64]);  permute_348 = None
    bmm_48: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(permute_349, view_449);  permute_349 = None
    bmm_49: "f32[64, 512, 512]" = torch.ops.aten.bmm.default(view_449, permute_350);  view_449 = permute_350 = None
    view_450: "f32[1, 64, 512, 64]" = torch.ops.aten.view.default(bmm_48, [1, 64, 512, 64]);  bmm_48 = None
    view_451: "f32[1, 64, 512, 512]" = torch.ops.aten.view.default(bmm_49, [1, 64, 512, 512]);  bmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    mul_276: "f32[1, 64, 512, 512]" = torch.ops.aten.mul.Tensor(view_451, alias_41);  view_451 = None
    sum_123: "f32[1, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_276, [-1], True)
    mul_277: "f32[1, 64, 512, 512]" = torch.ops.aten.mul.Tensor(alias_41, sum_123);  alias_41 = sum_123 = None
    sub_99: "f32[1, 64, 512, 512]" = torch.ops.aten.sub.Tensor(mul_276, mul_277);  mul_276 = mul_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:332, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_50: "f32[1, 64, 512, 512]" = torch.ops.aten.div.Tensor(sub_99, 8.0);  sub_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_452: "f32[64, 512, 512]" = torch.ops.aten.view.default(div_50, [64, 512, 512]);  div_50 = None
    bmm_50: "f32[64, 64, 512]" = torch.ops.aten.bmm.default(permute_351, view_452);  permute_351 = None
    bmm_51: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(view_452, permute_352);  view_452 = permute_352 = None
    view_453: "f32[1, 64, 64, 512]" = torch.ops.aten.view.default(bmm_50, [1, 64, 64, 512]);  bmm_50 = None
    view_454: "f32[1, 64, 512, 64]" = torch.ops.aten.view.default(bmm_51, [1, 64, 512, 64]);  bmm_51 = None
    permute_353: "f32[1, 64, 512, 64]" = torch.ops.aten.permute.default(view_453, [0, 1, 3, 2]);  view_453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_354: "f32[1, 512, 64, 64]" = torch.ops.aten.permute.default(view_450, [0, 2, 1, 3]);  view_450 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_51: "f32[1, 512, 64, 64]" = torch.ops.aten.clone.default(permute_354, memory_format = torch.contiguous_format);  permute_354 = None
    view_455: "f32[1, 512, 4096]" = torch.ops.aten.view.default(clone_51, [1, 512, 4096]);  clone_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_355: "f32[1, 512, 64, 64]" = torch.ops.aten.permute.default(permute_353, [0, 2, 1, 3]);  permute_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_456: "f32[1, 512, 4096]" = torch.ops.aten.view.default(permute_355, [1, 512, 4096]);  permute_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_356: "f32[1, 512, 64, 64]" = torch.ops.aten.permute.default(view_454, [0, 2, 1, 3]);  view_454 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_52: "f32[1, 512, 64, 64]" = torch.ops.aten.clone.default(permute_356, memory_format = torch.contiguous_format);  permute_356 = None
    view_457: "f32[1, 512, 4096]" = torch.ops.aten.view.default(clone_52, [1, 512, 4096]);  clone_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_458: "f32[512, 4096]" = torch.ops.aten.view.default(view_455, [512, 4096]);  view_455 = None
    mm_80: "f32[512, 4096]" = torch.ops.aten.mm.default(view_458, permute_159)
    permute_358: "f32[4096, 512]" = torch.ops.aten.permute.default(view_458, [1, 0])
    mm_81: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_358, view_112);  permute_358 = None
    permute_359: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_81, [1, 0]);  mm_81 = None
    sum_124: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_458, [0], True);  view_458 = None
    view_459: "f32[4096]" = torch.ops.aten.view.default(sum_124, [4096]);  sum_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_244: "f32[4096]" = torch.ops.aten.add.Tensor(add_222, view_459);  add_222 = view_459 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    permute_360: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_359, [1, 0]);  permute_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_245: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_223, permute_360);  add_223 = permute_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_460: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_80, [1, 512, 4096]);  mm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_246: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_274, view_460);  mul_274 = view_460 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_461: "f32[512, 4096]" = torch.ops.aten.view.default(view_456, [512, 4096]);  view_456 = None
    mm_82: "f32[512, 4096]" = torch.ops.aten.mm.default(view_461, permute_163)
    permute_362: "f32[4096, 512]" = torch.ops.aten.permute.default(view_461, [1, 0])
    mm_83: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_362, view_112);  permute_362 = None
    permute_363: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_83, [1, 0]);  mm_83 = None
    sum_125: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_461, [0], True);  view_461 = None
    view_462: "f32[4096]" = torch.ops.aten.view.default(sum_125, [4096]);  sum_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_247: "f32[4096]" = torch.ops.aten.add.Tensor(add_225, view_462);  add_225 = view_462 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    permute_364: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_363, [1, 0]);  permute_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_248: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_226, permute_364);  add_226 = permute_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_463: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_82, [1, 512, 4096]);  mm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_249: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_246, view_463);  add_246 = view_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_464: "f32[512, 4096]" = torch.ops.aten.view.default(view_457, [512, 4096]);  view_457 = None
    mm_84: "f32[512, 4096]" = torch.ops.aten.mm.default(view_464, permute_167)
    permute_366: "f32[4096, 512]" = torch.ops.aten.permute.default(view_464, [1, 0])
    mm_85: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_366, view_112);  permute_366 = view_112 = None
    permute_367: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_85, [1, 0]);  mm_85 = None
    sum_126: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_464, [0], True);  view_464 = None
    view_465: "f32[4096]" = torch.ops.aten.view.default(sum_126, [4096]);  sum_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_250: "f32[4096]" = torch.ops.aten.add.Tensor(add_228, view_465);  add_228 = view_465 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    permute_368: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_367, [1, 0]);  permute_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_251: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_229, permute_368);  add_229 = permute_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_466: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_84, [1, 512, 4096]);  mm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_252: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_249, view_466);  add_249 = view_466 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    mul_279: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_252, primals_22)
    mul_280: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_279, 4096)
    sum_127: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_279, [2], True)
    mul_281: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_279, mul_41);  mul_279 = None
    sum_128: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_281, [2], True);  mul_281 = None
    mul_282: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_41, sum_128);  sum_128 = None
    sub_101: "f32[1, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_280, sum_127);  mul_280 = sum_127 = None
    sub_102: "f32[1, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_101, mul_282);  sub_101 = mul_282 = None
    mul_283: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(div_51, sub_102);  div_51 = sub_102 = None
    mul_284: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_252, mul_41);  mul_41 = None
    sum_129: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_284, [0, 1]);  mul_284 = None
    sum_130: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_252, [0, 1]);  add_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_253: "f32[4096]" = torch.ops.aten.add.Tensor(add_231, sum_129);  add_231 = sum_129 = None
    add_254: "f32[4096]" = torch.ops.aten.add.Tensor(add_232, sum_130);  add_232 = sum_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_467: "f32[512, 4096]" = torch.ops.aten.view.default(mul_283, [512, 4096])
    mm_86: "f32[512, 16384]" = torch.ops.aten.mm.default(view_467, permute_138)
    permute_370: "f32[4096, 512]" = torch.ops.aten.permute.default(view_467, [1, 0])
    mm_87: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_370, view_110);  permute_370 = view_110 = None
    permute_371: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_87, [1, 0]);  mm_87 = None
    sum_131: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_467, [0], True);  view_467 = None
    view_468: "f32[4096]" = torch.ops.aten.view.default(sum_131, [4096]);  sum_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_255: "f32[4096]" = torch.ops.aten.add.Tensor(add_233, view_468);  add_233 = view_468 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    permute_372: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_371, [1, 0]);  permute_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_256: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_234, permute_372);  add_234 = permute_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_469: "f32[1, 512, 16384]" = torch.ops.aten.view.default(mm_86, [1, 512, 16384]);  mm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_285: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(view_469, mul_37);  mul_37 = None
    mul_286: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(view_469, add_45);  view_469 = add_45 = None
    alias_42: "f32[1, 512, 16384]" = torch.ops.aten.alias.default(alias_9);  alias_9 = None
    mul_287: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(alias_42, alias_42);  alias_42 = None
    sub_103: "f32[1, 512, 16384]" = torch.ops.aten.sub.Tensor(1, mul_287);  mul_287 = None
    mul_288: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_285, sub_103);  mul_285 = sub_103 = None
    mul_289: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_288, 0.7978845608028654);  mul_288 = None
    mul_290: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_289, 0.044715)
    pow_20: "f32[1, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_109, 2.0);  view_109 = None
    mul_291: "f32[1, 512, 16384]" = torch.ops.aten.mul.Scalar(pow_20, 3.0);  pow_20 = None
    mul_292: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_290, mul_291);  mul_290 = mul_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_257: "f32[1, 512, 16384]" = torch.ops.aten.add.Tensor(mul_289, mul_292);  mul_289 = mul_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_293: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_286, 0.5);  mul_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_258: "f32[1, 512, 16384]" = torch.ops.aten.add.Tensor(add_257, mul_293);  add_257 = mul_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_470: "f32[512, 16384]" = torch.ops.aten.view.default(add_258, [512, 16384]);  add_258 = None
    mm_88: "f32[512, 4096]" = torch.ops.aten.mm.default(view_470, permute_142)
    permute_374: "f32[16384, 512]" = torch.ops.aten.permute.default(view_470, [1, 0])
    mm_89: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_374, view_108);  permute_374 = view_108 = None
    permute_375: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_89, [1, 0]);  mm_89 = None
    sum_132: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_470, [0], True);  view_470 = None
    view_471: "f32[16384]" = torch.ops.aten.view.default(sum_132, [16384]);  sum_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_259: "f32[16384]" = torch.ops.aten.add.Tensor(add_237, view_471);  add_237 = view_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    permute_376: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_375, [1, 0]);  permute_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_260: "f32[16384, 4096]" = torch.ops.aten.add.Tensor(add_238, permute_376);  add_238 = permute_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_472: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_88, [1, 512, 4096]);  mm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_261: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_283, view_472);  mul_283 = view_472 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    mul_295: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_261, primals_16)
    mul_296: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_295, 4096)
    sum_133: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_295, [2], True)
    mul_297: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_295, mul_35);  mul_295 = None
    sum_134: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_297, [2], True);  mul_297 = None
    mul_298: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_35, sum_134);  sum_134 = None
    sub_105: "f32[1, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_296, sum_133);  mul_296 = sum_133 = None
    sub_106: "f32[1, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_105, mul_298);  sub_105 = mul_298 = None
    mul_299: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(div_52, sub_106);  div_52 = sub_106 = None
    mul_300: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_261, mul_35);  mul_35 = None
    sum_135: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_300, [0, 1]);  mul_300 = None
    sum_136: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_261, [0, 1]);  add_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_262: "f32[4096]" = torch.ops.aten.add.Tensor(add_240, sum_135);  add_240 = sum_135 = None
    add_263: "f32[4096]" = torch.ops.aten.add.Tensor(add_241, sum_136);  add_241 = sum_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_473: "f32[512, 4096]" = torch.ops.aten.view.default(mul_299, [512, 4096])
    mm_90: "f32[512, 4096]" = torch.ops.aten.mm.default(view_473, permute_146)
    permute_378: "f32[4096, 512]" = torch.ops.aten.permute.default(view_473, [1, 0])
    mm_91: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_378, view_106);  permute_378 = view_106 = None
    permute_379: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_91, [1, 0]);  mm_91 = None
    sum_137: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_473, [0], True);  view_473 = None
    view_474: "f32[4096]" = torch.ops.aten.view.default(sum_137, [4096]);  sum_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_264: "f32[4096]" = torch.ops.aten.add.Tensor(add_242, view_474);  add_242 = view_474 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    permute_380: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_379, [1, 0]);  permute_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_265: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_243, permute_380);  add_243 = permute_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_475: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_90, [1, 512, 4096]);  mm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    view_476: "f32[1, 512, 64, 64]" = torch.ops.aten.view.default(view_475, [1, 512, 64, 64]);  view_475 = None
    permute_381: "f32[1, 64, 512, 64]" = torch.ops.aten.permute.default(view_476, [0, 2, 1, 3]);  view_476 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_477: "f32[64, 512, 64]" = torch.ops.aten.view.default(permute_381, [64, 512, 64]);  permute_381 = None
    bmm_52: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(permute_382, view_477);  permute_382 = None
    bmm_53: "f32[64, 512, 512]" = torch.ops.aten.bmm.default(view_477, permute_383);  view_477 = permute_383 = None
    view_478: "f32[1, 64, 512, 64]" = torch.ops.aten.view.default(bmm_52, [1, 64, 512, 64]);  bmm_52 = None
    view_479: "f32[1, 64, 512, 512]" = torch.ops.aten.view.default(bmm_53, [1, 64, 512, 512]);  bmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    mul_301: "f32[1, 64, 512, 512]" = torch.ops.aten.mul.Tensor(view_479, alias_43);  view_479 = None
    sum_138: "f32[1, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_301, [-1], True)
    mul_302: "f32[1, 64, 512, 512]" = torch.ops.aten.mul.Tensor(alias_43, sum_138);  alias_43 = sum_138 = None
    sub_107: "f32[1, 64, 512, 512]" = torch.ops.aten.sub.Tensor(mul_301, mul_302);  mul_301 = mul_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:332, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_53: "f32[1, 64, 512, 512]" = torch.ops.aten.div.Tensor(sub_107, 8.0);  sub_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_480: "f32[64, 512, 512]" = torch.ops.aten.view.default(div_53, [64, 512, 512]);  div_53 = None
    bmm_54: "f32[64, 64, 512]" = torch.ops.aten.bmm.default(permute_384, view_480);  permute_384 = None
    bmm_55: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(view_480, permute_385);  view_480 = permute_385 = None
    view_481: "f32[1, 64, 64, 512]" = torch.ops.aten.view.default(bmm_54, [1, 64, 64, 512]);  bmm_54 = None
    view_482: "f32[1, 64, 512, 64]" = torch.ops.aten.view.default(bmm_55, [1, 64, 512, 64]);  bmm_55 = None
    permute_386: "f32[1, 64, 512, 64]" = torch.ops.aten.permute.default(view_481, [0, 1, 3, 2]);  view_481 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_387: "f32[1, 512, 64, 64]" = torch.ops.aten.permute.default(view_478, [0, 2, 1, 3]);  view_478 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_53: "f32[1, 512, 64, 64]" = torch.ops.aten.clone.default(permute_387, memory_format = torch.contiguous_format);  permute_387 = None
    view_483: "f32[1, 512, 4096]" = torch.ops.aten.view.default(clone_53, [1, 512, 4096]);  clone_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_388: "f32[1, 512, 64, 64]" = torch.ops.aten.permute.default(permute_386, [0, 2, 1, 3]);  permute_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_484: "f32[1, 512, 4096]" = torch.ops.aten.view.default(permute_388, [1, 512, 4096]);  permute_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_389: "f32[1, 512, 64, 64]" = torch.ops.aten.permute.default(view_482, [0, 2, 1, 3]);  view_482 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_54: "f32[1, 512, 64, 64]" = torch.ops.aten.clone.default(permute_389, memory_format = torch.contiguous_format);  permute_389 = None
    view_485: "f32[1, 512, 4096]" = torch.ops.aten.view.default(clone_54, [1, 512, 4096]);  clone_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_486: "f32[512, 4096]" = torch.ops.aten.view.default(view_483, [512, 4096]);  view_483 = None
    mm_92: "f32[512, 4096]" = torch.ops.aten.mm.default(view_486, permute_159)
    permute_391: "f32[4096, 512]" = torch.ops.aten.permute.default(view_486, [1, 0])
    mm_93: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_391, view_90);  permute_391 = None
    permute_392: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_93, [1, 0]);  mm_93 = None
    sum_139: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_486, [0], True);  view_486 = None
    view_487: "f32[4096]" = torch.ops.aten.view.default(sum_139, [4096]);  sum_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_266: "f32[4096]" = torch.ops.aten.add.Tensor(add_244, view_487);  add_244 = view_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    permute_393: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_392, [1, 0]);  permute_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_267: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_245, permute_393);  add_245 = permute_393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_488: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_92, [1, 512, 4096]);  mm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_268: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_299, view_488);  mul_299 = view_488 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_489: "f32[512, 4096]" = torch.ops.aten.view.default(view_484, [512, 4096]);  view_484 = None
    mm_94: "f32[512, 4096]" = torch.ops.aten.mm.default(view_489, permute_163)
    permute_395: "f32[4096, 512]" = torch.ops.aten.permute.default(view_489, [1, 0])
    mm_95: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_395, view_90);  permute_395 = None
    permute_396: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_95, [1, 0]);  mm_95 = None
    sum_140: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_489, [0], True);  view_489 = None
    view_490: "f32[4096]" = torch.ops.aten.view.default(sum_140, [4096]);  sum_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_269: "f32[4096]" = torch.ops.aten.add.Tensor(add_247, view_490);  add_247 = view_490 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    permute_397: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_396, [1, 0]);  permute_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_270: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_248, permute_397);  add_248 = permute_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_491: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_94, [1, 512, 4096]);  mm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_271: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_268, view_491);  add_268 = view_491 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_492: "f32[512, 4096]" = torch.ops.aten.view.default(view_485, [512, 4096]);  view_485 = None
    mm_96: "f32[512, 4096]" = torch.ops.aten.mm.default(view_492, permute_167)
    permute_399: "f32[4096, 512]" = torch.ops.aten.permute.default(view_492, [1, 0])
    mm_97: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_399, view_90);  permute_399 = view_90 = None
    permute_400: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_97, [1, 0]);  mm_97 = None
    sum_141: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_492, [0], True);  view_492 = None
    view_493: "f32[4096]" = torch.ops.aten.view.default(sum_141, [4096]);  sum_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_272: "f32[4096]" = torch.ops.aten.add.Tensor(add_250, view_493);  add_250 = view_493 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    permute_401: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_400, [1, 0]);  permute_400 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_273: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_251, permute_401);  add_251 = permute_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_494: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_96, [1, 512, 4096]);  mm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_274: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_271, view_494);  add_271 = view_494 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    mul_304: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_274, primals_22)
    mul_305: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_304, 4096)
    sum_142: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_304, [2], True)
    mul_306: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_304, mul_33);  mul_304 = None
    sum_143: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_306, [2], True);  mul_306 = None
    mul_307: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_33, sum_143);  sum_143 = None
    sub_109: "f32[1, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_305, sum_142);  mul_305 = sum_142 = None
    sub_110: "f32[1, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_109, mul_307);  sub_109 = mul_307 = None
    mul_308: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(div_54, sub_110);  div_54 = sub_110 = None
    mul_309: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_274, mul_33);  mul_33 = None
    sum_144: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_309, [0, 1]);  mul_309 = None
    sum_145: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_274, [0, 1]);  add_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_275: "f32[4096]" = torch.ops.aten.add.Tensor(add_253, sum_144);  add_253 = sum_144 = None
    add_276: "f32[4096]" = torch.ops.aten.add.Tensor(add_254, sum_145);  add_254 = sum_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_495: "f32[512, 4096]" = torch.ops.aten.view.default(mul_308, [512, 4096])
    mm_98: "f32[512, 16384]" = torch.ops.aten.mm.default(view_495, permute_138)
    permute_403: "f32[4096, 512]" = torch.ops.aten.permute.default(view_495, [1, 0])
    mm_99: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_403, view_88);  permute_403 = view_88 = None
    permute_404: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_99, [1, 0]);  mm_99 = None
    sum_146: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_495, [0], True);  view_495 = None
    view_496: "f32[4096]" = torch.ops.aten.view.default(sum_146, [4096]);  sum_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_277: "f32[4096]" = torch.ops.aten.add.Tensor(add_255, view_496);  add_255 = view_496 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    permute_405: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_404, [1, 0]);  permute_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_278: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_256, permute_405);  add_256 = permute_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_497: "f32[1, 512, 16384]" = torch.ops.aten.view.default(mm_98, [1, 512, 16384]);  mm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_310: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(view_497, mul_29);  mul_29 = None
    mul_311: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(view_497, add_36);  view_497 = add_36 = None
    alias_44: "f32[1, 512, 16384]" = torch.ops.aten.alias.default(alias_7);  alias_7 = None
    mul_312: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(alias_44, alias_44);  alias_44 = None
    sub_111: "f32[1, 512, 16384]" = torch.ops.aten.sub.Tensor(1, mul_312);  mul_312 = None
    mul_313: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_310, sub_111);  mul_310 = sub_111 = None
    mul_314: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_313, 0.7978845608028654);  mul_313 = None
    mul_315: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_314, 0.044715)
    pow_21: "f32[1, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_87, 2.0);  view_87 = None
    mul_316: "f32[1, 512, 16384]" = torch.ops.aten.mul.Scalar(pow_21, 3.0);  pow_21 = None
    mul_317: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_315, mul_316);  mul_315 = mul_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_279: "f32[1, 512, 16384]" = torch.ops.aten.add.Tensor(mul_314, mul_317);  mul_314 = mul_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_318: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_311, 0.5);  mul_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_280: "f32[1, 512, 16384]" = torch.ops.aten.add.Tensor(add_279, mul_318);  add_279 = mul_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_498: "f32[512, 16384]" = torch.ops.aten.view.default(add_280, [512, 16384]);  add_280 = None
    mm_100: "f32[512, 4096]" = torch.ops.aten.mm.default(view_498, permute_142)
    permute_407: "f32[16384, 512]" = torch.ops.aten.permute.default(view_498, [1, 0])
    mm_101: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_407, view_86);  permute_407 = view_86 = None
    permute_408: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_101, [1, 0]);  mm_101 = None
    sum_147: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_498, [0], True);  view_498 = None
    view_499: "f32[16384]" = torch.ops.aten.view.default(sum_147, [16384]);  sum_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_281: "f32[16384]" = torch.ops.aten.add.Tensor(add_259, view_499);  add_259 = view_499 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    permute_409: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_408, [1, 0]);  permute_408 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_282: "f32[16384, 4096]" = torch.ops.aten.add.Tensor(add_260, permute_409);  add_260 = permute_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_500: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_100, [1, 512, 4096]);  mm_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_283: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_308, view_500);  mul_308 = view_500 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    mul_320: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_283, primals_16)
    mul_321: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_320, 4096)
    sum_148: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_320, [2], True)
    mul_322: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_320, mul_27);  mul_320 = None
    sum_149: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_322, [2], True);  mul_322 = None
    mul_323: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_27, sum_149);  sum_149 = None
    sub_113: "f32[1, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_321, sum_148);  mul_321 = sum_148 = None
    sub_114: "f32[1, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_113, mul_323);  sub_113 = mul_323 = None
    mul_324: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(div_55, sub_114);  div_55 = sub_114 = None
    mul_325: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_283, mul_27);  mul_27 = None
    sum_150: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_325, [0, 1]);  mul_325 = None
    sum_151: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_283, [0, 1]);  add_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_284: "f32[4096]" = torch.ops.aten.add.Tensor(add_262, sum_150);  add_262 = sum_150 = None
    add_285: "f32[4096]" = torch.ops.aten.add.Tensor(add_263, sum_151);  add_263 = sum_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_501: "f32[512, 4096]" = torch.ops.aten.view.default(mul_324, [512, 4096])
    mm_102: "f32[512, 4096]" = torch.ops.aten.mm.default(view_501, permute_146)
    permute_411: "f32[4096, 512]" = torch.ops.aten.permute.default(view_501, [1, 0])
    mm_103: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_411, view_84);  permute_411 = view_84 = None
    permute_412: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_103, [1, 0]);  mm_103 = None
    sum_152: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_501, [0], True);  view_501 = None
    view_502: "f32[4096]" = torch.ops.aten.view.default(sum_152, [4096]);  sum_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_286: "f32[4096]" = torch.ops.aten.add.Tensor(add_264, view_502);  add_264 = view_502 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    permute_413: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_412, [1, 0]);  permute_412 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_287: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_265, permute_413);  add_265 = permute_413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_503: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_102, [1, 512, 4096]);  mm_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    view_504: "f32[1, 512, 64, 64]" = torch.ops.aten.view.default(view_503, [1, 512, 64, 64]);  view_503 = None
    permute_414: "f32[1, 64, 512, 64]" = torch.ops.aten.permute.default(view_504, [0, 2, 1, 3]);  view_504 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_505: "f32[64, 512, 64]" = torch.ops.aten.view.default(permute_414, [64, 512, 64]);  permute_414 = None
    bmm_56: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(permute_415, view_505);  permute_415 = None
    bmm_57: "f32[64, 512, 512]" = torch.ops.aten.bmm.default(view_505, permute_416);  view_505 = permute_416 = None
    view_506: "f32[1, 64, 512, 64]" = torch.ops.aten.view.default(bmm_56, [1, 64, 512, 64]);  bmm_56 = None
    view_507: "f32[1, 64, 512, 512]" = torch.ops.aten.view.default(bmm_57, [1, 64, 512, 512]);  bmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    mul_326: "f32[1, 64, 512, 512]" = torch.ops.aten.mul.Tensor(view_507, alias_45);  view_507 = None
    sum_153: "f32[1, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_326, [-1], True)
    mul_327: "f32[1, 64, 512, 512]" = torch.ops.aten.mul.Tensor(alias_45, sum_153);  alias_45 = sum_153 = None
    sub_115: "f32[1, 64, 512, 512]" = torch.ops.aten.sub.Tensor(mul_326, mul_327);  mul_326 = mul_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:332, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_56: "f32[1, 64, 512, 512]" = torch.ops.aten.div.Tensor(sub_115, 8.0);  sub_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_508: "f32[64, 512, 512]" = torch.ops.aten.view.default(div_56, [64, 512, 512]);  div_56 = None
    bmm_58: "f32[64, 64, 512]" = torch.ops.aten.bmm.default(permute_417, view_508);  permute_417 = None
    bmm_59: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(view_508, permute_418);  view_508 = permute_418 = None
    view_509: "f32[1, 64, 64, 512]" = torch.ops.aten.view.default(bmm_58, [1, 64, 64, 512]);  bmm_58 = None
    view_510: "f32[1, 64, 512, 64]" = torch.ops.aten.view.default(bmm_59, [1, 64, 512, 64]);  bmm_59 = None
    permute_419: "f32[1, 64, 512, 64]" = torch.ops.aten.permute.default(view_509, [0, 1, 3, 2]);  view_509 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_420: "f32[1, 512, 64, 64]" = torch.ops.aten.permute.default(view_506, [0, 2, 1, 3]);  view_506 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_55: "f32[1, 512, 64, 64]" = torch.ops.aten.clone.default(permute_420, memory_format = torch.contiguous_format);  permute_420 = None
    view_511: "f32[1, 512, 4096]" = torch.ops.aten.view.default(clone_55, [1, 512, 4096]);  clone_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_421: "f32[1, 512, 64, 64]" = torch.ops.aten.permute.default(permute_419, [0, 2, 1, 3]);  permute_419 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_512: "f32[1, 512, 4096]" = torch.ops.aten.view.default(permute_421, [1, 512, 4096]);  permute_421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_422: "f32[1, 512, 64, 64]" = torch.ops.aten.permute.default(view_510, [0, 2, 1, 3]);  view_510 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_56: "f32[1, 512, 64, 64]" = torch.ops.aten.clone.default(permute_422, memory_format = torch.contiguous_format);  permute_422 = None
    view_513: "f32[1, 512, 4096]" = torch.ops.aten.view.default(clone_56, [1, 512, 4096]);  clone_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_514: "f32[512, 4096]" = torch.ops.aten.view.default(view_511, [512, 4096]);  view_511 = None
    mm_104: "f32[512, 4096]" = torch.ops.aten.mm.default(view_514, permute_159)
    permute_424: "f32[4096, 512]" = torch.ops.aten.permute.default(view_514, [1, 0])
    mm_105: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_424, view_68);  permute_424 = None
    permute_425: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_105, [1, 0]);  mm_105 = None
    sum_154: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_514, [0], True);  view_514 = None
    view_515: "f32[4096]" = torch.ops.aten.view.default(sum_154, [4096]);  sum_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_288: "f32[4096]" = torch.ops.aten.add.Tensor(add_266, view_515);  add_266 = view_515 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    permute_426: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_425, [1, 0]);  permute_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_289: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_267, permute_426);  add_267 = permute_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_516: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_104, [1, 512, 4096]);  mm_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_290: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_324, view_516);  mul_324 = view_516 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_517: "f32[512, 4096]" = torch.ops.aten.view.default(view_512, [512, 4096]);  view_512 = None
    mm_106: "f32[512, 4096]" = torch.ops.aten.mm.default(view_517, permute_163)
    permute_428: "f32[4096, 512]" = torch.ops.aten.permute.default(view_517, [1, 0])
    mm_107: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_428, view_68);  permute_428 = None
    permute_429: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_107, [1, 0]);  mm_107 = None
    sum_155: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_517, [0], True);  view_517 = None
    view_518: "f32[4096]" = torch.ops.aten.view.default(sum_155, [4096]);  sum_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_291: "f32[4096]" = torch.ops.aten.add.Tensor(add_269, view_518);  add_269 = view_518 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    permute_430: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_429, [1, 0]);  permute_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_292: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_270, permute_430);  add_270 = permute_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_519: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_106, [1, 512, 4096]);  mm_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_293: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_290, view_519);  add_290 = view_519 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_520: "f32[512, 4096]" = torch.ops.aten.view.default(view_513, [512, 4096]);  view_513 = None
    mm_108: "f32[512, 4096]" = torch.ops.aten.mm.default(view_520, permute_167)
    permute_432: "f32[4096, 512]" = torch.ops.aten.permute.default(view_520, [1, 0])
    mm_109: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_432, view_68);  permute_432 = view_68 = None
    permute_433: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_109, [1, 0]);  mm_109 = None
    sum_156: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_520, [0], True);  view_520 = None
    view_521: "f32[4096]" = torch.ops.aten.view.default(sum_156, [4096]);  sum_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_294: "f32[4096]" = torch.ops.aten.add.Tensor(add_272, view_521);  add_272 = view_521 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    permute_434: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_433, [1, 0]);  permute_433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_295: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_273, permute_434);  add_273 = permute_434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_522: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_108, [1, 512, 4096]);  mm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_296: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_293, view_522);  add_293 = view_522 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    mul_329: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_296, primals_22)
    mul_330: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_329, 4096)
    sum_157: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_329, [2], True)
    mul_331: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_329, mul_25);  mul_329 = None
    sum_158: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_331, [2], True);  mul_331 = None
    mul_332: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_25, sum_158);  sum_158 = None
    sub_117: "f32[1, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_330, sum_157);  mul_330 = sum_157 = None
    sub_118: "f32[1, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_117, mul_332);  sub_117 = mul_332 = None
    mul_333: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(div_57, sub_118);  div_57 = sub_118 = None
    mul_334: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_296, mul_25);  mul_25 = None
    sum_159: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_334, [0, 1]);  mul_334 = None
    sum_160: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_296, [0, 1]);  add_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_297: "f32[4096]" = torch.ops.aten.add.Tensor(add_275, sum_159);  add_275 = sum_159 = None
    add_298: "f32[4096]" = torch.ops.aten.add.Tensor(add_276, sum_160);  add_276 = sum_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_523: "f32[512, 4096]" = torch.ops.aten.view.default(mul_333, [512, 4096])
    mm_110: "f32[512, 16384]" = torch.ops.aten.mm.default(view_523, permute_138)
    permute_436: "f32[4096, 512]" = torch.ops.aten.permute.default(view_523, [1, 0])
    mm_111: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_436, view_66);  permute_436 = view_66 = None
    permute_437: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_111, [1, 0]);  mm_111 = None
    sum_161: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_523, [0], True);  view_523 = None
    view_524: "f32[4096]" = torch.ops.aten.view.default(sum_161, [4096]);  sum_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_299: "f32[4096]" = torch.ops.aten.add.Tensor(add_277, view_524);  add_277 = view_524 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    permute_438: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_437, [1, 0]);  permute_437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_300: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_278, permute_438);  add_278 = permute_438 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_525: "f32[1, 512, 16384]" = torch.ops.aten.view.default(mm_110, [1, 512, 16384]);  mm_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_335: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(view_525, mul_21);  mul_21 = None
    mul_336: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(view_525, add_27);  view_525 = add_27 = None
    alias_46: "f32[1, 512, 16384]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    mul_337: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(alias_46, alias_46);  alias_46 = None
    sub_119: "f32[1, 512, 16384]" = torch.ops.aten.sub.Tensor(1, mul_337);  mul_337 = None
    mul_338: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_335, sub_119);  mul_335 = sub_119 = None
    mul_339: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_338, 0.7978845608028654);  mul_338 = None
    mul_340: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_339, 0.044715)
    pow_22: "f32[1, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_65, 2.0);  view_65 = None
    mul_341: "f32[1, 512, 16384]" = torch.ops.aten.mul.Scalar(pow_22, 3.0);  pow_22 = None
    mul_342: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_340, mul_341);  mul_340 = mul_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_301: "f32[1, 512, 16384]" = torch.ops.aten.add.Tensor(mul_339, mul_342);  mul_339 = mul_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_343: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_336, 0.5);  mul_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_302: "f32[1, 512, 16384]" = torch.ops.aten.add.Tensor(add_301, mul_343);  add_301 = mul_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_526: "f32[512, 16384]" = torch.ops.aten.view.default(add_302, [512, 16384]);  add_302 = None
    mm_112: "f32[512, 4096]" = torch.ops.aten.mm.default(view_526, permute_142)
    permute_440: "f32[16384, 512]" = torch.ops.aten.permute.default(view_526, [1, 0])
    mm_113: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_440, view_64);  permute_440 = view_64 = None
    permute_441: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_113, [1, 0]);  mm_113 = None
    sum_162: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_526, [0], True);  view_526 = None
    view_527: "f32[16384]" = torch.ops.aten.view.default(sum_162, [16384]);  sum_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_303: "f32[16384]" = torch.ops.aten.add.Tensor(add_281, view_527);  add_281 = view_527 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    permute_442: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_441, [1, 0]);  permute_441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_304: "f32[16384, 4096]" = torch.ops.aten.add.Tensor(add_282, permute_442);  add_282 = permute_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_528: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_112, [1, 512, 4096]);  mm_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_305: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_333, view_528);  mul_333 = view_528 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    mul_345: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_305, primals_16)
    mul_346: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_345, 4096)
    sum_163: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_345, [2], True)
    mul_347: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_345, mul_19);  mul_345 = None
    sum_164: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_347, [2], True);  mul_347 = None
    mul_348: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_19, sum_164);  sum_164 = None
    sub_121: "f32[1, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_346, sum_163);  mul_346 = sum_163 = None
    sub_122: "f32[1, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_121, mul_348);  sub_121 = mul_348 = None
    mul_349: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(div_58, sub_122);  div_58 = sub_122 = None
    mul_350: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_305, mul_19);  mul_19 = None
    sum_165: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_350, [0, 1]);  mul_350 = None
    sum_166: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_305, [0, 1]);  add_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_306: "f32[4096]" = torch.ops.aten.add.Tensor(add_284, sum_165);  add_284 = sum_165 = None
    add_307: "f32[4096]" = torch.ops.aten.add.Tensor(add_285, sum_166);  add_285 = sum_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_529: "f32[512, 4096]" = torch.ops.aten.view.default(mul_349, [512, 4096])
    mm_114: "f32[512, 4096]" = torch.ops.aten.mm.default(view_529, permute_146)
    permute_444: "f32[4096, 512]" = torch.ops.aten.permute.default(view_529, [1, 0])
    mm_115: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_444, view_62);  permute_444 = view_62 = None
    permute_445: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_115, [1, 0]);  mm_115 = None
    sum_167: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_529, [0], True);  view_529 = None
    view_530: "f32[4096]" = torch.ops.aten.view.default(sum_167, [4096]);  sum_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_308: "f32[4096]" = torch.ops.aten.add.Tensor(add_286, view_530);  add_286 = view_530 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    permute_446: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_445, [1, 0]);  permute_445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_309: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_287, permute_446);  add_287 = permute_446 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_531: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_114, [1, 512, 4096]);  mm_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    view_532: "f32[1, 512, 64, 64]" = torch.ops.aten.view.default(view_531, [1, 512, 64, 64]);  view_531 = None
    permute_447: "f32[1, 64, 512, 64]" = torch.ops.aten.permute.default(view_532, [0, 2, 1, 3]);  view_532 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_533: "f32[64, 512, 64]" = torch.ops.aten.view.default(permute_447, [64, 512, 64]);  permute_447 = None
    bmm_60: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(permute_448, view_533);  permute_448 = None
    bmm_61: "f32[64, 512, 512]" = torch.ops.aten.bmm.default(view_533, permute_449);  view_533 = permute_449 = None
    view_534: "f32[1, 64, 512, 64]" = torch.ops.aten.view.default(bmm_60, [1, 64, 512, 64]);  bmm_60 = None
    view_535: "f32[1, 64, 512, 512]" = torch.ops.aten.view.default(bmm_61, [1, 64, 512, 512]);  bmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    mul_351: "f32[1, 64, 512, 512]" = torch.ops.aten.mul.Tensor(view_535, alias_47);  view_535 = None
    sum_168: "f32[1, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_351, [-1], True)
    mul_352: "f32[1, 64, 512, 512]" = torch.ops.aten.mul.Tensor(alias_47, sum_168);  alias_47 = sum_168 = None
    sub_123: "f32[1, 64, 512, 512]" = torch.ops.aten.sub.Tensor(mul_351, mul_352);  mul_351 = mul_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:332, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_59: "f32[1, 64, 512, 512]" = torch.ops.aten.div.Tensor(sub_123, 8.0);  sub_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_536: "f32[64, 512, 512]" = torch.ops.aten.view.default(div_59, [64, 512, 512]);  div_59 = None
    bmm_62: "f32[64, 64, 512]" = torch.ops.aten.bmm.default(permute_450, view_536);  permute_450 = None
    bmm_63: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(view_536, permute_451);  view_536 = permute_451 = None
    view_537: "f32[1, 64, 64, 512]" = torch.ops.aten.view.default(bmm_62, [1, 64, 64, 512]);  bmm_62 = None
    view_538: "f32[1, 64, 512, 64]" = torch.ops.aten.view.default(bmm_63, [1, 64, 512, 64]);  bmm_63 = None
    permute_452: "f32[1, 64, 512, 64]" = torch.ops.aten.permute.default(view_537, [0, 1, 3, 2]);  view_537 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_453: "f32[1, 512, 64, 64]" = torch.ops.aten.permute.default(view_534, [0, 2, 1, 3]);  view_534 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_57: "f32[1, 512, 64, 64]" = torch.ops.aten.clone.default(permute_453, memory_format = torch.contiguous_format);  permute_453 = None
    view_539: "f32[1, 512, 4096]" = torch.ops.aten.view.default(clone_57, [1, 512, 4096]);  clone_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_454: "f32[1, 512, 64, 64]" = torch.ops.aten.permute.default(permute_452, [0, 2, 1, 3]);  permute_452 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_540: "f32[1, 512, 4096]" = torch.ops.aten.view.default(permute_454, [1, 512, 4096]);  permute_454 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_455: "f32[1, 512, 64, 64]" = torch.ops.aten.permute.default(view_538, [0, 2, 1, 3]);  view_538 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_58: "f32[1, 512, 64, 64]" = torch.ops.aten.clone.default(permute_455, memory_format = torch.contiguous_format);  permute_455 = None
    view_541: "f32[1, 512, 4096]" = torch.ops.aten.view.default(clone_58, [1, 512, 4096]);  clone_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_542: "f32[512, 4096]" = torch.ops.aten.view.default(view_539, [512, 4096]);  view_539 = None
    mm_116: "f32[512, 4096]" = torch.ops.aten.mm.default(view_542, permute_159)
    permute_457: "f32[4096, 512]" = torch.ops.aten.permute.default(view_542, [1, 0])
    mm_117: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_457, view_46);  permute_457 = None
    permute_458: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_117, [1, 0]);  mm_117 = None
    sum_169: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_542, [0], True);  view_542 = None
    view_543: "f32[4096]" = torch.ops.aten.view.default(sum_169, [4096]);  sum_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_310: "f32[4096]" = torch.ops.aten.add.Tensor(add_288, view_543);  add_288 = view_543 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    permute_459: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_458, [1, 0]);  permute_458 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_311: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_289, permute_459);  add_289 = permute_459 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_544: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_116, [1, 512, 4096]);  mm_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_312: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_349, view_544);  mul_349 = view_544 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_545: "f32[512, 4096]" = torch.ops.aten.view.default(view_540, [512, 4096]);  view_540 = None
    mm_118: "f32[512, 4096]" = torch.ops.aten.mm.default(view_545, permute_163)
    permute_461: "f32[4096, 512]" = torch.ops.aten.permute.default(view_545, [1, 0])
    mm_119: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_461, view_46);  permute_461 = None
    permute_462: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_119, [1, 0]);  mm_119 = None
    sum_170: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_545, [0], True);  view_545 = None
    view_546: "f32[4096]" = torch.ops.aten.view.default(sum_170, [4096]);  sum_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_313: "f32[4096]" = torch.ops.aten.add.Tensor(add_291, view_546);  add_291 = view_546 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    permute_463: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_462, [1, 0]);  permute_462 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_314: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_292, permute_463);  add_292 = permute_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_547: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_118, [1, 512, 4096]);  mm_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_315: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_312, view_547);  add_312 = view_547 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_548: "f32[512, 4096]" = torch.ops.aten.view.default(view_541, [512, 4096]);  view_541 = None
    mm_120: "f32[512, 4096]" = torch.ops.aten.mm.default(view_548, permute_167)
    permute_465: "f32[4096, 512]" = torch.ops.aten.permute.default(view_548, [1, 0])
    mm_121: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_465, view_46);  permute_465 = view_46 = None
    permute_466: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_121, [1, 0]);  mm_121 = None
    sum_171: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_548, [0], True);  view_548 = None
    view_549: "f32[4096]" = torch.ops.aten.view.default(sum_171, [4096]);  sum_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_316: "f32[4096]" = torch.ops.aten.add.Tensor(add_294, view_549);  add_294 = view_549 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    permute_467: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_466, [1, 0]);  permute_466 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_317: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_295, permute_467);  add_295 = permute_467 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_550: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_120, [1, 512, 4096]);  mm_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_318: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_315, view_550);  add_315 = view_550 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    mul_354: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_318, primals_22)
    mul_355: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_354, 4096)
    sum_172: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_354, [2], True)
    mul_356: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_354, mul_17);  mul_354 = None
    sum_173: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_356, [2], True);  mul_356 = None
    mul_357: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_17, sum_173);  sum_173 = None
    sub_125: "f32[1, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_355, sum_172);  mul_355 = sum_172 = None
    sub_126: "f32[1, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_125, mul_357);  sub_125 = mul_357 = None
    mul_358: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(div_60, sub_126);  div_60 = sub_126 = None
    mul_359: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_318, mul_17);  mul_17 = None
    sum_174: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_359, [0, 1]);  mul_359 = None
    sum_175: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_318, [0, 1]);  add_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_319: "f32[4096]" = torch.ops.aten.add.Tensor(add_297, sum_174);  add_297 = sum_174 = None
    add_320: "f32[4096]" = torch.ops.aten.add.Tensor(add_298, sum_175);  add_298 = sum_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_551: "f32[512, 4096]" = torch.ops.aten.view.default(mul_358, [512, 4096])
    mm_122: "f32[512, 16384]" = torch.ops.aten.mm.default(view_551, permute_138)
    permute_469: "f32[4096, 512]" = torch.ops.aten.permute.default(view_551, [1, 0])
    mm_123: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_469, view_44);  permute_469 = view_44 = None
    permute_470: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_123, [1, 0]);  mm_123 = None
    sum_176: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_551, [0], True);  view_551 = None
    view_552: "f32[4096]" = torch.ops.aten.view.default(sum_176, [4096]);  sum_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_321: "f32[4096]" = torch.ops.aten.add.Tensor(add_299, view_552);  add_299 = view_552 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    permute_471: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_470, [1, 0]);  permute_470 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_322: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_300, permute_471);  add_300 = permute_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_553: "f32[1, 512, 16384]" = torch.ops.aten.view.default(mm_122, [1, 512, 16384]);  mm_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_360: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(view_553, mul_13);  mul_13 = None
    mul_361: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(view_553, add_18);  view_553 = add_18 = None
    alias_48: "f32[1, 512, 16384]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    mul_362: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(alias_48, alias_48);  alias_48 = None
    sub_127: "f32[1, 512, 16384]" = torch.ops.aten.sub.Tensor(1, mul_362);  mul_362 = None
    mul_363: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_360, sub_127);  mul_360 = sub_127 = None
    mul_364: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_363, 0.7978845608028654);  mul_363 = None
    mul_365: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_364, 0.044715)
    pow_23: "f32[1, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_43, 2.0);  view_43 = None
    mul_366: "f32[1, 512, 16384]" = torch.ops.aten.mul.Scalar(pow_23, 3.0);  pow_23 = None
    mul_367: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_365, mul_366);  mul_365 = mul_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_323: "f32[1, 512, 16384]" = torch.ops.aten.add.Tensor(mul_364, mul_367);  mul_364 = mul_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_368: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_361, 0.5);  mul_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_324: "f32[1, 512, 16384]" = torch.ops.aten.add.Tensor(add_323, mul_368);  add_323 = mul_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_554: "f32[512, 16384]" = torch.ops.aten.view.default(add_324, [512, 16384]);  add_324 = None
    mm_124: "f32[512, 4096]" = torch.ops.aten.mm.default(view_554, permute_142)
    permute_473: "f32[16384, 512]" = torch.ops.aten.permute.default(view_554, [1, 0])
    mm_125: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_473, view_42);  permute_473 = view_42 = None
    permute_474: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_125, [1, 0]);  mm_125 = None
    sum_177: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_554, [0], True);  view_554 = None
    view_555: "f32[16384]" = torch.ops.aten.view.default(sum_177, [16384]);  sum_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_325: "f32[16384]" = torch.ops.aten.add.Tensor(add_303, view_555);  add_303 = view_555 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    permute_475: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_474, [1, 0]);  permute_474 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_326: "f32[16384, 4096]" = torch.ops.aten.add.Tensor(add_304, permute_475);  add_304 = permute_475 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_556: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_124, [1, 512, 4096]);  mm_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_327: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_358, view_556);  mul_358 = view_556 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    mul_370: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_327, primals_16)
    mul_371: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_370, 4096)
    sum_178: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_370, [2], True)
    mul_372: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_370, mul_11);  mul_370 = None
    sum_179: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_372, [2], True);  mul_372 = None
    mul_373: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_11, sum_179);  sum_179 = None
    sub_129: "f32[1, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_371, sum_178);  mul_371 = sum_178 = None
    sub_130: "f32[1, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_129, mul_373);  sub_129 = mul_373 = None
    mul_374: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(div_61, sub_130);  div_61 = sub_130 = None
    mul_375: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_327, mul_11);  mul_11 = None
    sum_180: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_375, [0, 1]);  mul_375 = None
    sum_181: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_327, [0, 1]);  add_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_328: "f32[4096]" = torch.ops.aten.add.Tensor(add_306, sum_180);  add_306 = sum_180 = None
    add_329: "f32[4096]" = torch.ops.aten.add.Tensor(add_307, sum_181);  add_307 = sum_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_557: "f32[512, 4096]" = torch.ops.aten.view.default(mul_374, [512, 4096])
    mm_126: "f32[512, 4096]" = torch.ops.aten.mm.default(view_557, permute_146)
    permute_477: "f32[4096, 512]" = torch.ops.aten.permute.default(view_557, [1, 0])
    mm_127: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_477, view_40);  permute_477 = view_40 = None
    permute_478: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_127, [1, 0]);  mm_127 = None
    sum_182: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_557, [0], True);  view_557 = None
    view_558: "f32[4096]" = torch.ops.aten.view.default(sum_182, [4096]);  sum_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_330: "f32[4096]" = torch.ops.aten.add.Tensor(add_308, view_558);  add_308 = view_558 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    permute_479: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_478, [1, 0]);  permute_478 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_331: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_309, permute_479);  add_309 = permute_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_559: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_126, [1, 512, 4096]);  mm_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    view_560: "f32[1, 512, 64, 64]" = torch.ops.aten.view.default(view_559, [1, 512, 64, 64]);  view_559 = None
    permute_480: "f32[1, 64, 512, 64]" = torch.ops.aten.permute.default(view_560, [0, 2, 1, 3]);  view_560 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_561: "f32[64, 512, 64]" = torch.ops.aten.view.default(permute_480, [64, 512, 64]);  permute_480 = None
    bmm_64: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(permute_481, view_561);  permute_481 = None
    bmm_65: "f32[64, 512, 512]" = torch.ops.aten.bmm.default(view_561, permute_482);  view_561 = permute_482 = None
    view_562: "f32[1, 64, 512, 64]" = torch.ops.aten.view.default(bmm_64, [1, 64, 512, 64]);  bmm_64 = None
    view_563: "f32[1, 64, 512, 512]" = torch.ops.aten.view.default(bmm_65, [1, 64, 512, 512]);  bmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    mul_376: "f32[1, 64, 512, 512]" = torch.ops.aten.mul.Tensor(view_563, alias_49);  view_563 = None
    sum_183: "f32[1, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_376, [-1], True)
    mul_377: "f32[1, 64, 512, 512]" = torch.ops.aten.mul.Tensor(alias_49, sum_183);  alias_49 = sum_183 = None
    sub_131: "f32[1, 64, 512, 512]" = torch.ops.aten.sub.Tensor(mul_376, mul_377);  mul_376 = mul_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:332, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_62: "f32[1, 64, 512, 512]" = torch.ops.aten.div.Tensor(sub_131, 8.0);  sub_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_564: "f32[64, 512, 512]" = torch.ops.aten.view.default(div_62, [64, 512, 512]);  div_62 = None
    bmm_66: "f32[64, 64, 512]" = torch.ops.aten.bmm.default(permute_483, view_564);  permute_483 = None
    bmm_67: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(view_564, permute_484);  view_564 = permute_484 = None
    view_565: "f32[1, 64, 64, 512]" = torch.ops.aten.view.default(bmm_66, [1, 64, 64, 512]);  bmm_66 = None
    view_566: "f32[1, 64, 512, 64]" = torch.ops.aten.view.default(bmm_67, [1, 64, 512, 64]);  bmm_67 = None
    permute_485: "f32[1, 64, 512, 64]" = torch.ops.aten.permute.default(view_565, [0, 1, 3, 2]);  view_565 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_486: "f32[1, 512, 64, 64]" = torch.ops.aten.permute.default(view_562, [0, 2, 1, 3]);  view_562 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_59: "f32[1, 512, 64, 64]" = torch.ops.aten.clone.default(permute_486, memory_format = torch.contiguous_format);  permute_486 = None
    view_567: "f32[1, 512, 4096]" = torch.ops.aten.view.default(clone_59, [1, 512, 4096]);  clone_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_487: "f32[1, 512, 64, 64]" = torch.ops.aten.permute.default(permute_485, [0, 2, 1, 3]);  permute_485 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_568: "f32[1, 512, 4096]" = torch.ops.aten.view.default(permute_487, [1, 512, 4096]);  permute_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_488: "f32[1, 512, 64, 64]" = torch.ops.aten.permute.default(view_566, [0, 2, 1, 3]);  view_566 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_60: "f32[1, 512, 64, 64]" = torch.ops.aten.clone.default(permute_488, memory_format = torch.contiguous_format);  permute_488 = None
    view_569: "f32[1, 512, 4096]" = torch.ops.aten.view.default(clone_60, [1, 512, 4096]);  clone_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_570: "f32[512, 4096]" = torch.ops.aten.view.default(view_567, [512, 4096]);  view_567 = None
    mm_128: "f32[512, 4096]" = torch.ops.aten.mm.default(view_570, permute_159)
    permute_490: "f32[4096, 512]" = torch.ops.aten.permute.default(view_570, [1, 0])
    mm_129: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_490, view_24);  permute_490 = None
    permute_491: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_129, [1, 0]);  mm_129 = None
    sum_184: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_570, [0], True);  view_570 = None
    view_571: "f32[4096]" = torch.ops.aten.view.default(sum_184, [4096]);  sum_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_332: "f32[4096]" = torch.ops.aten.add.Tensor(add_310, view_571);  add_310 = view_571 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    permute_492: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_491, [1, 0]);  permute_491 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_333: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_311, permute_492);  add_311 = permute_492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_572: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_128, [1, 512, 4096]);  mm_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_334: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_374, view_572);  mul_374 = view_572 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_573: "f32[512, 4096]" = torch.ops.aten.view.default(view_568, [512, 4096]);  view_568 = None
    mm_130: "f32[512, 4096]" = torch.ops.aten.mm.default(view_573, permute_163)
    permute_494: "f32[4096, 512]" = torch.ops.aten.permute.default(view_573, [1, 0])
    mm_131: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_494, view_24);  permute_494 = None
    permute_495: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_131, [1, 0]);  mm_131 = None
    sum_185: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_573, [0], True);  view_573 = None
    view_574: "f32[4096]" = torch.ops.aten.view.default(sum_185, [4096]);  sum_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_335: "f32[4096]" = torch.ops.aten.add.Tensor(add_313, view_574);  add_313 = view_574 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    permute_496: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_495, [1, 0]);  permute_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_336: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_314, permute_496);  add_314 = permute_496 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_575: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_130, [1, 512, 4096]);  mm_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_337: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_334, view_575);  add_334 = view_575 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_576: "f32[512, 4096]" = torch.ops.aten.view.default(view_569, [512, 4096]);  view_569 = None
    mm_132: "f32[512, 4096]" = torch.ops.aten.mm.default(view_576, permute_167)
    permute_498: "f32[4096, 512]" = torch.ops.aten.permute.default(view_576, [1, 0])
    mm_133: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_498, view_24);  permute_498 = view_24 = None
    permute_499: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_133, [1, 0]);  mm_133 = None
    sum_186: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_576, [0], True);  view_576 = None
    view_577: "f32[4096]" = torch.ops.aten.view.default(sum_186, [4096]);  sum_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_338: "f32[4096]" = torch.ops.aten.add.Tensor(add_316, view_577);  add_316 = view_577 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    permute_500: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_499, [1, 0]);  permute_499 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_339: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_317, permute_500);  add_317 = permute_500 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_578: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_132, [1, 512, 4096]);  mm_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_340: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_337, view_578);  add_337 = view_578 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    mul_379: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_340, primals_22);  primals_22 = None
    mul_380: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_379, 4096)
    sum_187: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_379, [2], True)
    mul_381: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_379, mul_9);  mul_379 = None
    sum_188: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_381, [2], True);  mul_381 = None
    mul_382: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_9, sum_188);  sum_188 = None
    sub_133: "f32[1, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_380, sum_187);  mul_380 = sum_187 = None
    sub_134: "f32[1, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_133, mul_382);  sub_133 = mul_382 = None
    mul_383: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(div_63, sub_134);  div_63 = sub_134 = None
    mul_384: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_340, mul_9);  mul_9 = None
    sum_189: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_384, [0, 1]);  mul_384 = None
    sum_190: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_340, [0, 1]);  add_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_341: "f32[4096]" = torch.ops.aten.add.Tensor(add_319, sum_189);  add_319 = sum_189 = None
    add_342: "f32[4096]" = torch.ops.aten.add.Tensor(add_320, sum_190);  add_320 = sum_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_579: "f32[512, 4096]" = torch.ops.aten.view.default(mul_383, [512, 4096])
    mm_134: "f32[512, 16384]" = torch.ops.aten.mm.default(view_579, permute_138);  permute_138 = None
    permute_502: "f32[4096, 512]" = torch.ops.aten.permute.default(view_579, [1, 0])
    mm_135: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_502, view_22);  permute_502 = view_22 = None
    permute_503: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_135, [1, 0]);  mm_135 = None
    sum_191: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_579, [0], True);  view_579 = None
    view_580: "f32[4096]" = torch.ops.aten.view.default(sum_191, [4096]);  sum_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_343: "f32[4096]" = torch.ops.aten.add.Tensor(add_321, view_580);  add_321 = view_580 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    permute_504: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_503, [1, 0]);  permute_503 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_344: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_322, permute_504);  add_322 = permute_504 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_581: "f32[1, 512, 16384]" = torch.ops.aten.view.default(mm_134, [1, 512, 16384]);  mm_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_385: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(view_581, mul_5);  mul_5 = None
    mul_386: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(view_581, add_9);  view_581 = add_9 = None
    alias_50: "f32[1, 512, 16384]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    mul_387: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(alias_50, alias_50);  alias_50 = None
    sub_135: "f32[1, 512, 16384]" = torch.ops.aten.sub.Tensor(1, mul_387);  mul_387 = None
    mul_388: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_385, sub_135);  mul_385 = sub_135 = None
    mul_389: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_388, 0.7978845608028654);  mul_388 = None
    mul_390: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_389, 0.044715)
    pow_24: "f32[1, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_21, 2.0);  view_21 = None
    mul_391: "f32[1, 512, 16384]" = torch.ops.aten.mul.Scalar(pow_24, 3.0);  pow_24 = None
    mul_392: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_390, mul_391);  mul_390 = mul_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_345: "f32[1, 512, 16384]" = torch.ops.aten.add.Tensor(mul_389, mul_392);  mul_389 = mul_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_393: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_386, 0.5);  mul_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_346: "f32[1, 512, 16384]" = torch.ops.aten.add.Tensor(add_345, mul_393);  add_345 = mul_393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_582: "f32[512, 16384]" = torch.ops.aten.view.default(add_346, [512, 16384]);  add_346 = None
    mm_136: "f32[512, 4096]" = torch.ops.aten.mm.default(view_582, permute_142);  permute_142 = None
    permute_506: "f32[16384, 512]" = torch.ops.aten.permute.default(view_582, [1, 0])
    mm_137: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_506, view_20);  permute_506 = view_20 = None
    permute_507: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_137, [1, 0]);  mm_137 = None
    sum_192: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_582, [0], True);  view_582 = None
    view_583: "f32[16384]" = torch.ops.aten.view.default(sum_192, [16384]);  sum_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_347: "f32[16384]" = torch.ops.aten.add.Tensor(add_325, view_583);  add_325 = view_583 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    permute_508: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_507, [1, 0]);  permute_507 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_348: "f32[16384, 4096]" = torch.ops.aten.add.Tensor(add_326, permute_508);  add_326 = permute_508 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_584: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_136, [1, 512, 4096]);  mm_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_349: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_383, view_584);  mul_383 = view_584 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    mul_395: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_349, primals_16);  primals_16 = None
    mul_396: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_395, 4096)
    sum_193: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_395, [2], True)
    mul_397: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_395, mul_3);  mul_395 = None
    sum_194: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_397, [2], True);  mul_397 = None
    mul_398: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_3, sum_194);  sum_194 = None
    sub_137: "f32[1, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_396, sum_193);  mul_396 = sum_193 = None
    sub_138: "f32[1, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_137, mul_398);  sub_137 = mul_398 = None
    mul_399: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(div_64, sub_138);  div_64 = sub_138 = None
    mul_400: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_349, mul_3);  mul_3 = None
    sum_195: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_400, [0, 1]);  mul_400 = None
    sum_196: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_349, [0, 1]);  add_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_350: "f32[4096]" = torch.ops.aten.add.Tensor(add_328, sum_195);  add_328 = sum_195 = None
    add_351: "f32[4096]" = torch.ops.aten.add.Tensor(add_329, sum_196);  add_329 = sum_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_585: "f32[512, 4096]" = torch.ops.aten.view.default(mul_399, [512, 4096])
    mm_138: "f32[512, 4096]" = torch.ops.aten.mm.default(view_585, permute_146);  permute_146 = None
    permute_510: "f32[4096, 512]" = torch.ops.aten.permute.default(view_585, [1, 0])
    mm_139: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_510, view_18);  permute_510 = view_18 = None
    permute_511: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_139, [1, 0]);  mm_139 = None
    sum_197: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_585, [0], True);  view_585 = None
    view_586: "f32[4096]" = torch.ops.aten.view.default(sum_197, [4096]);  sum_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_352: "f32[4096]" = torch.ops.aten.add.Tensor(add_330, view_586);  add_330 = view_586 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    permute_512: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_511, [1, 0]);  permute_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_353: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_331, permute_512);  add_331 = permute_512 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_587: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_138, [1, 512, 4096]);  mm_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    view_588: "f32[1, 512, 64, 64]" = torch.ops.aten.view.default(view_587, [1, 512, 64, 64]);  view_587 = None
    permute_513: "f32[1, 64, 512, 64]" = torch.ops.aten.permute.default(view_588, [0, 2, 1, 3]);  view_588 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_589: "f32[64, 512, 64]" = torch.ops.aten.view.default(permute_513, [64, 512, 64]);  permute_513 = None
    bmm_68: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(permute_514, view_589);  permute_514 = None
    bmm_69: "f32[64, 512, 512]" = torch.ops.aten.bmm.default(view_589, permute_515);  view_589 = permute_515 = None
    view_590: "f32[1, 64, 512, 64]" = torch.ops.aten.view.default(bmm_68, [1, 64, 512, 64]);  bmm_68 = None
    view_591: "f32[1, 64, 512, 512]" = torch.ops.aten.view.default(bmm_69, [1, 64, 512, 512]);  bmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    mul_401: "f32[1, 64, 512, 512]" = torch.ops.aten.mul.Tensor(view_591, alias_51);  view_591 = None
    sum_198: "f32[1, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_401, [-1], True)
    mul_402: "f32[1, 64, 512, 512]" = torch.ops.aten.mul.Tensor(alias_51, sum_198);  alias_51 = sum_198 = None
    sub_139: "f32[1, 64, 512, 512]" = torch.ops.aten.sub.Tensor(mul_401, mul_402);  mul_401 = mul_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:332, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_65: "f32[1, 64, 512, 512]" = torch.ops.aten.div.Tensor(sub_139, 8.0);  sub_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_592: "f32[64, 512, 512]" = torch.ops.aten.view.default(div_65, [64, 512, 512]);  div_65 = None
    bmm_70: "f32[64, 64, 512]" = torch.ops.aten.bmm.default(permute_516, view_592);  permute_516 = None
    bmm_71: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(view_592, permute_517);  view_592 = permute_517 = None
    view_593: "f32[1, 64, 64, 512]" = torch.ops.aten.view.default(bmm_70, [1, 64, 64, 512]);  bmm_70 = None
    view_594: "f32[1, 64, 512, 64]" = torch.ops.aten.view.default(bmm_71, [1, 64, 512, 64]);  bmm_71 = None
    permute_518: "f32[1, 64, 512, 64]" = torch.ops.aten.permute.default(view_593, [0, 1, 3, 2]);  view_593 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_519: "f32[1, 512, 64, 64]" = torch.ops.aten.permute.default(view_590, [0, 2, 1, 3]);  view_590 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_61: "f32[1, 512, 64, 64]" = torch.ops.aten.clone.default(permute_519, memory_format = torch.contiguous_format);  permute_519 = None
    view_595: "f32[1, 512, 4096]" = torch.ops.aten.view.default(clone_61, [1, 512, 4096]);  clone_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_520: "f32[1, 512, 64, 64]" = torch.ops.aten.permute.default(permute_518, [0, 2, 1, 3]);  permute_518 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_596: "f32[1, 512, 4096]" = torch.ops.aten.view.default(permute_520, [1, 512, 4096]);  permute_520 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_521: "f32[1, 512, 64, 64]" = torch.ops.aten.permute.default(view_594, [0, 2, 1, 3]);  view_594 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_62: "f32[1, 512, 64, 64]" = torch.ops.aten.clone.default(permute_521, memory_format = torch.contiguous_format);  permute_521 = None
    view_597: "f32[1, 512, 4096]" = torch.ops.aten.view.default(clone_62, [1, 512, 4096]);  clone_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_598: "f32[512, 4096]" = torch.ops.aten.view.default(view_595, [512, 4096]);  view_595 = None
    mm_140: "f32[512, 4096]" = torch.ops.aten.mm.default(view_598, permute_159);  permute_159 = None
    permute_523: "f32[4096, 512]" = torch.ops.aten.permute.default(view_598, [1, 0])
    mm_141: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_523, view_2);  permute_523 = None
    permute_524: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_141, [1, 0]);  mm_141 = None
    sum_199: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_598, [0], True);  view_598 = None
    view_599: "f32[4096]" = torch.ops.aten.view.default(sum_199, [4096]);  sum_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_354: "f32[4096]" = torch.ops.aten.add.Tensor(add_332, view_599);  add_332 = view_599 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    permute_525: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_524, [1, 0]);  permute_524 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_355: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_333, permute_525);  add_333 = permute_525 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_600: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_140, [1, 512, 4096]);  mm_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_356: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_399, view_600);  mul_399 = view_600 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_601: "f32[512, 4096]" = torch.ops.aten.view.default(view_596, [512, 4096]);  view_596 = None
    mm_142: "f32[512, 4096]" = torch.ops.aten.mm.default(view_601, permute_163);  permute_163 = None
    permute_527: "f32[4096, 512]" = torch.ops.aten.permute.default(view_601, [1, 0])
    mm_143: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_527, view_2);  permute_527 = None
    permute_528: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_143, [1, 0]);  mm_143 = None
    sum_200: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_601, [0], True);  view_601 = None
    view_602: "f32[4096]" = torch.ops.aten.view.default(sum_200, [4096]);  sum_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_357: "f32[4096]" = torch.ops.aten.add.Tensor(add_335, view_602);  add_335 = view_602 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    permute_529: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_528, [1, 0]);  permute_528 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_358: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_336, permute_529);  add_336 = permute_529 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_603: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_142, [1, 512, 4096]);  mm_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_359: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_356, view_603);  add_356 = view_603 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_604: "f32[512, 4096]" = torch.ops.aten.view.default(view_597, [512, 4096]);  view_597 = None
    mm_144: "f32[512, 4096]" = torch.ops.aten.mm.default(view_604, permute_167);  permute_167 = None
    permute_531: "f32[4096, 512]" = torch.ops.aten.permute.default(view_604, [1, 0])
    mm_145: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_531, view_2);  permute_531 = view_2 = None
    permute_532: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_145, [1, 0]);  mm_145 = None
    sum_201: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_604, [0], True);  view_604 = None
    view_605: "f32[4096]" = torch.ops.aten.view.default(sum_201, [4096]);  sum_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_360: "f32[4096]" = torch.ops.aten.add.Tensor(add_338, view_605);  add_338 = view_605 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    permute_533: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_532, [1, 0]);  permute_532 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_361: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_339, permute_533);  add_339 = permute_533 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_606: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_144, [1, 512, 4096]);  mm_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_362: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_359, view_606);  add_359 = view_606 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:467, code: hidden_states = self.embedding_hidden_mapping_in(hidden_states)
    view_607: "f32[512, 4096]" = torch.ops.aten.view.default(add_362, [512, 4096]);  add_362 = None
    mm_146: "f32[512, 128]" = torch.ops.aten.mm.default(view_607, permute_534);  permute_534 = None
    permute_535: "f32[4096, 512]" = torch.ops.aten.permute.default(view_607, [1, 0])
    mm_147: "f32[4096, 128]" = torch.ops.aten.mm.default(permute_535, view);  permute_535 = view = None
    permute_536: "f32[128, 4096]" = torch.ops.aten.permute.default(mm_147, [1, 0]);  mm_147 = None
    sum_202: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_607, [0], True);  view_607 = None
    view_608: "f32[4096]" = torch.ops.aten.view.default(sum_202, [4096]);  sum_202 = None
    permute_537: "f32[4096, 128]" = torch.ops.aten.permute.default(permute_536, [1, 0]);  permute_536 = None
    view_609: "f32[1, 512, 128]" = torch.ops.aten.view.default(mm_146, [1, 512, 128]);  mm_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:257, code: embeddings = self.LayerNorm(embeddings)
    mul_404: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(view_609, primals_4);  primals_4 = None
    mul_405: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(mul_404, 128)
    sum_203: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_404, [2], True)
    mul_406: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(mul_404, mul_1);  mul_404 = None
    sum_204: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_406, [2], True);  mul_406 = None
    mul_407: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(mul_1, sum_204);  sum_204 = None
    sub_141: "f32[1, 512, 128]" = torch.ops.aten.sub.Tensor(mul_405, sum_203);  mul_405 = sum_203 = None
    sub_142: "f32[1, 512, 128]" = torch.ops.aten.sub.Tensor(sub_141, mul_407);  sub_141 = mul_407 = None
    mul_408: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(div_66, sub_142);  div_66 = sub_142 = None
    mul_409: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(view_609, mul_1);  mul_1 = None
    sum_205: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_409, [0, 1]);  mul_409 = None
    sum_206: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_609, [0, 1]);  view_609 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:255, code: position_embeddings = self.position_embeddings(position_ids)
    eq: "b8[1, 512]" = torch.ops.aten.eq.Scalar(slice_2, -1)
    unsqueeze_8: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq, -1);  eq = None
    where_8: "f32[1, 512, 128]" = torch.ops.aten.where.self(unsqueeze_8, full_default_2, mul_408);  unsqueeze_8 = None
    full_default_12: "f32[512, 128]" = torch.ops.aten.full.default([512, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put: "f32[512, 128]" = torch.ops.aten._unsafe_index_put.default(full_default_12, [slice_2], where_8, True);  full_default_12 = slice_2 = where_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:251, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
    eq_1: "b8[1, 512]" = torch.ops.aten.eq.Scalar(expand, -1)
    unsqueeze_9: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_1, -1);  eq_1 = None
    where_9: "f32[1, 512, 128]" = torch.ops.aten.where.self(unsqueeze_9, full_default_2, mul_408);  unsqueeze_9 = None
    full_default_14: "f32[2, 128]" = torch.ops.aten.full.default([2, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put_1: "f32[2, 128]" = torch.ops.aten._unsafe_index_put.default(full_default_14, [expand], where_9, True);  full_default_14 = expand = where_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:250, code: inputs_embeds = self.word_embeddings(input_ids)
    eq_2: "b8[1, 512]" = torch.ops.aten.eq.Scalar(primals_28, 0)
    unsqueeze_10: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_2, -1);  eq_2 = None
    where_10: "f32[1, 512, 128]" = torch.ops.aten.where.self(unsqueeze_10, full_default_2, mul_408);  unsqueeze_10 = full_default_2 = mul_408 = None
    full_default_16: "f32[30000, 128]" = torch.ops.aten.full.default([30000, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put_2: "f32[30000, 128]" = torch.ops.aten._unsafe_index_put.default(full_default_16, [primals_28], where_10, True);  full_default_16 = primals_28 = where_10 = None
    return [_unsafe_index_put_2, _unsafe_index_put_1, _unsafe_index_put, sum_205, sum_206, permute_537, view_608, add_361, add_360, add_358, add_357, add_355, add_354, add_353, add_352, add_350, add_351, add_348, add_347, add_344, add_343, add_341, add_342, permute_137, view_269, None, None, None, None, None]
    