from __future__ import annotations



def forward(self, primals_4: "f32[128]", primals_16: "f32[768]", primals_22: "f32[768]", primals_26: "f32[128]", primals_32: "i64[4, 512]", expand: "i64[4, 512]", slice_2: "i64[1, 512]", mul_1: "f32[4, 512, 128]", view: "f32[2048, 128]", view_2: "f32[2048, 768]", view_18: "f32[2048, 768]", mul_3: "f32[4, 512, 768]", view_20: "f32[2048, 768]", addmm_5: "f32[2048, 3072]", tanh: "f32[4, 512, 3072]", view_22: "f32[2048, 3072]", mul_9: "f32[4, 512, 768]", view_24: "f32[2048, 768]", view_40: "f32[2048, 768]", mul_11: "f32[4, 512, 768]", view_42: "f32[2048, 768]", addmm_11: "f32[2048, 3072]", tanh_1: "f32[4, 512, 3072]", view_44: "f32[2048, 3072]", mul_17: "f32[4, 512, 768]", view_46: "f32[2048, 768]", view_62: "f32[2048, 768]", mul_19: "f32[4, 512, 768]", view_64: "f32[2048, 768]", addmm_17: "f32[2048, 3072]", tanh_2: "f32[4, 512, 3072]", view_66: "f32[2048, 3072]", mul_25: "f32[4, 512, 768]", view_68: "f32[2048, 768]", view_84: "f32[2048, 768]", mul_27: "f32[4, 512, 768]", view_86: "f32[2048, 768]", addmm_23: "f32[2048, 3072]", tanh_3: "f32[4, 512, 3072]", view_88: "f32[2048, 3072]", mul_33: "f32[4, 512, 768]", view_90: "f32[2048, 768]", view_106: "f32[2048, 768]", mul_35: "f32[4, 512, 768]", view_108: "f32[2048, 768]", addmm_29: "f32[2048, 3072]", tanh_4: "f32[4, 512, 3072]", view_110: "f32[2048, 3072]", mul_41: "f32[4, 512, 768]", view_112: "f32[2048, 768]", view_128: "f32[2048, 768]", mul_43: "f32[4, 512, 768]", view_130: "f32[2048, 768]", addmm_35: "f32[2048, 3072]", tanh_5: "f32[4, 512, 3072]", view_132: "f32[2048, 3072]", mul_49: "f32[4, 512, 768]", view_134: "f32[2048, 768]", view_150: "f32[2048, 768]", mul_51: "f32[4, 512, 768]", view_152: "f32[2048, 768]", addmm_41: "f32[2048, 3072]", tanh_6: "f32[4, 512, 3072]", view_154: "f32[2048, 3072]", mul_57: "f32[4, 512, 768]", view_156: "f32[2048, 768]", view_172: "f32[2048, 768]", mul_59: "f32[4, 512, 768]", view_174: "f32[2048, 768]", addmm_47: "f32[2048, 3072]", tanh_7: "f32[4, 512, 3072]", view_176: "f32[2048, 3072]", mul_65: "f32[4, 512, 768]", view_178: "f32[2048, 768]", view_194: "f32[2048, 768]", mul_67: "f32[4, 512, 768]", view_196: "f32[2048, 768]", addmm_53: "f32[2048, 3072]", tanh_8: "f32[4, 512, 3072]", view_198: "f32[2048, 3072]", mul_73: "f32[4, 512, 768]", view_200: "f32[2048, 768]", view_216: "f32[2048, 768]", mul_75: "f32[4, 512, 768]", view_218: "f32[2048, 768]", addmm_59: "f32[2048, 3072]", tanh_9: "f32[4, 512, 3072]", view_220: "f32[2048, 3072]", mul_81: "f32[4, 512, 768]", view_222: "f32[2048, 768]", view_238: "f32[2048, 768]", mul_83: "f32[4, 512, 768]", view_240: "f32[2048, 768]", addmm_65: "f32[2048, 3072]", tanh_10: "f32[4, 512, 3072]", view_242: "f32[2048, 3072]", mul_89: "f32[4, 512, 768]", view_244: "f32[2048, 768]", view_260: "f32[2048, 768]", mul_91: "f32[4, 512, 768]", view_262: "f32[2048, 768]", addmm_71: "f32[2048, 3072]", tanh_11: "f32[4, 512, 3072]", view_264: "f32[2048, 3072]", mul_97: "f32[4, 512, 768]", view_266: "f32[2048, 768]", addmm_73: "f32[2048, 128]", tanh_12: "f32[4, 512, 128]", getitem_51: "f32[4, 512, 1]", rsqrt_25: "f32[4, 512, 1]", view_268: "f32[2048, 128]", permute_135: "f32[30000, 128]", permute_139: "f32[128, 768]", div_25: "f32[4, 512, 1]", permute_143: "f32[768, 3072]", permute_147: "f32[3072, 768]", div_26: "f32[4, 512, 1]", permute_151: "f32[768, 768]", permute_156: "f32[48, 512, 512]", permute_157: "f32[48, 64, 512]", alias_27: "f32[4, 12, 512, 512]", permute_158: "f32[48, 64, 512]", permute_159: "f32[48, 512, 64]", permute_164: "f32[768, 768]", permute_168: "f32[768, 768]", permute_172: "f32[768, 768]", div_28: "f32[4, 512, 1]", div_29: "f32[4, 512, 1]", permute_189: "f32[48, 512, 512]", permute_190: "f32[48, 64, 512]", alias_29: "f32[4, 12, 512, 512]", permute_191: "f32[48, 64, 512]", permute_192: "f32[48, 512, 64]", div_31: "f32[4, 512, 1]", div_32: "f32[4, 512, 1]", permute_222: "f32[48, 512, 512]", permute_223: "f32[48, 64, 512]", alias_31: "f32[4, 12, 512, 512]", permute_224: "f32[48, 64, 512]", permute_225: "f32[48, 512, 64]", div_34: "f32[4, 512, 1]", div_35: "f32[4, 512, 1]", permute_255: "f32[48, 512, 512]", permute_256: "f32[48, 64, 512]", alias_33: "f32[4, 12, 512, 512]", permute_257: "f32[48, 64, 512]", permute_258: "f32[48, 512, 64]", div_37: "f32[4, 512, 1]", div_38: "f32[4, 512, 1]", permute_288: "f32[48, 512, 512]", permute_289: "f32[48, 64, 512]", alias_35: "f32[4, 12, 512, 512]", permute_290: "f32[48, 64, 512]", permute_291: "f32[48, 512, 64]", div_40: "f32[4, 512, 1]", div_41: "f32[4, 512, 1]", permute_321: "f32[48, 512, 512]", permute_322: "f32[48, 64, 512]", alias_37: "f32[4, 12, 512, 512]", permute_323: "f32[48, 64, 512]", permute_324: "f32[48, 512, 64]", div_43: "f32[4, 512, 1]", div_44: "f32[4, 512, 1]", permute_354: "f32[48, 512, 512]", permute_355: "f32[48, 64, 512]", alias_39: "f32[4, 12, 512, 512]", permute_356: "f32[48, 64, 512]", permute_357: "f32[48, 512, 64]", div_46: "f32[4, 512, 1]", div_47: "f32[4, 512, 1]", permute_387: "f32[48, 512, 512]", permute_388: "f32[48, 64, 512]", alias_41: "f32[4, 12, 512, 512]", permute_389: "f32[48, 64, 512]", permute_390: "f32[48, 512, 64]", div_49: "f32[4, 512, 1]", div_50: "f32[4, 512, 1]", permute_420: "f32[48, 512, 512]", permute_421: "f32[48, 64, 512]", alias_43: "f32[4, 12, 512, 512]", permute_422: "f32[48, 64, 512]", permute_423: "f32[48, 512, 64]", div_52: "f32[4, 512, 1]", div_53: "f32[4, 512, 1]", permute_453: "f32[48, 512, 512]", permute_454: "f32[48, 64, 512]", alias_45: "f32[4, 12, 512, 512]", permute_455: "f32[48, 64, 512]", permute_456: "f32[48, 512, 64]", div_55: "f32[4, 512, 1]", div_56: "f32[4, 512, 1]", permute_486: "f32[48, 512, 512]", permute_487: "f32[48, 64, 512]", alias_47: "f32[4, 12, 512, 512]", permute_488: "f32[48, 64, 512]", permute_489: "f32[48, 512, 64]", div_58: "f32[4, 512, 1]", div_59: "f32[4, 512, 1]", permute_519: "f32[48, 512, 512]", permute_520: "f32[48, 64, 512]", alias_49: "f32[4, 12, 512, 512]", permute_521: "f32[48, 64, 512]", permute_522: "f32[48, 512, 64]", permute_539: "f32[768, 128]", div_61: "f32[4, 512, 1]", tangents_1: "f32[4, 512, 30000]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_21: "f32[4, 512, 3072]" = torch.ops.aten.view.default(addmm_5, [4, 512, 3072]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_5: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_21, 0.5)
    alias_1: "f32[4, 512, 3072]" = torch.ops.aten.alias.default(tanh)
    add_9: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_43: "f32[4, 512, 3072]" = torch.ops.aten.view.default(addmm_11, [4, 512, 3072]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_13: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_43, 0.5)
    alias_3: "f32[4, 512, 3072]" = torch.ops.aten.alias.default(tanh_1)
    add_18: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_1, 1.0);  tanh_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_65: "f32[4, 512, 3072]" = torch.ops.aten.view.default(addmm_17, [4, 512, 3072]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_21: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_65, 0.5)
    alias_5: "f32[4, 512, 3072]" = torch.ops.aten.alias.default(tanh_2)
    add_27: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_2, 1.0);  tanh_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_87: "f32[4, 512, 3072]" = torch.ops.aten.view.default(addmm_23, [4, 512, 3072]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_29: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_87, 0.5)
    alias_7: "f32[4, 512, 3072]" = torch.ops.aten.alias.default(tanh_3)
    add_36: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_3, 1.0);  tanh_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_109: "f32[4, 512, 3072]" = torch.ops.aten.view.default(addmm_29, [4, 512, 3072]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_37: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_109, 0.5)
    alias_9: "f32[4, 512, 3072]" = torch.ops.aten.alias.default(tanh_4)
    add_45: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_4, 1.0);  tanh_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_131: "f32[4, 512, 3072]" = torch.ops.aten.view.default(addmm_35, [4, 512, 3072]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_45: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_131, 0.5)
    alias_11: "f32[4, 512, 3072]" = torch.ops.aten.alias.default(tanh_5)
    add_54: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_5, 1.0);  tanh_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_153: "f32[4, 512, 3072]" = torch.ops.aten.view.default(addmm_41, [4, 512, 3072]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_53: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_153, 0.5)
    alias_13: "f32[4, 512, 3072]" = torch.ops.aten.alias.default(tanh_6)
    add_63: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_6, 1.0);  tanh_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_175: "f32[4, 512, 3072]" = torch.ops.aten.view.default(addmm_47, [4, 512, 3072]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_61: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_175, 0.5)
    alias_15: "f32[4, 512, 3072]" = torch.ops.aten.alias.default(tanh_7)
    add_72: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_7, 1.0);  tanh_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_197: "f32[4, 512, 3072]" = torch.ops.aten.view.default(addmm_53, [4, 512, 3072]);  addmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_69: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_197, 0.5)
    alias_17: "f32[4, 512, 3072]" = torch.ops.aten.alias.default(tanh_8)
    add_81: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_8, 1.0);  tanh_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_219: "f32[4, 512, 3072]" = torch.ops.aten.view.default(addmm_59, [4, 512, 3072]);  addmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_77: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_219, 0.5)
    alias_19: "f32[4, 512, 3072]" = torch.ops.aten.alias.default(tanh_9)
    add_90: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_9, 1.0);  tanh_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_241: "f32[4, 512, 3072]" = torch.ops.aten.view.default(addmm_65, [4, 512, 3072]);  addmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_85: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_241, 0.5)
    alias_21: "f32[4, 512, 3072]" = torch.ops.aten.alias.default(tanh_10)
    add_99: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_10, 1.0);  tanh_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_263: "f32[4, 512, 3072]" = torch.ops.aten.view.default(addmm_71, [4, 512, 3072]);  addmm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_93: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_263, 0.5)
    alias_23: "f32[4, 512, 3072]" = torch.ops.aten.alias.default(tanh_11)
    add_108: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_11, 1.0);  tanh_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:880, code: hidden_states = self.dense(hidden_states)
    view_267: "f32[4, 512, 128]" = torch.ops.aten.view.default(addmm_73, [4, 512, 128]);  addmm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_99: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(view_267, 0.5)
    alias_24: "f32[4, 512, 128]" = torch.ops.aten.alias.default(tanh_12)
    add_113: "f32[4, 512, 128]" = torch.ops.aten.add.Tensor(tanh_12, 1.0);  tanh_12 = None
    mul_102: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(mul_99, add_113)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:882, code: hidden_states = self.LayerNorm(hidden_states)
    sub_38: "f32[4, 512, 128]" = torch.ops.aten.sub.Tensor(mul_102, getitem_51);  mul_102 = getitem_51 = None
    mul_103: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_25);  sub_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:883, code: hidden_states = self.decoder(hidden_states)
    view_270: "f32[2048, 30000]" = torch.ops.aten.view.default(tangents_1, [2048, 30000]);  tangents_1 = None
    mm: "f32[2048, 128]" = torch.ops.aten.mm.default(view_270, permute_135);  permute_135 = None
    permute_136: "f32[30000, 2048]" = torch.ops.aten.permute.default(view_270, [1, 0])
    mm_1: "f32[30000, 128]" = torch.ops.aten.mm.default(permute_136, view_268);  permute_136 = view_268 = None
    permute_137: "f32[128, 30000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_13: "f32[1, 30000]" = torch.ops.aten.sum.dim_IntList(view_270, [0], True);  view_270 = None
    view_271: "f32[30000]" = torch.ops.aten.view.default(sum_13, [30000]);  sum_13 = None
    permute_138: "f32[30000, 128]" = torch.ops.aten.permute.default(permute_137, [1, 0]);  permute_137 = None
    view_272: "f32[4, 512, 128]" = torch.ops.aten.view.default(mm, [4, 512, 128]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:882, code: hidden_states = self.LayerNorm(hidden_states)
    mul_106: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(view_272, primals_26);  primals_26 = None
    mul_107: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(mul_106, 128)
    sum_14: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_106, [2], True)
    mul_108: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(mul_106, mul_103);  mul_106 = None
    sum_15: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_108, [2], True);  mul_108 = None
    mul_109: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(mul_103, sum_15);  sum_15 = None
    sub_40: "f32[4, 512, 128]" = torch.ops.aten.sub.Tensor(mul_107, sum_14);  mul_107 = sum_14 = None
    sub_41: "f32[4, 512, 128]" = torch.ops.aten.sub.Tensor(sub_40, mul_109);  sub_40 = mul_109 = None
    div_24: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 128);  rsqrt_25 = None
    mul_110: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(div_24, sub_41);  div_24 = sub_41 = None
    mul_111: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(view_272, mul_103);  mul_103 = None
    sum_16: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_111, [0, 1]);  mul_111 = None
    sum_17: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_272, [0, 1]);  view_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_112: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(mul_110, mul_99);  mul_99 = None
    mul_113: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(mul_110, add_113);  mul_110 = add_113 = None
    alias_25: "f32[4, 512, 128]" = torch.ops.aten.alias.default(alias_24);  alias_24 = None
    mul_114: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(alias_25, alias_25);  alias_25 = None
    sub_42: "f32[4, 512, 128]" = torch.ops.aten.sub.Tensor(1, mul_114);  mul_114 = None
    mul_115: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(mul_112, sub_42);  mul_112 = sub_42 = None
    mul_116: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(mul_115, 0.7978845608028654);  mul_115 = None
    mul_117: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(mul_116, 0.044715)
    pow_14: "f32[4, 512, 128]" = torch.ops.aten.pow.Tensor_Scalar(view_267, 2.0);  view_267 = None
    mul_118: "f32[4, 512, 128]" = torch.ops.aten.mul.Scalar(pow_14, 3.0);  pow_14 = None
    mul_119: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(mul_117, mul_118);  mul_117 = mul_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_116: "f32[4, 512, 128]" = torch.ops.aten.add.Tensor(mul_116, mul_119);  mul_116 = mul_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_120: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(mul_113, 0.5);  mul_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_117: "f32[4, 512, 128]" = torch.ops.aten.add.Tensor(add_116, mul_120);  add_116 = mul_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:880, code: hidden_states = self.dense(hidden_states)
    view_273: "f32[2048, 128]" = torch.ops.aten.view.default(add_117, [2048, 128]);  add_117 = None
    mm_2: "f32[2048, 768]" = torch.ops.aten.mm.default(view_273, permute_139);  permute_139 = None
    permute_140: "f32[128, 2048]" = torch.ops.aten.permute.default(view_273, [1, 0])
    mm_3: "f32[128, 768]" = torch.ops.aten.mm.default(permute_140, view_266);  permute_140 = view_266 = None
    permute_141: "f32[768, 128]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
    sum_18: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_273, [0], True);  view_273 = None
    view_274: "f32[128]" = torch.ops.aten.view.default(sum_18, [128]);  sum_18 = None
    permute_142: "f32[128, 768]" = torch.ops.aten.permute.default(permute_141, [1, 0]);  permute_141 = None
    view_275: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_2, [4, 512, 768]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    mul_122: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_275, primals_22)
    mul_123: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_122, 768)
    sum_19: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_122, [2], True)
    mul_124: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_122, mul_97);  mul_122 = None
    sum_20: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_124, [2], True);  mul_124 = None
    mul_125: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_97, sum_20);  sum_20 = None
    sub_44: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_123, sum_19);  mul_123 = sum_19 = None
    sub_45: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_44, mul_125);  sub_44 = mul_125 = None
    mul_126: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_25, sub_45);  div_25 = sub_45 = None
    mul_127: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_275, mul_97);  mul_97 = None
    sum_21: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_127, [0, 1]);  mul_127 = None
    sum_22: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_275, [0, 1]);  view_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_276: "f32[2048, 768]" = torch.ops.aten.view.default(mul_126, [2048, 768])
    mm_4: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_276, permute_143)
    permute_144: "f32[768, 2048]" = torch.ops.aten.permute.default(view_276, [1, 0])
    mm_5: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_144, view_264);  permute_144 = view_264 = None
    permute_145: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
    sum_23: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_276, [0], True);  view_276 = None
    view_277: "f32[768]" = torch.ops.aten.view.default(sum_23, [768]);  sum_23 = None
    permute_146: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_145, [1, 0]);  permute_145 = None
    view_278: "f32[4, 512, 3072]" = torch.ops.aten.view.default(mm_4, [4, 512, 3072]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_128: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_278, mul_93);  mul_93 = None
    mul_129: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_278, add_108);  view_278 = add_108 = None
    alias_26: "f32[4, 512, 3072]" = torch.ops.aten.alias.default(alias_23);  alias_23 = None
    mul_130: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(alias_26, alias_26);  alias_26 = None
    sub_46: "f32[4, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_130);  mul_130 = None
    mul_131: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_128, sub_46);  mul_128 = sub_46 = None
    mul_132: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_131, 0.7978845608028654);  mul_131 = None
    mul_133: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_132, 0.044715)
    pow_15: "f32[4, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_263, 2.0);  view_263 = None
    mul_134: "f32[4, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_15, 3.0);  pow_15 = None
    mul_135: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_133, mul_134);  mul_133 = mul_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_118: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_132, mul_135);  mul_132 = mul_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_136: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_129, 0.5);  mul_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_119: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(add_118, mul_136);  add_118 = mul_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_279: "f32[2048, 3072]" = torch.ops.aten.view.default(add_119, [2048, 3072]);  add_119 = None
    mm_6: "f32[2048, 768]" = torch.ops.aten.mm.default(view_279, permute_147)
    permute_148: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_279, [1, 0])
    mm_7: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_148, view_262);  permute_148 = view_262 = None
    permute_149: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
    sum_24: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_279, [0], True);  view_279 = None
    view_280: "f32[3072]" = torch.ops.aten.view.default(sum_24, [3072]);  sum_24 = None
    permute_150: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_149, [1, 0]);  permute_149 = None
    view_281: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_6, [4, 512, 768]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_120: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_126, view_281);  mul_126 = view_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    mul_138: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_120, primals_16)
    mul_139: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_138, 768)
    sum_25: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_138, [2], True)
    mul_140: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_138, mul_91);  mul_138 = None
    sum_26: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_140, [2], True);  mul_140 = None
    mul_141: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_91, sum_26);  sum_26 = None
    sub_48: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_139, sum_25);  mul_139 = sum_25 = None
    sub_49: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_48, mul_141);  sub_48 = mul_141 = None
    mul_142: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_26, sub_49);  div_26 = sub_49 = None
    mul_143: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_120, mul_91);  mul_91 = None
    sum_27: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_143, [0, 1]);  mul_143 = None
    sum_28: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_120, [0, 1]);  add_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_282: "f32[2048, 768]" = torch.ops.aten.view.default(mul_142, [2048, 768])
    mm_8: "f32[2048, 768]" = torch.ops.aten.mm.default(view_282, permute_151)
    permute_152: "f32[768, 2048]" = torch.ops.aten.permute.default(view_282, [1, 0])
    mm_9: "f32[768, 768]" = torch.ops.aten.mm.default(permute_152, view_260);  permute_152 = view_260 = None
    permute_153: "f32[768, 768]" = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
    sum_29: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_282, [0], True);  view_282 = None
    view_283: "f32[768]" = torch.ops.aten.view.default(sum_29, [768]);  sum_29 = None
    permute_154: "f32[768, 768]" = torch.ops.aten.permute.default(permute_153, [1, 0]);  permute_153 = None
    view_284: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_8, [4, 512, 768]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    view_285: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_284, [4, 512, 12, 64]);  view_284 = None
    permute_155: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_285, [0, 2, 1, 3]);  view_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    clone_73: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_155, memory_format = torch.contiguous_format);  permute_155 = None
    view_286: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_73, [48, 512, 64]);  clone_73 = None
    bmm_24: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_156, view_286);  permute_156 = None
    bmm_25: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_286, permute_157);  view_286 = permute_157 = None
    view_287: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_24, [4, 12, 512, 64]);  bmm_24 = None
    view_288: "f32[4, 12, 512, 512]" = torch.ops.aten.view.default(bmm_25, [4, 12, 512, 512]);  bmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    mul_144: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_288, alias_27);  view_288 = None
    sum_30: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_144, [-1], True)
    mul_145: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_27, sum_30);  alias_27 = sum_30 = None
    sub_50: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_144, mul_145);  mul_144 = mul_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:332, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_27: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(sub_50, 8.0);  sub_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_289: "f32[48, 512, 512]" = torch.ops.aten.view.default(div_27, [48, 512, 512]);  div_27 = None
    bmm_26: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_158, view_289);  permute_158 = None
    bmm_27: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_289, permute_159);  view_289 = permute_159 = None
    view_290: "f32[4, 12, 64, 512]" = torch.ops.aten.view.default(bmm_26, [4, 12, 64, 512]);  bmm_26 = None
    view_291: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_27, [4, 12, 512, 64]);  bmm_27 = None
    permute_160: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_290, [0, 1, 3, 2]);  view_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_161: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_287, [0, 2, 1, 3]);  view_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_74: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None
    view_292: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_74, [4, 512, 768]);  clone_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_162: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(permute_160, [0, 2, 1, 3]);  permute_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_293: "f32[4, 512, 768]" = torch.ops.aten.view.default(permute_162, [4, 512, 768]);  permute_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_163: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_291, [0, 2, 1, 3]);  view_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_75: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_163, memory_format = torch.contiguous_format);  permute_163 = None
    view_294: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_75, [4, 512, 768]);  clone_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_295: "f32[2048, 768]" = torch.ops.aten.view.default(view_292, [2048, 768]);  view_292 = None
    mm_10: "f32[2048, 768]" = torch.ops.aten.mm.default(view_295, permute_164)
    permute_165: "f32[768, 2048]" = torch.ops.aten.permute.default(view_295, [1, 0])
    mm_11: "f32[768, 768]" = torch.ops.aten.mm.default(permute_165, view_244);  permute_165 = None
    permute_166: "f32[768, 768]" = torch.ops.aten.permute.default(mm_11, [1, 0]);  mm_11 = None
    sum_31: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_295, [0], True);  view_295 = None
    view_296: "f32[768]" = torch.ops.aten.view.default(sum_31, [768]);  sum_31 = None
    permute_167: "f32[768, 768]" = torch.ops.aten.permute.default(permute_166, [1, 0]);  permute_166 = None
    view_297: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_10, [4, 512, 768]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_121: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_142, view_297);  mul_142 = view_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    clone_76: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_293, memory_format = torch.contiguous_format);  view_293 = None
    view_298: "f32[2048, 768]" = torch.ops.aten.view.default(clone_76, [2048, 768]);  clone_76 = None
    mm_12: "f32[2048, 768]" = torch.ops.aten.mm.default(view_298, permute_168)
    permute_169: "f32[768, 2048]" = torch.ops.aten.permute.default(view_298, [1, 0])
    mm_13: "f32[768, 768]" = torch.ops.aten.mm.default(permute_169, view_244);  permute_169 = None
    permute_170: "f32[768, 768]" = torch.ops.aten.permute.default(mm_13, [1, 0]);  mm_13 = None
    sum_32: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_298, [0], True);  view_298 = None
    view_299: "f32[768]" = torch.ops.aten.view.default(sum_32, [768]);  sum_32 = None
    permute_171: "f32[768, 768]" = torch.ops.aten.permute.default(permute_170, [1, 0]);  permute_170 = None
    view_300: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_12, [4, 512, 768]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_122: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_121, view_300);  add_121 = view_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_301: "f32[2048, 768]" = torch.ops.aten.view.default(view_294, [2048, 768]);  view_294 = None
    mm_14: "f32[2048, 768]" = torch.ops.aten.mm.default(view_301, permute_172)
    permute_173: "f32[768, 2048]" = torch.ops.aten.permute.default(view_301, [1, 0])
    mm_15: "f32[768, 768]" = torch.ops.aten.mm.default(permute_173, view_244);  permute_173 = view_244 = None
    permute_174: "f32[768, 768]" = torch.ops.aten.permute.default(mm_15, [1, 0]);  mm_15 = None
    sum_33: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_301, [0], True);  view_301 = None
    view_302: "f32[768]" = torch.ops.aten.view.default(sum_33, [768]);  sum_33 = None
    permute_175: "f32[768, 768]" = torch.ops.aten.permute.default(permute_174, [1, 0]);  permute_174 = None
    view_303: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_14, [4, 512, 768]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_123: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_122, view_303);  add_122 = view_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    mul_147: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_123, primals_22)
    mul_148: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_147, 768)
    sum_34: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_147, [2], True)
    mul_149: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_147, mul_89);  mul_147 = None
    sum_35: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_149, [2], True);  mul_149 = None
    mul_150: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_89, sum_35);  sum_35 = None
    sub_52: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_148, sum_34);  mul_148 = sum_34 = None
    sub_53: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_52, mul_150);  sub_52 = mul_150 = None
    mul_151: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_28, sub_53);  div_28 = sub_53 = None
    mul_152: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_123, mul_89);  mul_89 = None
    sum_36: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_152, [0, 1]);  mul_152 = None
    sum_37: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_123, [0, 1]);  add_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_124: "f32[768]" = torch.ops.aten.add.Tensor(sum_21, sum_36);  sum_21 = sum_36 = None
    add_125: "f32[768]" = torch.ops.aten.add.Tensor(sum_22, sum_37);  sum_22 = sum_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_304: "f32[2048, 768]" = torch.ops.aten.view.default(mul_151, [2048, 768])
    mm_16: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_304, permute_143)
    permute_177: "f32[768, 2048]" = torch.ops.aten.permute.default(view_304, [1, 0])
    mm_17: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_177, view_242);  permute_177 = view_242 = None
    permute_178: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_17, [1, 0]);  mm_17 = None
    sum_38: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_304, [0], True);  view_304 = None
    view_305: "f32[768]" = torch.ops.aten.view.default(sum_38, [768]);  sum_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_126: "f32[768]" = torch.ops.aten.add.Tensor(view_277, view_305);  view_277 = view_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    permute_179: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_178, [1, 0]);  permute_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_127: "f32[768, 3072]" = torch.ops.aten.add.Tensor(permute_146, permute_179);  permute_146 = permute_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_306: "f32[4, 512, 3072]" = torch.ops.aten.view.default(mm_16, [4, 512, 3072]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_153: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_306, mul_85);  mul_85 = None
    mul_154: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_306, add_99);  view_306 = add_99 = None
    alias_28: "f32[4, 512, 3072]" = torch.ops.aten.alias.default(alias_21);  alias_21 = None
    mul_155: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(alias_28, alias_28);  alias_28 = None
    sub_54: "f32[4, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_155);  mul_155 = None
    mul_156: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_153, sub_54);  mul_153 = sub_54 = None
    mul_157: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_156, 0.7978845608028654);  mul_156 = None
    mul_158: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_157, 0.044715)
    pow_16: "f32[4, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_241, 2.0);  view_241 = None
    mul_159: "f32[4, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_16, 3.0);  pow_16 = None
    mul_160: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_158, mul_159);  mul_158 = mul_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_128: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_157, mul_160);  mul_157 = mul_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_161: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_154, 0.5);  mul_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_129: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(add_128, mul_161);  add_128 = mul_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_307: "f32[2048, 3072]" = torch.ops.aten.view.default(add_129, [2048, 3072]);  add_129 = None
    mm_18: "f32[2048, 768]" = torch.ops.aten.mm.default(view_307, permute_147)
    permute_181: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_307, [1, 0])
    mm_19: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_181, view_240);  permute_181 = view_240 = None
    permute_182: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_19, [1, 0]);  mm_19 = None
    sum_39: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_307, [0], True);  view_307 = None
    view_308: "f32[3072]" = torch.ops.aten.view.default(sum_39, [3072]);  sum_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_130: "f32[3072]" = torch.ops.aten.add.Tensor(view_280, view_308);  view_280 = view_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    permute_183: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_182, [1, 0]);  permute_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_131: "f32[3072, 768]" = torch.ops.aten.add.Tensor(permute_150, permute_183);  permute_150 = permute_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_309: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_18, [4, 512, 768]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_132: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_151, view_309);  mul_151 = view_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    mul_163: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_132, primals_16)
    mul_164: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_163, 768)
    sum_40: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_163, [2], True)
    mul_165: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_163, mul_83);  mul_163 = None
    sum_41: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_165, [2], True);  mul_165 = None
    mul_166: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_83, sum_41);  sum_41 = None
    sub_56: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_164, sum_40);  mul_164 = sum_40 = None
    sub_57: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_56, mul_166);  sub_56 = mul_166 = None
    mul_167: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_29, sub_57);  div_29 = sub_57 = None
    mul_168: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_132, mul_83);  mul_83 = None
    sum_42: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_168, [0, 1]);  mul_168 = None
    sum_43: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_132, [0, 1]);  add_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_133: "f32[768]" = torch.ops.aten.add.Tensor(sum_27, sum_42);  sum_27 = sum_42 = None
    add_134: "f32[768]" = torch.ops.aten.add.Tensor(sum_28, sum_43);  sum_28 = sum_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_310: "f32[2048, 768]" = torch.ops.aten.view.default(mul_167, [2048, 768])
    mm_20: "f32[2048, 768]" = torch.ops.aten.mm.default(view_310, permute_151)
    permute_185: "f32[768, 2048]" = torch.ops.aten.permute.default(view_310, [1, 0])
    mm_21: "f32[768, 768]" = torch.ops.aten.mm.default(permute_185, view_238);  permute_185 = view_238 = None
    permute_186: "f32[768, 768]" = torch.ops.aten.permute.default(mm_21, [1, 0]);  mm_21 = None
    sum_44: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_310, [0], True);  view_310 = None
    view_311: "f32[768]" = torch.ops.aten.view.default(sum_44, [768]);  sum_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_135: "f32[768]" = torch.ops.aten.add.Tensor(view_283, view_311);  view_283 = view_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    permute_187: "f32[768, 768]" = torch.ops.aten.permute.default(permute_186, [1, 0]);  permute_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_136: "f32[768, 768]" = torch.ops.aten.add.Tensor(permute_154, permute_187);  permute_154 = permute_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_312: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_20, [4, 512, 768]);  mm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    view_313: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_312, [4, 512, 12, 64]);  view_312 = None
    permute_188: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_313, [0, 2, 1, 3]);  view_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    clone_77: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_188, memory_format = torch.contiguous_format);  permute_188 = None
    view_314: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_77, [48, 512, 64]);  clone_77 = None
    bmm_28: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_189, view_314);  permute_189 = None
    bmm_29: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_314, permute_190);  view_314 = permute_190 = None
    view_315: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_28, [4, 12, 512, 64]);  bmm_28 = None
    view_316: "f32[4, 12, 512, 512]" = torch.ops.aten.view.default(bmm_29, [4, 12, 512, 512]);  bmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    mul_169: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_316, alias_29);  view_316 = None
    sum_45: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_169, [-1], True)
    mul_170: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_29, sum_45);  alias_29 = sum_45 = None
    sub_58: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_169, mul_170);  mul_169 = mul_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:332, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_30: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(sub_58, 8.0);  sub_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_317: "f32[48, 512, 512]" = torch.ops.aten.view.default(div_30, [48, 512, 512]);  div_30 = None
    bmm_30: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_191, view_317);  permute_191 = None
    bmm_31: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_317, permute_192);  view_317 = permute_192 = None
    view_318: "f32[4, 12, 64, 512]" = torch.ops.aten.view.default(bmm_30, [4, 12, 64, 512]);  bmm_30 = None
    view_319: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_31, [4, 12, 512, 64]);  bmm_31 = None
    permute_193: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_318, [0, 1, 3, 2]);  view_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_194: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_315, [0, 2, 1, 3]);  view_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_78: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_194, memory_format = torch.contiguous_format);  permute_194 = None
    view_320: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_78, [4, 512, 768]);  clone_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_195: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(permute_193, [0, 2, 1, 3]);  permute_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_321: "f32[4, 512, 768]" = torch.ops.aten.view.default(permute_195, [4, 512, 768]);  permute_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_196: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_319, [0, 2, 1, 3]);  view_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_79: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_196, memory_format = torch.contiguous_format);  permute_196 = None
    view_322: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_79, [4, 512, 768]);  clone_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_323: "f32[2048, 768]" = torch.ops.aten.view.default(view_320, [2048, 768]);  view_320 = None
    mm_22: "f32[2048, 768]" = torch.ops.aten.mm.default(view_323, permute_164)
    permute_198: "f32[768, 2048]" = torch.ops.aten.permute.default(view_323, [1, 0])
    mm_23: "f32[768, 768]" = torch.ops.aten.mm.default(permute_198, view_222);  permute_198 = None
    permute_199: "f32[768, 768]" = torch.ops.aten.permute.default(mm_23, [1, 0]);  mm_23 = None
    sum_46: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_323, [0], True);  view_323 = None
    view_324: "f32[768]" = torch.ops.aten.view.default(sum_46, [768]);  sum_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_137: "f32[768]" = torch.ops.aten.add.Tensor(view_296, view_324);  view_296 = view_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    permute_200: "f32[768, 768]" = torch.ops.aten.permute.default(permute_199, [1, 0]);  permute_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_138: "f32[768, 768]" = torch.ops.aten.add.Tensor(permute_167, permute_200);  permute_167 = permute_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_325: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_22, [4, 512, 768]);  mm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_139: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_167, view_325);  mul_167 = view_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    clone_80: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_321, memory_format = torch.contiguous_format);  view_321 = None
    view_326: "f32[2048, 768]" = torch.ops.aten.view.default(clone_80, [2048, 768]);  clone_80 = None
    mm_24: "f32[2048, 768]" = torch.ops.aten.mm.default(view_326, permute_168)
    permute_202: "f32[768, 2048]" = torch.ops.aten.permute.default(view_326, [1, 0])
    mm_25: "f32[768, 768]" = torch.ops.aten.mm.default(permute_202, view_222);  permute_202 = None
    permute_203: "f32[768, 768]" = torch.ops.aten.permute.default(mm_25, [1, 0]);  mm_25 = None
    sum_47: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_326, [0], True);  view_326 = None
    view_327: "f32[768]" = torch.ops.aten.view.default(sum_47, [768]);  sum_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_140: "f32[768]" = torch.ops.aten.add.Tensor(view_299, view_327);  view_299 = view_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    permute_204: "f32[768, 768]" = torch.ops.aten.permute.default(permute_203, [1, 0]);  permute_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_141: "f32[768, 768]" = torch.ops.aten.add.Tensor(permute_171, permute_204);  permute_171 = permute_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_328: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_24, [4, 512, 768]);  mm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_142: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_139, view_328);  add_139 = view_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_329: "f32[2048, 768]" = torch.ops.aten.view.default(view_322, [2048, 768]);  view_322 = None
    mm_26: "f32[2048, 768]" = torch.ops.aten.mm.default(view_329, permute_172)
    permute_206: "f32[768, 2048]" = torch.ops.aten.permute.default(view_329, [1, 0])
    mm_27: "f32[768, 768]" = torch.ops.aten.mm.default(permute_206, view_222);  permute_206 = view_222 = None
    permute_207: "f32[768, 768]" = torch.ops.aten.permute.default(mm_27, [1, 0]);  mm_27 = None
    sum_48: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_329, [0], True);  view_329 = None
    view_330: "f32[768]" = torch.ops.aten.view.default(sum_48, [768]);  sum_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_143: "f32[768]" = torch.ops.aten.add.Tensor(view_302, view_330);  view_302 = view_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    permute_208: "f32[768, 768]" = torch.ops.aten.permute.default(permute_207, [1, 0]);  permute_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_144: "f32[768, 768]" = torch.ops.aten.add.Tensor(permute_175, permute_208);  permute_175 = permute_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_331: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_26, [4, 512, 768]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_145: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_142, view_331);  add_142 = view_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    mul_172: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_145, primals_22)
    mul_173: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_172, 768)
    sum_49: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_172, [2], True)
    mul_174: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_172, mul_81);  mul_172 = None
    sum_50: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_174, [2], True);  mul_174 = None
    mul_175: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_81, sum_50);  sum_50 = None
    sub_60: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_173, sum_49);  mul_173 = sum_49 = None
    sub_61: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_60, mul_175);  sub_60 = mul_175 = None
    mul_176: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_31, sub_61);  div_31 = sub_61 = None
    mul_177: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_145, mul_81);  mul_81 = None
    sum_51: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_177, [0, 1]);  mul_177 = None
    sum_52: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_145, [0, 1]);  add_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_146: "f32[768]" = torch.ops.aten.add.Tensor(add_124, sum_51);  add_124 = sum_51 = None
    add_147: "f32[768]" = torch.ops.aten.add.Tensor(add_125, sum_52);  add_125 = sum_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_332: "f32[2048, 768]" = torch.ops.aten.view.default(mul_176, [2048, 768])
    mm_28: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_332, permute_143)
    permute_210: "f32[768, 2048]" = torch.ops.aten.permute.default(view_332, [1, 0])
    mm_29: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_210, view_220);  permute_210 = view_220 = None
    permute_211: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_29, [1, 0]);  mm_29 = None
    sum_53: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_332, [0], True);  view_332 = None
    view_333: "f32[768]" = torch.ops.aten.view.default(sum_53, [768]);  sum_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_148: "f32[768]" = torch.ops.aten.add.Tensor(add_126, view_333);  add_126 = view_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    permute_212: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_211, [1, 0]);  permute_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_149: "f32[768, 3072]" = torch.ops.aten.add.Tensor(add_127, permute_212);  add_127 = permute_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_334: "f32[4, 512, 3072]" = torch.ops.aten.view.default(mm_28, [4, 512, 3072]);  mm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_178: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_334, mul_77);  mul_77 = None
    mul_179: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_334, add_90);  view_334 = add_90 = None
    alias_30: "f32[4, 512, 3072]" = torch.ops.aten.alias.default(alias_19);  alias_19 = None
    mul_180: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(alias_30, alias_30);  alias_30 = None
    sub_62: "f32[4, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_180);  mul_180 = None
    mul_181: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_178, sub_62);  mul_178 = sub_62 = None
    mul_182: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_181, 0.7978845608028654);  mul_181 = None
    mul_183: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_182, 0.044715)
    pow_17: "f32[4, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_219, 2.0);  view_219 = None
    mul_184: "f32[4, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_17, 3.0);  pow_17 = None
    mul_185: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_183, mul_184);  mul_183 = mul_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_150: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_182, mul_185);  mul_182 = mul_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_186: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_179, 0.5);  mul_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_151: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(add_150, mul_186);  add_150 = mul_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_335: "f32[2048, 3072]" = torch.ops.aten.view.default(add_151, [2048, 3072]);  add_151 = None
    mm_30: "f32[2048, 768]" = torch.ops.aten.mm.default(view_335, permute_147)
    permute_214: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_335, [1, 0])
    mm_31: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_214, view_218);  permute_214 = view_218 = None
    permute_215: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_31, [1, 0]);  mm_31 = None
    sum_54: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_335, [0], True);  view_335 = None
    view_336: "f32[3072]" = torch.ops.aten.view.default(sum_54, [3072]);  sum_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_152: "f32[3072]" = torch.ops.aten.add.Tensor(add_130, view_336);  add_130 = view_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    permute_216: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_215, [1, 0]);  permute_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_153: "f32[3072, 768]" = torch.ops.aten.add.Tensor(add_131, permute_216);  add_131 = permute_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_337: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_30, [4, 512, 768]);  mm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_154: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_176, view_337);  mul_176 = view_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    mul_188: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_154, primals_16)
    mul_189: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_188, 768)
    sum_55: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_188, [2], True)
    mul_190: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_188, mul_75);  mul_188 = None
    sum_56: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_190, [2], True);  mul_190 = None
    mul_191: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_75, sum_56);  sum_56 = None
    sub_64: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_189, sum_55);  mul_189 = sum_55 = None
    sub_65: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_64, mul_191);  sub_64 = mul_191 = None
    mul_192: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_32, sub_65);  div_32 = sub_65 = None
    mul_193: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_154, mul_75);  mul_75 = None
    sum_57: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_193, [0, 1]);  mul_193 = None
    sum_58: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_154, [0, 1]);  add_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_155: "f32[768]" = torch.ops.aten.add.Tensor(add_133, sum_57);  add_133 = sum_57 = None
    add_156: "f32[768]" = torch.ops.aten.add.Tensor(add_134, sum_58);  add_134 = sum_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_338: "f32[2048, 768]" = torch.ops.aten.view.default(mul_192, [2048, 768])
    mm_32: "f32[2048, 768]" = torch.ops.aten.mm.default(view_338, permute_151)
    permute_218: "f32[768, 2048]" = torch.ops.aten.permute.default(view_338, [1, 0])
    mm_33: "f32[768, 768]" = torch.ops.aten.mm.default(permute_218, view_216);  permute_218 = view_216 = None
    permute_219: "f32[768, 768]" = torch.ops.aten.permute.default(mm_33, [1, 0]);  mm_33 = None
    sum_59: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_338, [0], True);  view_338 = None
    view_339: "f32[768]" = torch.ops.aten.view.default(sum_59, [768]);  sum_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_157: "f32[768]" = torch.ops.aten.add.Tensor(add_135, view_339);  add_135 = view_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    permute_220: "f32[768, 768]" = torch.ops.aten.permute.default(permute_219, [1, 0]);  permute_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_158: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_136, permute_220);  add_136 = permute_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_340: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_32, [4, 512, 768]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    view_341: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_340, [4, 512, 12, 64]);  view_340 = None
    permute_221: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_341, [0, 2, 1, 3]);  view_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    clone_81: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_221, memory_format = torch.contiguous_format);  permute_221 = None
    view_342: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_81, [48, 512, 64]);  clone_81 = None
    bmm_32: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_222, view_342);  permute_222 = None
    bmm_33: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_342, permute_223);  view_342 = permute_223 = None
    view_343: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_32, [4, 12, 512, 64]);  bmm_32 = None
    view_344: "f32[4, 12, 512, 512]" = torch.ops.aten.view.default(bmm_33, [4, 12, 512, 512]);  bmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    mul_194: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_344, alias_31);  view_344 = None
    sum_60: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_194, [-1], True)
    mul_195: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_31, sum_60);  alias_31 = sum_60 = None
    sub_66: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_194, mul_195);  mul_194 = mul_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:332, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_33: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(sub_66, 8.0);  sub_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_345: "f32[48, 512, 512]" = torch.ops.aten.view.default(div_33, [48, 512, 512]);  div_33 = None
    bmm_34: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_224, view_345);  permute_224 = None
    bmm_35: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_345, permute_225);  view_345 = permute_225 = None
    view_346: "f32[4, 12, 64, 512]" = torch.ops.aten.view.default(bmm_34, [4, 12, 64, 512]);  bmm_34 = None
    view_347: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_35, [4, 12, 512, 64]);  bmm_35 = None
    permute_226: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_346, [0, 1, 3, 2]);  view_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_227: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_343, [0, 2, 1, 3]);  view_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_82: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_227, memory_format = torch.contiguous_format);  permute_227 = None
    view_348: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_82, [4, 512, 768]);  clone_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_228: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(permute_226, [0, 2, 1, 3]);  permute_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_349: "f32[4, 512, 768]" = torch.ops.aten.view.default(permute_228, [4, 512, 768]);  permute_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_229: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_347, [0, 2, 1, 3]);  view_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_83: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_229, memory_format = torch.contiguous_format);  permute_229 = None
    view_350: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_83, [4, 512, 768]);  clone_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_351: "f32[2048, 768]" = torch.ops.aten.view.default(view_348, [2048, 768]);  view_348 = None
    mm_34: "f32[2048, 768]" = torch.ops.aten.mm.default(view_351, permute_164)
    permute_231: "f32[768, 2048]" = torch.ops.aten.permute.default(view_351, [1, 0])
    mm_35: "f32[768, 768]" = torch.ops.aten.mm.default(permute_231, view_200);  permute_231 = None
    permute_232: "f32[768, 768]" = torch.ops.aten.permute.default(mm_35, [1, 0]);  mm_35 = None
    sum_61: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_351, [0], True);  view_351 = None
    view_352: "f32[768]" = torch.ops.aten.view.default(sum_61, [768]);  sum_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_159: "f32[768]" = torch.ops.aten.add.Tensor(add_137, view_352);  add_137 = view_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    permute_233: "f32[768, 768]" = torch.ops.aten.permute.default(permute_232, [1, 0]);  permute_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_160: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_138, permute_233);  add_138 = permute_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_353: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_34, [4, 512, 768]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_161: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_192, view_353);  mul_192 = view_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    clone_84: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_349, memory_format = torch.contiguous_format);  view_349 = None
    view_354: "f32[2048, 768]" = torch.ops.aten.view.default(clone_84, [2048, 768]);  clone_84 = None
    mm_36: "f32[2048, 768]" = torch.ops.aten.mm.default(view_354, permute_168)
    permute_235: "f32[768, 2048]" = torch.ops.aten.permute.default(view_354, [1, 0])
    mm_37: "f32[768, 768]" = torch.ops.aten.mm.default(permute_235, view_200);  permute_235 = None
    permute_236: "f32[768, 768]" = torch.ops.aten.permute.default(mm_37, [1, 0]);  mm_37 = None
    sum_62: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_354, [0], True);  view_354 = None
    view_355: "f32[768]" = torch.ops.aten.view.default(sum_62, [768]);  sum_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_162: "f32[768]" = torch.ops.aten.add.Tensor(add_140, view_355);  add_140 = view_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    permute_237: "f32[768, 768]" = torch.ops.aten.permute.default(permute_236, [1, 0]);  permute_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_163: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_141, permute_237);  add_141 = permute_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_356: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_36, [4, 512, 768]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_164: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_161, view_356);  add_161 = view_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_357: "f32[2048, 768]" = torch.ops.aten.view.default(view_350, [2048, 768]);  view_350 = None
    mm_38: "f32[2048, 768]" = torch.ops.aten.mm.default(view_357, permute_172)
    permute_239: "f32[768, 2048]" = torch.ops.aten.permute.default(view_357, [1, 0])
    mm_39: "f32[768, 768]" = torch.ops.aten.mm.default(permute_239, view_200);  permute_239 = view_200 = None
    permute_240: "f32[768, 768]" = torch.ops.aten.permute.default(mm_39, [1, 0]);  mm_39 = None
    sum_63: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_357, [0], True);  view_357 = None
    view_358: "f32[768]" = torch.ops.aten.view.default(sum_63, [768]);  sum_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_165: "f32[768]" = torch.ops.aten.add.Tensor(add_143, view_358);  add_143 = view_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    permute_241: "f32[768, 768]" = torch.ops.aten.permute.default(permute_240, [1, 0]);  permute_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_166: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_144, permute_241);  add_144 = permute_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_359: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_38, [4, 512, 768]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_167: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_164, view_359);  add_164 = view_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    mul_197: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_167, primals_22)
    mul_198: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_197, 768)
    sum_64: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_197, [2], True)
    mul_199: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_197, mul_73);  mul_197 = None
    sum_65: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_199, [2], True);  mul_199 = None
    mul_200: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_73, sum_65);  sum_65 = None
    sub_68: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_198, sum_64);  mul_198 = sum_64 = None
    sub_69: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_68, mul_200);  sub_68 = mul_200 = None
    mul_201: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_34, sub_69);  div_34 = sub_69 = None
    mul_202: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_167, mul_73);  mul_73 = None
    sum_66: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_202, [0, 1]);  mul_202 = None
    sum_67: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_167, [0, 1]);  add_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_168: "f32[768]" = torch.ops.aten.add.Tensor(add_146, sum_66);  add_146 = sum_66 = None
    add_169: "f32[768]" = torch.ops.aten.add.Tensor(add_147, sum_67);  add_147 = sum_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_360: "f32[2048, 768]" = torch.ops.aten.view.default(mul_201, [2048, 768])
    mm_40: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_360, permute_143)
    permute_243: "f32[768, 2048]" = torch.ops.aten.permute.default(view_360, [1, 0])
    mm_41: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_243, view_198);  permute_243 = view_198 = None
    permute_244: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_41, [1, 0]);  mm_41 = None
    sum_68: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_360, [0], True);  view_360 = None
    view_361: "f32[768]" = torch.ops.aten.view.default(sum_68, [768]);  sum_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_170: "f32[768]" = torch.ops.aten.add.Tensor(add_148, view_361);  add_148 = view_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    permute_245: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_244, [1, 0]);  permute_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_171: "f32[768, 3072]" = torch.ops.aten.add.Tensor(add_149, permute_245);  add_149 = permute_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_362: "f32[4, 512, 3072]" = torch.ops.aten.view.default(mm_40, [4, 512, 3072]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_203: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_362, mul_69);  mul_69 = None
    mul_204: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_362, add_81);  view_362 = add_81 = None
    alias_32: "f32[4, 512, 3072]" = torch.ops.aten.alias.default(alias_17);  alias_17 = None
    mul_205: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(alias_32, alias_32);  alias_32 = None
    sub_70: "f32[4, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_205);  mul_205 = None
    mul_206: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_203, sub_70);  mul_203 = sub_70 = None
    mul_207: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_206, 0.7978845608028654);  mul_206 = None
    mul_208: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_207, 0.044715)
    pow_18: "f32[4, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_197, 2.0);  view_197 = None
    mul_209: "f32[4, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_18, 3.0);  pow_18 = None
    mul_210: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_208, mul_209);  mul_208 = mul_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_172: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_207, mul_210);  mul_207 = mul_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_211: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_204, 0.5);  mul_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_173: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(add_172, mul_211);  add_172 = mul_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_363: "f32[2048, 3072]" = torch.ops.aten.view.default(add_173, [2048, 3072]);  add_173 = None
    mm_42: "f32[2048, 768]" = torch.ops.aten.mm.default(view_363, permute_147)
    permute_247: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_363, [1, 0])
    mm_43: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_247, view_196);  permute_247 = view_196 = None
    permute_248: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_43, [1, 0]);  mm_43 = None
    sum_69: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_363, [0], True);  view_363 = None
    view_364: "f32[3072]" = torch.ops.aten.view.default(sum_69, [3072]);  sum_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_174: "f32[3072]" = torch.ops.aten.add.Tensor(add_152, view_364);  add_152 = view_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    permute_249: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_248, [1, 0]);  permute_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_175: "f32[3072, 768]" = torch.ops.aten.add.Tensor(add_153, permute_249);  add_153 = permute_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_365: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_42, [4, 512, 768]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_176: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_201, view_365);  mul_201 = view_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    mul_213: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_176, primals_16)
    mul_214: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_213, 768)
    sum_70: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_213, [2], True)
    mul_215: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_213, mul_67);  mul_213 = None
    sum_71: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_215, [2], True);  mul_215 = None
    mul_216: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_67, sum_71);  sum_71 = None
    sub_72: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_214, sum_70);  mul_214 = sum_70 = None
    sub_73: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_72, mul_216);  sub_72 = mul_216 = None
    mul_217: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_35, sub_73);  div_35 = sub_73 = None
    mul_218: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_176, mul_67);  mul_67 = None
    sum_72: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_218, [0, 1]);  mul_218 = None
    sum_73: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_176, [0, 1]);  add_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_177: "f32[768]" = torch.ops.aten.add.Tensor(add_155, sum_72);  add_155 = sum_72 = None
    add_178: "f32[768]" = torch.ops.aten.add.Tensor(add_156, sum_73);  add_156 = sum_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_366: "f32[2048, 768]" = torch.ops.aten.view.default(mul_217, [2048, 768])
    mm_44: "f32[2048, 768]" = torch.ops.aten.mm.default(view_366, permute_151)
    permute_251: "f32[768, 2048]" = torch.ops.aten.permute.default(view_366, [1, 0])
    mm_45: "f32[768, 768]" = torch.ops.aten.mm.default(permute_251, view_194);  permute_251 = view_194 = None
    permute_252: "f32[768, 768]" = torch.ops.aten.permute.default(mm_45, [1, 0]);  mm_45 = None
    sum_74: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_366, [0], True);  view_366 = None
    view_367: "f32[768]" = torch.ops.aten.view.default(sum_74, [768]);  sum_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_179: "f32[768]" = torch.ops.aten.add.Tensor(add_157, view_367);  add_157 = view_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    permute_253: "f32[768, 768]" = torch.ops.aten.permute.default(permute_252, [1, 0]);  permute_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_180: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_158, permute_253);  add_158 = permute_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_368: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_44, [4, 512, 768]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    view_369: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_368, [4, 512, 12, 64]);  view_368 = None
    permute_254: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_369, [0, 2, 1, 3]);  view_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    clone_85: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_254, memory_format = torch.contiguous_format);  permute_254 = None
    view_370: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_85, [48, 512, 64]);  clone_85 = None
    bmm_36: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_255, view_370);  permute_255 = None
    bmm_37: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_370, permute_256);  view_370 = permute_256 = None
    view_371: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_36, [4, 12, 512, 64]);  bmm_36 = None
    view_372: "f32[4, 12, 512, 512]" = torch.ops.aten.view.default(bmm_37, [4, 12, 512, 512]);  bmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    mul_219: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_372, alias_33);  view_372 = None
    sum_75: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_219, [-1], True)
    mul_220: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_33, sum_75);  alias_33 = sum_75 = None
    sub_74: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_219, mul_220);  mul_219 = mul_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:332, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_36: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(sub_74, 8.0);  sub_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_373: "f32[48, 512, 512]" = torch.ops.aten.view.default(div_36, [48, 512, 512]);  div_36 = None
    bmm_38: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_257, view_373);  permute_257 = None
    bmm_39: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_373, permute_258);  view_373 = permute_258 = None
    view_374: "f32[4, 12, 64, 512]" = torch.ops.aten.view.default(bmm_38, [4, 12, 64, 512]);  bmm_38 = None
    view_375: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_39, [4, 12, 512, 64]);  bmm_39 = None
    permute_259: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_374, [0, 1, 3, 2]);  view_374 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_260: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_371, [0, 2, 1, 3]);  view_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_86: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_260, memory_format = torch.contiguous_format);  permute_260 = None
    view_376: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_86, [4, 512, 768]);  clone_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_261: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(permute_259, [0, 2, 1, 3]);  permute_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_377: "f32[4, 512, 768]" = torch.ops.aten.view.default(permute_261, [4, 512, 768]);  permute_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_262: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_375, [0, 2, 1, 3]);  view_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_87: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_262, memory_format = torch.contiguous_format);  permute_262 = None
    view_378: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_87, [4, 512, 768]);  clone_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_379: "f32[2048, 768]" = torch.ops.aten.view.default(view_376, [2048, 768]);  view_376 = None
    mm_46: "f32[2048, 768]" = torch.ops.aten.mm.default(view_379, permute_164)
    permute_264: "f32[768, 2048]" = torch.ops.aten.permute.default(view_379, [1, 0])
    mm_47: "f32[768, 768]" = torch.ops.aten.mm.default(permute_264, view_178);  permute_264 = None
    permute_265: "f32[768, 768]" = torch.ops.aten.permute.default(mm_47, [1, 0]);  mm_47 = None
    sum_76: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_379, [0], True);  view_379 = None
    view_380: "f32[768]" = torch.ops.aten.view.default(sum_76, [768]);  sum_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_181: "f32[768]" = torch.ops.aten.add.Tensor(add_159, view_380);  add_159 = view_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    permute_266: "f32[768, 768]" = torch.ops.aten.permute.default(permute_265, [1, 0]);  permute_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_182: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_160, permute_266);  add_160 = permute_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_381: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_46, [4, 512, 768]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_183: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_217, view_381);  mul_217 = view_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    clone_88: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_377, memory_format = torch.contiguous_format);  view_377 = None
    view_382: "f32[2048, 768]" = torch.ops.aten.view.default(clone_88, [2048, 768]);  clone_88 = None
    mm_48: "f32[2048, 768]" = torch.ops.aten.mm.default(view_382, permute_168)
    permute_268: "f32[768, 2048]" = torch.ops.aten.permute.default(view_382, [1, 0])
    mm_49: "f32[768, 768]" = torch.ops.aten.mm.default(permute_268, view_178);  permute_268 = None
    permute_269: "f32[768, 768]" = torch.ops.aten.permute.default(mm_49, [1, 0]);  mm_49 = None
    sum_77: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_382, [0], True);  view_382 = None
    view_383: "f32[768]" = torch.ops.aten.view.default(sum_77, [768]);  sum_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_184: "f32[768]" = torch.ops.aten.add.Tensor(add_162, view_383);  add_162 = view_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    permute_270: "f32[768, 768]" = torch.ops.aten.permute.default(permute_269, [1, 0]);  permute_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_185: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_163, permute_270);  add_163 = permute_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_384: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_48, [4, 512, 768]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_186: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_183, view_384);  add_183 = view_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_385: "f32[2048, 768]" = torch.ops.aten.view.default(view_378, [2048, 768]);  view_378 = None
    mm_50: "f32[2048, 768]" = torch.ops.aten.mm.default(view_385, permute_172)
    permute_272: "f32[768, 2048]" = torch.ops.aten.permute.default(view_385, [1, 0])
    mm_51: "f32[768, 768]" = torch.ops.aten.mm.default(permute_272, view_178);  permute_272 = view_178 = None
    permute_273: "f32[768, 768]" = torch.ops.aten.permute.default(mm_51, [1, 0]);  mm_51 = None
    sum_78: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_385, [0], True);  view_385 = None
    view_386: "f32[768]" = torch.ops.aten.view.default(sum_78, [768]);  sum_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_187: "f32[768]" = torch.ops.aten.add.Tensor(add_165, view_386);  add_165 = view_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    permute_274: "f32[768, 768]" = torch.ops.aten.permute.default(permute_273, [1, 0]);  permute_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_188: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_166, permute_274);  add_166 = permute_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_387: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_50, [4, 512, 768]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_189: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_186, view_387);  add_186 = view_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    mul_222: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_189, primals_22)
    mul_223: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_222, 768)
    sum_79: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_222, [2], True)
    mul_224: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_222, mul_65);  mul_222 = None
    sum_80: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_224, [2], True);  mul_224 = None
    mul_225: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_65, sum_80);  sum_80 = None
    sub_76: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_223, sum_79);  mul_223 = sum_79 = None
    sub_77: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_76, mul_225);  sub_76 = mul_225 = None
    mul_226: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_37, sub_77);  div_37 = sub_77 = None
    mul_227: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_189, mul_65);  mul_65 = None
    sum_81: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_227, [0, 1]);  mul_227 = None
    sum_82: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_189, [0, 1]);  add_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_190: "f32[768]" = torch.ops.aten.add.Tensor(add_168, sum_81);  add_168 = sum_81 = None
    add_191: "f32[768]" = torch.ops.aten.add.Tensor(add_169, sum_82);  add_169 = sum_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_388: "f32[2048, 768]" = torch.ops.aten.view.default(mul_226, [2048, 768])
    mm_52: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_388, permute_143)
    permute_276: "f32[768, 2048]" = torch.ops.aten.permute.default(view_388, [1, 0])
    mm_53: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_276, view_176);  permute_276 = view_176 = None
    permute_277: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_53, [1, 0]);  mm_53 = None
    sum_83: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_388, [0], True);  view_388 = None
    view_389: "f32[768]" = torch.ops.aten.view.default(sum_83, [768]);  sum_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_192: "f32[768]" = torch.ops.aten.add.Tensor(add_170, view_389);  add_170 = view_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    permute_278: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_277, [1, 0]);  permute_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_193: "f32[768, 3072]" = torch.ops.aten.add.Tensor(add_171, permute_278);  add_171 = permute_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_390: "f32[4, 512, 3072]" = torch.ops.aten.view.default(mm_52, [4, 512, 3072]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_228: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_390, mul_61);  mul_61 = None
    mul_229: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_390, add_72);  view_390 = add_72 = None
    alias_34: "f32[4, 512, 3072]" = torch.ops.aten.alias.default(alias_15);  alias_15 = None
    mul_230: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(alias_34, alias_34);  alias_34 = None
    sub_78: "f32[4, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_230);  mul_230 = None
    mul_231: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_228, sub_78);  mul_228 = sub_78 = None
    mul_232: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_231, 0.7978845608028654);  mul_231 = None
    mul_233: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_232, 0.044715)
    pow_19: "f32[4, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_175, 2.0);  view_175 = None
    mul_234: "f32[4, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_19, 3.0);  pow_19 = None
    mul_235: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_233, mul_234);  mul_233 = mul_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_194: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_232, mul_235);  mul_232 = mul_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_236: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_229, 0.5);  mul_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_195: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(add_194, mul_236);  add_194 = mul_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_391: "f32[2048, 3072]" = torch.ops.aten.view.default(add_195, [2048, 3072]);  add_195 = None
    mm_54: "f32[2048, 768]" = torch.ops.aten.mm.default(view_391, permute_147)
    permute_280: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_391, [1, 0])
    mm_55: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_280, view_174);  permute_280 = view_174 = None
    permute_281: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_55, [1, 0]);  mm_55 = None
    sum_84: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_391, [0], True);  view_391 = None
    view_392: "f32[3072]" = torch.ops.aten.view.default(sum_84, [3072]);  sum_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_196: "f32[3072]" = torch.ops.aten.add.Tensor(add_174, view_392);  add_174 = view_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    permute_282: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_281, [1, 0]);  permute_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_197: "f32[3072, 768]" = torch.ops.aten.add.Tensor(add_175, permute_282);  add_175 = permute_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_393: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_54, [4, 512, 768]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_198: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_226, view_393);  mul_226 = view_393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    mul_238: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_198, primals_16)
    mul_239: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_238, 768)
    sum_85: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_238, [2], True)
    mul_240: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_238, mul_59);  mul_238 = None
    sum_86: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_240, [2], True);  mul_240 = None
    mul_241: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_59, sum_86);  sum_86 = None
    sub_80: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_239, sum_85);  mul_239 = sum_85 = None
    sub_81: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_80, mul_241);  sub_80 = mul_241 = None
    mul_242: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_38, sub_81);  div_38 = sub_81 = None
    mul_243: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_198, mul_59);  mul_59 = None
    sum_87: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_243, [0, 1]);  mul_243 = None
    sum_88: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_198, [0, 1]);  add_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_199: "f32[768]" = torch.ops.aten.add.Tensor(add_177, sum_87);  add_177 = sum_87 = None
    add_200: "f32[768]" = torch.ops.aten.add.Tensor(add_178, sum_88);  add_178 = sum_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_394: "f32[2048, 768]" = torch.ops.aten.view.default(mul_242, [2048, 768])
    mm_56: "f32[2048, 768]" = torch.ops.aten.mm.default(view_394, permute_151)
    permute_284: "f32[768, 2048]" = torch.ops.aten.permute.default(view_394, [1, 0])
    mm_57: "f32[768, 768]" = torch.ops.aten.mm.default(permute_284, view_172);  permute_284 = view_172 = None
    permute_285: "f32[768, 768]" = torch.ops.aten.permute.default(mm_57, [1, 0]);  mm_57 = None
    sum_89: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_394, [0], True);  view_394 = None
    view_395: "f32[768]" = torch.ops.aten.view.default(sum_89, [768]);  sum_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_201: "f32[768]" = torch.ops.aten.add.Tensor(add_179, view_395);  add_179 = view_395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    permute_286: "f32[768, 768]" = torch.ops.aten.permute.default(permute_285, [1, 0]);  permute_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_202: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_180, permute_286);  add_180 = permute_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_396: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_56, [4, 512, 768]);  mm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    view_397: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_396, [4, 512, 12, 64]);  view_396 = None
    permute_287: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_397, [0, 2, 1, 3]);  view_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    clone_89: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_287, memory_format = torch.contiguous_format);  permute_287 = None
    view_398: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_89, [48, 512, 64]);  clone_89 = None
    bmm_40: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_288, view_398);  permute_288 = None
    bmm_41: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_398, permute_289);  view_398 = permute_289 = None
    view_399: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_40, [4, 12, 512, 64]);  bmm_40 = None
    view_400: "f32[4, 12, 512, 512]" = torch.ops.aten.view.default(bmm_41, [4, 12, 512, 512]);  bmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    mul_244: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_400, alias_35);  view_400 = None
    sum_90: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_244, [-1], True)
    mul_245: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_35, sum_90);  alias_35 = sum_90 = None
    sub_82: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_244, mul_245);  mul_244 = mul_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:332, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_39: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(sub_82, 8.0);  sub_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_401: "f32[48, 512, 512]" = torch.ops.aten.view.default(div_39, [48, 512, 512]);  div_39 = None
    bmm_42: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_290, view_401);  permute_290 = None
    bmm_43: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_401, permute_291);  view_401 = permute_291 = None
    view_402: "f32[4, 12, 64, 512]" = torch.ops.aten.view.default(bmm_42, [4, 12, 64, 512]);  bmm_42 = None
    view_403: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_43, [4, 12, 512, 64]);  bmm_43 = None
    permute_292: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_402, [0, 1, 3, 2]);  view_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_293: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_399, [0, 2, 1, 3]);  view_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_90: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_293, memory_format = torch.contiguous_format);  permute_293 = None
    view_404: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_90, [4, 512, 768]);  clone_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_294: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(permute_292, [0, 2, 1, 3]);  permute_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_405: "f32[4, 512, 768]" = torch.ops.aten.view.default(permute_294, [4, 512, 768]);  permute_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_295: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_403, [0, 2, 1, 3]);  view_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_91: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_295, memory_format = torch.contiguous_format);  permute_295 = None
    view_406: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_91, [4, 512, 768]);  clone_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_407: "f32[2048, 768]" = torch.ops.aten.view.default(view_404, [2048, 768]);  view_404 = None
    mm_58: "f32[2048, 768]" = torch.ops.aten.mm.default(view_407, permute_164)
    permute_297: "f32[768, 2048]" = torch.ops.aten.permute.default(view_407, [1, 0])
    mm_59: "f32[768, 768]" = torch.ops.aten.mm.default(permute_297, view_156);  permute_297 = None
    permute_298: "f32[768, 768]" = torch.ops.aten.permute.default(mm_59, [1, 0]);  mm_59 = None
    sum_91: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_407, [0], True);  view_407 = None
    view_408: "f32[768]" = torch.ops.aten.view.default(sum_91, [768]);  sum_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_203: "f32[768]" = torch.ops.aten.add.Tensor(add_181, view_408);  add_181 = view_408 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    permute_299: "f32[768, 768]" = torch.ops.aten.permute.default(permute_298, [1, 0]);  permute_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_204: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_182, permute_299);  add_182 = permute_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_409: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_58, [4, 512, 768]);  mm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_205: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_242, view_409);  mul_242 = view_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    clone_92: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_405, memory_format = torch.contiguous_format);  view_405 = None
    view_410: "f32[2048, 768]" = torch.ops.aten.view.default(clone_92, [2048, 768]);  clone_92 = None
    mm_60: "f32[2048, 768]" = torch.ops.aten.mm.default(view_410, permute_168)
    permute_301: "f32[768, 2048]" = torch.ops.aten.permute.default(view_410, [1, 0])
    mm_61: "f32[768, 768]" = torch.ops.aten.mm.default(permute_301, view_156);  permute_301 = None
    permute_302: "f32[768, 768]" = torch.ops.aten.permute.default(mm_61, [1, 0]);  mm_61 = None
    sum_92: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_410, [0], True);  view_410 = None
    view_411: "f32[768]" = torch.ops.aten.view.default(sum_92, [768]);  sum_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_206: "f32[768]" = torch.ops.aten.add.Tensor(add_184, view_411);  add_184 = view_411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    permute_303: "f32[768, 768]" = torch.ops.aten.permute.default(permute_302, [1, 0]);  permute_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_207: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_185, permute_303);  add_185 = permute_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_412: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_60, [4, 512, 768]);  mm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_208: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_205, view_412);  add_205 = view_412 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_413: "f32[2048, 768]" = torch.ops.aten.view.default(view_406, [2048, 768]);  view_406 = None
    mm_62: "f32[2048, 768]" = torch.ops.aten.mm.default(view_413, permute_172)
    permute_305: "f32[768, 2048]" = torch.ops.aten.permute.default(view_413, [1, 0])
    mm_63: "f32[768, 768]" = torch.ops.aten.mm.default(permute_305, view_156);  permute_305 = view_156 = None
    permute_306: "f32[768, 768]" = torch.ops.aten.permute.default(mm_63, [1, 0]);  mm_63 = None
    sum_93: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_413, [0], True);  view_413 = None
    view_414: "f32[768]" = torch.ops.aten.view.default(sum_93, [768]);  sum_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_209: "f32[768]" = torch.ops.aten.add.Tensor(add_187, view_414);  add_187 = view_414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    permute_307: "f32[768, 768]" = torch.ops.aten.permute.default(permute_306, [1, 0]);  permute_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_210: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_188, permute_307);  add_188 = permute_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_415: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_62, [4, 512, 768]);  mm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_211: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_208, view_415);  add_208 = view_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    mul_247: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_211, primals_22)
    mul_248: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_247, 768)
    sum_94: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_247, [2], True)
    mul_249: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_247, mul_57);  mul_247 = None
    sum_95: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_249, [2], True);  mul_249 = None
    mul_250: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_57, sum_95);  sum_95 = None
    sub_84: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_248, sum_94);  mul_248 = sum_94 = None
    sub_85: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_84, mul_250);  sub_84 = mul_250 = None
    mul_251: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_40, sub_85);  div_40 = sub_85 = None
    mul_252: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_211, mul_57);  mul_57 = None
    sum_96: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_252, [0, 1]);  mul_252 = None
    sum_97: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_211, [0, 1]);  add_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_212: "f32[768]" = torch.ops.aten.add.Tensor(add_190, sum_96);  add_190 = sum_96 = None
    add_213: "f32[768]" = torch.ops.aten.add.Tensor(add_191, sum_97);  add_191 = sum_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_416: "f32[2048, 768]" = torch.ops.aten.view.default(mul_251, [2048, 768])
    mm_64: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_416, permute_143)
    permute_309: "f32[768, 2048]" = torch.ops.aten.permute.default(view_416, [1, 0])
    mm_65: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_309, view_154);  permute_309 = view_154 = None
    permute_310: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_65, [1, 0]);  mm_65 = None
    sum_98: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_416, [0], True);  view_416 = None
    view_417: "f32[768]" = torch.ops.aten.view.default(sum_98, [768]);  sum_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_214: "f32[768]" = torch.ops.aten.add.Tensor(add_192, view_417);  add_192 = view_417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    permute_311: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_310, [1, 0]);  permute_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_215: "f32[768, 3072]" = torch.ops.aten.add.Tensor(add_193, permute_311);  add_193 = permute_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_418: "f32[4, 512, 3072]" = torch.ops.aten.view.default(mm_64, [4, 512, 3072]);  mm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_253: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_418, mul_53);  mul_53 = None
    mul_254: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_418, add_63);  view_418 = add_63 = None
    alias_36: "f32[4, 512, 3072]" = torch.ops.aten.alias.default(alias_13);  alias_13 = None
    mul_255: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(alias_36, alias_36);  alias_36 = None
    sub_86: "f32[4, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_255);  mul_255 = None
    mul_256: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_253, sub_86);  mul_253 = sub_86 = None
    mul_257: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_256, 0.7978845608028654);  mul_256 = None
    mul_258: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_257, 0.044715)
    pow_20: "f32[4, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_153, 2.0);  view_153 = None
    mul_259: "f32[4, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_20, 3.0);  pow_20 = None
    mul_260: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_258, mul_259);  mul_258 = mul_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_216: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_257, mul_260);  mul_257 = mul_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_261: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_254, 0.5);  mul_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_217: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(add_216, mul_261);  add_216 = mul_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_419: "f32[2048, 3072]" = torch.ops.aten.view.default(add_217, [2048, 3072]);  add_217 = None
    mm_66: "f32[2048, 768]" = torch.ops.aten.mm.default(view_419, permute_147)
    permute_313: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_419, [1, 0])
    mm_67: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_313, view_152);  permute_313 = view_152 = None
    permute_314: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_67, [1, 0]);  mm_67 = None
    sum_99: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_419, [0], True);  view_419 = None
    view_420: "f32[3072]" = torch.ops.aten.view.default(sum_99, [3072]);  sum_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_218: "f32[3072]" = torch.ops.aten.add.Tensor(add_196, view_420);  add_196 = view_420 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    permute_315: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_314, [1, 0]);  permute_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_219: "f32[3072, 768]" = torch.ops.aten.add.Tensor(add_197, permute_315);  add_197 = permute_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_421: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_66, [4, 512, 768]);  mm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_220: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_251, view_421);  mul_251 = view_421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    mul_263: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_220, primals_16)
    mul_264: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_263, 768)
    sum_100: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_263, [2], True)
    mul_265: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_263, mul_51);  mul_263 = None
    sum_101: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_265, [2], True);  mul_265 = None
    mul_266: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_51, sum_101);  sum_101 = None
    sub_88: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_264, sum_100);  mul_264 = sum_100 = None
    sub_89: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_88, mul_266);  sub_88 = mul_266 = None
    mul_267: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_41, sub_89);  div_41 = sub_89 = None
    mul_268: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_220, mul_51);  mul_51 = None
    sum_102: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_268, [0, 1]);  mul_268 = None
    sum_103: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_220, [0, 1]);  add_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_221: "f32[768]" = torch.ops.aten.add.Tensor(add_199, sum_102);  add_199 = sum_102 = None
    add_222: "f32[768]" = torch.ops.aten.add.Tensor(add_200, sum_103);  add_200 = sum_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_422: "f32[2048, 768]" = torch.ops.aten.view.default(mul_267, [2048, 768])
    mm_68: "f32[2048, 768]" = torch.ops.aten.mm.default(view_422, permute_151)
    permute_317: "f32[768, 2048]" = torch.ops.aten.permute.default(view_422, [1, 0])
    mm_69: "f32[768, 768]" = torch.ops.aten.mm.default(permute_317, view_150);  permute_317 = view_150 = None
    permute_318: "f32[768, 768]" = torch.ops.aten.permute.default(mm_69, [1, 0]);  mm_69 = None
    sum_104: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_422, [0], True);  view_422 = None
    view_423: "f32[768]" = torch.ops.aten.view.default(sum_104, [768]);  sum_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_223: "f32[768]" = torch.ops.aten.add.Tensor(add_201, view_423);  add_201 = view_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    permute_319: "f32[768, 768]" = torch.ops.aten.permute.default(permute_318, [1, 0]);  permute_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_224: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_202, permute_319);  add_202 = permute_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_424: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_68, [4, 512, 768]);  mm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    view_425: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_424, [4, 512, 12, 64]);  view_424 = None
    permute_320: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_425, [0, 2, 1, 3]);  view_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    clone_93: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_320, memory_format = torch.contiguous_format);  permute_320 = None
    view_426: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_93, [48, 512, 64]);  clone_93 = None
    bmm_44: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_321, view_426);  permute_321 = None
    bmm_45: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_426, permute_322);  view_426 = permute_322 = None
    view_427: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_44, [4, 12, 512, 64]);  bmm_44 = None
    view_428: "f32[4, 12, 512, 512]" = torch.ops.aten.view.default(bmm_45, [4, 12, 512, 512]);  bmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    mul_269: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_428, alias_37);  view_428 = None
    sum_105: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_269, [-1], True)
    mul_270: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_37, sum_105);  alias_37 = sum_105 = None
    sub_90: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_269, mul_270);  mul_269 = mul_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:332, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_42: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(sub_90, 8.0);  sub_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_429: "f32[48, 512, 512]" = torch.ops.aten.view.default(div_42, [48, 512, 512]);  div_42 = None
    bmm_46: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_323, view_429);  permute_323 = None
    bmm_47: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_429, permute_324);  view_429 = permute_324 = None
    view_430: "f32[4, 12, 64, 512]" = torch.ops.aten.view.default(bmm_46, [4, 12, 64, 512]);  bmm_46 = None
    view_431: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_47, [4, 12, 512, 64]);  bmm_47 = None
    permute_325: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_430, [0, 1, 3, 2]);  view_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_326: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_427, [0, 2, 1, 3]);  view_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_94: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_326, memory_format = torch.contiguous_format);  permute_326 = None
    view_432: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_94, [4, 512, 768]);  clone_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_327: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(permute_325, [0, 2, 1, 3]);  permute_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_433: "f32[4, 512, 768]" = torch.ops.aten.view.default(permute_327, [4, 512, 768]);  permute_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_328: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_431, [0, 2, 1, 3]);  view_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_95: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_328, memory_format = torch.contiguous_format);  permute_328 = None
    view_434: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_95, [4, 512, 768]);  clone_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_435: "f32[2048, 768]" = torch.ops.aten.view.default(view_432, [2048, 768]);  view_432 = None
    mm_70: "f32[2048, 768]" = torch.ops.aten.mm.default(view_435, permute_164)
    permute_330: "f32[768, 2048]" = torch.ops.aten.permute.default(view_435, [1, 0])
    mm_71: "f32[768, 768]" = torch.ops.aten.mm.default(permute_330, view_134);  permute_330 = None
    permute_331: "f32[768, 768]" = torch.ops.aten.permute.default(mm_71, [1, 0]);  mm_71 = None
    sum_106: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_435, [0], True);  view_435 = None
    view_436: "f32[768]" = torch.ops.aten.view.default(sum_106, [768]);  sum_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_225: "f32[768]" = torch.ops.aten.add.Tensor(add_203, view_436);  add_203 = view_436 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    permute_332: "f32[768, 768]" = torch.ops.aten.permute.default(permute_331, [1, 0]);  permute_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_226: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_204, permute_332);  add_204 = permute_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_437: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_70, [4, 512, 768]);  mm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_227: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_267, view_437);  mul_267 = view_437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    clone_96: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_433, memory_format = torch.contiguous_format);  view_433 = None
    view_438: "f32[2048, 768]" = torch.ops.aten.view.default(clone_96, [2048, 768]);  clone_96 = None
    mm_72: "f32[2048, 768]" = torch.ops.aten.mm.default(view_438, permute_168)
    permute_334: "f32[768, 2048]" = torch.ops.aten.permute.default(view_438, [1, 0])
    mm_73: "f32[768, 768]" = torch.ops.aten.mm.default(permute_334, view_134);  permute_334 = None
    permute_335: "f32[768, 768]" = torch.ops.aten.permute.default(mm_73, [1, 0]);  mm_73 = None
    sum_107: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_438, [0], True);  view_438 = None
    view_439: "f32[768]" = torch.ops.aten.view.default(sum_107, [768]);  sum_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_228: "f32[768]" = torch.ops.aten.add.Tensor(add_206, view_439);  add_206 = view_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    permute_336: "f32[768, 768]" = torch.ops.aten.permute.default(permute_335, [1, 0]);  permute_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_229: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_207, permute_336);  add_207 = permute_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_440: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_72, [4, 512, 768]);  mm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_230: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_227, view_440);  add_227 = view_440 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_441: "f32[2048, 768]" = torch.ops.aten.view.default(view_434, [2048, 768]);  view_434 = None
    mm_74: "f32[2048, 768]" = torch.ops.aten.mm.default(view_441, permute_172)
    permute_338: "f32[768, 2048]" = torch.ops.aten.permute.default(view_441, [1, 0])
    mm_75: "f32[768, 768]" = torch.ops.aten.mm.default(permute_338, view_134);  permute_338 = view_134 = None
    permute_339: "f32[768, 768]" = torch.ops.aten.permute.default(mm_75, [1, 0]);  mm_75 = None
    sum_108: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_441, [0], True);  view_441 = None
    view_442: "f32[768]" = torch.ops.aten.view.default(sum_108, [768]);  sum_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_231: "f32[768]" = torch.ops.aten.add.Tensor(add_209, view_442);  add_209 = view_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    permute_340: "f32[768, 768]" = torch.ops.aten.permute.default(permute_339, [1, 0]);  permute_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_232: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_210, permute_340);  add_210 = permute_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_443: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_74, [4, 512, 768]);  mm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_233: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_230, view_443);  add_230 = view_443 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    mul_272: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_233, primals_22)
    mul_273: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_272, 768)
    sum_109: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_272, [2], True)
    mul_274: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_272, mul_49);  mul_272 = None
    sum_110: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_274, [2], True);  mul_274 = None
    mul_275: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_49, sum_110);  sum_110 = None
    sub_92: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_273, sum_109);  mul_273 = sum_109 = None
    sub_93: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_92, mul_275);  sub_92 = mul_275 = None
    mul_276: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_43, sub_93);  div_43 = sub_93 = None
    mul_277: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_233, mul_49);  mul_49 = None
    sum_111: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_277, [0, 1]);  mul_277 = None
    sum_112: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_233, [0, 1]);  add_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_234: "f32[768]" = torch.ops.aten.add.Tensor(add_212, sum_111);  add_212 = sum_111 = None
    add_235: "f32[768]" = torch.ops.aten.add.Tensor(add_213, sum_112);  add_213 = sum_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_444: "f32[2048, 768]" = torch.ops.aten.view.default(mul_276, [2048, 768])
    mm_76: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_444, permute_143)
    permute_342: "f32[768, 2048]" = torch.ops.aten.permute.default(view_444, [1, 0])
    mm_77: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_342, view_132);  permute_342 = view_132 = None
    permute_343: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_77, [1, 0]);  mm_77 = None
    sum_113: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_444, [0], True);  view_444 = None
    view_445: "f32[768]" = torch.ops.aten.view.default(sum_113, [768]);  sum_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_236: "f32[768]" = torch.ops.aten.add.Tensor(add_214, view_445);  add_214 = view_445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    permute_344: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_343, [1, 0]);  permute_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_237: "f32[768, 3072]" = torch.ops.aten.add.Tensor(add_215, permute_344);  add_215 = permute_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_446: "f32[4, 512, 3072]" = torch.ops.aten.view.default(mm_76, [4, 512, 3072]);  mm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_278: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_446, mul_45);  mul_45 = None
    mul_279: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_446, add_54);  view_446 = add_54 = None
    alias_38: "f32[4, 512, 3072]" = torch.ops.aten.alias.default(alias_11);  alias_11 = None
    mul_280: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(alias_38, alias_38);  alias_38 = None
    sub_94: "f32[4, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_280);  mul_280 = None
    mul_281: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_278, sub_94);  mul_278 = sub_94 = None
    mul_282: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_281, 0.7978845608028654);  mul_281 = None
    mul_283: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_282, 0.044715)
    pow_21: "f32[4, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_131, 2.0);  view_131 = None
    mul_284: "f32[4, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_21, 3.0);  pow_21 = None
    mul_285: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_283, mul_284);  mul_283 = mul_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_238: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_282, mul_285);  mul_282 = mul_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_286: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_279, 0.5);  mul_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_239: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(add_238, mul_286);  add_238 = mul_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_447: "f32[2048, 3072]" = torch.ops.aten.view.default(add_239, [2048, 3072]);  add_239 = None
    mm_78: "f32[2048, 768]" = torch.ops.aten.mm.default(view_447, permute_147)
    permute_346: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_447, [1, 0])
    mm_79: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_346, view_130);  permute_346 = view_130 = None
    permute_347: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_79, [1, 0]);  mm_79 = None
    sum_114: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_447, [0], True);  view_447 = None
    view_448: "f32[3072]" = torch.ops.aten.view.default(sum_114, [3072]);  sum_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_240: "f32[3072]" = torch.ops.aten.add.Tensor(add_218, view_448);  add_218 = view_448 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    permute_348: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_347, [1, 0]);  permute_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_241: "f32[3072, 768]" = torch.ops.aten.add.Tensor(add_219, permute_348);  add_219 = permute_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_449: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_78, [4, 512, 768]);  mm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_242: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_276, view_449);  mul_276 = view_449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    mul_288: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_242, primals_16)
    mul_289: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_288, 768)
    sum_115: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_288, [2], True)
    mul_290: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_288, mul_43);  mul_288 = None
    sum_116: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_290, [2], True);  mul_290 = None
    mul_291: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_43, sum_116);  sum_116 = None
    sub_96: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_289, sum_115);  mul_289 = sum_115 = None
    sub_97: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_96, mul_291);  sub_96 = mul_291 = None
    mul_292: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_44, sub_97);  div_44 = sub_97 = None
    mul_293: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_242, mul_43);  mul_43 = None
    sum_117: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_293, [0, 1]);  mul_293 = None
    sum_118: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_242, [0, 1]);  add_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_243: "f32[768]" = torch.ops.aten.add.Tensor(add_221, sum_117);  add_221 = sum_117 = None
    add_244: "f32[768]" = torch.ops.aten.add.Tensor(add_222, sum_118);  add_222 = sum_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_450: "f32[2048, 768]" = torch.ops.aten.view.default(mul_292, [2048, 768])
    mm_80: "f32[2048, 768]" = torch.ops.aten.mm.default(view_450, permute_151)
    permute_350: "f32[768, 2048]" = torch.ops.aten.permute.default(view_450, [1, 0])
    mm_81: "f32[768, 768]" = torch.ops.aten.mm.default(permute_350, view_128);  permute_350 = view_128 = None
    permute_351: "f32[768, 768]" = torch.ops.aten.permute.default(mm_81, [1, 0]);  mm_81 = None
    sum_119: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_450, [0], True);  view_450 = None
    view_451: "f32[768]" = torch.ops.aten.view.default(sum_119, [768]);  sum_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_245: "f32[768]" = torch.ops.aten.add.Tensor(add_223, view_451);  add_223 = view_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    permute_352: "f32[768, 768]" = torch.ops.aten.permute.default(permute_351, [1, 0]);  permute_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_246: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_224, permute_352);  add_224 = permute_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_452: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_80, [4, 512, 768]);  mm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    view_453: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_452, [4, 512, 12, 64]);  view_452 = None
    permute_353: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_453, [0, 2, 1, 3]);  view_453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    clone_97: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_353, memory_format = torch.contiguous_format);  permute_353 = None
    view_454: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_97, [48, 512, 64]);  clone_97 = None
    bmm_48: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_354, view_454);  permute_354 = None
    bmm_49: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_454, permute_355);  view_454 = permute_355 = None
    view_455: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_48, [4, 12, 512, 64]);  bmm_48 = None
    view_456: "f32[4, 12, 512, 512]" = torch.ops.aten.view.default(bmm_49, [4, 12, 512, 512]);  bmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    mul_294: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_456, alias_39);  view_456 = None
    sum_120: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_294, [-1], True)
    mul_295: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_39, sum_120);  alias_39 = sum_120 = None
    sub_98: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_294, mul_295);  mul_294 = mul_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:332, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_45: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(sub_98, 8.0);  sub_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_457: "f32[48, 512, 512]" = torch.ops.aten.view.default(div_45, [48, 512, 512]);  div_45 = None
    bmm_50: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_356, view_457);  permute_356 = None
    bmm_51: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_457, permute_357);  view_457 = permute_357 = None
    view_458: "f32[4, 12, 64, 512]" = torch.ops.aten.view.default(bmm_50, [4, 12, 64, 512]);  bmm_50 = None
    view_459: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_51, [4, 12, 512, 64]);  bmm_51 = None
    permute_358: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_458, [0, 1, 3, 2]);  view_458 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_359: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_455, [0, 2, 1, 3]);  view_455 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_98: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_359, memory_format = torch.contiguous_format);  permute_359 = None
    view_460: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_98, [4, 512, 768]);  clone_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_360: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(permute_358, [0, 2, 1, 3]);  permute_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_461: "f32[4, 512, 768]" = torch.ops.aten.view.default(permute_360, [4, 512, 768]);  permute_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_361: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_459, [0, 2, 1, 3]);  view_459 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_99: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_361, memory_format = torch.contiguous_format);  permute_361 = None
    view_462: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_99, [4, 512, 768]);  clone_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_463: "f32[2048, 768]" = torch.ops.aten.view.default(view_460, [2048, 768]);  view_460 = None
    mm_82: "f32[2048, 768]" = torch.ops.aten.mm.default(view_463, permute_164)
    permute_363: "f32[768, 2048]" = torch.ops.aten.permute.default(view_463, [1, 0])
    mm_83: "f32[768, 768]" = torch.ops.aten.mm.default(permute_363, view_112);  permute_363 = None
    permute_364: "f32[768, 768]" = torch.ops.aten.permute.default(mm_83, [1, 0]);  mm_83 = None
    sum_121: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_463, [0], True);  view_463 = None
    view_464: "f32[768]" = torch.ops.aten.view.default(sum_121, [768]);  sum_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_247: "f32[768]" = torch.ops.aten.add.Tensor(add_225, view_464);  add_225 = view_464 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    permute_365: "f32[768, 768]" = torch.ops.aten.permute.default(permute_364, [1, 0]);  permute_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_248: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_226, permute_365);  add_226 = permute_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_465: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_82, [4, 512, 768]);  mm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_249: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_292, view_465);  mul_292 = view_465 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    clone_100: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_461, memory_format = torch.contiguous_format);  view_461 = None
    view_466: "f32[2048, 768]" = torch.ops.aten.view.default(clone_100, [2048, 768]);  clone_100 = None
    mm_84: "f32[2048, 768]" = torch.ops.aten.mm.default(view_466, permute_168)
    permute_367: "f32[768, 2048]" = torch.ops.aten.permute.default(view_466, [1, 0])
    mm_85: "f32[768, 768]" = torch.ops.aten.mm.default(permute_367, view_112);  permute_367 = None
    permute_368: "f32[768, 768]" = torch.ops.aten.permute.default(mm_85, [1, 0]);  mm_85 = None
    sum_122: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_466, [0], True);  view_466 = None
    view_467: "f32[768]" = torch.ops.aten.view.default(sum_122, [768]);  sum_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_250: "f32[768]" = torch.ops.aten.add.Tensor(add_228, view_467);  add_228 = view_467 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    permute_369: "f32[768, 768]" = torch.ops.aten.permute.default(permute_368, [1, 0]);  permute_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_251: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_229, permute_369);  add_229 = permute_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_468: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_84, [4, 512, 768]);  mm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_252: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_249, view_468);  add_249 = view_468 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_469: "f32[2048, 768]" = torch.ops.aten.view.default(view_462, [2048, 768]);  view_462 = None
    mm_86: "f32[2048, 768]" = torch.ops.aten.mm.default(view_469, permute_172)
    permute_371: "f32[768, 2048]" = torch.ops.aten.permute.default(view_469, [1, 0])
    mm_87: "f32[768, 768]" = torch.ops.aten.mm.default(permute_371, view_112);  permute_371 = view_112 = None
    permute_372: "f32[768, 768]" = torch.ops.aten.permute.default(mm_87, [1, 0]);  mm_87 = None
    sum_123: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_469, [0], True);  view_469 = None
    view_470: "f32[768]" = torch.ops.aten.view.default(sum_123, [768]);  sum_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_253: "f32[768]" = torch.ops.aten.add.Tensor(add_231, view_470);  add_231 = view_470 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    permute_373: "f32[768, 768]" = torch.ops.aten.permute.default(permute_372, [1, 0]);  permute_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_254: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_232, permute_373);  add_232 = permute_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_471: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_86, [4, 512, 768]);  mm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_255: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_252, view_471);  add_252 = view_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    mul_297: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_255, primals_22)
    mul_298: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_297, 768)
    sum_124: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_297, [2], True)
    mul_299: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_297, mul_41);  mul_297 = None
    sum_125: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_299, [2], True);  mul_299 = None
    mul_300: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_41, sum_125);  sum_125 = None
    sub_100: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_298, sum_124);  mul_298 = sum_124 = None
    sub_101: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_100, mul_300);  sub_100 = mul_300 = None
    mul_301: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_46, sub_101);  div_46 = sub_101 = None
    mul_302: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_255, mul_41);  mul_41 = None
    sum_126: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_302, [0, 1]);  mul_302 = None
    sum_127: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_255, [0, 1]);  add_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_256: "f32[768]" = torch.ops.aten.add.Tensor(add_234, sum_126);  add_234 = sum_126 = None
    add_257: "f32[768]" = torch.ops.aten.add.Tensor(add_235, sum_127);  add_235 = sum_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_472: "f32[2048, 768]" = torch.ops.aten.view.default(mul_301, [2048, 768])
    mm_88: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_472, permute_143)
    permute_375: "f32[768, 2048]" = torch.ops.aten.permute.default(view_472, [1, 0])
    mm_89: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_375, view_110);  permute_375 = view_110 = None
    permute_376: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_89, [1, 0]);  mm_89 = None
    sum_128: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_472, [0], True);  view_472 = None
    view_473: "f32[768]" = torch.ops.aten.view.default(sum_128, [768]);  sum_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_258: "f32[768]" = torch.ops.aten.add.Tensor(add_236, view_473);  add_236 = view_473 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    permute_377: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_376, [1, 0]);  permute_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_259: "f32[768, 3072]" = torch.ops.aten.add.Tensor(add_237, permute_377);  add_237 = permute_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_474: "f32[4, 512, 3072]" = torch.ops.aten.view.default(mm_88, [4, 512, 3072]);  mm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_303: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_474, mul_37);  mul_37 = None
    mul_304: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_474, add_45);  view_474 = add_45 = None
    alias_40: "f32[4, 512, 3072]" = torch.ops.aten.alias.default(alias_9);  alias_9 = None
    mul_305: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(alias_40, alias_40);  alias_40 = None
    sub_102: "f32[4, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_305);  mul_305 = None
    mul_306: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_303, sub_102);  mul_303 = sub_102 = None
    mul_307: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_306, 0.7978845608028654);  mul_306 = None
    mul_308: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_307, 0.044715)
    pow_22: "f32[4, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_109, 2.0);  view_109 = None
    mul_309: "f32[4, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_22, 3.0);  pow_22 = None
    mul_310: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_308, mul_309);  mul_308 = mul_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_260: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_307, mul_310);  mul_307 = mul_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_311: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_304, 0.5);  mul_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_261: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(add_260, mul_311);  add_260 = mul_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_475: "f32[2048, 3072]" = torch.ops.aten.view.default(add_261, [2048, 3072]);  add_261 = None
    mm_90: "f32[2048, 768]" = torch.ops.aten.mm.default(view_475, permute_147)
    permute_379: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_475, [1, 0])
    mm_91: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_379, view_108);  permute_379 = view_108 = None
    permute_380: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_91, [1, 0]);  mm_91 = None
    sum_129: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_475, [0], True);  view_475 = None
    view_476: "f32[3072]" = torch.ops.aten.view.default(sum_129, [3072]);  sum_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_262: "f32[3072]" = torch.ops.aten.add.Tensor(add_240, view_476);  add_240 = view_476 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    permute_381: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_380, [1, 0]);  permute_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_263: "f32[3072, 768]" = torch.ops.aten.add.Tensor(add_241, permute_381);  add_241 = permute_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_477: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_90, [4, 512, 768]);  mm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_264: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_301, view_477);  mul_301 = view_477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    mul_313: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_264, primals_16)
    mul_314: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_313, 768)
    sum_130: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_313, [2], True)
    mul_315: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_313, mul_35);  mul_313 = None
    sum_131: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_315, [2], True);  mul_315 = None
    mul_316: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_35, sum_131);  sum_131 = None
    sub_104: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_314, sum_130);  mul_314 = sum_130 = None
    sub_105: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_104, mul_316);  sub_104 = mul_316 = None
    mul_317: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_47, sub_105);  div_47 = sub_105 = None
    mul_318: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_264, mul_35);  mul_35 = None
    sum_132: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_318, [0, 1]);  mul_318 = None
    sum_133: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_264, [0, 1]);  add_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_265: "f32[768]" = torch.ops.aten.add.Tensor(add_243, sum_132);  add_243 = sum_132 = None
    add_266: "f32[768]" = torch.ops.aten.add.Tensor(add_244, sum_133);  add_244 = sum_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_478: "f32[2048, 768]" = torch.ops.aten.view.default(mul_317, [2048, 768])
    mm_92: "f32[2048, 768]" = torch.ops.aten.mm.default(view_478, permute_151)
    permute_383: "f32[768, 2048]" = torch.ops.aten.permute.default(view_478, [1, 0])
    mm_93: "f32[768, 768]" = torch.ops.aten.mm.default(permute_383, view_106);  permute_383 = view_106 = None
    permute_384: "f32[768, 768]" = torch.ops.aten.permute.default(mm_93, [1, 0]);  mm_93 = None
    sum_134: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_478, [0], True);  view_478 = None
    view_479: "f32[768]" = torch.ops.aten.view.default(sum_134, [768]);  sum_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_267: "f32[768]" = torch.ops.aten.add.Tensor(add_245, view_479);  add_245 = view_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    permute_385: "f32[768, 768]" = torch.ops.aten.permute.default(permute_384, [1, 0]);  permute_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_268: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_246, permute_385);  add_246 = permute_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_480: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_92, [4, 512, 768]);  mm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    view_481: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_480, [4, 512, 12, 64]);  view_480 = None
    permute_386: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_481, [0, 2, 1, 3]);  view_481 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    clone_101: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_386, memory_format = torch.contiguous_format);  permute_386 = None
    view_482: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_101, [48, 512, 64]);  clone_101 = None
    bmm_52: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_387, view_482);  permute_387 = None
    bmm_53: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_482, permute_388);  view_482 = permute_388 = None
    view_483: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_52, [4, 12, 512, 64]);  bmm_52 = None
    view_484: "f32[4, 12, 512, 512]" = torch.ops.aten.view.default(bmm_53, [4, 12, 512, 512]);  bmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    mul_319: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_484, alias_41);  view_484 = None
    sum_135: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_319, [-1], True)
    mul_320: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_41, sum_135);  alias_41 = sum_135 = None
    sub_106: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_319, mul_320);  mul_319 = mul_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:332, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_48: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(sub_106, 8.0);  sub_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_485: "f32[48, 512, 512]" = torch.ops.aten.view.default(div_48, [48, 512, 512]);  div_48 = None
    bmm_54: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_389, view_485);  permute_389 = None
    bmm_55: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_485, permute_390);  view_485 = permute_390 = None
    view_486: "f32[4, 12, 64, 512]" = torch.ops.aten.view.default(bmm_54, [4, 12, 64, 512]);  bmm_54 = None
    view_487: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_55, [4, 12, 512, 64]);  bmm_55 = None
    permute_391: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_486, [0, 1, 3, 2]);  view_486 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_392: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_483, [0, 2, 1, 3]);  view_483 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_102: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_392, memory_format = torch.contiguous_format);  permute_392 = None
    view_488: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_102, [4, 512, 768]);  clone_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_393: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(permute_391, [0, 2, 1, 3]);  permute_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_489: "f32[4, 512, 768]" = torch.ops.aten.view.default(permute_393, [4, 512, 768]);  permute_393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_394: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_487, [0, 2, 1, 3]);  view_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_103: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_394, memory_format = torch.contiguous_format);  permute_394 = None
    view_490: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_103, [4, 512, 768]);  clone_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_491: "f32[2048, 768]" = torch.ops.aten.view.default(view_488, [2048, 768]);  view_488 = None
    mm_94: "f32[2048, 768]" = torch.ops.aten.mm.default(view_491, permute_164)
    permute_396: "f32[768, 2048]" = torch.ops.aten.permute.default(view_491, [1, 0])
    mm_95: "f32[768, 768]" = torch.ops.aten.mm.default(permute_396, view_90);  permute_396 = None
    permute_397: "f32[768, 768]" = torch.ops.aten.permute.default(mm_95, [1, 0]);  mm_95 = None
    sum_136: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_491, [0], True);  view_491 = None
    view_492: "f32[768]" = torch.ops.aten.view.default(sum_136, [768]);  sum_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_269: "f32[768]" = torch.ops.aten.add.Tensor(add_247, view_492);  add_247 = view_492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    permute_398: "f32[768, 768]" = torch.ops.aten.permute.default(permute_397, [1, 0]);  permute_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_270: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_248, permute_398);  add_248 = permute_398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_493: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_94, [4, 512, 768]);  mm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_271: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_317, view_493);  mul_317 = view_493 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    clone_104: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_489, memory_format = torch.contiguous_format);  view_489 = None
    view_494: "f32[2048, 768]" = torch.ops.aten.view.default(clone_104, [2048, 768]);  clone_104 = None
    mm_96: "f32[2048, 768]" = torch.ops.aten.mm.default(view_494, permute_168)
    permute_400: "f32[768, 2048]" = torch.ops.aten.permute.default(view_494, [1, 0])
    mm_97: "f32[768, 768]" = torch.ops.aten.mm.default(permute_400, view_90);  permute_400 = None
    permute_401: "f32[768, 768]" = torch.ops.aten.permute.default(mm_97, [1, 0]);  mm_97 = None
    sum_137: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_494, [0], True);  view_494 = None
    view_495: "f32[768]" = torch.ops.aten.view.default(sum_137, [768]);  sum_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_272: "f32[768]" = torch.ops.aten.add.Tensor(add_250, view_495);  add_250 = view_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    permute_402: "f32[768, 768]" = torch.ops.aten.permute.default(permute_401, [1, 0]);  permute_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_273: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_251, permute_402);  add_251 = permute_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_496: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_96, [4, 512, 768]);  mm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_274: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_271, view_496);  add_271 = view_496 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_497: "f32[2048, 768]" = torch.ops.aten.view.default(view_490, [2048, 768]);  view_490 = None
    mm_98: "f32[2048, 768]" = torch.ops.aten.mm.default(view_497, permute_172)
    permute_404: "f32[768, 2048]" = torch.ops.aten.permute.default(view_497, [1, 0])
    mm_99: "f32[768, 768]" = torch.ops.aten.mm.default(permute_404, view_90);  permute_404 = view_90 = None
    permute_405: "f32[768, 768]" = torch.ops.aten.permute.default(mm_99, [1, 0]);  mm_99 = None
    sum_138: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_497, [0], True);  view_497 = None
    view_498: "f32[768]" = torch.ops.aten.view.default(sum_138, [768]);  sum_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_275: "f32[768]" = torch.ops.aten.add.Tensor(add_253, view_498);  add_253 = view_498 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    permute_406: "f32[768, 768]" = torch.ops.aten.permute.default(permute_405, [1, 0]);  permute_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_276: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_254, permute_406);  add_254 = permute_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_499: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_98, [4, 512, 768]);  mm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_277: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_274, view_499);  add_274 = view_499 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    mul_322: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_277, primals_22)
    mul_323: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_322, 768)
    sum_139: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_322, [2], True)
    mul_324: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_322, mul_33);  mul_322 = None
    sum_140: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_324, [2], True);  mul_324 = None
    mul_325: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_33, sum_140);  sum_140 = None
    sub_108: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_323, sum_139);  mul_323 = sum_139 = None
    sub_109: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_108, mul_325);  sub_108 = mul_325 = None
    mul_326: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_49, sub_109);  div_49 = sub_109 = None
    mul_327: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_277, mul_33);  mul_33 = None
    sum_141: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_327, [0, 1]);  mul_327 = None
    sum_142: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_277, [0, 1]);  add_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_278: "f32[768]" = torch.ops.aten.add.Tensor(add_256, sum_141);  add_256 = sum_141 = None
    add_279: "f32[768]" = torch.ops.aten.add.Tensor(add_257, sum_142);  add_257 = sum_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_500: "f32[2048, 768]" = torch.ops.aten.view.default(mul_326, [2048, 768])
    mm_100: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_500, permute_143)
    permute_408: "f32[768, 2048]" = torch.ops.aten.permute.default(view_500, [1, 0])
    mm_101: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_408, view_88);  permute_408 = view_88 = None
    permute_409: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_101, [1, 0]);  mm_101 = None
    sum_143: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_500, [0], True);  view_500 = None
    view_501: "f32[768]" = torch.ops.aten.view.default(sum_143, [768]);  sum_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_280: "f32[768]" = torch.ops.aten.add.Tensor(add_258, view_501);  add_258 = view_501 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    permute_410: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_409, [1, 0]);  permute_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_281: "f32[768, 3072]" = torch.ops.aten.add.Tensor(add_259, permute_410);  add_259 = permute_410 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_502: "f32[4, 512, 3072]" = torch.ops.aten.view.default(mm_100, [4, 512, 3072]);  mm_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_328: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_502, mul_29);  mul_29 = None
    mul_329: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_502, add_36);  view_502 = add_36 = None
    alias_42: "f32[4, 512, 3072]" = torch.ops.aten.alias.default(alias_7);  alias_7 = None
    mul_330: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(alias_42, alias_42);  alias_42 = None
    sub_110: "f32[4, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_330);  mul_330 = None
    mul_331: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_328, sub_110);  mul_328 = sub_110 = None
    mul_332: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_331, 0.7978845608028654);  mul_331 = None
    mul_333: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_332, 0.044715)
    pow_23: "f32[4, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_87, 2.0);  view_87 = None
    mul_334: "f32[4, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_23, 3.0);  pow_23 = None
    mul_335: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_333, mul_334);  mul_333 = mul_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_282: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_332, mul_335);  mul_332 = mul_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_336: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_329, 0.5);  mul_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_283: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(add_282, mul_336);  add_282 = mul_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_503: "f32[2048, 3072]" = torch.ops.aten.view.default(add_283, [2048, 3072]);  add_283 = None
    mm_102: "f32[2048, 768]" = torch.ops.aten.mm.default(view_503, permute_147)
    permute_412: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_503, [1, 0])
    mm_103: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_412, view_86);  permute_412 = view_86 = None
    permute_413: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_103, [1, 0]);  mm_103 = None
    sum_144: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_503, [0], True);  view_503 = None
    view_504: "f32[3072]" = torch.ops.aten.view.default(sum_144, [3072]);  sum_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_284: "f32[3072]" = torch.ops.aten.add.Tensor(add_262, view_504);  add_262 = view_504 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    permute_414: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_413, [1, 0]);  permute_413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_285: "f32[3072, 768]" = torch.ops.aten.add.Tensor(add_263, permute_414);  add_263 = permute_414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_505: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_102, [4, 512, 768]);  mm_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_286: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_326, view_505);  mul_326 = view_505 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    mul_338: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_286, primals_16)
    mul_339: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_338, 768)
    sum_145: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_338, [2], True)
    mul_340: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_338, mul_27);  mul_338 = None
    sum_146: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_340, [2], True);  mul_340 = None
    mul_341: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_27, sum_146);  sum_146 = None
    sub_112: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_339, sum_145);  mul_339 = sum_145 = None
    sub_113: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_112, mul_341);  sub_112 = mul_341 = None
    mul_342: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_50, sub_113);  div_50 = sub_113 = None
    mul_343: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_286, mul_27);  mul_27 = None
    sum_147: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_343, [0, 1]);  mul_343 = None
    sum_148: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_286, [0, 1]);  add_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_287: "f32[768]" = torch.ops.aten.add.Tensor(add_265, sum_147);  add_265 = sum_147 = None
    add_288: "f32[768]" = torch.ops.aten.add.Tensor(add_266, sum_148);  add_266 = sum_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_506: "f32[2048, 768]" = torch.ops.aten.view.default(mul_342, [2048, 768])
    mm_104: "f32[2048, 768]" = torch.ops.aten.mm.default(view_506, permute_151)
    permute_416: "f32[768, 2048]" = torch.ops.aten.permute.default(view_506, [1, 0])
    mm_105: "f32[768, 768]" = torch.ops.aten.mm.default(permute_416, view_84);  permute_416 = view_84 = None
    permute_417: "f32[768, 768]" = torch.ops.aten.permute.default(mm_105, [1, 0]);  mm_105 = None
    sum_149: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_506, [0], True);  view_506 = None
    view_507: "f32[768]" = torch.ops.aten.view.default(sum_149, [768]);  sum_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_289: "f32[768]" = torch.ops.aten.add.Tensor(add_267, view_507);  add_267 = view_507 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    permute_418: "f32[768, 768]" = torch.ops.aten.permute.default(permute_417, [1, 0]);  permute_417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_290: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_268, permute_418);  add_268 = permute_418 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_508: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_104, [4, 512, 768]);  mm_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    view_509: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_508, [4, 512, 12, 64]);  view_508 = None
    permute_419: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_509, [0, 2, 1, 3]);  view_509 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    clone_105: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_419, memory_format = torch.contiguous_format);  permute_419 = None
    view_510: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_105, [48, 512, 64]);  clone_105 = None
    bmm_56: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_420, view_510);  permute_420 = None
    bmm_57: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_510, permute_421);  view_510 = permute_421 = None
    view_511: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_56, [4, 12, 512, 64]);  bmm_56 = None
    view_512: "f32[4, 12, 512, 512]" = torch.ops.aten.view.default(bmm_57, [4, 12, 512, 512]);  bmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    mul_344: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_512, alias_43);  view_512 = None
    sum_150: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_344, [-1], True)
    mul_345: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_43, sum_150);  alias_43 = sum_150 = None
    sub_114: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_344, mul_345);  mul_344 = mul_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:332, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_51: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(sub_114, 8.0);  sub_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_513: "f32[48, 512, 512]" = torch.ops.aten.view.default(div_51, [48, 512, 512]);  div_51 = None
    bmm_58: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_422, view_513);  permute_422 = None
    bmm_59: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_513, permute_423);  view_513 = permute_423 = None
    view_514: "f32[4, 12, 64, 512]" = torch.ops.aten.view.default(bmm_58, [4, 12, 64, 512]);  bmm_58 = None
    view_515: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_59, [4, 12, 512, 64]);  bmm_59 = None
    permute_424: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_514, [0, 1, 3, 2]);  view_514 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_425: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_511, [0, 2, 1, 3]);  view_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_106: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_425, memory_format = torch.contiguous_format);  permute_425 = None
    view_516: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_106, [4, 512, 768]);  clone_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_426: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(permute_424, [0, 2, 1, 3]);  permute_424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_517: "f32[4, 512, 768]" = torch.ops.aten.view.default(permute_426, [4, 512, 768]);  permute_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_427: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_515, [0, 2, 1, 3]);  view_515 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_107: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_427, memory_format = torch.contiguous_format);  permute_427 = None
    view_518: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_107, [4, 512, 768]);  clone_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_519: "f32[2048, 768]" = torch.ops.aten.view.default(view_516, [2048, 768]);  view_516 = None
    mm_106: "f32[2048, 768]" = torch.ops.aten.mm.default(view_519, permute_164)
    permute_429: "f32[768, 2048]" = torch.ops.aten.permute.default(view_519, [1, 0])
    mm_107: "f32[768, 768]" = torch.ops.aten.mm.default(permute_429, view_68);  permute_429 = None
    permute_430: "f32[768, 768]" = torch.ops.aten.permute.default(mm_107, [1, 0]);  mm_107 = None
    sum_151: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_519, [0], True);  view_519 = None
    view_520: "f32[768]" = torch.ops.aten.view.default(sum_151, [768]);  sum_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_291: "f32[768]" = torch.ops.aten.add.Tensor(add_269, view_520);  add_269 = view_520 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    permute_431: "f32[768, 768]" = torch.ops.aten.permute.default(permute_430, [1, 0]);  permute_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_292: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_270, permute_431);  add_270 = permute_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_521: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_106, [4, 512, 768]);  mm_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_293: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_342, view_521);  mul_342 = view_521 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    clone_108: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_517, memory_format = torch.contiguous_format);  view_517 = None
    view_522: "f32[2048, 768]" = torch.ops.aten.view.default(clone_108, [2048, 768]);  clone_108 = None
    mm_108: "f32[2048, 768]" = torch.ops.aten.mm.default(view_522, permute_168)
    permute_433: "f32[768, 2048]" = torch.ops.aten.permute.default(view_522, [1, 0])
    mm_109: "f32[768, 768]" = torch.ops.aten.mm.default(permute_433, view_68);  permute_433 = None
    permute_434: "f32[768, 768]" = torch.ops.aten.permute.default(mm_109, [1, 0]);  mm_109 = None
    sum_152: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_522, [0], True);  view_522 = None
    view_523: "f32[768]" = torch.ops.aten.view.default(sum_152, [768]);  sum_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_294: "f32[768]" = torch.ops.aten.add.Tensor(add_272, view_523);  add_272 = view_523 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    permute_435: "f32[768, 768]" = torch.ops.aten.permute.default(permute_434, [1, 0]);  permute_434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_295: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_273, permute_435);  add_273 = permute_435 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_524: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_108, [4, 512, 768]);  mm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_296: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_293, view_524);  add_293 = view_524 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_525: "f32[2048, 768]" = torch.ops.aten.view.default(view_518, [2048, 768]);  view_518 = None
    mm_110: "f32[2048, 768]" = torch.ops.aten.mm.default(view_525, permute_172)
    permute_437: "f32[768, 2048]" = torch.ops.aten.permute.default(view_525, [1, 0])
    mm_111: "f32[768, 768]" = torch.ops.aten.mm.default(permute_437, view_68);  permute_437 = view_68 = None
    permute_438: "f32[768, 768]" = torch.ops.aten.permute.default(mm_111, [1, 0]);  mm_111 = None
    sum_153: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_525, [0], True);  view_525 = None
    view_526: "f32[768]" = torch.ops.aten.view.default(sum_153, [768]);  sum_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_297: "f32[768]" = torch.ops.aten.add.Tensor(add_275, view_526);  add_275 = view_526 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    permute_439: "f32[768, 768]" = torch.ops.aten.permute.default(permute_438, [1, 0]);  permute_438 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_298: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_276, permute_439);  add_276 = permute_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_527: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_110, [4, 512, 768]);  mm_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_299: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_296, view_527);  add_296 = view_527 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    mul_347: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_299, primals_22)
    mul_348: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_347, 768)
    sum_154: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_347, [2], True)
    mul_349: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_347, mul_25);  mul_347 = None
    sum_155: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_349, [2], True);  mul_349 = None
    mul_350: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_25, sum_155);  sum_155 = None
    sub_116: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_348, sum_154);  mul_348 = sum_154 = None
    sub_117: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_116, mul_350);  sub_116 = mul_350 = None
    mul_351: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_52, sub_117);  div_52 = sub_117 = None
    mul_352: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_299, mul_25);  mul_25 = None
    sum_156: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_352, [0, 1]);  mul_352 = None
    sum_157: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_299, [0, 1]);  add_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_300: "f32[768]" = torch.ops.aten.add.Tensor(add_278, sum_156);  add_278 = sum_156 = None
    add_301: "f32[768]" = torch.ops.aten.add.Tensor(add_279, sum_157);  add_279 = sum_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_528: "f32[2048, 768]" = torch.ops.aten.view.default(mul_351, [2048, 768])
    mm_112: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_528, permute_143)
    permute_441: "f32[768, 2048]" = torch.ops.aten.permute.default(view_528, [1, 0])
    mm_113: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_441, view_66);  permute_441 = view_66 = None
    permute_442: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_113, [1, 0]);  mm_113 = None
    sum_158: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_528, [0], True);  view_528 = None
    view_529: "f32[768]" = torch.ops.aten.view.default(sum_158, [768]);  sum_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_302: "f32[768]" = torch.ops.aten.add.Tensor(add_280, view_529);  add_280 = view_529 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    permute_443: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_442, [1, 0]);  permute_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_303: "f32[768, 3072]" = torch.ops.aten.add.Tensor(add_281, permute_443);  add_281 = permute_443 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_530: "f32[4, 512, 3072]" = torch.ops.aten.view.default(mm_112, [4, 512, 3072]);  mm_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_353: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_530, mul_21);  mul_21 = None
    mul_354: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_530, add_27);  view_530 = add_27 = None
    alias_44: "f32[4, 512, 3072]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    mul_355: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(alias_44, alias_44);  alias_44 = None
    sub_118: "f32[4, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_355);  mul_355 = None
    mul_356: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_353, sub_118);  mul_353 = sub_118 = None
    mul_357: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_356, 0.7978845608028654);  mul_356 = None
    mul_358: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_357, 0.044715)
    pow_24: "f32[4, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_65, 2.0);  view_65 = None
    mul_359: "f32[4, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_24, 3.0);  pow_24 = None
    mul_360: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_358, mul_359);  mul_358 = mul_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_304: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_357, mul_360);  mul_357 = mul_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_361: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_354, 0.5);  mul_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_305: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(add_304, mul_361);  add_304 = mul_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_531: "f32[2048, 3072]" = torch.ops.aten.view.default(add_305, [2048, 3072]);  add_305 = None
    mm_114: "f32[2048, 768]" = torch.ops.aten.mm.default(view_531, permute_147)
    permute_445: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_531, [1, 0])
    mm_115: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_445, view_64);  permute_445 = view_64 = None
    permute_446: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_115, [1, 0]);  mm_115 = None
    sum_159: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_531, [0], True);  view_531 = None
    view_532: "f32[3072]" = torch.ops.aten.view.default(sum_159, [3072]);  sum_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_306: "f32[3072]" = torch.ops.aten.add.Tensor(add_284, view_532);  add_284 = view_532 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    permute_447: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_446, [1, 0]);  permute_446 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_307: "f32[3072, 768]" = torch.ops.aten.add.Tensor(add_285, permute_447);  add_285 = permute_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_533: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_114, [4, 512, 768]);  mm_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_308: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_351, view_533);  mul_351 = view_533 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    mul_363: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_308, primals_16)
    mul_364: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_363, 768)
    sum_160: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_363, [2], True)
    mul_365: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_363, mul_19);  mul_363 = None
    sum_161: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_365, [2], True);  mul_365 = None
    mul_366: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_19, sum_161);  sum_161 = None
    sub_120: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_364, sum_160);  mul_364 = sum_160 = None
    sub_121: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_120, mul_366);  sub_120 = mul_366 = None
    mul_367: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_53, sub_121);  div_53 = sub_121 = None
    mul_368: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_308, mul_19);  mul_19 = None
    sum_162: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_368, [0, 1]);  mul_368 = None
    sum_163: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_308, [0, 1]);  add_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_309: "f32[768]" = torch.ops.aten.add.Tensor(add_287, sum_162);  add_287 = sum_162 = None
    add_310: "f32[768]" = torch.ops.aten.add.Tensor(add_288, sum_163);  add_288 = sum_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_534: "f32[2048, 768]" = torch.ops.aten.view.default(mul_367, [2048, 768])
    mm_116: "f32[2048, 768]" = torch.ops.aten.mm.default(view_534, permute_151)
    permute_449: "f32[768, 2048]" = torch.ops.aten.permute.default(view_534, [1, 0])
    mm_117: "f32[768, 768]" = torch.ops.aten.mm.default(permute_449, view_62);  permute_449 = view_62 = None
    permute_450: "f32[768, 768]" = torch.ops.aten.permute.default(mm_117, [1, 0]);  mm_117 = None
    sum_164: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_534, [0], True);  view_534 = None
    view_535: "f32[768]" = torch.ops.aten.view.default(sum_164, [768]);  sum_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_311: "f32[768]" = torch.ops.aten.add.Tensor(add_289, view_535);  add_289 = view_535 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    permute_451: "f32[768, 768]" = torch.ops.aten.permute.default(permute_450, [1, 0]);  permute_450 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_312: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_290, permute_451);  add_290 = permute_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_536: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_116, [4, 512, 768]);  mm_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    view_537: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_536, [4, 512, 12, 64]);  view_536 = None
    permute_452: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_537, [0, 2, 1, 3]);  view_537 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    clone_109: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_452, memory_format = torch.contiguous_format);  permute_452 = None
    view_538: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_109, [48, 512, 64]);  clone_109 = None
    bmm_60: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_453, view_538);  permute_453 = None
    bmm_61: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_538, permute_454);  view_538 = permute_454 = None
    view_539: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_60, [4, 12, 512, 64]);  bmm_60 = None
    view_540: "f32[4, 12, 512, 512]" = torch.ops.aten.view.default(bmm_61, [4, 12, 512, 512]);  bmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    mul_369: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_540, alias_45);  view_540 = None
    sum_165: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_369, [-1], True)
    mul_370: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_45, sum_165);  alias_45 = sum_165 = None
    sub_122: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_369, mul_370);  mul_369 = mul_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:332, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_54: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(sub_122, 8.0);  sub_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_541: "f32[48, 512, 512]" = torch.ops.aten.view.default(div_54, [48, 512, 512]);  div_54 = None
    bmm_62: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_455, view_541);  permute_455 = None
    bmm_63: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_541, permute_456);  view_541 = permute_456 = None
    view_542: "f32[4, 12, 64, 512]" = torch.ops.aten.view.default(bmm_62, [4, 12, 64, 512]);  bmm_62 = None
    view_543: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_63, [4, 12, 512, 64]);  bmm_63 = None
    permute_457: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_542, [0, 1, 3, 2]);  view_542 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_458: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_539, [0, 2, 1, 3]);  view_539 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_110: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_458, memory_format = torch.contiguous_format);  permute_458 = None
    view_544: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_110, [4, 512, 768]);  clone_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_459: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(permute_457, [0, 2, 1, 3]);  permute_457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_545: "f32[4, 512, 768]" = torch.ops.aten.view.default(permute_459, [4, 512, 768]);  permute_459 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_460: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_543, [0, 2, 1, 3]);  view_543 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_111: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_460, memory_format = torch.contiguous_format);  permute_460 = None
    view_546: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_111, [4, 512, 768]);  clone_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_547: "f32[2048, 768]" = torch.ops.aten.view.default(view_544, [2048, 768]);  view_544 = None
    mm_118: "f32[2048, 768]" = torch.ops.aten.mm.default(view_547, permute_164)
    permute_462: "f32[768, 2048]" = torch.ops.aten.permute.default(view_547, [1, 0])
    mm_119: "f32[768, 768]" = torch.ops.aten.mm.default(permute_462, view_46);  permute_462 = None
    permute_463: "f32[768, 768]" = torch.ops.aten.permute.default(mm_119, [1, 0]);  mm_119 = None
    sum_166: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_547, [0], True);  view_547 = None
    view_548: "f32[768]" = torch.ops.aten.view.default(sum_166, [768]);  sum_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_313: "f32[768]" = torch.ops.aten.add.Tensor(add_291, view_548);  add_291 = view_548 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    permute_464: "f32[768, 768]" = torch.ops.aten.permute.default(permute_463, [1, 0]);  permute_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_314: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_292, permute_464);  add_292 = permute_464 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_549: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_118, [4, 512, 768]);  mm_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_315: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_367, view_549);  mul_367 = view_549 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    clone_112: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_545, memory_format = torch.contiguous_format);  view_545 = None
    view_550: "f32[2048, 768]" = torch.ops.aten.view.default(clone_112, [2048, 768]);  clone_112 = None
    mm_120: "f32[2048, 768]" = torch.ops.aten.mm.default(view_550, permute_168)
    permute_466: "f32[768, 2048]" = torch.ops.aten.permute.default(view_550, [1, 0])
    mm_121: "f32[768, 768]" = torch.ops.aten.mm.default(permute_466, view_46);  permute_466 = None
    permute_467: "f32[768, 768]" = torch.ops.aten.permute.default(mm_121, [1, 0]);  mm_121 = None
    sum_167: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_550, [0], True);  view_550 = None
    view_551: "f32[768]" = torch.ops.aten.view.default(sum_167, [768]);  sum_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_316: "f32[768]" = torch.ops.aten.add.Tensor(add_294, view_551);  add_294 = view_551 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    permute_468: "f32[768, 768]" = torch.ops.aten.permute.default(permute_467, [1, 0]);  permute_467 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_317: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_295, permute_468);  add_295 = permute_468 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_552: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_120, [4, 512, 768]);  mm_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_318: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_315, view_552);  add_315 = view_552 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_553: "f32[2048, 768]" = torch.ops.aten.view.default(view_546, [2048, 768]);  view_546 = None
    mm_122: "f32[2048, 768]" = torch.ops.aten.mm.default(view_553, permute_172)
    permute_470: "f32[768, 2048]" = torch.ops.aten.permute.default(view_553, [1, 0])
    mm_123: "f32[768, 768]" = torch.ops.aten.mm.default(permute_470, view_46);  permute_470 = view_46 = None
    permute_471: "f32[768, 768]" = torch.ops.aten.permute.default(mm_123, [1, 0]);  mm_123 = None
    sum_168: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_553, [0], True);  view_553 = None
    view_554: "f32[768]" = torch.ops.aten.view.default(sum_168, [768]);  sum_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_319: "f32[768]" = torch.ops.aten.add.Tensor(add_297, view_554);  add_297 = view_554 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    permute_472: "f32[768, 768]" = torch.ops.aten.permute.default(permute_471, [1, 0]);  permute_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_320: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_298, permute_472);  add_298 = permute_472 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_555: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_122, [4, 512, 768]);  mm_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_321: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_318, view_555);  add_318 = view_555 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    mul_372: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_321, primals_22)
    mul_373: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_372, 768)
    sum_169: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_372, [2], True)
    mul_374: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_372, mul_17);  mul_372 = None
    sum_170: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_374, [2], True);  mul_374 = None
    mul_375: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_17, sum_170);  sum_170 = None
    sub_124: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_373, sum_169);  mul_373 = sum_169 = None
    sub_125: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_124, mul_375);  sub_124 = mul_375 = None
    mul_376: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_55, sub_125);  div_55 = sub_125 = None
    mul_377: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_321, mul_17);  mul_17 = None
    sum_171: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_377, [0, 1]);  mul_377 = None
    sum_172: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_321, [0, 1]);  add_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_322: "f32[768]" = torch.ops.aten.add.Tensor(add_300, sum_171);  add_300 = sum_171 = None
    add_323: "f32[768]" = torch.ops.aten.add.Tensor(add_301, sum_172);  add_301 = sum_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_556: "f32[2048, 768]" = torch.ops.aten.view.default(mul_376, [2048, 768])
    mm_124: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_556, permute_143)
    permute_474: "f32[768, 2048]" = torch.ops.aten.permute.default(view_556, [1, 0])
    mm_125: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_474, view_44);  permute_474 = view_44 = None
    permute_475: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_125, [1, 0]);  mm_125 = None
    sum_173: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_556, [0], True);  view_556 = None
    view_557: "f32[768]" = torch.ops.aten.view.default(sum_173, [768]);  sum_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_324: "f32[768]" = torch.ops.aten.add.Tensor(add_302, view_557);  add_302 = view_557 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    permute_476: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_475, [1, 0]);  permute_475 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_325: "f32[768, 3072]" = torch.ops.aten.add.Tensor(add_303, permute_476);  add_303 = permute_476 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_558: "f32[4, 512, 3072]" = torch.ops.aten.view.default(mm_124, [4, 512, 3072]);  mm_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_378: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_558, mul_13);  mul_13 = None
    mul_379: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_558, add_18);  view_558 = add_18 = None
    alias_46: "f32[4, 512, 3072]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    mul_380: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(alias_46, alias_46);  alias_46 = None
    sub_126: "f32[4, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_380);  mul_380 = None
    mul_381: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_378, sub_126);  mul_378 = sub_126 = None
    mul_382: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_381, 0.7978845608028654);  mul_381 = None
    mul_383: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_382, 0.044715)
    pow_25: "f32[4, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_43, 2.0);  view_43 = None
    mul_384: "f32[4, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_25, 3.0);  pow_25 = None
    mul_385: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_383, mul_384);  mul_383 = mul_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_326: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_382, mul_385);  mul_382 = mul_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_386: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_379, 0.5);  mul_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_327: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(add_326, mul_386);  add_326 = mul_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_559: "f32[2048, 3072]" = torch.ops.aten.view.default(add_327, [2048, 3072]);  add_327 = None
    mm_126: "f32[2048, 768]" = torch.ops.aten.mm.default(view_559, permute_147)
    permute_478: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_559, [1, 0])
    mm_127: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_478, view_42);  permute_478 = view_42 = None
    permute_479: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_127, [1, 0]);  mm_127 = None
    sum_174: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_559, [0], True);  view_559 = None
    view_560: "f32[3072]" = torch.ops.aten.view.default(sum_174, [3072]);  sum_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_328: "f32[3072]" = torch.ops.aten.add.Tensor(add_306, view_560);  add_306 = view_560 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    permute_480: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_479, [1, 0]);  permute_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_329: "f32[3072, 768]" = torch.ops.aten.add.Tensor(add_307, permute_480);  add_307 = permute_480 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_561: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_126, [4, 512, 768]);  mm_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_330: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_376, view_561);  mul_376 = view_561 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    mul_388: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_330, primals_16)
    mul_389: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_388, 768)
    sum_175: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_388, [2], True)
    mul_390: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_388, mul_11);  mul_388 = None
    sum_176: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_390, [2], True);  mul_390 = None
    mul_391: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_11, sum_176);  sum_176 = None
    sub_128: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_389, sum_175);  mul_389 = sum_175 = None
    sub_129: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_128, mul_391);  sub_128 = mul_391 = None
    mul_392: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_56, sub_129);  div_56 = sub_129 = None
    mul_393: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_330, mul_11);  mul_11 = None
    sum_177: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_393, [0, 1]);  mul_393 = None
    sum_178: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_330, [0, 1]);  add_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_331: "f32[768]" = torch.ops.aten.add.Tensor(add_309, sum_177);  add_309 = sum_177 = None
    add_332: "f32[768]" = torch.ops.aten.add.Tensor(add_310, sum_178);  add_310 = sum_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_562: "f32[2048, 768]" = torch.ops.aten.view.default(mul_392, [2048, 768])
    mm_128: "f32[2048, 768]" = torch.ops.aten.mm.default(view_562, permute_151)
    permute_482: "f32[768, 2048]" = torch.ops.aten.permute.default(view_562, [1, 0])
    mm_129: "f32[768, 768]" = torch.ops.aten.mm.default(permute_482, view_40);  permute_482 = view_40 = None
    permute_483: "f32[768, 768]" = torch.ops.aten.permute.default(mm_129, [1, 0]);  mm_129 = None
    sum_179: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_562, [0], True);  view_562 = None
    view_563: "f32[768]" = torch.ops.aten.view.default(sum_179, [768]);  sum_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_333: "f32[768]" = torch.ops.aten.add.Tensor(add_311, view_563);  add_311 = view_563 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    permute_484: "f32[768, 768]" = torch.ops.aten.permute.default(permute_483, [1, 0]);  permute_483 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_334: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_312, permute_484);  add_312 = permute_484 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_564: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_128, [4, 512, 768]);  mm_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    view_565: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_564, [4, 512, 12, 64]);  view_564 = None
    permute_485: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_565, [0, 2, 1, 3]);  view_565 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    clone_113: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_485, memory_format = torch.contiguous_format);  permute_485 = None
    view_566: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_113, [48, 512, 64]);  clone_113 = None
    bmm_64: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_486, view_566);  permute_486 = None
    bmm_65: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_566, permute_487);  view_566 = permute_487 = None
    view_567: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_64, [4, 12, 512, 64]);  bmm_64 = None
    view_568: "f32[4, 12, 512, 512]" = torch.ops.aten.view.default(bmm_65, [4, 12, 512, 512]);  bmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    mul_394: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_568, alias_47);  view_568 = None
    sum_180: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_394, [-1], True)
    mul_395: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_47, sum_180);  alias_47 = sum_180 = None
    sub_130: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_394, mul_395);  mul_394 = mul_395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:332, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_57: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(sub_130, 8.0);  sub_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_569: "f32[48, 512, 512]" = torch.ops.aten.view.default(div_57, [48, 512, 512]);  div_57 = None
    bmm_66: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_488, view_569);  permute_488 = None
    bmm_67: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_569, permute_489);  view_569 = permute_489 = None
    view_570: "f32[4, 12, 64, 512]" = torch.ops.aten.view.default(bmm_66, [4, 12, 64, 512]);  bmm_66 = None
    view_571: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_67, [4, 12, 512, 64]);  bmm_67 = None
    permute_490: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_570, [0, 1, 3, 2]);  view_570 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_491: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_567, [0, 2, 1, 3]);  view_567 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_114: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_491, memory_format = torch.contiguous_format);  permute_491 = None
    view_572: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_114, [4, 512, 768]);  clone_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_492: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(permute_490, [0, 2, 1, 3]);  permute_490 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_573: "f32[4, 512, 768]" = torch.ops.aten.view.default(permute_492, [4, 512, 768]);  permute_492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_493: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_571, [0, 2, 1, 3]);  view_571 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_115: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_493, memory_format = torch.contiguous_format);  permute_493 = None
    view_574: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_115, [4, 512, 768]);  clone_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_575: "f32[2048, 768]" = torch.ops.aten.view.default(view_572, [2048, 768]);  view_572 = None
    mm_130: "f32[2048, 768]" = torch.ops.aten.mm.default(view_575, permute_164)
    permute_495: "f32[768, 2048]" = torch.ops.aten.permute.default(view_575, [1, 0])
    mm_131: "f32[768, 768]" = torch.ops.aten.mm.default(permute_495, view_24);  permute_495 = None
    permute_496: "f32[768, 768]" = torch.ops.aten.permute.default(mm_131, [1, 0]);  mm_131 = None
    sum_181: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_575, [0], True);  view_575 = None
    view_576: "f32[768]" = torch.ops.aten.view.default(sum_181, [768]);  sum_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_335: "f32[768]" = torch.ops.aten.add.Tensor(add_313, view_576);  add_313 = view_576 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    permute_497: "f32[768, 768]" = torch.ops.aten.permute.default(permute_496, [1, 0]);  permute_496 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_336: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_314, permute_497);  add_314 = permute_497 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_577: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_130, [4, 512, 768]);  mm_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_337: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_392, view_577);  mul_392 = view_577 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    clone_116: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_573, memory_format = torch.contiguous_format);  view_573 = None
    view_578: "f32[2048, 768]" = torch.ops.aten.view.default(clone_116, [2048, 768]);  clone_116 = None
    mm_132: "f32[2048, 768]" = torch.ops.aten.mm.default(view_578, permute_168)
    permute_499: "f32[768, 2048]" = torch.ops.aten.permute.default(view_578, [1, 0])
    mm_133: "f32[768, 768]" = torch.ops.aten.mm.default(permute_499, view_24);  permute_499 = None
    permute_500: "f32[768, 768]" = torch.ops.aten.permute.default(mm_133, [1, 0]);  mm_133 = None
    sum_182: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_578, [0], True);  view_578 = None
    view_579: "f32[768]" = torch.ops.aten.view.default(sum_182, [768]);  sum_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_338: "f32[768]" = torch.ops.aten.add.Tensor(add_316, view_579);  add_316 = view_579 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    permute_501: "f32[768, 768]" = torch.ops.aten.permute.default(permute_500, [1, 0]);  permute_500 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_339: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_317, permute_501);  add_317 = permute_501 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_580: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_132, [4, 512, 768]);  mm_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_340: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_337, view_580);  add_337 = view_580 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_581: "f32[2048, 768]" = torch.ops.aten.view.default(view_574, [2048, 768]);  view_574 = None
    mm_134: "f32[2048, 768]" = torch.ops.aten.mm.default(view_581, permute_172)
    permute_503: "f32[768, 2048]" = torch.ops.aten.permute.default(view_581, [1, 0])
    mm_135: "f32[768, 768]" = torch.ops.aten.mm.default(permute_503, view_24);  permute_503 = view_24 = None
    permute_504: "f32[768, 768]" = torch.ops.aten.permute.default(mm_135, [1, 0]);  mm_135 = None
    sum_183: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_581, [0], True);  view_581 = None
    view_582: "f32[768]" = torch.ops.aten.view.default(sum_183, [768]);  sum_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_341: "f32[768]" = torch.ops.aten.add.Tensor(add_319, view_582);  add_319 = view_582 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    permute_505: "f32[768, 768]" = torch.ops.aten.permute.default(permute_504, [1, 0]);  permute_504 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_342: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_320, permute_505);  add_320 = permute_505 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_583: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_134, [4, 512, 768]);  mm_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_343: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_340, view_583);  add_340 = view_583 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    mul_397: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_343, primals_22);  primals_22 = None
    mul_398: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_397, 768)
    sum_184: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_397, [2], True)
    mul_399: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_397, mul_9);  mul_397 = None
    sum_185: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_399, [2], True);  mul_399 = None
    mul_400: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_9, sum_185);  sum_185 = None
    sub_132: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_398, sum_184);  mul_398 = sum_184 = None
    sub_133: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_132, mul_400);  sub_132 = mul_400 = None
    mul_401: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_58, sub_133);  div_58 = sub_133 = None
    mul_402: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_343, mul_9);  mul_9 = None
    sum_186: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_402, [0, 1]);  mul_402 = None
    sum_187: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_343, [0, 1]);  add_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_344: "f32[768]" = torch.ops.aten.add.Tensor(add_322, sum_186);  add_322 = sum_186 = None
    add_345: "f32[768]" = torch.ops.aten.add.Tensor(add_323, sum_187);  add_323 = sum_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_584: "f32[2048, 768]" = torch.ops.aten.view.default(mul_401, [2048, 768])
    mm_136: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_584, permute_143);  permute_143 = None
    permute_507: "f32[768, 2048]" = torch.ops.aten.permute.default(view_584, [1, 0])
    mm_137: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_507, view_22);  permute_507 = view_22 = None
    permute_508: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_137, [1, 0]);  mm_137 = None
    sum_188: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_584, [0], True);  view_584 = None
    view_585: "f32[768]" = torch.ops.aten.view.default(sum_188, [768]);  sum_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_346: "f32[768]" = torch.ops.aten.add.Tensor(add_324, view_585);  add_324 = view_585 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    permute_509: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_508, [1, 0]);  permute_508 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_347: "f32[768, 3072]" = torch.ops.aten.add.Tensor(add_325, permute_509);  add_325 = permute_509 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_586: "f32[4, 512, 3072]" = torch.ops.aten.view.default(mm_136, [4, 512, 3072]);  mm_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_403: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_586, mul_5);  mul_5 = None
    mul_404: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_586, add_9);  view_586 = add_9 = None
    alias_48: "f32[4, 512, 3072]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    mul_405: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(alias_48, alias_48);  alias_48 = None
    sub_134: "f32[4, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_405);  mul_405 = None
    mul_406: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_403, sub_134);  mul_403 = sub_134 = None
    mul_407: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_406, 0.7978845608028654);  mul_406 = None
    mul_408: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_407, 0.044715)
    pow_26: "f32[4, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_21, 2.0);  view_21 = None
    mul_409: "f32[4, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_26, 3.0);  pow_26 = None
    mul_410: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_408, mul_409);  mul_408 = mul_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_348: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_407, mul_410);  mul_407 = mul_410 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_411: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_404, 0.5);  mul_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_349: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(add_348, mul_411);  add_348 = mul_411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_587: "f32[2048, 3072]" = torch.ops.aten.view.default(add_349, [2048, 3072]);  add_349 = None
    mm_138: "f32[2048, 768]" = torch.ops.aten.mm.default(view_587, permute_147);  permute_147 = None
    permute_511: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_587, [1, 0])
    mm_139: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_511, view_20);  permute_511 = view_20 = None
    permute_512: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_139, [1, 0]);  mm_139 = None
    sum_189: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_587, [0], True);  view_587 = None
    view_588: "f32[3072]" = torch.ops.aten.view.default(sum_189, [3072]);  sum_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_350: "f32[3072]" = torch.ops.aten.add.Tensor(add_328, view_588);  add_328 = view_588 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    permute_513: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_512, [1, 0]);  permute_512 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_351: "f32[3072, 768]" = torch.ops.aten.add.Tensor(add_329, permute_513);  add_329 = permute_513 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_589: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_138, [4, 512, 768]);  mm_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_352: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_401, view_589);  mul_401 = view_589 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    mul_413: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_352, primals_16);  primals_16 = None
    mul_414: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_413, 768)
    sum_190: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_413, [2], True)
    mul_415: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_413, mul_3);  mul_413 = None
    sum_191: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_415, [2], True);  mul_415 = None
    mul_416: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_3, sum_191);  sum_191 = None
    sub_136: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_414, sum_190);  mul_414 = sum_190 = None
    sub_137: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_136, mul_416);  sub_136 = mul_416 = None
    mul_417: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_59, sub_137);  div_59 = sub_137 = None
    mul_418: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_352, mul_3);  mul_3 = None
    sum_192: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_418, [0, 1]);  mul_418 = None
    sum_193: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_352, [0, 1]);  add_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_353: "f32[768]" = torch.ops.aten.add.Tensor(add_331, sum_192);  add_331 = sum_192 = None
    add_354: "f32[768]" = torch.ops.aten.add.Tensor(add_332, sum_193);  add_332 = sum_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_590: "f32[2048, 768]" = torch.ops.aten.view.default(mul_417, [2048, 768])
    mm_140: "f32[2048, 768]" = torch.ops.aten.mm.default(view_590, permute_151);  permute_151 = None
    permute_515: "f32[768, 2048]" = torch.ops.aten.permute.default(view_590, [1, 0])
    mm_141: "f32[768, 768]" = torch.ops.aten.mm.default(permute_515, view_18);  permute_515 = view_18 = None
    permute_516: "f32[768, 768]" = torch.ops.aten.permute.default(mm_141, [1, 0]);  mm_141 = None
    sum_194: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_590, [0], True);  view_590 = None
    view_591: "f32[768]" = torch.ops.aten.view.default(sum_194, [768]);  sum_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_355: "f32[768]" = torch.ops.aten.add.Tensor(add_333, view_591);  add_333 = view_591 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    permute_517: "f32[768, 768]" = torch.ops.aten.permute.default(permute_516, [1, 0]);  permute_516 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_356: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_334, permute_517);  add_334 = permute_517 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_592: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_140, [4, 512, 768]);  mm_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    view_593: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_592, [4, 512, 12, 64]);  view_592 = None
    permute_518: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_593, [0, 2, 1, 3]);  view_593 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    clone_117: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_518, memory_format = torch.contiguous_format);  permute_518 = None
    view_594: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_117, [48, 512, 64]);  clone_117 = None
    bmm_68: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_519, view_594);  permute_519 = None
    bmm_69: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_594, permute_520);  view_594 = permute_520 = None
    view_595: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_68, [4, 12, 512, 64]);  bmm_68 = None
    view_596: "f32[4, 12, 512, 512]" = torch.ops.aten.view.default(bmm_69, [4, 12, 512, 512]);  bmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    mul_419: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_596, alias_49);  view_596 = None
    sum_195: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_419, [-1], True)
    mul_420: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_49, sum_195);  alias_49 = sum_195 = None
    sub_138: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_419, mul_420);  mul_419 = mul_420 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:332, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_60: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(sub_138, 8.0);  sub_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_597: "f32[48, 512, 512]" = torch.ops.aten.view.default(div_60, [48, 512, 512]);  div_60 = None
    bmm_70: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_521, view_597);  permute_521 = None
    bmm_71: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_597, permute_522);  view_597 = permute_522 = None
    view_598: "f32[4, 12, 64, 512]" = torch.ops.aten.view.default(bmm_70, [4, 12, 64, 512]);  bmm_70 = None
    view_599: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_71, [4, 12, 512, 64]);  bmm_71 = None
    permute_523: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_598, [0, 1, 3, 2]);  view_598 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_524: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_595, [0, 2, 1, 3]);  view_595 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_118: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_524, memory_format = torch.contiguous_format);  permute_524 = None
    view_600: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_118, [4, 512, 768]);  clone_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_525: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(permute_523, [0, 2, 1, 3]);  permute_523 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_601: "f32[4, 512, 768]" = torch.ops.aten.view.default(permute_525, [4, 512, 768]);  permute_525 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_526: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_599, [0, 2, 1, 3]);  view_599 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_119: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_526, memory_format = torch.contiguous_format);  permute_526 = None
    view_602: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_119, [4, 512, 768]);  clone_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_603: "f32[2048, 768]" = torch.ops.aten.view.default(view_600, [2048, 768]);  view_600 = None
    mm_142: "f32[2048, 768]" = torch.ops.aten.mm.default(view_603, permute_164);  permute_164 = None
    permute_528: "f32[768, 2048]" = torch.ops.aten.permute.default(view_603, [1, 0])
    mm_143: "f32[768, 768]" = torch.ops.aten.mm.default(permute_528, view_2);  permute_528 = None
    permute_529: "f32[768, 768]" = torch.ops.aten.permute.default(mm_143, [1, 0]);  mm_143 = None
    sum_196: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_603, [0], True);  view_603 = None
    view_604: "f32[768]" = torch.ops.aten.view.default(sum_196, [768]);  sum_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_357: "f32[768]" = torch.ops.aten.add.Tensor(add_335, view_604);  add_335 = view_604 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    permute_530: "f32[768, 768]" = torch.ops.aten.permute.default(permute_529, [1, 0]);  permute_529 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_358: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_336, permute_530);  add_336 = permute_530 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_605: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_142, [4, 512, 768]);  mm_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_359: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_417, view_605);  mul_417 = view_605 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    clone_120: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_601, memory_format = torch.contiguous_format);  view_601 = None
    view_606: "f32[2048, 768]" = torch.ops.aten.view.default(clone_120, [2048, 768]);  clone_120 = None
    mm_144: "f32[2048, 768]" = torch.ops.aten.mm.default(view_606, permute_168);  permute_168 = None
    permute_532: "f32[768, 2048]" = torch.ops.aten.permute.default(view_606, [1, 0])
    mm_145: "f32[768, 768]" = torch.ops.aten.mm.default(permute_532, view_2);  permute_532 = None
    permute_533: "f32[768, 768]" = torch.ops.aten.permute.default(mm_145, [1, 0]);  mm_145 = None
    sum_197: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_606, [0], True);  view_606 = None
    view_607: "f32[768]" = torch.ops.aten.view.default(sum_197, [768]);  sum_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_360: "f32[768]" = torch.ops.aten.add.Tensor(add_338, view_607);  add_338 = view_607 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    permute_534: "f32[768, 768]" = torch.ops.aten.permute.default(permute_533, [1, 0]);  permute_533 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_361: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_339, permute_534);  add_339 = permute_534 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_608: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_144, [4, 512, 768]);  mm_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_362: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_359, view_608);  add_359 = view_608 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_609: "f32[2048, 768]" = torch.ops.aten.view.default(view_602, [2048, 768]);  view_602 = None
    mm_146: "f32[2048, 768]" = torch.ops.aten.mm.default(view_609, permute_172);  permute_172 = None
    permute_536: "f32[768, 2048]" = torch.ops.aten.permute.default(view_609, [1, 0])
    mm_147: "f32[768, 768]" = torch.ops.aten.mm.default(permute_536, view_2);  permute_536 = view_2 = None
    permute_537: "f32[768, 768]" = torch.ops.aten.permute.default(mm_147, [1, 0]);  mm_147 = None
    sum_198: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_609, [0], True);  view_609 = None
    view_610: "f32[768]" = torch.ops.aten.view.default(sum_198, [768]);  sum_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_363: "f32[768]" = torch.ops.aten.add.Tensor(add_341, view_610);  add_341 = view_610 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    permute_538: "f32[768, 768]" = torch.ops.aten.permute.default(permute_537, [1, 0]);  permute_537 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_364: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_342, permute_538);  add_342 = permute_538 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_611: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_146, [4, 512, 768]);  mm_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_365: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_362, view_611);  add_362 = view_611 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:467, code: hidden_states = self.embedding_hidden_mapping_in(hidden_states)
    view_612: "f32[2048, 768]" = torch.ops.aten.view.default(add_365, [2048, 768]);  add_365 = None
    mm_148: "f32[2048, 128]" = torch.ops.aten.mm.default(view_612, permute_539);  permute_539 = None
    permute_540: "f32[768, 2048]" = torch.ops.aten.permute.default(view_612, [1, 0])
    mm_149: "f32[768, 128]" = torch.ops.aten.mm.default(permute_540, view);  permute_540 = view = None
    permute_541: "f32[128, 768]" = torch.ops.aten.permute.default(mm_149, [1, 0]);  mm_149 = None
    sum_199: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_612, [0], True);  view_612 = None
    view_613: "f32[768]" = torch.ops.aten.view.default(sum_199, [768]);  sum_199 = None
    permute_542: "f32[768, 128]" = torch.ops.aten.permute.default(permute_541, [1, 0]);  permute_541 = None
    view_614: "f32[4, 512, 128]" = torch.ops.aten.view.default(mm_148, [4, 512, 128]);  mm_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:257, code: embeddings = self.LayerNorm(embeddings)
    mul_422: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(view_614, primals_4);  primals_4 = None
    mul_423: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(mul_422, 128)
    sum_200: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_422, [2], True)
    mul_424: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(mul_422, mul_1);  mul_422 = None
    sum_201: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_424, [2], True);  mul_424 = None
    mul_425: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(mul_1, sum_201);  sum_201 = None
    sub_140: "f32[4, 512, 128]" = torch.ops.aten.sub.Tensor(mul_423, sum_200);  mul_423 = sum_200 = None
    sub_141: "f32[4, 512, 128]" = torch.ops.aten.sub.Tensor(sub_140, mul_425);  sub_140 = mul_425 = None
    mul_426: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(div_61, sub_141);  div_61 = sub_141 = None
    mul_427: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(view_614, mul_1);  mul_1 = None
    sum_202: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_427, [0, 1]);  mul_427 = None
    sum_203: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_614, [0, 1]);  view_614 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:256, code: embeddings += position_embeddings
    sum_204: "f32[1, 512, 128]" = torch.ops.aten.sum.dim_IntList(mul_426, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:255, code: position_embeddings = self.position_embeddings(position_ids)
    eq: "b8[1, 512]" = torch.ops.aten.eq.Scalar(slice_2, -1)
    unsqueeze_2: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq, -1);  eq = None
    full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where: "f32[1, 512, 128]" = torch.ops.aten.where.self(unsqueeze_2, full_default_1, sum_204);  unsqueeze_2 = sum_204 = None
    full_default_2: "f32[512, 128]" = torch.ops.aten.full.default([512, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put: "f32[512, 128]" = torch.ops.aten._unsafe_index_put.default(full_default_2, [slice_2], where, True);  full_default_2 = slice_2 = where = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:251, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
    eq_1: "b8[4, 512]" = torch.ops.aten.eq.Scalar(expand, -1)
    unsqueeze_3: "b8[4, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_1, -1);  eq_1 = None
    where_1: "f32[4, 512, 128]" = torch.ops.aten.where.self(unsqueeze_3, full_default_1, mul_426);  unsqueeze_3 = None
    full_default_4: "f32[2, 128]" = torch.ops.aten.full.default([2, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put_1: "f32[2, 128]" = torch.ops.aten._unsafe_index_put.default(full_default_4, [expand], where_1, True);  full_default_4 = expand = where_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:250, code: inputs_embeds = self.word_embeddings(input_ids)
    eq_2: "b8[4, 512]" = torch.ops.aten.eq.Scalar(primals_32, 0)
    unsqueeze_4: "b8[4, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_2, -1);  eq_2 = None
    where_2: "f32[4, 512, 128]" = torch.ops.aten.where.self(unsqueeze_4, full_default_1, mul_426);  unsqueeze_4 = full_default_1 = mul_426 = None
    full_default_6: "f32[30000, 128]" = torch.ops.aten.full.default([30000, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put_2: "f32[30000, 128]" = torch.ops.aten._unsafe_index_put.default(full_default_6, [primals_32], where_2, True);  full_default_6 = primals_32 = where_2 = None
    return [_unsafe_index_put_2, _unsafe_index_put_1, _unsafe_index_put, sum_202, sum_203, permute_542, view_613, add_364, add_363, add_361, add_360, add_358, add_357, add_356, add_355, add_353, add_354, add_351, add_350, add_347, add_346, add_344, add_345, permute_142, view_274, sum_16, sum_17, permute_138, view_271, None, None, None]
    