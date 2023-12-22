from __future__ import annotations



def forward(self, primals_99: "f32[768]", primals_101: "f32[768]", primals_103: "f32[768]", primals_105: "f32[768]", primals_107: "f32[768]", primals_109: "f32[768]", primals_111: "f32[768]", primals_113: "f32[768]", primals_115: "f32[768]", primals_117: "f32[768]", primals_119: "f32[768]", primals_121: "f32[768]", primals_123: "f32[768]", primals_125: "f32[768]", primals_127: "f32[768]", primals_129: "f32[768]", primals_131: "f32[768]", primals_133: "f32[768]", primals_135: "f32[768]", primals_137: "f32[768]", primals_139: "f32[768]", primals_141: "f32[768]", primals_143: "f32[768]", primals_145: "f32[768]", primals_147: "f32[768]", primals_150: "b8[1, 1, 1024, 1024]", primals_151: "b8[1, 1, 1024, 1024]", primals_152: "b8[1, 1, 1024, 1024]", primals_153: "b8[1, 1, 1024, 1024]", primals_154: "b8[1, 1, 1024, 1024]", primals_155: "b8[1, 1, 1024, 1024]", primals_156: "b8[1, 1, 1024, 1024]", primals_157: "b8[1, 1, 1024, 1024]", primals_158: "b8[1, 1, 1024, 1024]", primals_159: "b8[1, 1, 1024, 1024]", primals_160: "b8[1, 1, 1024, 1024]", primals_161: "b8[1, 1, 1024, 1024]", view: "i64[1, 1024]", view_1: "i64[1, 1024]", getitem_1: "b8[1, 1024, 768]", mul: "f32[1, 1024, 768]", getitem_8: "b8[1, 12, 1024, 1024]", getitem_10: "b8[1, 1024, 768]", mul_2: "f32[1, 1024, 768]", addmm_2: "f32[1024, 3072]", tanh: "f32[1, 1024, 3072]", getitem_14: "b8[1, 1024, 768]", mul_8: "f32[1, 1024, 768]", getitem_21: "b8[1, 12, 1024, 1024]", getitem_23: "b8[1, 1024, 768]", mul_10: "f32[1, 1024, 768]", addmm_6: "f32[1024, 3072]", tanh_1: "f32[1, 1024, 3072]", getitem_27: "b8[1, 1024, 768]", mul_16: "f32[1, 1024, 768]", getitem_34: "b8[1, 12, 1024, 1024]", getitem_36: "b8[1, 1024, 768]", mul_18: "f32[1, 1024, 768]", addmm_10: "f32[1024, 3072]", tanh_2: "f32[1, 1024, 3072]", getitem_40: "b8[1, 1024, 768]", mul_24: "f32[1, 1024, 768]", getitem_47: "b8[1, 12, 1024, 1024]", getitem_49: "b8[1, 1024, 768]", mul_26: "f32[1, 1024, 768]", addmm_14: "f32[1024, 3072]", tanh_3: "f32[1, 1024, 3072]", getitem_53: "b8[1, 1024, 768]", mul_32: "f32[1, 1024, 768]", getitem_60: "b8[1, 12, 1024, 1024]", getitem_62: "b8[1, 1024, 768]", mul_34: "f32[1, 1024, 768]", addmm_18: "f32[1024, 3072]", tanh_4: "f32[1, 1024, 3072]", getitem_66: "b8[1, 1024, 768]", mul_40: "f32[1, 1024, 768]", getitem_73: "b8[1, 12, 1024, 1024]", getitem_75: "b8[1, 1024, 768]", mul_42: "f32[1, 1024, 768]", addmm_22: "f32[1024, 3072]", tanh_5: "f32[1, 1024, 3072]", getitem_79: "b8[1, 1024, 768]", mul_48: "f32[1, 1024, 768]", getitem_86: "b8[1, 12, 1024, 1024]", getitem_88: "b8[1, 1024, 768]", mul_50: "f32[1, 1024, 768]", addmm_26: "f32[1024, 3072]", tanh_6: "f32[1, 1024, 3072]", getitem_92: "b8[1, 1024, 768]", mul_56: "f32[1, 1024, 768]", getitem_99: "b8[1, 12, 1024, 1024]", getitem_101: "b8[1, 1024, 768]", mul_58: "f32[1, 1024, 768]", addmm_30: "f32[1024, 3072]", tanh_7: "f32[1, 1024, 3072]", getitem_105: "b8[1, 1024, 768]", mul_64: "f32[1, 1024, 768]", getitem_112: "b8[1, 12, 1024, 1024]", getitem_114: "b8[1, 1024, 768]", mul_66: "f32[1, 1024, 768]", addmm_34: "f32[1024, 3072]", tanh_8: "f32[1, 1024, 3072]", getitem_118: "b8[1, 1024, 768]", mul_72: "f32[1, 1024, 768]", getitem_125: "b8[1, 12, 1024, 1024]", getitem_127: "b8[1, 1024, 768]", mul_74: "f32[1, 1024, 768]", addmm_38: "f32[1024, 3072]", tanh_9: "f32[1, 1024, 3072]", getitem_131: "b8[1, 1024, 768]", mul_80: "f32[1, 1024, 768]", getitem_138: "b8[1, 12, 1024, 1024]", getitem_140: "b8[1, 1024, 768]", mul_82: "f32[1, 1024, 768]", addmm_42: "f32[1024, 3072]", tanh_10: "f32[1, 1024, 3072]", getitem_144: "b8[1, 1024, 768]", mul_88: "f32[1, 1024, 768]", getitem_151: "b8[1, 12, 1024, 1024]", getitem_153: "b8[1, 1024, 768]", mul_90: "f32[1, 1024, 768]", addmm_46: "f32[1024, 3072]", tanh_11: "f32[1, 1024, 3072]", getitem_157: "b8[1, 1024, 768]", mul_96: "f32[1, 1024, 768]", view_219: "f32[1024, 768]", sub_37: "i64[1]", full_default_24: "i64[1]", permute_63: "f32[2, 768]", div_24: "f32[1, 1024, 1]", permute_65: "f32[768, 3072]", permute_66: "f32[3072, 1024]", permute_67: "f32[3072, 768]", permute_68: "f32[768, 1024]", div_25: "f32[1, 1024, 1]", permute_69: "f32[768, 768]", permute_70: "f32[768, 1024]", permute_72: "f32[12, 1024, 1024]", permute_73: "f32[12, 64, 1024]", alias_25: "f32[1, 12, 1024, 1024]", permute_74: "f32[12, 64, 1024]", permute_75: "f32[12, 1024, 64]", permute_80: "f32[2304, 768]", permute_81: "f32[768, 1024]", div_27: "f32[1, 1024, 1]", permute_82: "f32[768, 3072]", permute_83: "f32[3072, 1024]", permute_84: "f32[3072, 768]", permute_85: "f32[768, 1024]", div_28: "f32[1, 1024, 1]", permute_86: "f32[768, 768]", permute_87: "f32[768, 1024]", permute_89: "f32[12, 1024, 1024]", permute_90: "f32[12, 64, 1024]", alias_27: "f32[1, 12, 1024, 1024]", permute_91: "f32[12, 64, 1024]", permute_92: "f32[12, 1024, 64]", permute_97: "f32[2304, 768]", permute_98: "f32[768, 1024]", div_30: "f32[1, 1024, 1]", permute_99: "f32[768, 3072]", permute_100: "f32[3072, 1024]", permute_101: "f32[3072, 768]", permute_102: "f32[768, 1024]", div_31: "f32[1, 1024, 1]", permute_103: "f32[768, 768]", permute_104: "f32[768, 1024]", permute_106: "f32[12, 1024, 1024]", permute_107: "f32[12, 64, 1024]", alias_29: "f32[1, 12, 1024, 1024]", permute_108: "f32[12, 64, 1024]", permute_109: "f32[12, 1024, 64]", permute_114: "f32[2304, 768]", permute_115: "f32[768, 1024]", div_33: "f32[1, 1024, 1]", permute_116: "f32[768, 3072]", permute_117: "f32[3072, 1024]", permute_118: "f32[3072, 768]", permute_119: "f32[768, 1024]", div_34: "f32[1, 1024, 1]", permute_120: "f32[768, 768]", permute_121: "f32[768, 1024]", permute_123: "f32[12, 1024, 1024]", permute_124: "f32[12, 64, 1024]", alias_31: "f32[1, 12, 1024, 1024]", permute_125: "f32[12, 64, 1024]", permute_126: "f32[12, 1024, 64]", permute_131: "f32[2304, 768]", permute_132: "f32[768, 1024]", div_36: "f32[1, 1024, 1]", permute_133: "f32[768, 3072]", permute_134: "f32[3072, 1024]", permute_135: "f32[3072, 768]", permute_136: "f32[768, 1024]", div_37: "f32[1, 1024, 1]", permute_137: "f32[768, 768]", permute_138: "f32[768, 1024]", permute_140: "f32[12, 1024, 1024]", permute_141: "f32[12, 64, 1024]", alias_33: "f32[1, 12, 1024, 1024]", permute_142: "f32[12, 64, 1024]", permute_143: "f32[12, 1024, 64]", permute_148: "f32[2304, 768]", permute_149: "f32[768, 1024]", div_39: "f32[1, 1024, 1]", permute_150: "f32[768, 3072]", permute_151: "f32[3072, 1024]", permute_152: "f32[3072, 768]", permute_153: "f32[768, 1024]", div_40: "f32[1, 1024, 1]", permute_154: "f32[768, 768]", permute_155: "f32[768, 1024]", permute_157: "f32[12, 1024, 1024]", permute_158: "f32[12, 64, 1024]", alias_35: "f32[1, 12, 1024, 1024]", permute_159: "f32[12, 64, 1024]", permute_160: "f32[12, 1024, 64]", permute_165: "f32[2304, 768]", permute_166: "f32[768, 1024]", div_42: "f32[1, 1024, 1]", permute_167: "f32[768, 3072]", permute_168: "f32[3072, 1024]", permute_169: "f32[3072, 768]", permute_170: "f32[768, 1024]", div_43: "f32[1, 1024, 1]", permute_171: "f32[768, 768]", permute_172: "f32[768, 1024]", permute_174: "f32[12, 1024, 1024]", permute_175: "f32[12, 64, 1024]", alias_37: "f32[1, 12, 1024, 1024]", permute_176: "f32[12, 64, 1024]", permute_177: "f32[12, 1024, 64]", permute_182: "f32[2304, 768]", permute_183: "f32[768, 1024]", div_45: "f32[1, 1024, 1]", permute_184: "f32[768, 3072]", permute_185: "f32[3072, 1024]", permute_186: "f32[3072, 768]", permute_187: "f32[768, 1024]", div_46: "f32[1, 1024, 1]", permute_188: "f32[768, 768]", permute_189: "f32[768, 1024]", permute_191: "f32[12, 1024, 1024]", permute_192: "f32[12, 64, 1024]", alias_39: "f32[1, 12, 1024, 1024]", permute_193: "f32[12, 64, 1024]", permute_194: "f32[12, 1024, 64]", permute_199: "f32[2304, 768]", permute_200: "f32[768, 1024]", div_48: "f32[1, 1024, 1]", permute_201: "f32[768, 3072]", permute_202: "f32[3072, 1024]", permute_203: "f32[3072, 768]", permute_204: "f32[768, 1024]", div_49: "f32[1, 1024, 1]", permute_205: "f32[768, 768]", permute_206: "f32[768, 1024]", permute_208: "f32[12, 1024, 1024]", permute_209: "f32[12, 64, 1024]", alias_41: "f32[1, 12, 1024, 1024]", permute_210: "f32[12, 64, 1024]", permute_211: "f32[12, 1024, 64]", permute_216: "f32[2304, 768]", permute_217: "f32[768, 1024]", div_51: "f32[1, 1024, 1]", permute_218: "f32[768, 3072]", permute_219: "f32[3072, 1024]", permute_220: "f32[3072, 768]", permute_221: "f32[768, 1024]", div_52: "f32[1, 1024, 1]", permute_222: "f32[768, 768]", permute_223: "f32[768, 1024]", permute_225: "f32[12, 1024, 1024]", permute_226: "f32[12, 64, 1024]", alias_43: "f32[1, 12, 1024, 1024]", permute_227: "f32[12, 64, 1024]", permute_228: "f32[12, 1024, 64]", permute_233: "f32[2304, 768]", permute_234: "f32[768, 1024]", div_54: "f32[1, 1024, 1]", permute_235: "f32[768, 3072]", permute_236: "f32[3072, 1024]", permute_237: "f32[3072, 768]", permute_238: "f32[768, 1024]", div_55: "f32[1, 1024, 1]", permute_239: "f32[768, 768]", permute_240: "f32[768, 1024]", permute_242: "f32[12, 1024, 1024]", permute_243: "f32[12, 64, 1024]", alias_45: "f32[1, 12, 1024, 1024]", permute_244: "f32[12, 64, 1024]", permute_245: "f32[12, 1024, 64]", permute_250: "f32[2304, 768]", permute_251: "f32[768, 1024]", div_57: "f32[1, 1024, 1]", permute_252: "f32[768, 3072]", permute_253: "f32[3072, 1024]", permute_254: "f32[3072, 768]", permute_255: "f32[768, 1024]", div_58: "f32[1, 1024, 1]", permute_256: "f32[768, 768]", permute_257: "f32[768, 1024]", permute_259: "f32[12, 1024, 1024]", permute_260: "f32[12, 64, 1024]", alias_47: "f32[1, 12, 1024, 1024]", permute_261: "f32[12, 64, 1024]", permute_262: "f32[12, 1024, 64]", permute_267: "f32[2304, 768]", permute_268: "f32[768, 1024]", div_60: "f32[1, 1024, 1]", tangents_1: "f32[1, 1024, 768]", tangents_2: "f32[1, 12, 1024, 64]", tangents_3: "f32[1, 12, 1024, 64]", tangents_4: "f32[1, 12, 1024, 64]", tangents_5: "f32[1, 12, 1024, 64]", tangents_6: "f32[1, 12, 1024, 64]", tangents_7: "f32[1, 12, 1024, 64]", tangents_8: "f32[1, 12, 1024, 64]", tangents_9: "f32[1, 12, 1024, 64]", tangents_10: "f32[1, 12, 1024, 64]", tangents_11: "f32[1, 12, 1024, 64]", tangents_12: "f32[1, 12, 1024, 64]", tangents_13: "f32[1, 12, 1024, 64]", tangents_14: "f32[1, 12, 1024, 64]", tangents_15: "f32[1, 12, 1024, 64]", tangents_16: "f32[1, 12, 1024, 64]", tangents_17: "f32[1, 12, 1024, 64]", tangents_18: "f32[1, 12, 1024, 64]", tangents_19: "f32[1, 12, 1024, 64]", tangents_20: "f32[1, 12, 1024, 64]", tangents_21: "f32[1, 12, 1024, 64]", tangents_22: "f32[1, 12, 1024, 64]", tangents_23: "f32[1, 12, 1024, 64]", tangents_24: "f32[1, 12, 1024, 64]", tangents_25: "f32[1, 12, 1024, 64]", tangents_26: "f32[1, 2]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_default: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_17: "f32[1, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_2, [1, 1024, 3072]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_4: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_17, 0.5)
    add_7: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh, 1.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_35: "f32[1, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_6, [1, 1024, 3072]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_12: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_35, 0.5)
    add_15: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_1, 1.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_53: "f32[1, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_10, [1, 1024, 3072]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_20: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_53, 0.5)
    add_23: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_2, 1.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_71: "f32[1, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_14, [1, 1024, 3072]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_28: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_71, 0.5)
    add_31: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_3, 1.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_89: "f32[1, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_18, [1, 1024, 3072]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_36: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_89, 0.5)
    add_39: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_4, 1.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_107: "f32[1, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_22, [1, 1024, 3072]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_44: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_107, 0.5)
    add_47: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_5, 1.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_125: "f32[1, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_26, [1, 1024, 3072]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_52: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_125, 0.5)
    add_55: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_6, 1.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_143: "f32[1, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_30, [1, 1024, 3072]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_60: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_143, 0.5)
    add_63: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_7, 1.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_161: "f32[1, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_34, [1, 1024, 3072]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_68: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_161, 0.5)
    add_71: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_8, 1.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_179: "f32[1, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_38, [1, 1024, 3072]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_76: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_179, 0.5)
    add_79: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_9, 1.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_197: "f32[1, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_42, [1, 1024, 3072]);  addmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_84: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_197, 0.5)
    add_87: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_10, 1.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_215: "f32[1, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_46, [1, 1024, 3072]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_92: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_215, 0.5)
    add_95: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_11, 1.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1457, code: pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
    full_default_25: "f32[1, 1024, 2]" = torch.ops.aten.full.default([1, 1024, 2], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    index_put: "f32[1, 1024, 2]" = torch.ops.aten.index_put_.default(full_default_25, [full_default_24, sub_37], tangents_26, True);  full_default_25 = full_default_24 = sub_37 = tangents_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1433, code: logits = self.score(hidden_states)
    view_221: "f32[1024, 2]" = torch.ops.aten.reshape.default(index_put, [1024, 2]);  index_put = None
    permute_61: "f32[2, 1024]" = torch.ops.aten.permute.default(view_221, [1, 0])
    mm_1: "f32[2, 768]" = torch.ops.aten.mm.default(permute_61, view_219);  permute_61 = view_219 = None
    permute_62: "f32[768, 2]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    mm_2: "f32[1024, 768]" = torch.ops.aten.mm.default(view_221, permute_63);  view_221 = permute_63 = None
    view_222: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(mm_2, [1, 1024, 768]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1433, code: logits = self.score(hidden_states)
    add_99: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(tangents_1, view_222);  tangents_1 = view_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1433, code: logits = self.score(hidden_states)
    permute_64: "f32[2, 768]" = torch.ops.aten.permute.default(permute_62, [1, 0]);  permute_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:926, code: hidden_states = self.ln_f(hidden_states)
    mul_99: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_99, primals_147);  primals_147 = None
    mul_100: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_99, 768)
    sum_13: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_99, [2], True)
    mul_101: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_99, mul_96);  mul_99 = None
    sum_14: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_101, [2], True);  mul_101 = None
    mul_102: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_96, sum_14);  sum_14 = None
    sub_39: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_100, sum_13);  mul_100 = sum_13 = None
    sub_40: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_39, mul_102);  sub_39 = mul_102 = None
    mul_103: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_24, sub_40);  div_24 = sub_40 = None
    mul_104: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_99, mul_96);  mul_96 = None
    sum_15: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_104, [0, 1]);  mul_104 = None
    sum_16: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_99, [0, 1]);  add_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_13: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_157, torch.float32);  getitem_157 = None
    mul_105: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_13, 1.1111111111111112);  convert_element_type_13 = None
    mul_106: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_103, mul_105);  mul_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_224: "f32[1024, 768]" = torch.ops.aten.reshape.default(mul_106, [1024, 768]);  mul_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_3: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_224, permute_65);  permute_65 = None
    mm_4: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_66, view_224);  permute_66 = None
    sum_17: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_224, [0], True);  view_224 = None
    view_225: "f32[768]" = torch.ops.aten.reshape.default(sum_17, [768]);  sum_17 = None
    view_226: "f32[1, 1024, 3072]" = torch.ops.aten.reshape.default(mm_3, [1, 1024, 3072]);  mm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_107: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_226, mul_92);  mul_92 = None
    mul_108: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_226, add_95);  view_226 = add_95 = None
    mul_109: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(tanh_11, tanh_11);  tanh_11 = None
    sub_41: "f32[1, 1024, 3072]" = torch.ops.aten.sub.Tensor(1, mul_109);  mul_109 = None
    mul_110: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_107, sub_41);  mul_107 = sub_41 = None
    mul_111: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_110, 0.7978845608028654);  mul_110 = None
    mul_112: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_111, 0.044715)
    pow_13: "f32[1, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_215, 2.0);  view_215 = None
    mul_113: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Scalar(pow_13, 3.0);  pow_13 = None
    mul_114: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_112, mul_113);  mul_112 = mul_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_100: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_111, mul_114);  mul_111 = mul_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_115: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_108, 0.5);  mul_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_101: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(add_100, mul_115);  add_100 = mul_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_227: "f32[1024, 3072]" = torch.ops.aten.reshape.default(add_101, [1024, 3072]);  add_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_5: "f32[1024, 768]" = torch.ops.aten.mm.default(view_227, permute_67);  permute_67 = None
    mm_6: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_68, view_227);  permute_68 = None
    sum_18: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_227, [0], True);  view_227 = None
    view_228: "f32[3072]" = torch.ops.aten.reshape.default(sum_18, [3072]);  sum_18 = None
    view_229: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(mm_5, [1, 1024, 768]);  mm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    mul_117: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_229, primals_145);  primals_145 = None
    mul_118: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_117, 768)
    sum_19: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_117, [2], True)
    mul_119: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_117, mul_90);  mul_117 = None
    sum_20: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_119, [2], True);  mul_119 = None
    mul_120: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_90, sum_20);  sum_20 = None
    sub_43: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_118, sum_19);  mul_118 = sum_19 = None
    sub_44: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_43, mul_120);  sub_43 = mul_120 = None
    mul_121: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_25, sub_44);  div_25 = sub_44 = None
    mul_122: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_229, mul_90);  mul_90 = None
    sum_21: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_122, [0, 1]);  mul_122 = None
    sum_22: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_229, [0, 1]);  view_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    add_102: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_103, mul_121);  mul_103 = mul_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    convert_element_type_14: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_153, torch.float32);  getitem_153 = None
    mul_123: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_14, 1.1111111111111112);  convert_element_type_14 = None
    mul_124: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_102, mul_123);  mul_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_230: "f32[1024, 768]" = torch.ops.aten.reshape.default(mul_124, [1024, 768]);  mul_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_7: "f32[1024, 768]" = torch.ops.aten.mm.default(view_230, permute_69);  permute_69 = None
    mm_8: "f32[768, 768]" = torch.ops.aten.mm.default(permute_70, view_230);  permute_70 = None
    sum_23: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_230, [0], True);  view_230 = None
    view_231: "f32[768]" = torch.ops.aten.reshape.default(sum_23, [768]);  sum_23 = None
    view_232: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(mm_7, [1, 1024, 768]);  mm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_233: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_232, [1, 1024, 12, 64]);  view_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_71: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_233, [0, 2, 1, 3]);  view_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    view_234: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(permute_71, [12, 1024, 64]);  permute_71 = None
    bmm_24: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(permute_72, view_234);  permute_72 = None
    bmm_25: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_234, permute_73);  view_234 = permute_73 = None
    view_235: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_24, [1, 12, 1024, 64]);  bmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    add_103: "f32[1, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_25, view_235);  tangents_25 = view_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    view_236: "f32[1, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_25, [1, 12, 1024, 1024]);  bmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    convert_element_type_15: "f32[1, 12, 1024, 1024]" = torch.ops.prims.convert_element_type.default(getitem_151, torch.float32);  getitem_151 = None
    mul_125: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_15, 1.1111111111111112);  convert_element_type_15 = None
    mul_126: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_236, mul_125);  view_236 = mul_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_127: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_126, alias_25);  mul_126 = None
    sum_24: "f32[1, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_127, [-1], True)
    mul_128: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_25, sum_24);  alias_25 = sum_24 = None
    sub_45: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_127, mul_128);  mul_127 = mul_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    full_default_26: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where_12: "f32[1, 12, 1024, 1024]" = torch.ops.aten.where.self(primals_161, sub_45, full_default_26);  primals_161 = sub_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    div_26: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(where_12, full_default);  where_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_237: "f32[12, 1024, 1024]" = torch.ops.aten.reshape.default(div_26, [12, 1024, 1024]);  div_26 = None
    bmm_26: "f32[12, 64, 1024]" = torch.ops.aten.bmm.default(permute_74, view_237);  permute_74 = None
    bmm_27: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(view_237, permute_75);  view_237 = permute_75 = None
    view_238: "f32[1, 12, 64, 1024]" = torch.ops.aten.reshape.default(bmm_26, [1, 12, 64, 1024]);  bmm_26 = None
    view_239: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_27, [1, 12, 1024, 64]);  bmm_27 = None
    permute_76: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_238, [0, 1, 3, 2]);  view_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_104: "f32[1, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_24, permute_76);  tangents_24 = permute_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_77: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(add_103, [0, 2, 1, 3]);  add_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_15: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_77, memory_format = torch.contiguous_format);  permute_77 = None
    view_240: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_15, [1, 1024, 768]);  clone_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_78: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(add_104, [0, 2, 1, 3]);  add_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_16: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_78, memory_format = torch.contiguous_format);  permute_78 = None
    view_241: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_16, [1, 1024, 768]);  clone_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_79: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_239, [0, 2, 1, 3]);  view_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_17: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_79, memory_format = torch.contiguous_format);  permute_79 = None
    view_242: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_17, [1, 1024, 768]);  clone_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    cat: "f32[1, 1024, 2304]" = torch.ops.aten.cat.default([view_242, view_241, view_240], 2);  view_242 = view_241 = view_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_243: "f32[1024, 2304]" = torch.ops.aten.reshape.default(cat, [1024, 2304]);  cat = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_9: "f32[1024, 768]" = torch.ops.aten.mm.default(view_243, permute_80);  permute_80 = None
    mm_10: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_81, view_243);  permute_81 = None
    sum_25: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_243, [0], True);  view_243 = None
    view_244: "f32[2304]" = torch.ops.aten.reshape.default(sum_25, [2304]);  sum_25 = None
    view_245: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(mm_9, [1, 1024, 768]);  mm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    mul_130: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_245, primals_143);  primals_143 = None
    mul_131: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_130, 768)
    sum_26: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_130, [2], True)
    mul_132: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_130, mul_88);  mul_130 = None
    sum_27: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_132, [2], True);  mul_132 = None
    mul_133: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_88, sum_27);  sum_27 = None
    sub_47: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_131, sum_26);  mul_131 = sum_26 = None
    sub_48: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_47, mul_133);  sub_47 = mul_133 = None
    mul_134: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_27, sub_48);  div_27 = sub_48 = None
    mul_135: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_245, mul_88);  mul_88 = None
    sum_28: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_135, [0, 1]);  mul_135 = None
    sum_29: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_245, [0, 1]);  view_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    add_105: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_102, mul_134);  add_102 = mul_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_16: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_144, torch.float32);  getitem_144 = None
    mul_136: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_16, 1.1111111111111112);  convert_element_type_16 = None
    mul_137: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_105, mul_136);  mul_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_246: "f32[1024, 768]" = torch.ops.aten.reshape.default(mul_137, [1024, 768]);  mul_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_11: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_246, permute_82);  permute_82 = None
    mm_12: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_83, view_246);  permute_83 = None
    sum_30: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_246, [0], True);  view_246 = None
    view_247: "f32[768]" = torch.ops.aten.reshape.default(sum_30, [768]);  sum_30 = None
    view_248: "f32[1, 1024, 3072]" = torch.ops.aten.reshape.default(mm_11, [1, 1024, 3072]);  mm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_138: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_248, mul_84);  mul_84 = None
    mul_139: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_248, add_87);  view_248 = add_87 = None
    mul_140: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(tanh_10, tanh_10);  tanh_10 = None
    sub_49: "f32[1, 1024, 3072]" = torch.ops.aten.sub.Tensor(1, mul_140);  mul_140 = None
    mul_141: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_138, sub_49);  mul_138 = sub_49 = None
    mul_142: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_141, 0.7978845608028654);  mul_141 = None
    mul_143: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_142, 0.044715)
    pow_14: "f32[1, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_197, 2.0);  view_197 = None
    mul_144: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Scalar(pow_14, 3.0);  pow_14 = None
    mul_145: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_143, mul_144);  mul_143 = mul_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_106: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_142, mul_145);  mul_142 = mul_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_146: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_139, 0.5);  mul_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_107: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(add_106, mul_146);  add_106 = mul_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_249: "f32[1024, 3072]" = torch.ops.aten.reshape.default(add_107, [1024, 3072]);  add_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_13: "f32[1024, 768]" = torch.ops.aten.mm.default(view_249, permute_84);  permute_84 = None
    mm_14: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_85, view_249);  permute_85 = None
    sum_31: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_249, [0], True);  view_249 = None
    view_250: "f32[3072]" = torch.ops.aten.reshape.default(sum_31, [3072]);  sum_31 = None
    view_251: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(mm_13, [1, 1024, 768]);  mm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    mul_148: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_251, primals_141);  primals_141 = None
    mul_149: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_148, 768)
    sum_32: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_148, [2], True)
    mul_150: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_148, mul_82);  mul_148 = None
    sum_33: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_150, [2], True);  mul_150 = None
    mul_151: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_82, sum_33);  sum_33 = None
    sub_51: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_149, sum_32);  mul_149 = sum_32 = None
    sub_52: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_51, mul_151);  sub_51 = mul_151 = None
    mul_152: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_28, sub_52);  div_28 = sub_52 = None
    mul_153: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_251, mul_82);  mul_82 = None
    sum_34: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_153, [0, 1]);  mul_153 = None
    sum_35: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_251, [0, 1]);  view_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    add_108: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_105, mul_152);  add_105 = mul_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    convert_element_type_17: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_140, torch.float32);  getitem_140 = None
    mul_154: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_17, 1.1111111111111112);  convert_element_type_17 = None
    mul_155: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_108, mul_154);  mul_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_252: "f32[1024, 768]" = torch.ops.aten.reshape.default(mul_155, [1024, 768]);  mul_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_15: "f32[1024, 768]" = torch.ops.aten.mm.default(view_252, permute_86);  permute_86 = None
    mm_16: "f32[768, 768]" = torch.ops.aten.mm.default(permute_87, view_252);  permute_87 = None
    sum_36: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_252, [0], True);  view_252 = None
    view_253: "f32[768]" = torch.ops.aten.reshape.default(sum_36, [768]);  sum_36 = None
    view_254: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(mm_15, [1, 1024, 768]);  mm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_255: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_254, [1, 1024, 12, 64]);  view_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_88: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_255, [0, 2, 1, 3]);  view_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    view_256: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(permute_88, [12, 1024, 64]);  permute_88 = None
    bmm_28: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(permute_89, view_256);  permute_89 = None
    bmm_29: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_256, permute_90);  view_256 = permute_90 = None
    view_257: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_28, [1, 12, 1024, 64]);  bmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    add_109: "f32[1, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_23, view_257);  tangents_23 = view_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    view_258: "f32[1, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_29, [1, 12, 1024, 1024]);  bmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    convert_element_type_18: "f32[1, 12, 1024, 1024]" = torch.ops.prims.convert_element_type.default(getitem_138, torch.float32);  getitem_138 = None
    mul_156: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_18, 1.1111111111111112);  convert_element_type_18 = None
    mul_157: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_258, mul_156);  view_258 = mul_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_158: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_157, alias_27);  mul_157 = None
    sum_37: "f32[1, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_158, [-1], True)
    mul_159: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_27, sum_37);  alias_27 = sum_37 = None
    sub_53: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_158, mul_159);  mul_158 = mul_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_13: "f32[1, 12, 1024, 1024]" = torch.ops.aten.where.self(primals_160, sub_53, full_default_26);  primals_160 = sub_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    div_29: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(where_13, full_default);  where_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_259: "f32[12, 1024, 1024]" = torch.ops.aten.reshape.default(div_29, [12, 1024, 1024]);  div_29 = None
    bmm_30: "f32[12, 64, 1024]" = torch.ops.aten.bmm.default(permute_91, view_259);  permute_91 = None
    bmm_31: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(view_259, permute_92);  view_259 = permute_92 = None
    view_260: "f32[1, 12, 64, 1024]" = torch.ops.aten.reshape.default(bmm_30, [1, 12, 64, 1024]);  bmm_30 = None
    view_261: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_31, [1, 12, 1024, 64]);  bmm_31 = None
    permute_93: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_260, [0, 1, 3, 2]);  view_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_110: "f32[1, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_22, permute_93);  tangents_22 = permute_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_94: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(add_109, [0, 2, 1, 3]);  add_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_21: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_94, memory_format = torch.contiguous_format);  permute_94 = None
    view_262: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_21, [1, 1024, 768]);  clone_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_95: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(add_110, [0, 2, 1, 3]);  add_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_22: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
    view_263: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_22, [1, 1024, 768]);  clone_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_96: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_261, [0, 2, 1, 3]);  view_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_23: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_96, memory_format = torch.contiguous_format);  permute_96 = None
    view_264: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_23, [1, 1024, 768]);  clone_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    cat_1: "f32[1, 1024, 2304]" = torch.ops.aten.cat.default([view_264, view_263, view_262], 2);  view_264 = view_263 = view_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_265: "f32[1024, 2304]" = torch.ops.aten.reshape.default(cat_1, [1024, 2304]);  cat_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_17: "f32[1024, 768]" = torch.ops.aten.mm.default(view_265, permute_97);  permute_97 = None
    mm_18: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_98, view_265);  permute_98 = None
    sum_38: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_265, [0], True);  view_265 = None
    view_266: "f32[2304]" = torch.ops.aten.reshape.default(sum_38, [2304]);  sum_38 = None
    view_267: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(mm_17, [1, 1024, 768]);  mm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    mul_161: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_267, primals_139);  primals_139 = None
    mul_162: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_161, 768)
    sum_39: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_161, [2], True)
    mul_163: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_161, mul_80);  mul_161 = None
    sum_40: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_163, [2], True);  mul_163 = None
    mul_164: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_80, sum_40);  sum_40 = None
    sub_55: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_162, sum_39);  mul_162 = sum_39 = None
    sub_56: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_55, mul_164);  sub_55 = mul_164 = None
    mul_165: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_30, sub_56);  div_30 = sub_56 = None
    mul_166: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_267, mul_80);  mul_80 = None
    sum_41: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_166, [0, 1]);  mul_166 = None
    sum_42: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_267, [0, 1]);  view_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    add_111: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_108, mul_165);  add_108 = mul_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_19: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_131, torch.float32);  getitem_131 = None
    mul_167: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_19, 1.1111111111111112);  convert_element_type_19 = None
    mul_168: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_111, mul_167);  mul_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_268: "f32[1024, 768]" = torch.ops.aten.reshape.default(mul_168, [1024, 768]);  mul_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_19: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_268, permute_99);  permute_99 = None
    mm_20: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_100, view_268);  permute_100 = None
    sum_43: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_268, [0], True);  view_268 = None
    view_269: "f32[768]" = torch.ops.aten.reshape.default(sum_43, [768]);  sum_43 = None
    view_270: "f32[1, 1024, 3072]" = torch.ops.aten.reshape.default(mm_19, [1, 1024, 3072]);  mm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_169: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_270, mul_76);  mul_76 = None
    mul_170: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_270, add_79);  view_270 = add_79 = None
    mul_171: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(tanh_9, tanh_9);  tanh_9 = None
    sub_57: "f32[1, 1024, 3072]" = torch.ops.aten.sub.Tensor(1, mul_171);  mul_171 = None
    mul_172: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_169, sub_57);  mul_169 = sub_57 = None
    mul_173: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_172, 0.7978845608028654);  mul_172 = None
    mul_174: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_173, 0.044715)
    pow_15: "f32[1, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_179, 2.0);  view_179 = None
    mul_175: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Scalar(pow_15, 3.0);  pow_15 = None
    mul_176: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_174, mul_175);  mul_174 = mul_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_112: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_173, mul_176);  mul_173 = mul_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_177: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_170, 0.5);  mul_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_113: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(add_112, mul_177);  add_112 = mul_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_271: "f32[1024, 3072]" = torch.ops.aten.reshape.default(add_113, [1024, 3072]);  add_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_21: "f32[1024, 768]" = torch.ops.aten.mm.default(view_271, permute_101);  permute_101 = None
    mm_22: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_102, view_271);  permute_102 = None
    sum_44: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_271, [0], True);  view_271 = None
    view_272: "f32[3072]" = torch.ops.aten.reshape.default(sum_44, [3072]);  sum_44 = None
    view_273: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(mm_21, [1, 1024, 768]);  mm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    mul_179: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_273, primals_137);  primals_137 = None
    mul_180: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_179, 768)
    sum_45: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_179, [2], True)
    mul_181: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_179, mul_74);  mul_179 = None
    sum_46: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_181, [2], True);  mul_181 = None
    mul_182: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_74, sum_46);  sum_46 = None
    sub_59: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_180, sum_45);  mul_180 = sum_45 = None
    sub_60: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_59, mul_182);  sub_59 = mul_182 = None
    mul_183: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_31, sub_60);  div_31 = sub_60 = None
    mul_184: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_273, mul_74);  mul_74 = None
    sum_47: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_184, [0, 1]);  mul_184 = None
    sum_48: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_273, [0, 1]);  view_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    add_114: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_111, mul_183);  add_111 = mul_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    convert_element_type_20: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_127, torch.float32);  getitem_127 = None
    mul_185: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_20, 1.1111111111111112);  convert_element_type_20 = None
    mul_186: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_114, mul_185);  mul_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_274: "f32[1024, 768]" = torch.ops.aten.reshape.default(mul_186, [1024, 768]);  mul_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_23: "f32[1024, 768]" = torch.ops.aten.mm.default(view_274, permute_103);  permute_103 = None
    mm_24: "f32[768, 768]" = torch.ops.aten.mm.default(permute_104, view_274);  permute_104 = None
    sum_49: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_274, [0], True);  view_274 = None
    view_275: "f32[768]" = torch.ops.aten.reshape.default(sum_49, [768]);  sum_49 = None
    view_276: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(mm_23, [1, 1024, 768]);  mm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_277: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_276, [1, 1024, 12, 64]);  view_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_105: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_277, [0, 2, 1, 3]);  view_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    view_278: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(permute_105, [12, 1024, 64]);  permute_105 = None
    bmm_32: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(permute_106, view_278);  permute_106 = None
    bmm_33: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_278, permute_107);  view_278 = permute_107 = None
    view_279: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_32, [1, 12, 1024, 64]);  bmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    add_115: "f32[1, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_21, view_279);  tangents_21 = view_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    view_280: "f32[1, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_33, [1, 12, 1024, 1024]);  bmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    convert_element_type_21: "f32[1, 12, 1024, 1024]" = torch.ops.prims.convert_element_type.default(getitem_125, torch.float32);  getitem_125 = None
    mul_187: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_21, 1.1111111111111112);  convert_element_type_21 = None
    mul_188: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_280, mul_187);  view_280 = mul_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_189: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_188, alias_29);  mul_188 = None
    sum_50: "f32[1, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_189, [-1], True)
    mul_190: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_29, sum_50);  alias_29 = sum_50 = None
    sub_61: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_189, mul_190);  mul_189 = mul_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_14: "f32[1, 12, 1024, 1024]" = torch.ops.aten.where.self(primals_159, sub_61, full_default_26);  primals_159 = sub_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    div_32: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(where_14, full_default);  where_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_281: "f32[12, 1024, 1024]" = torch.ops.aten.reshape.default(div_32, [12, 1024, 1024]);  div_32 = None
    bmm_34: "f32[12, 64, 1024]" = torch.ops.aten.bmm.default(permute_108, view_281);  permute_108 = None
    bmm_35: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(view_281, permute_109);  view_281 = permute_109 = None
    view_282: "f32[1, 12, 64, 1024]" = torch.ops.aten.reshape.default(bmm_34, [1, 12, 64, 1024]);  bmm_34 = None
    view_283: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_35, [1, 12, 1024, 64]);  bmm_35 = None
    permute_110: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_282, [0, 1, 3, 2]);  view_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_116: "f32[1, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_20, permute_110);  tangents_20 = permute_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_111: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(add_115, [0, 2, 1, 3]);  add_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_27: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_111, memory_format = torch.contiguous_format);  permute_111 = None
    view_284: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_27, [1, 1024, 768]);  clone_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_112: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(add_116, [0, 2, 1, 3]);  add_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_28: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_112, memory_format = torch.contiguous_format);  permute_112 = None
    view_285: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_28, [1, 1024, 768]);  clone_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_113: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_283, [0, 2, 1, 3]);  view_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_29: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_113, memory_format = torch.contiguous_format);  permute_113 = None
    view_286: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_29, [1, 1024, 768]);  clone_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    cat_2: "f32[1, 1024, 2304]" = torch.ops.aten.cat.default([view_286, view_285, view_284], 2);  view_286 = view_285 = view_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_287: "f32[1024, 2304]" = torch.ops.aten.reshape.default(cat_2, [1024, 2304]);  cat_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_25: "f32[1024, 768]" = torch.ops.aten.mm.default(view_287, permute_114);  permute_114 = None
    mm_26: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_115, view_287);  permute_115 = None
    sum_51: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_287, [0], True);  view_287 = None
    view_288: "f32[2304]" = torch.ops.aten.reshape.default(sum_51, [2304]);  sum_51 = None
    view_289: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(mm_25, [1, 1024, 768]);  mm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    mul_192: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_289, primals_135);  primals_135 = None
    mul_193: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_192, 768)
    sum_52: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_192, [2], True)
    mul_194: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_192, mul_72);  mul_192 = None
    sum_53: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_194, [2], True);  mul_194 = None
    mul_195: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_72, sum_53);  sum_53 = None
    sub_63: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_193, sum_52);  mul_193 = sum_52 = None
    sub_64: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_63, mul_195);  sub_63 = mul_195 = None
    mul_196: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_33, sub_64);  div_33 = sub_64 = None
    mul_197: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_289, mul_72);  mul_72 = None
    sum_54: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_197, [0, 1]);  mul_197 = None
    sum_55: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_289, [0, 1]);  view_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    add_117: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_114, mul_196);  add_114 = mul_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_22: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_118, torch.float32);  getitem_118 = None
    mul_198: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_22, 1.1111111111111112);  convert_element_type_22 = None
    mul_199: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_117, mul_198);  mul_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_290: "f32[1024, 768]" = torch.ops.aten.reshape.default(mul_199, [1024, 768]);  mul_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_27: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_290, permute_116);  permute_116 = None
    mm_28: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_117, view_290);  permute_117 = None
    sum_56: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_290, [0], True);  view_290 = None
    view_291: "f32[768]" = torch.ops.aten.reshape.default(sum_56, [768]);  sum_56 = None
    view_292: "f32[1, 1024, 3072]" = torch.ops.aten.reshape.default(mm_27, [1, 1024, 3072]);  mm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_200: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_292, mul_68);  mul_68 = None
    mul_201: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_292, add_71);  view_292 = add_71 = None
    mul_202: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(tanh_8, tanh_8);  tanh_8 = None
    sub_65: "f32[1, 1024, 3072]" = torch.ops.aten.sub.Tensor(1, mul_202);  mul_202 = None
    mul_203: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_200, sub_65);  mul_200 = sub_65 = None
    mul_204: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_203, 0.7978845608028654);  mul_203 = None
    mul_205: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_204, 0.044715)
    pow_16: "f32[1, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_161, 2.0);  view_161 = None
    mul_206: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Scalar(pow_16, 3.0);  pow_16 = None
    mul_207: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_205, mul_206);  mul_205 = mul_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_118: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_204, mul_207);  mul_204 = mul_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_208: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_201, 0.5);  mul_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_119: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(add_118, mul_208);  add_118 = mul_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_293: "f32[1024, 3072]" = torch.ops.aten.reshape.default(add_119, [1024, 3072]);  add_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_29: "f32[1024, 768]" = torch.ops.aten.mm.default(view_293, permute_118);  permute_118 = None
    mm_30: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_119, view_293);  permute_119 = None
    sum_57: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_293, [0], True);  view_293 = None
    view_294: "f32[3072]" = torch.ops.aten.reshape.default(sum_57, [3072]);  sum_57 = None
    view_295: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(mm_29, [1, 1024, 768]);  mm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    mul_210: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_295, primals_133);  primals_133 = None
    mul_211: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_210, 768)
    sum_58: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_210, [2], True)
    mul_212: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_210, mul_66);  mul_210 = None
    sum_59: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_212, [2], True);  mul_212 = None
    mul_213: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_66, sum_59);  sum_59 = None
    sub_67: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_211, sum_58);  mul_211 = sum_58 = None
    sub_68: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_67, mul_213);  sub_67 = mul_213 = None
    mul_214: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_34, sub_68);  div_34 = sub_68 = None
    mul_215: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_295, mul_66);  mul_66 = None
    sum_60: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_215, [0, 1]);  mul_215 = None
    sum_61: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_295, [0, 1]);  view_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    add_120: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_117, mul_214);  add_117 = mul_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    convert_element_type_23: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_114, torch.float32);  getitem_114 = None
    mul_216: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_23, 1.1111111111111112);  convert_element_type_23 = None
    mul_217: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_120, mul_216);  mul_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_296: "f32[1024, 768]" = torch.ops.aten.reshape.default(mul_217, [1024, 768]);  mul_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_31: "f32[1024, 768]" = torch.ops.aten.mm.default(view_296, permute_120);  permute_120 = None
    mm_32: "f32[768, 768]" = torch.ops.aten.mm.default(permute_121, view_296);  permute_121 = None
    sum_62: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_296, [0], True);  view_296 = None
    view_297: "f32[768]" = torch.ops.aten.reshape.default(sum_62, [768]);  sum_62 = None
    view_298: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(mm_31, [1, 1024, 768]);  mm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_299: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_298, [1, 1024, 12, 64]);  view_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_122: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_299, [0, 2, 1, 3]);  view_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    view_300: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(permute_122, [12, 1024, 64]);  permute_122 = None
    bmm_36: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(permute_123, view_300);  permute_123 = None
    bmm_37: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_300, permute_124);  view_300 = permute_124 = None
    view_301: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_36, [1, 12, 1024, 64]);  bmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    add_121: "f32[1, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_19, view_301);  tangents_19 = view_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    view_302: "f32[1, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_37, [1, 12, 1024, 1024]);  bmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    convert_element_type_24: "f32[1, 12, 1024, 1024]" = torch.ops.prims.convert_element_type.default(getitem_112, torch.float32);  getitem_112 = None
    mul_218: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_24, 1.1111111111111112);  convert_element_type_24 = None
    mul_219: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_302, mul_218);  view_302 = mul_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_220: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_219, alias_31);  mul_219 = None
    sum_63: "f32[1, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_220, [-1], True)
    mul_221: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_31, sum_63);  alias_31 = sum_63 = None
    sub_69: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_220, mul_221);  mul_220 = mul_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_15: "f32[1, 12, 1024, 1024]" = torch.ops.aten.where.self(primals_158, sub_69, full_default_26);  primals_158 = sub_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    div_35: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(where_15, full_default);  where_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_303: "f32[12, 1024, 1024]" = torch.ops.aten.reshape.default(div_35, [12, 1024, 1024]);  div_35 = None
    bmm_38: "f32[12, 64, 1024]" = torch.ops.aten.bmm.default(permute_125, view_303);  permute_125 = None
    bmm_39: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(view_303, permute_126);  view_303 = permute_126 = None
    view_304: "f32[1, 12, 64, 1024]" = torch.ops.aten.reshape.default(bmm_38, [1, 12, 64, 1024]);  bmm_38 = None
    view_305: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_39, [1, 12, 1024, 64]);  bmm_39 = None
    permute_127: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_304, [0, 1, 3, 2]);  view_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_122: "f32[1, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_18, permute_127);  tangents_18 = permute_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_128: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(add_121, [0, 2, 1, 3]);  add_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_33: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
    view_306: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_33, [1, 1024, 768]);  clone_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_129: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(add_122, [0, 2, 1, 3]);  add_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_34: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_129, memory_format = torch.contiguous_format);  permute_129 = None
    view_307: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_34, [1, 1024, 768]);  clone_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_130: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_305, [0, 2, 1, 3]);  view_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_35: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_130, memory_format = torch.contiguous_format);  permute_130 = None
    view_308: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_35, [1, 1024, 768]);  clone_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    cat_3: "f32[1, 1024, 2304]" = torch.ops.aten.cat.default([view_308, view_307, view_306], 2);  view_308 = view_307 = view_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_309: "f32[1024, 2304]" = torch.ops.aten.reshape.default(cat_3, [1024, 2304]);  cat_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_33: "f32[1024, 768]" = torch.ops.aten.mm.default(view_309, permute_131);  permute_131 = None
    mm_34: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_132, view_309);  permute_132 = None
    sum_64: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_309, [0], True);  view_309 = None
    view_310: "f32[2304]" = torch.ops.aten.reshape.default(sum_64, [2304]);  sum_64 = None
    view_311: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(mm_33, [1, 1024, 768]);  mm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    mul_223: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_311, primals_131);  primals_131 = None
    mul_224: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_223, 768)
    sum_65: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_223, [2], True)
    mul_225: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_223, mul_64);  mul_223 = None
    sum_66: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_225, [2], True);  mul_225 = None
    mul_226: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_64, sum_66);  sum_66 = None
    sub_71: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_224, sum_65);  mul_224 = sum_65 = None
    sub_72: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_71, mul_226);  sub_71 = mul_226 = None
    mul_227: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_36, sub_72);  div_36 = sub_72 = None
    mul_228: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_311, mul_64);  mul_64 = None
    sum_67: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_228, [0, 1]);  mul_228 = None
    sum_68: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_311, [0, 1]);  view_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    add_123: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_120, mul_227);  add_120 = mul_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_25: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_105, torch.float32);  getitem_105 = None
    mul_229: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_25, 1.1111111111111112);  convert_element_type_25 = None
    mul_230: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_123, mul_229);  mul_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_312: "f32[1024, 768]" = torch.ops.aten.reshape.default(mul_230, [1024, 768]);  mul_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_35: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_312, permute_133);  permute_133 = None
    mm_36: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_134, view_312);  permute_134 = None
    sum_69: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_312, [0], True);  view_312 = None
    view_313: "f32[768]" = torch.ops.aten.reshape.default(sum_69, [768]);  sum_69 = None
    view_314: "f32[1, 1024, 3072]" = torch.ops.aten.reshape.default(mm_35, [1, 1024, 3072]);  mm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_231: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_314, mul_60);  mul_60 = None
    mul_232: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_314, add_63);  view_314 = add_63 = None
    mul_233: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(tanh_7, tanh_7);  tanh_7 = None
    sub_73: "f32[1, 1024, 3072]" = torch.ops.aten.sub.Tensor(1, mul_233);  mul_233 = None
    mul_234: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_231, sub_73);  mul_231 = sub_73 = None
    mul_235: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_234, 0.7978845608028654);  mul_234 = None
    mul_236: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_235, 0.044715)
    pow_17: "f32[1, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_143, 2.0);  view_143 = None
    mul_237: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Scalar(pow_17, 3.0);  pow_17 = None
    mul_238: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_236, mul_237);  mul_236 = mul_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_124: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_235, mul_238);  mul_235 = mul_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_239: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_232, 0.5);  mul_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_125: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(add_124, mul_239);  add_124 = mul_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_315: "f32[1024, 3072]" = torch.ops.aten.reshape.default(add_125, [1024, 3072]);  add_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_37: "f32[1024, 768]" = torch.ops.aten.mm.default(view_315, permute_135);  permute_135 = None
    mm_38: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_136, view_315);  permute_136 = None
    sum_70: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_315, [0], True);  view_315 = None
    view_316: "f32[3072]" = torch.ops.aten.reshape.default(sum_70, [3072]);  sum_70 = None
    view_317: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(mm_37, [1, 1024, 768]);  mm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    mul_241: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_317, primals_129);  primals_129 = None
    mul_242: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_241, 768)
    sum_71: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_241, [2], True)
    mul_243: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_241, mul_58);  mul_241 = None
    sum_72: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_243, [2], True);  mul_243 = None
    mul_244: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_58, sum_72);  sum_72 = None
    sub_75: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_242, sum_71);  mul_242 = sum_71 = None
    sub_76: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_75, mul_244);  sub_75 = mul_244 = None
    mul_245: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_37, sub_76);  div_37 = sub_76 = None
    mul_246: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_317, mul_58);  mul_58 = None
    sum_73: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_246, [0, 1]);  mul_246 = None
    sum_74: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_317, [0, 1]);  view_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    add_126: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_123, mul_245);  add_123 = mul_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    convert_element_type_26: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_101, torch.float32);  getitem_101 = None
    mul_247: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_26, 1.1111111111111112);  convert_element_type_26 = None
    mul_248: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_126, mul_247);  mul_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_318: "f32[1024, 768]" = torch.ops.aten.reshape.default(mul_248, [1024, 768]);  mul_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_39: "f32[1024, 768]" = torch.ops.aten.mm.default(view_318, permute_137);  permute_137 = None
    mm_40: "f32[768, 768]" = torch.ops.aten.mm.default(permute_138, view_318);  permute_138 = None
    sum_75: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_318, [0], True);  view_318 = None
    view_319: "f32[768]" = torch.ops.aten.reshape.default(sum_75, [768]);  sum_75 = None
    view_320: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(mm_39, [1, 1024, 768]);  mm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_321: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_320, [1, 1024, 12, 64]);  view_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_139: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_321, [0, 2, 1, 3]);  view_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    view_322: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(permute_139, [12, 1024, 64]);  permute_139 = None
    bmm_40: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(permute_140, view_322);  permute_140 = None
    bmm_41: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_322, permute_141);  view_322 = permute_141 = None
    view_323: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_40, [1, 12, 1024, 64]);  bmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    add_127: "f32[1, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_17, view_323);  tangents_17 = view_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    view_324: "f32[1, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_41, [1, 12, 1024, 1024]);  bmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    convert_element_type_27: "f32[1, 12, 1024, 1024]" = torch.ops.prims.convert_element_type.default(getitem_99, torch.float32);  getitem_99 = None
    mul_249: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_27, 1.1111111111111112);  convert_element_type_27 = None
    mul_250: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_324, mul_249);  view_324 = mul_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_251: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_250, alias_33);  mul_250 = None
    sum_76: "f32[1, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_251, [-1], True)
    mul_252: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_33, sum_76);  alias_33 = sum_76 = None
    sub_77: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_251, mul_252);  mul_251 = mul_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_16: "f32[1, 12, 1024, 1024]" = torch.ops.aten.where.self(primals_157, sub_77, full_default_26);  primals_157 = sub_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    div_38: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(where_16, full_default);  where_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_325: "f32[12, 1024, 1024]" = torch.ops.aten.reshape.default(div_38, [12, 1024, 1024]);  div_38 = None
    bmm_42: "f32[12, 64, 1024]" = torch.ops.aten.bmm.default(permute_142, view_325);  permute_142 = None
    bmm_43: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(view_325, permute_143);  view_325 = permute_143 = None
    view_326: "f32[1, 12, 64, 1024]" = torch.ops.aten.reshape.default(bmm_42, [1, 12, 64, 1024]);  bmm_42 = None
    view_327: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_43, [1, 12, 1024, 64]);  bmm_43 = None
    permute_144: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_326, [0, 1, 3, 2]);  view_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_128: "f32[1, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_16, permute_144);  tangents_16 = permute_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_145: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(add_127, [0, 2, 1, 3]);  add_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_39: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_145, memory_format = torch.contiguous_format);  permute_145 = None
    view_328: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_39, [1, 1024, 768]);  clone_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_146: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(add_128, [0, 2, 1, 3]);  add_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_40: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_146, memory_format = torch.contiguous_format);  permute_146 = None
    view_329: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_40, [1, 1024, 768]);  clone_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_147: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_327, [0, 2, 1, 3]);  view_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_41: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_147, memory_format = torch.contiguous_format);  permute_147 = None
    view_330: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_41, [1, 1024, 768]);  clone_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    cat_4: "f32[1, 1024, 2304]" = torch.ops.aten.cat.default([view_330, view_329, view_328], 2);  view_330 = view_329 = view_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_331: "f32[1024, 2304]" = torch.ops.aten.reshape.default(cat_4, [1024, 2304]);  cat_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_41: "f32[1024, 768]" = torch.ops.aten.mm.default(view_331, permute_148);  permute_148 = None
    mm_42: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_149, view_331);  permute_149 = None
    sum_77: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_331, [0], True);  view_331 = None
    view_332: "f32[2304]" = torch.ops.aten.reshape.default(sum_77, [2304]);  sum_77 = None
    view_333: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(mm_41, [1, 1024, 768]);  mm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    mul_254: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_333, primals_127);  primals_127 = None
    mul_255: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_254, 768)
    sum_78: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_254, [2], True)
    mul_256: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_254, mul_56);  mul_254 = None
    sum_79: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_256, [2], True);  mul_256 = None
    mul_257: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_56, sum_79);  sum_79 = None
    sub_79: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_255, sum_78);  mul_255 = sum_78 = None
    sub_80: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_79, mul_257);  sub_79 = mul_257 = None
    mul_258: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_39, sub_80);  div_39 = sub_80 = None
    mul_259: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_333, mul_56);  mul_56 = None
    sum_80: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_259, [0, 1]);  mul_259 = None
    sum_81: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_333, [0, 1]);  view_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    add_129: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_126, mul_258);  add_126 = mul_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_28: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_92, torch.float32);  getitem_92 = None
    mul_260: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_28, 1.1111111111111112);  convert_element_type_28 = None
    mul_261: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_129, mul_260);  mul_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_334: "f32[1024, 768]" = torch.ops.aten.reshape.default(mul_261, [1024, 768]);  mul_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_43: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_334, permute_150);  permute_150 = None
    mm_44: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_151, view_334);  permute_151 = None
    sum_82: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_334, [0], True);  view_334 = None
    view_335: "f32[768]" = torch.ops.aten.reshape.default(sum_82, [768]);  sum_82 = None
    view_336: "f32[1, 1024, 3072]" = torch.ops.aten.reshape.default(mm_43, [1, 1024, 3072]);  mm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_262: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_336, mul_52);  mul_52 = None
    mul_263: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_336, add_55);  view_336 = add_55 = None
    mul_264: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(tanh_6, tanh_6);  tanh_6 = None
    sub_81: "f32[1, 1024, 3072]" = torch.ops.aten.sub.Tensor(1, mul_264);  mul_264 = None
    mul_265: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_262, sub_81);  mul_262 = sub_81 = None
    mul_266: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_265, 0.7978845608028654);  mul_265 = None
    mul_267: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_266, 0.044715)
    pow_18: "f32[1, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_125, 2.0);  view_125 = None
    mul_268: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Scalar(pow_18, 3.0);  pow_18 = None
    mul_269: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_267, mul_268);  mul_267 = mul_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_130: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_266, mul_269);  mul_266 = mul_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_270: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_263, 0.5);  mul_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_131: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(add_130, mul_270);  add_130 = mul_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_337: "f32[1024, 3072]" = torch.ops.aten.reshape.default(add_131, [1024, 3072]);  add_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_45: "f32[1024, 768]" = torch.ops.aten.mm.default(view_337, permute_152);  permute_152 = None
    mm_46: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_153, view_337);  permute_153 = None
    sum_83: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_337, [0], True);  view_337 = None
    view_338: "f32[3072]" = torch.ops.aten.reshape.default(sum_83, [3072]);  sum_83 = None
    view_339: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(mm_45, [1, 1024, 768]);  mm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    mul_272: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_339, primals_125);  primals_125 = None
    mul_273: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_272, 768)
    sum_84: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_272, [2], True)
    mul_274: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_272, mul_50);  mul_272 = None
    sum_85: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_274, [2], True);  mul_274 = None
    mul_275: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_50, sum_85);  sum_85 = None
    sub_83: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_273, sum_84);  mul_273 = sum_84 = None
    sub_84: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_83, mul_275);  sub_83 = mul_275 = None
    mul_276: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_40, sub_84);  div_40 = sub_84 = None
    mul_277: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_339, mul_50);  mul_50 = None
    sum_86: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_277, [0, 1]);  mul_277 = None
    sum_87: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_339, [0, 1]);  view_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    add_132: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_129, mul_276);  add_129 = mul_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    convert_element_type_29: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_88, torch.float32);  getitem_88 = None
    mul_278: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_29, 1.1111111111111112);  convert_element_type_29 = None
    mul_279: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_132, mul_278);  mul_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_340: "f32[1024, 768]" = torch.ops.aten.reshape.default(mul_279, [1024, 768]);  mul_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_47: "f32[1024, 768]" = torch.ops.aten.mm.default(view_340, permute_154);  permute_154 = None
    mm_48: "f32[768, 768]" = torch.ops.aten.mm.default(permute_155, view_340);  permute_155 = None
    sum_88: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_340, [0], True);  view_340 = None
    view_341: "f32[768]" = torch.ops.aten.reshape.default(sum_88, [768]);  sum_88 = None
    view_342: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(mm_47, [1, 1024, 768]);  mm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_343: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_342, [1, 1024, 12, 64]);  view_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_156: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_343, [0, 2, 1, 3]);  view_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    view_344: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(permute_156, [12, 1024, 64]);  permute_156 = None
    bmm_44: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(permute_157, view_344);  permute_157 = None
    bmm_45: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_344, permute_158);  view_344 = permute_158 = None
    view_345: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_44, [1, 12, 1024, 64]);  bmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    add_133: "f32[1, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_15, view_345);  tangents_15 = view_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    view_346: "f32[1, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_45, [1, 12, 1024, 1024]);  bmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    convert_element_type_30: "f32[1, 12, 1024, 1024]" = torch.ops.prims.convert_element_type.default(getitem_86, torch.float32);  getitem_86 = None
    mul_280: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_30, 1.1111111111111112);  convert_element_type_30 = None
    mul_281: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_346, mul_280);  view_346 = mul_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_282: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_281, alias_35);  mul_281 = None
    sum_89: "f32[1, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_282, [-1], True)
    mul_283: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_35, sum_89);  alias_35 = sum_89 = None
    sub_85: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_282, mul_283);  mul_282 = mul_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_17: "f32[1, 12, 1024, 1024]" = torch.ops.aten.where.self(primals_156, sub_85, full_default_26);  primals_156 = sub_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    div_41: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(where_17, full_default);  where_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_347: "f32[12, 1024, 1024]" = torch.ops.aten.reshape.default(div_41, [12, 1024, 1024]);  div_41 = None
    bmm_46: "f32[12, 64, 1024]" = torch.ops.aten.bmm.default(permute_159, view_347);  permute_159 = None
    bmm_47: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(view_347, permute_160);  view_347 = permute_160 = None
    view_348: "f32[1, 12, 64, 1024]" = torch.ops.aten.reshape.default(bmm_46, [1, 12, 64, 1024]);  bmm_46 = None
    view_349: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_47, [1, 12, 1024, 64]);  bmm_47 = None
    permute_161: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_348, [0, 1, 3, 2]);  view_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_134: "f32[1, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_14, permute_161);  tangents_14 = permute_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_162: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(add_133, [0, 2, 1, 3]);  add_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_45: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_162, memory_format = torch.contiguous_format);  permute_162 = None
    view_350: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_45, [1, 1024, 768]);  clone_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_163: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(add_134, [0, 2, 1, 3]);  add_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_46: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_163, memory_format = torch.contiguous_format);  permute_163 = None
    view_351: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_46, [1, 1024, 768]);  clone_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_164: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_349, [0, 2, 1, 3]);  view_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_47: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_164, memory_format = torch.contiguous_format);  permute_164 = None
    view_352: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_47, [1, 1024, 768]);  clone_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    cat_5: "f32[1, 1024, 2304]" = torch.ops.aten.cat.default([view_352, view_351, view_350], 2);  view_352 = view_351 = view_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_353: "f32[1024, 2304]" = torch.ops.aten.reshape.default(cat_5, [1024, 2304]);  cat_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_49: "f32[1024, 768]" = torch.ops.aten.mm.default(view_353, permute_165);  permute_165 = None
    mm_50: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_166, view_353);  permute_166 = None
    sum_90: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_353, [0], True);  view_353 = None
    view_354: "f32[2304]" = torch.ops.aten.reshape.default(sum_90, [2304]);  sum_90 = None
    view_355: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(mm_49, [1, 1024, 768]);  mm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    mul_285: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_355, primals_123);  primals_123 = None
    mul_286: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_285, 768)
    sum_91: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_285, [2], True)
    mul_287: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_285, mul_48);  mul_285 = None
    sum_92: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_287, [2], True);  mul_287 = None
    mul_288: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_48, sum_92);  sum_92 = None
    sub_87: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_286, sum_91);  mul_286 = sum_91 = None
    sub_88: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_87, mul_288);  sub_87 = mul_288 = None
    mul_289: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_42, sub_88);  div_42 = sub_88 = None
    mul_290: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_355, mul_48);  mul_48 = None
    sum_93: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_290, [0, 1]);  mul_290 = None
    sum_94: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_355, [0, 1]);  view_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    add_135: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_132, mul_289);  add_132 = mul_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_31: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_79, torch.float32);  getitem_79 = None
    mul_291: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_31, 1.1111111111111112);  convert_element_type_31 = None
    mul_292: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_135, mul_291);  mul_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_356: "f32[1024, 768]" = torch.ops.aten.reshape.default(mul_292, [1024, 768]);  mul_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_51: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_356, permute_167);  permute_167 = None
    mm_52: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_168, view_356);  permute_168 = None
    sum_95: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_356, [0], True);  view_356 = None
    view_357: "f32[768]" = torch.ops.aten.reshape.default(sum_95, [768]);  sum_95 = None
    view_358: "f32[1, 1024, 3072]" = torch.ops.aten.reshape.default(mm_51, [1, 1024, 3072]);  mm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_293: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_358, mul_44);  mul_44 = None
    mul_294: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_358, add_47);  view_358 = add_47 = None
    mul_295: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(tanh_5, tanh_5);  tanh_5 = None
    sub_89: "f32[1, 1024, 3072]" = torch.ops.aten.sub.Tensor(1, mul_295);  mul_295 = None
    mul_296: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_293, sub_89);  mul_293 = sub_89 = None
    mul_297: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_296, 0.7978845608028654);  mul_296 = None
    mul_298: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_297, 0.044715)
    pow_19: "f32[1, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_107, 2.0);  view_107 = None
    mul_299: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Scalar(pow_19, 3.0);  pow_19 = None
    mul_300: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_298, mul_299);  mul_298 = mul_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_136: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_297, mul_300);  mul_297 = mul_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_301: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_294, 0.5);  mul_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_137: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(add_136, mul_301);  add_136 = mul_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_359: "f32[1024, 3072]" = torch.ops.aten.reshape.default(add_137, [1024, 3072]);  add_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_53: "f32[1024, 768]" = torch.ops.aten.mm.default(view_359, permute_169);  permute_169 = None
    mm_54: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_170, view_359);  permute_170 = None
    sum_96: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_359, [0], True);  view_359 = None
    view_360: "f32[3072]" = torch.ops.aten.reshape.default(sum_96, [3072]);  sum_96 = None
    view_361: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(mm_53, [1, 1024, 768]);  mm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    mul_303: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_361, primals_121);  primals_121 = None
    mul_304: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_303, 768)
    sum_97: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_303, [2], True)
    mul_305: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_303, mul_42);  mul_303 = None
    sum_98: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_305, [2], True);  mul_305 = None
    mul_306: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_42, sum_98);  sum_98 = None
    sub_91: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_304, sum_97);  mul_304 = sum_97 = None
    sub_92: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_91, mul_306);  sub_91 = mul_306 = None
    mul_307: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_43, sub_92);  div_43 = sub_92 = None
    mul_308: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_361, mul_42);  mul_42 = None
    sum_99: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_308, [0, 1]);  mul_308 = None
    sum_100: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_361, [0, 1]);  view_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    add_138: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_135, mul_307);  add_135 = mul_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    convert_element_type_32: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_75, torch.float32);  getitem_75 = None
    mul_309: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_32, 1.1111111111111112);  convert_element_type_32 = None
    mul_310: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_138, mul_309);  mul_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_362: "f32[1024, 768]" = torch.ops.aten.reshape.default(mul_310, [1024, 768]);  mul_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_55: "f32[1024, 768]" = torch.ops.aten.mm.default(view_362, permute_171);  permute_171 = None
    mm_56: "f32[768, 768]" = torch.ops.aten.mm.default(permute_172, view_362);  permute_172 = None
    sum_101: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_362, [0], True);  view_362 = None
    view_363: "f32[768]" = torch.ops.aten.reshape.default(sum_101, [768]);  sum_101 = None
    view_364: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(mm_55, [1, 1024, 768]);  mm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_365: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_364, [1, 1024, 12, 64]);  view_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_173: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_365, [0, 2, 1, 3]);  view_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    view_366: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(permute_173, [12, 1024, 64]);  permute_173 = None
    bmm_48: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(permute_174, view_366);  permute_174 = None
    bmm_49: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_366, permute_175);  view_366 = permute_175 = None
    view_367: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_48, [1, 12, 1024, 64]);  bmm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    add_139: "f32[1, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_13, view_367);  tangents_13 = view_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    view_368: "f32[1, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_49, [1, 12, 1024, 1024]);  bmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    convert_element_type_33: "f32[1, 12, 1024, 1024]" = torch.ops.prims.convert_element_type.default(getitem_73, torch.float32);  getitem_73 = None
    mul_311: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_33, 1.1111111111111112);  convert_element_type_33 = None
    mul_312: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_368, mul_311);  view_368 = mul_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_313: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_312, alias_37);  mul_312 = None
    sum_102: "f32[1, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_313, [-1], True)
    mul_314: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_37, sum_102);  alias_37 = sum_102 = None
    sub_93: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_313, mul_314);  mul_313 = mul_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_18: "f32[1, 12, 1024, 1024]" = torch.ops.aten.where.self(primals_155, sub_93, full_default_26);  primals_155 = sub_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    div_44: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(where_18, full_default);  where_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_369: "f32[12, 1024, 1024]" = torch.ops.aten.reshape.default(div_44, [12, 1024, 1024]);  div_44 = None
    bmm_50: "f32[12, 64, 1024]" = torch.ops.aten.bmm.default(permute_176, view_369);  permute_176 = None
    bmm_51: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(view_369, permute_177);  view_369 = permute_177 = None
    view_370: "f32[1, 12, 64, 1024]" = torch.ops.aten.reshape.default(bmm_50, [1, 12, 64, 1024]);  bmm_50 = None
    view_371: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_51, [1, 12, 1024, 64]);  bmm_51 = None
    permute_178: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_370, [0, 1, 3, 2]);  view_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_140: "f32[1, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_12, permute_178);  tangents_12 = permute_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_179: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(add_139, [0, 2, 1, 3]);  add_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_51: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_179, memory_format = torch.contiguous_format);  permute_179 = None
    view_372: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_51, [1, 1024, 768]);  clone_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_180: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(add_140, [0, 2, 1, 3]);  add_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_52: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_180, memory_format = torch.contiguous_format);  permute_180 = None
    view_373: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_52, [1, 1024, 768]);  clone_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_181: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_371, [0, 2, 1, 3]);  view_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_53: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_181, memory_format = torch.contiguous_format);  permute_181 = None
    view_374: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_53, [1, 1024, 768]);  clone_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    cat_6: "f32[1, 1024, 2304]" = torch.ops.aten.cat.default([view_374, view_373, view_372], 2);  view_374 = view_373 = view_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_375: "f32[1024, 2304]" = torch.ops.aten.reshape.default(cat_6, [1024, 2304]);  cat_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_57: "f32[1024, 768]" = torch.ops.aten.mm.default(view_375, permute_182);  permute_182 = None
    mm_58: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_183, view_375);  permute_183 = None
    sum_103: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_375, [0], True);  view_375 = None
    view_376: "f32[2304]" = torch.ops.aten.reshape.default(sum_103, [2304]);  sum_103 = None
    view_377: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(mm_57, [1, 1024, 768]);  mm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    mul_316: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_377, primals_119);  primals_119 = None
    mul_317: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_316, 768)
    sum_104: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_316, [2], True)
    mul_318: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_316, mul_40);  mul_316 = None
    sum_105: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_318, [2], True);  mul_318 = None
    mul_319: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_40, sum_105);  sum_105 = None
    sub_95: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_317, sum_104);  mul_317 = sum_104 = None
    sub_96: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_95, mul_319);  sub_95 = mul_319 = None
    mul_320: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_45, sub_96);  div_45 = sub_96 = None
    mul_321: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_377, mul_40);  mul_40 = None
    sum_106: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_321, [0, 1]);  mul_321 = None
    sum_107: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_377, [0, 1]);  view_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    add_141: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_138, mul_320);  add_138 = mul_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_34: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_66, torch.float32);  getitem_66 = None
    mul_322: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_34, 1.1111111111111112);  convert_element_type_34 = None
    mul_323: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_141, mul_322);  mul_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_378: "f32[1024, 768]" = torch.ops.aten.reshape.default(mul_323, [1024, 768]);  mul_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_59: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_378, permute_184);  permute_184 = None
    mm_60: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_185, view_378);  permute_185 = None
    sum_108: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_378, [0], True);  view_378 = None
    view_379: "f32[768]" = torch.ops.aten.reshape.default(sum_108, [768]);  sum_108 = None
    view_380: "f32[1, 1024, 3072]" = torch.ops.aten.reshape.default(mm_59, [1, 1024, 3072]);  mm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_324: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_380, mul_36);  mul_36 = None
    mul_325: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_380, add_39);  view_380 = add_39 = None
    mul_326: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(tanh_4, tanh_4);  tanh_4 = None
    sub_97: "f32[1, 1024, 3072]" = torch.ops.aten.sub.Tensor(1, mul_326);  mul_326 = None
    mul_327: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_324, sub_97);  mul_324 = sub_97 = None
    mul_328: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_327, 0.7978845608028654);  mul_327 = None
    mul_329: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_328, 0.044715)
    pow_20: "f32[1, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_89, 2.0);  view_89 = None
    mul_330: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Scalar(pow_20, 3.0);  pow_20 = None
    mul_331: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_329, mul_330);  mul_329 = mul_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_142: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_328, mul_331);  mul_328 = mul_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_332: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_325, 0.5);  mul_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_143: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(add_142, mul_332);  add_142 = mul_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_381: "f32[1024, 3072]" = torch.ops.aten.reshape.default(add_143, [1024, 3072]);  add_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_61: "f32[1024, 768]" = torch.ops.aten.mm.default(view_381, permute_186);  permute_186 = None
    mm_62: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_187, view_381);  permute_187 = None
    sum_109: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_381, [0], True);  view_381 = None
    view_382: "f32[3072]" = torch.ops.aten.reshape.default(sum_109, [3072]);  sum_109 = None
    view_383: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(mm_61, [1, 1024, 768]);  mm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    mul_334: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_383, primals_117);  primals_117 = None
    mul_335: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_334, 768)
    sum_110: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_334, [2], True)
    mul_336: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_334, mul_34);  mul_334 = None
    sum_111: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_336, [2], True);  mul_336 = None
    mul_337: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_34, sum_111);  sum_111 = None
    sub_99: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_335, sum_110);  mul_335 = sum_110 = None
    sub_100: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_99, mul_337);  sub_99 = mul_337 = None
    mul_338: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_46, sub_100);  div_46 = sub_100 = None
    mul_339: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_383, mul_34);  mul_34 = None
    sum_112: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_339, [0, 1]);  mul_339 = None
    sum_113: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_383, [0, 1]);  view_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    add_144: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_141, mul_338);  add_141 = mul_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    convert_element_type_35: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_62, torch.float32);  getitem_62 = None
    mul_340: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_35, 1.1111111111111112);  convert_element_type_35 = None
    mul_341: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_144, mul_340);  mul_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_384: "f32[1024, 768]" = torch.ops.aten.reshape.default(mul_341, [1024, 768]);  mul_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_63: "f32[1024, 768]" = torch.ops.aten.mm.default(view_384, permute_188);  permute_188 = None
    mm_64: "f32[768, 768]" = torch.ops.aten.mm.default(permute_189, view_384);  permute_189 = None
    sum_114: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_384, [0], True);  view_384 = None
    view_385: "f32[768]" = torch.ops.aten.reshape.default(sum_114, [768]);  sum_114 = None
    view_386: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(mm_63, [1, 1024, 768]);  mm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_387: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_386, [1, 1024, 12, 64]);  view_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_190: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_387, [0, 2, 1, 3]);  view_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    view_388: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(permute_190, [12, 1024, 64]);  permute_190 = None
    bmm_52: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(permute_191, view_388);  permute_191 = None
    bmm_53: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_388, permute_192);  view_388 = permute_192 = None
    view_389: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_52, [1, 12, 1024, 64]);  bmm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    add_145: "f32[1, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_11, view_389);  tangents_11 = view_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    view_390: "f32[1, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_53, [1, 12, 1024, 1024]);  bmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    convert_element_type_36: "f32[1, 12, 1024, 1024]" = torch.ops.prims.convert_element_type.default(getitem_60, torch.float32);  getitem_60 = None
    mul_342: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_36, 1.1111111111111112);  convert_element_type_36 = None
    mul_343: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_390, mul_342);  view_390 = mul_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_344: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_343, alias_39);  mul_343 = None
    sum_115: "f32[1, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_344, [-1], True)
    mul_345: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_39, sum_115);  alias_39 = sum_115 = None
    sub_101: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_344, mul_345);  mul_344 = mul_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_19: "f32[1, 12, 1024, 1024]" = torch.ops.aten.where.self(primals_154, sub_101, full_default_26);  primals_154 = sub_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    div_47: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(where_19, full_default);  where_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_391: "f32[12, 1024, 1024]" = torch.ops.aten.reshape.default(div_47, [12, 1024, 1024]);  div_47 = None
    bmm_54: "f32[12, 64, 1024]" = torch.ops.aten.bmm.default(permute_193, view_391);  permute_193 = None
    bmm_55: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(view_391, permute_194);  view_391 = permute_194 = None
    view_392: "f32[1, 12, 64, 1024]" = torch.ops.aten.reshape.default(bmm_54, [1, 12, 64, 1024]);  bmm_54 = None
    view_393: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_55, [1, 12, 1024, 64]);  bmm_55 = None
    permute_195: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_392, [0, 1, 3, 2]);  view_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_146: "f32[1, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_10, permute_195);  tangents_10 = permute_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_196: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(add_145, [0, 2, 1, 3]);  add_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_57: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_196, memory_format = torch.contiguous_format);  permute_196 = None
    view_394: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_57, [1, 1024, 768]);  clone_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_197: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(add_146, [0, 2, 1, 3]);  add_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_58: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_197, memory_format = torch.contiguous_format);  permute_197 = None
    view_395: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_58, [1, 1024, 768]);  clone_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_198: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_393, [0, 2, 1, 3]);  view_393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_59: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_198, memory_format = torch.contiguous_format);  permute_198 = None
    view_396: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_59, [1, 1024, 768]);  clone_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    cat_7: "f32[1, 1024, 2304]" = torch.ops.aten.cat.default([view_396, view_395, view_394], 2);  view_396 = view_395 = view_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_397: "f32[1024, 2304]" = torch.ops.aten.reshape.default(cat_7, [1024, 2304]);  cat_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_65: "f32[1024, 768]" = torch.ops.aten.mm.default(view_397, permute_199);  permute_199 = None
    mm_66: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_200, view_397);  permute_200 = None
    sum_116: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_397, [0], True);  view_397 = None
    view_398: "f32[2304]" = torch.ops.aten.reshape.default(sum_116, [2304]);  sum_116 = None
    view_399: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(mm_65, [1, 1024, 768]);  mm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    mul_347: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_399, primals_115);  primals_115 = None
    mul_348: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_347, 768)
    sum_117: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_347, [2], True)
    mul_349: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_347, mul_32);  mul_347 = None
    sum_118: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_349, [2], True);  mul_349 = None
    mul_350: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_32, sum_118);  sum_118 = None
    sub_103: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_348, sum_117);  mul_348 = sum_117 = None
    sub_104: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_103, mul_350);  sub_103 = mul_350 = None
    mul_351: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_48, sub_104);  div_48 = sub_104 = None
    mul_352: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_399, mul_32);  mul_32 = None
    sum_119: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_352, [0, 1]);  mul_352 = None
    sum_120: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_399, [0, 1]);  view_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    add_147: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_144, mul_351);  add_144 = mul_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_37: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_53, torch.float32);  getitem_53 = None
    mul_353: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_37, 1.1111111111111112);  convert_element_type_37 = None
    mul_354: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_147, mul_353);  mul_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_400: "f32[1024, 768]" = torch.ops.aten.reshape.default(mul_354, [1024, 768]);  mul_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_67: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_400, permute_201);  permute_201 = None
    mm_68: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_202, view_400);  permute_202 = None
    sum_121: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_400, [0], True);  view_400 = None
    view_401: "f32[768]" = torch.ops.aten.reshape.default(sum_121, [768]);  sum_121 = None
    view_402: "f32[1, 1024, 3072]" = torch.ops.aten.reshape.default(mm_67, [1, 1024, 3072]);  mm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_355: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_402, mul_28);  mul_28 = None
    mul_356: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_402, add_31);  view_402 = add_31 = None
    mul_357: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(tanh_3, tanh_3);  tanh_3 = None
    sub_105: "f32[1, 1024, 3072]" = torch.ops.aten.sub.Tensor(1, mul_357);  mul_357 = None
    mul_358: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_355, sub_105);  mul_355 = sub_105 = None
    mul_359: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_358, 0.7978845608028654);  mul_358 = None
    mul_360: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_359, 0.044715)
    pow_21: "f32[1, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_71, 2.0);  view_71 = None
    mul_361: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Scalar(pow_21, 3.0);  pow_21 = None
    mul_362: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_360, mul_361);  mul_360 = mul_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_148: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_359, mul_362);  mul_359 = mul_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_363: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_356, 0.5);  mul_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_149: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(add_148, mul_363);  add_148 = mul_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_403: "f32[1024, 3072]" = torch.ops.aten.reshape.default(add_149, [1024, 3072]);  add_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_69: "f32[1024, 768]" = torch.ops.aten.mm.default(view_403, permute_203);  permute_203 = None
    mm_70: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_204, view_403);  permute_204 = None
    sum_122: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_403, [0], True);  view_403 = None
    view_404: "f32[3072]" = torch.ops.aten.reshape.default(sum_122, [3072]);  sum_122 = None
    view_405: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(mm_69, [1, 1024, 768]);  mm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    mul_365: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_405, primals_113);  primals_113 = None
    mul_366: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_365, 768)
    sum_123: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_365, [2], True)
    mul_367: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_365, mul_26);  mul_365 = None
    sum_124: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_367, [2], True);  mul_367 = None
    mul_368: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_26, sum_124);  sum_124 = None
    sub_107: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_366, sum_123);  mul_366 = sum_123 = None
    sub_108: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_107, mul_368);  sub_107 = mul_368 = None
    mul_369: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_49, sub_108);  div_49 = sub_108 = None
    mul_370: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_405, mul_26);  mul_26 = None
    sum_125: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_370, [0, 1]);  mul_370 = None
    sum_126: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_405, [0, 1]);  view_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    add_150: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_147, mul_369);  add_147 = mul_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    convert_element_type_38: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_49, torch.float32);  getitem_49 = None
    mul_371: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_38, 1.1111111111111112);  convert_element_type_38 = None
    mul_372: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_150, mul_371);  mul_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_406: "f32[1024, 768]" = torch.ops.aten.reshape.default(mul_372, [1024, 768]);  mul_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_71: "f32[1024, 768]" = torch.ops.aten.mm.default(view_406, permute_205);  permute_205 = None
    mm_72: "f32[768, 768]" = torch.ops.aten.mm.default(permute_206, view_406);  permute_206 = None
    sum_127: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_406, [0], True);  view_406 = None
    view_407: "f32[768]" = torch.ops.aten.reshape.default(sum_127, [768]);  sum_127 = None
    view_408: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(mm_71, [1, 1024, 768]);  mm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_409: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_408, [1, 1024, 12, 64]);  view_408 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_207: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_409, [0, 2, 1, 3]);  view_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    view_410: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(permute_207, [12, 1024, 64]);  permute_207 = None
    bmm_56: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(permute_208, view_410);  permute_208 = None
    bmm_57: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_410, permute_209);  view_410 = permute_209 = None
    view_411: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_56, [1, 12, 1024, 64]);  bmm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    add_151: "f32[1, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_9, view_411);  tangents_9 = view_411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    view_412: "f32[1, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_57, [1, 12, 1024, 1024]);  bmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    convert_element_type_39: "f32[1, 12, 1024, 1024]" = torch.ops.prims.convert_element_type.default(getitem_47, torch.float32);  getitem_47 = None
    mul_373: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_39, 1.1111111111111112);  convert_element_type_39 = None
    mul_374: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_412, mul_373);  view_412 = mul_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_375: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_374, alias_41);  mul_374 = None
    sum_128: "f32[1, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_375, [-1], True)
    mul_376: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_41, sum_128);  alias_41 = sum_128 = None
    sub_109: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_375, mul_376);  mul_375 = mul_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_20: "f32[1, 12, 1024, 1024]" = torch.ops.aten.where.self(primals_153, sub_109, full_default_26);  primals_153 = sub_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    div_50: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(where_20, full_default);  where_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_413: "f32[12, 1024, 1024]" = torch.ops.aten.reshape.default(div_50, [12, 1024, 1024]);  div_50 = None
    bmm_58: "f32[12, 64, 1024]" = torch.ops.aten.bmm.default(permute_210, view_413);  permute_210 = None
    bmm_59: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(view_413, permute_211);  view_413 = permute_211 = None
    view_414: "f32[1, 12, 64, 1024]" = torch.ops.aten.reshape.default(bmm_58, [1, 12, 64, 1024]);  bmm_58 = None
    view_415: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_59, [1, 12, 1024, 64]);  bmm_59 = None
    permute_212: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_414, [0, 1, 3, 2]);  view_414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_152: "f32[1, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_8, permute_212);  tangents_8 = permute_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_213: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(add_151, [0, 2, 1, 3]);  add_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_63: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_213, memory_format = torch.contiguous_format);  permute_213 = None
    view_416: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_63, [1, 1024, 768]);  clone_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_214: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(add_152, [0, 2, 1, 3]);  add_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_64: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_214, memory_format = torch.contiguous_format);  permute_214 = None
    view_417: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_64, [1, 1024, 768]);  clone_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_215: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_415, [0, 2, 1, 3]);  view_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_65: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_215, memory_format = torch.contiguous_format);  permute_215 = None
    view_418: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_65, [1, 1024, 768]);  clone_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    cat_8: "f32[1, 1024, 2304]" = torch.ops.aten.cat.default([view_418, view_417, view_416], 2);  view_418 = view_417 = view_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_419: "f32[1024, 2304]" = torch.ops.aten.reshape.default(cat_8, [1024, 2304]);  cat_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_73: "f32[1024, 768]" = torch.ops.aten.mm.default(view_419, permute_216);  permute_216 = None
    mm_74: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_217, view_419);  permute_217 = None
    sum_129: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_419, [0], True);  view_419 = None
    view_420: "f32[2304]" = torch.ops.aten.reshape.default(sum_129, [2304]);  sum_129 = None
    view_421: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(mm_73, [1, 1024, 768]);  mm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    mul_378: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_421, primals_111);  primals_111 = None
    mul_379: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_378, 768)
    sum_130: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_378, [2], True)
    mul_380: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_378, mul_24);  mul_378 = None
    sum_131: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_380, [2], True);  mul_380 = None
    mul_381: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_24, sum_131);  sum_131 = None
    sub_111: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_379, sum_130);  mul_379 = sum_130 = None
    sub_112: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_111, mul_381);  sub_111 = mul_381 = None
    mul_382: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_51, sub_112);  div_51 = sub_112 = None
    mul_383: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_421, mul_24);  mul_24 = None
    sum_132: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_383, [0, 1]);  mul_383 = None
    sum_133: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_421, [0, 1]);  view_421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    add_153: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_150, mul_382);  add_150 = mul_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_40: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_40, torch.float32);  getitem_40 = None
    mul_384: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_40, 1.1111111111111112);  convert_element_type_40 = None
    mul_385: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_153, mul_384);  mul_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_422: "f32[1024, 768]" = torch.ops.aten.reshape.default(mul_385, [1024, 768]);  mul_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_75: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_422, permute_218);  permute_218 = None
    mm_76: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_219, view_422);  permute_219 = None
    sum_134: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_422, [0], True);  view_422 = None
    view_423: "f32[768]" = torch.ops.aten.reshape.default(sum_134, [768]);  sum_134 = None
    view_424: "f32[1, 1024, 3072]" = torch.ops.aten.reshape.default(mm_75, [1, 1024, 3072]);  mm_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_386: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_424, mul_20);  mul_20 = None
    mul_387: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_424, add_23);  view_424 = add_23 = None
    mul_388: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(tanh_2, tanh_2);  tanh_2 = None
    sub_113: "f32[1, 1024, 3072]" = torch.ops.aten.sub.Tensor(1, mul_388);  mul_388 = None
    mul_389: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_386, sub_113);  mul_386 = sub_113 = None
    mul_390: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_389, 0.7978845608028654);  mul_389 = None
    mul_391: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_390, 0.044715)
    pow_22: "f32[1, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_53, 2.0);  view_53 = None
    mul_392: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Scalar(pow_22, 3.0);  pow_22 = None
    mul_393: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_391, mul_392);  mul_391 = mul_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_154: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_390, mul_393);  mul_390 = mul_393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_394: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_387, 0.5);  mul_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_155: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(add_154, mul_394);  add_154 = mul_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_425: "f32[1024, 3072]" = torch.ops.aten.reshape.default(add_155, [1024, 3072]);  add_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_77: "f32[1024, 768]" = torch.ops.aten.mm.default(view_425, permute_220);  permute_220 = None
    mm_78: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_221, view_425);  permute_221 = None
    sum_135: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_425, [0], True);  view_425 = None
    view_426: "f32[3072]" = torch.ops.aten.reshape.default(sum_135, [3072]);  sum_135 = None
    view_427: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(mm_77, [1, 1024, 768]);  mm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    mul_396: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_427, primals_109);  primals_109 = None
    mul_397: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_396, 768)
    sum_136: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_396, [2], True)
    mul_398: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_396, mul_18);  mul_396 = None
    sum_137: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_398, [2], True);  mul_398 = None
    mul_399: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_18, sum_137);  sum_137 = None
    sub_115: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_397, sum_136);  mul_397 = sum_136 = None
    sub_116: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_115, mul_399);  sub_115 = mul_399 = None
    mul_400: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_52, sub_116);  div_52 = sub_116 = None
    mul_401: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_427, mul_18);  mul_18 = None
    sum_138: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_401, [0, 1]);  mul_401 = None
    sum_139: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_427, [0, 1]);  view_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    add_156: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_153, mul_400);  add_153 = mul_400 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    convert_element_type_41: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_36, torch.float32);  getitem_36 = None
    mul_402: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_41, 1.1111111111111112);  convert_element_type_41 = None
    mul_403: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_156, mul_402);  mul_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_428: "f32[1024, 768]" = torch.ops.aten.reshape.default(mul_403, [1024, 768]);  mul_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_79: "f32[1024, 768]" = torch.ops.aten.mm.default(view_428, permute_222);  permute_222 = None
    mm_80: "f32[768, 768]" = torch.ops.aten.mm.default(permute_223, view_428);  permute_223 = None
    sum_140: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_428, [0], True);  view_428 = None
    view_429: "f32[768]" = torch.ops.aten.reshape.default(sum_140, [768]);  sum_140 = None
    view_430: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(mm_79, [1, 1024, 768]);  mm_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_431: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_430, [1, 1024, 12, 64]);  view_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_224: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_431, [0, 2, 1, 3]);  view_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    view_432: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(permute_224, [12, 1024, 64]);  permute_224 = None
    bmm_60: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(permute_225, view_432);  permute_225 = None
    bmm_61: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_432, permute_226);  view_432 = permute_226 = None
    view_433: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_60, [1, 12, 1024, 64]);  bmm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    add_157: "f32[1, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_7, view_433);  tangents_7 = view_433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    view_434: "f32[1, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_61, [1, 12, 1024, 1024]);  bmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    convert_element_type_42: "f32[1, 12, 1024, 1024]" = torch.ops.prims.convert_element_type.default(getitem_34, torch.float32);  getitem_34 = None
    mul_404: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_42, 1.1111111111111112);  convert_element_type_42 = None
    mul_405: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_434, mul_404);  view_434 = mul_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_406: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_405, alias_43);  mul_405 = None
    sum_141: "f32[1, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_406, [-1], True)
    mul_407: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_43, sum_141);  alias_43 = sum_141 = None
    sub_117: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_406, mul_407);  mul_406 = mul_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_21: "f32[1, 12, 1024, 1024]" = torch.ops.aten.where.self(primals_152, sub_117, full_default_26);  primals_152 = sub_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    div_53: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(where_21, full_default);  where_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_435: "f32[12, 1024, 1024]" = torch.ops.aten.reshape.default(div_53, [12, 1024, 1024]);  div_53 = None
    bmm_62: "f32[12, 64, 1024]" = torch.ops.aten.bmm.default(permute_227, view_435);  permute_227 = None
    bmm_63: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(view_435, permute_228);  view_435 = permute_228 = None
    view_436: "f32[1, 12, 64, 1024]" = torch.ops.aten.reshape.default(bmm_62, [1, 12, 64, 1024]);  bmm_62 = None
    view_437: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_63, [1, 12, 1024, 64]);  bmm_63 = None
    permute_229: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_436, [0, 1, 3, 2]);  view_436 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_158: "f32[1, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_6, permute_229);  tangents_6 = permute_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_230: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(add_157, [0, 2, 1, 3]);  add_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_69: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_230, memory_format = torch.contiguous_format);  permute_230 = None
    view_438: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_69, [1, 1024, 768]);  clone_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_231: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(add_158, [0, 2, 1, 3]);  add_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_70: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_231, memory_format = torch.contiguous_format);  permute_231 = None
    view_439: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_70, [1, 1024, 768]);  clone_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_232: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_437, [0, 2, 1, 3]);  view_437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_71: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_232, memory_format = torch.contiguous_format);  permute_232 = None
    view_440: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_71, [1, 1024, 768]);  clone_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    cat_9: "f32[1, 1024, 2304]" = torch.ops.aten.cat.default([view_440, view_439, view_438], 2);  view_440 = view_439 = view_438 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_441: "f32[1024, 2304]" = torch.ops.aten.reshape.default(cat_9, [1024, 2304]);  cat_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_81: "f32[1024, 768]" = torch.ops.aten.mm.default(view_441, permute_233);  permute_233 = None
    mm_82: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_234, view_441);  permute_234 = None
    sum_142: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_441, [0], True);  view_441 = None
    view_442: "f32[2304]" = torch.ops.aten.reshape.default(sum_142, [2304]);  sum_142 = None
    view_443: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(mm_81, [1, 1024, 768]);  mm_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    mul_409: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_443, primals_107);  primals_107 = None
    mul_410: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_409, 768)
    sum_143: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_409, [2], True)
    mul_411: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_409, mul_16);  mul_409 = None
    sum_144: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_411, [2], True);  mul_411 = None
    mul_412: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_16, sum_144);  sum_144 = None
    sub_119: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_410, sum_143);  mul_410 = sum_143 = None
    sub_120: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_119, mul_412);  sub_119 = mul_412 = None
    mul_413: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_54, sub_120);  div_54 = sub_120 = None
    mul_414: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_443, mul_16);  mul_16 = None
    sum_145: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_414, [0, 1]);  mul_414 = None
    sum_146: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_443, [0, 1]);  view_443 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    add_159: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_156, mul_413);  add_156 = mul_413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_43: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_27, torch.float32);  getitem_27 = None
    mul_415: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_43, 1.1111111111111112);  convert_element_type_43 = None
    mul_416: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_159, mul_415);  mul_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_444: "f32[1024, 768]" = torch.ops.aten.reshape.default(mul_416, [1024, 768]);  mul_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_83: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_444, permute_235);  permute_235 = None
    mm_84: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_236, view_444);  permute_236 = None
    sum_147: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_444, [0], True);  view_444 = None
    view_445: "f32[768]" = torch.ops.aten.reshape.default(sum_147, [768]);  sum_147 = None
    view_446: "f32[1, 1024, 3072]" = torch.ops.aten.reshape.default(mm_83, [1, 1024, 3072]);  mm_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_417: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_446, mul_12);  mul_12 = None
    mul_418: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_446, add_15);  view_446 = add_15 = None
    mul_419: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(tanh_1, tanh_1);  tanh_1 = None
    sub_121: "f32[1, 1024, 3072]" = torch.ops.aten.sub.Tensor(1, mul_419);  mul_419 = None
    mul_420: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_417, sub_121);  mul_417 = sub_121 = None
    mul_421: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_420, 0.7978845608028654);  mul_420 = None
    mul_422: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_421, 0.044715)
    pow_23: "f32[1, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_35, 2.0);  view_35 = None
    mul_423: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Scalar(pow_23, 3.0);  pow_23 = None
    mul_424: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_422, mul_423);  mul_422 = mul_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_160: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_421, mul_424);  mul_421 = mul_424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_425: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_418, 0.5);  mul_418 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_161: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(add_160, mul_425);  add_160 = mul_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_447: "f32[1024, 3072]" = torch.ops.aten.reshape.default(add_161, [1024, 3072]);  add_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_85: "f32[1024, 768]" = torch.ops.aten.mm.default(view_447, permute_237);  permute_237 = None
    mm_86: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_238, view_447);  permute_238 = None
    sum_148: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_447, [0], True);  view_447 = None
    view_448: "f32[3072]" = torch.ops.aten.reshape.default(sum_148, [3072]);  sum_148 = None
    view_449: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(mm_85, [1, 1024, 768]);  mm_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    mul_427: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_449, primals_105);  primals_105 = None
    mul_428: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_427, 768)
    sum_149: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_427, [2], True)
    mul_429: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_427, mul_10);  mul_427 = None
    sum_150: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_429, [2], True);  mul_429 = None
    mul_430: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_10, sum_150);  sum_150 = None
    sub_123: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_428, sum_149);  mul_428 = sum_149 = None
    sub_124: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_123, mul_430);  sub_123 = mul_430 = None
    mul_431: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_55, sub_124);  div_55 = sub_124 = None
    mul_432: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_449, mul_10);  mul_10 = None
    sum_151: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_432, [0, 1]);  mul_432 = None
    sum_152: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_449, [0, 1]);  view_449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    add_162: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_159, mul_431);  add_159 = mul_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    convert_element_type_44: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_23, torch.float32);  getitem_23 = None
    mul_433: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_44, 1.1111111111111112);  convert_element_type_44 = None
    mul_434: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_162, mul_433);  mul_433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_450: "f32[1024, 768]" = torch.ops.aten.reshape.default(mul_434, [1024, 768]);  mul_434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_87: "f32[1024, 768]" = torch.ops.aten.mm.default(view_450, permute_239);  permute_239 = None
    mm_88: "f32[768, 768]" = torch.ops.aten.mm.default(permute_240, view_450);  permute_240 = None
    sum_153: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_450, [0], True);  view_450 = None
    view_451: "f32[768]" = torch.ops.aten.reshape.default(sum_153, [768]);  sum_153 = None
    view_452: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(mm_87, [1, 1024, 768]);  mm_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_453: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_452, [1, 1024, 12, 64]);  view_452 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_241: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_453, [0, 2, 1, 3]);  view_453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    view_454: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(permute_241, [12, 1024, 64]);  permute_241 = None
    bmm_64: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(permute_242, view_454);  permute_242 = None
    bmm_65: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_454, permute_243);  view_454 = permute_243 = None
    view_455: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_64, [1, 12, 1024, 64]);  bmm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    add_163: "f32[1, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_5, view_455);  tangents_5 = view_455 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    view_456: "f32[1, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_65, [1, 12, 1024, 1024]);  bmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    convert_element_type_45: "f32[1, 12, 1024, 1024]" = torch.ops.prims.convert_element_type.default(getitem_21, torch.float32);  getitem_21 = None
    mul_435: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_45, 1.1111111111111112);  convert_element_type_45 = None
    mul_436: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_456, mul_435);  view_456 = mul_435 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_437: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_436, alias_45);  mul_436 = None
    sum_154: "f32[1, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_437, [-1], True)
    mul_438: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_45, sum_154);  alias_45 = sum_154 = None
    sub_125: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_437, mul_438);  mul_437 = mul_438 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_22: "f32[1, 12, 1024, 1024]" = torch.ops.aten.where.self(primals_151, sub_125, full_default_26);  primals_151 = sub_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    div_56: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(where_22, full_default);  where_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_457: "f32[12, 1024, 1024]" = torch.ops.aten.reshape.default(div_56, [12, 1024, 1024]);  div_56 = None
    bmm_66: "f32[12, 64, 1024]" = torch.ops.aten.bmm.default(permute_244, view_457);  permute_244 = None
    bmm_67: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(view_457, permute_245);  view_457 = permute_245 = None
    view_458: "f32[1, 12, 64, 1024]" = torch.ops.aten.reshape.default(bmm_66, [1, 12, 64, 1024]);  bmm_66 = None
    view_459: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_67, [1, 12, 1024, 64]);  bmm_67 = None
    permute_246: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_458, [0, 1, 3, 2]);  view_458 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_164: "f32[1, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_4, permute_246);  tangents_4 = permute_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_247: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(add_163, [0, 2, 1, 3]);  add_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_75: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_247, memory_format = torch.contiguous_format);  permute_247 = None
    view_460: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_75, [1, 1024, 768]);  clone_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_248: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(add_164, [0, 2, 1, 3]);  add_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_76: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_248, memory_format = torch.contiguous_format);  permute_248 = None
    view_461: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_76, [1, 1024, 768]);  clone_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_249: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_459, [0, 2, 1, 3]);  view_459 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_77: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_249, memory_format = torch.contiguous_format);  permute_249 = None
    view_462: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_77, [1, 1024, 768]);  clone_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    cat_10: "f32[1, 1024, 2304]" = torch.ops.aten.cat.default([view_462, view_461, view_460], 2);  view_462 = view_461 = view_460 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_463: "f32[1024, 2304]" = torch.ops.aten.reshape.default(cat_10, [1024, 2304]);  cat_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_89: "f32[1024, 768]" = torch.ops.aten.mm.default(view_463, permute_250);  permute_250 = None
    mm_90: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_251, view_463);  permute_251 = None
    sum_155: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_463, [0], True);  view_463 = None
    view_464: "f32[2304]" = torch.ops.aten.reshape.default(sum_155, [2304]);  sum_155 = None
    view_465: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(mm_89, [1, 1024, 768]);  mm_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    mul_440: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_465, primals_103);  primals_103 = None
    mul_441: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_440, 768)
    sum_156: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_440, [2], True)
    mul_442: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_440, mul_8);  mul_440 = None
    sum_157: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_442, [2], True);  mul_442 = None
    mul_443: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_8, sum_157);  sum_157 = None
    sub_127: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_441, sum_156);  mul_441 = sum_156 = None
    sub_128: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_127, mul_443);  sub_127 = mul_443 = None
    mul_444: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_57, sub_128);  div_57 = sub_128 = None
    mul_445: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_465, mul_8);  mul_8 = None
    sum_158: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_445, [0, 1]);  mul_445 = None
    sum_159: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_465, [0, 1]);  view_465 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    add_165: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_162, mul_444);  add_162 = mul_444 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_46: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_14, torch.float32);  getitem_14 = None
    mul_446: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_46, 1.1111111111111112);  convert_element_type_46 = None
    mul_447: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_165, mul_446);  mul_446 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_466: "f32[1024, 768]" = torch.ops.aten.reshape.default(mul_447, [1024, 768]);  mul_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_91: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_466, permute_252);  permute_252 = None
    mm_92: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_253, view_466);  permute_253 = None
    sum_160: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_466, [0], True);  view_466 = None
    view_467: "f32[768]" = torch.ops.aten.reshape.default(sum_160, [768]);  sum_160 = None
    view_468: "f32[1, 1024, 3072]" = torch.ops.aten.reshape.default(mm_91, [1, 1024, 3072]);  mm_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_448: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_468, mul_4);  mul_4 = None
    mul_449: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_468, add_7);  view_468 = add_7 = None
    mul_450: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(tanh, tanh);  tanh = None
    sub_129: "f32[1, 1024, 3072]" = torch.ops.aten.sub.Tensor(1, mul_450);  mul_450 = None
    mul_451: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_448, sub_129);  mul_448 = sub_129 = None
    mul_452: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_451, 0.7978845608028654);  mul_451 = None
    mul_453: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_452, 0.044715)
    pow_24: "f32[1, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_17, 2.0);  view_17 = None
    mul_454: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Scalar(pow_24, 3.0);  pow_24 = None
    mul_455: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_453, mul_454);  mul_453 = mul_454 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_166: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_452, mul_455);  mul_452 = mul_455 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_456: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_449, 0.5);  mul_449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_167: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(add_166, mul_456);  add_166 = mul_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_469: "f32[1024, 3072]" = torch.ops.aten.reshape.default(add_167, [1024, 3072]);  add_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_93: "f32[1024, 768]" = torch.ops.aten.mm.default(view_469, permute_254);  permute_254 = None
    mm_94: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_255, view_469);  permute_255 = None
    sum_161: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_469, [0], True);  view_469 = None
    view_470: "f32[3072]" = torch.ops.aten.reshape.default(sum_161, [3072]);  sum_161 = None
    view_471: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(mm_93, [1, 1024, 768]);  mm_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    mul_458: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_471, primals_101);  primals_101 = None
    mul_459: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_458, 768)
    sum_162: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_458, [2], True)
    mul_460: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_458, mul_2);  mul_458 = None
    sum_163: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_460, [2], True);  mul_460 = None
    mul_461: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2, sum_163);  sum_163 = None
    sub_131: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_459, sum_162);  mul_459 = sum_162 = None
    sub_132: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_131, mul_461);  sub_131 = mul_461 = None
    mul_462: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_58, sub_132);  div_58 = sub_132 = None
    mul_463: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_471, mul_2);  mul_2 = None
    sum_164: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_463, [0, 1]);  mul_463 = None
    sum_165: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_471, [0, 1]);  view_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    add_168: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_165, mul_462);  add_165 = mul_462 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    convert_element_type_47: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_10, torch.float32);  getitem_10 = None
    mul_464: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_47, 1.1111111111111112);  convert_element_type_47 = None
    mul_465: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_168, mul_464);  mul_464 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_472: "f32[1024, 768]" = torch.ops.aten.reshape.default(mul_465, [1024, 768]);  mul_465 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_95: "f32[1024, 768]" = torch.ops.aten.mm.default(view_472, permute_256);  permute_256 = None
    mm_96: "f32[768, 768]" = torch.ops.aten.mm.default(permute_257, view_472);  permute_257 = None
    sum_166: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_472, [0], True);  view_472 = None
    view_473: "f32[768]" = torch.ops.aten.reshape.default(sum_166, [768]);  sum_166 = None
    view_474: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(mm_95, [1, 1024, 768]);  mm_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_475: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_474, [1, 1024, 12, 64]);  view_474 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_258: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_475, [0, 2, 1, 3]);  view_475 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    view_476: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(permute_258, [12, 1024, 64]);  permute_258 = None
    bmm_68: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(permute_259, view_476);  permute_259 = None
    bmm_69: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_476, permute_260);  view_476 = permute_260 = None
    view_477: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_68, [1, 12, 1024, 64]);  bmm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    add_169: "f32[1, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_3, view_477);  tangents_3 = view_477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    view_478: "f32[1, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_69, [1, 12, 1024, 1024]);  bmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    convert_element_type_48: "f32[1, 12, 1024, 1024]" = torch.ops.prims.convert_element_type.default(getitem_8, torch.float32);  getitem_8 = None
    mul_466: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_48, 1.1111111111111112);  convert_element_type_48 = None
    mul_467: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_478, mul_466);  view_478 = mul_466 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_468: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_467, alias_47);  mul_467 = None
    sum_167: "f32[1, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_468, [-1], True)
    mul_469: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_47, sum_167);  alias_47 = sum_167 = None
    sub_133: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_468, mul_469);  mul_468 = mul_469 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_23: "f32[1, 12, 1024, 1024]" = torch.ops.aten.where.self(primals_150, sub_133, full_default_26);  primals_150 = sub_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    div_59: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(where_23, full_default);  where_23 = full_default = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_479: "f32[12, 1024, 1024]" = torch.ops.aten.reshape.default(div_59, [12, 1024, 1024]);  div_59 = None
    bmm_70: "f32[12, 64, 1024]" = torch.ops.aten.bmm.default(permute_261, view_479);  permute_261 = None
    bmm_71: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(view_479, permute_262);  view_479 = permute_262 = None
    view_480: "f32[1, 12, 64, 1024]" = torch.ops.aten.reshape.default(bmm_70, [1, 12, 64, 1024]);  bmm_70 = None
    view_481: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_71, [1, 12, 1024, 64]);  bmm_71 = None
    permute_263: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_480, [0, 1, 3, 2]);  view_480 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_170: "f32[1, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_2, permute_263);  tangents_2 = permute_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_264: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(add_169, [0, 2, 1, 3]);  add_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_81: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_264, memory_format = torch.contiguous_format);  permute_264 = None
    view_482: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_81, [1, 1024, 768]);  clone_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_265: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(add_170, [0, 2, 1, 3]);  add_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_82: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_265, memory_format = torch.contiguous_format);  permute_265 = None
    view_483: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_82, [1, 1024, 768]);  clone_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_266: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_481, [0, 2, 1, 3]);  view_481 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_83: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_266, memory_format = torch.contiguous_format);  permute_266 = None
    view_484: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_83, [1, 1024, 768]);  clone_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    cat_11: "f32[1, 1024, 2304]" = torch.ops.aten.cat.default([view_484, view_483, view_482], 2);  view_484 = view_483 = view_482 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_485: "f32[1024, 2304]" = torch.ops.aten.reshape.default(cat_11, [1024, 2304]);  cat_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_97: "f32[1024, 768]" = torch.ops.aten.mm.default(view_485, permute_267);  permute_267 = None
    mm_98: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_268, view_485);  permute_268 = None
    sum_168: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_485, [0], True);  view_485 = None
    view_486: "f32[2304]" = torch.ops.aten.reshape.default(sum_168, [2304]);  sum_168 = None
    view_487: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(mm_97, [1, 1024, 768]);  mm_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    mul_471: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_487, primals_99);  primals_99 = None
    mul_472: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_471, 768)
    sum_169: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_471, [2], True)
    mul_473: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_471, mul);  mul_471 = None
    sum_170: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_473, [2], True);  mul_473 = None
    mul_474: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul, sum_170);  sum_170 = None
    sub_135: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_472, sum_169);  mul_472 = sum_169 = None
    sub_136: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_135, mul_474);  sub_135 = mul_474 = None
    mul_475: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_60, sub_136);  div_60 = sub_136 = None
    mul_476: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_487, mul);  mul = None
    sum_171: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_476, [0, 1]);  mul_476 = None
    sum_172: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_487, [0, 1]);  view_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    add_171: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_168, mul_475);  add_168 = mul_475 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:851, code: hidden_states = self.drop(hidden_states)
    convert_element_type_49: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_1, torch.float32);  getitem_1 = None
    mul_477: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_49, 1.1111111111111112);  convert_element_type_49 = None
    mul_478: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_171, mul_477);  add_171 = mul_477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:844, code: position_embeds = self.wpe(position_ids)
    full_default_38: "b8[1, 1024, 1]" = torch.ops.aten.full.default([1, 1024, 1], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where_24: "f32[1, 1024, 768]" = torch.ops.aten.where.self(full_default_38, full_default_26, mul_478);  full_default_38 = None
    full_default_40: "f32[1024, 768]" = torch.ops.aten.full.default([1024, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put: "f32[1024, 768]" = torch.ops.prims._unsafe_index_put_.default(full_default_40, [view_1], where_24, True);  full_default_40 = view_1 = where_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:843, code: inputs_embeds = self.wte(input_ids)
    eq_2: "b8[1, 1024]" = torch.ops.aten.eq.Scalar(view, -1)
    unsqueeze_2: "b8[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(eq_2, -1);  eq_2 = None
    where_25: "f32[1, 1024, 768]" = torch.ops.aten.where.self(unsqueeze_2, full_default_26, mul_478);  unsqueeze_2 = full_default_26 = mul_478 = None
    full_default_42: "f32[50257, 768]" = torch.ops.aten.full.default([50257, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put_1: "f32[50257, 768]" = torch.ops.prims._unsafe_index_put_.default(full_default_42, [view], where_25, True);  full_default_42 = view = where_25 = None
    return [view_486, mm_98, view_473, mm_96, view_470, mm_94, view_467, mm_92, view_464, mm_90, view_451, mm_88, view_448, mm_86, view_445, mm_84, view_442, mm_82, view_429, mm_80, view_426, mm_78, view_423, mm_76, view_420, mm_74, view_407, mm_72, view_404, mm_70, view_401, mm_68, view_398, mm_66, view_385, mm_64, view_382, mm_62, view_379, mm_60, view_376, mm_58, view_363, mm_56, view_360, mm_54, view_357, mm_52, view_354, mm_50, view_341, mm_48, view_338, mm_46, view_335, mm_44, view_332, mm_42, view_319, mm_40, view_316, mm_38, view_313, mm_36, view_310, mm_34, view_297, mm_32, view_294, mm_30, view_291, mm_28, view_288, mm_26, view_275, mm_24, view_272, mm_22, view_269, mm_20, view_266, mm_18, view_253, mm_16, view_250, mm_14, view_247, mm_12, view_244, mm_10, view_231, mm_8, view_228, mm_6, view_225, mm_4, _unsafe_index_put_1, _unsafe_index_put, sum_171, sum_172, sum_164, sum_165, sum_158, sum_159, sum_151, sum_152, sum_145, sum_146, sum_138, sum_139, sum_132, sum_133, sum_125, sum_126, sum_119, sum_120, sum_112, sum_113, sum_106, sum_107, sum_99, sum_100, sum_93, sum_94, sum_86, sum_87, sum_80, sum_81, sum_73, sum_74, sum_67, sum_68, sum_60, sum_61, sum_54, sum_55, sum_47, sum_48, sum_41, sum_42, sum_34, sum_35, sum_28, sum_29, sum_21, sum_22, sum_15, sum_16, permute_64, None, None, None, None, None, None, None, None, None, None, None, None, None]
    