from __future__ import annotations



def forward(self, primals_1: "f32[384]", primals_2: "f32[1, 1, 384]", primals_3: "f32[1, 1, 384]", primals_4: "f32[384]", primals_5: "f32[1, 1, 384]", primals_6: "f32[1, 1, 384]", primals_7: "f32[384]", primals_8: "f32[1, 1, 384]", primals_9: "f32[1, 1, 384]", primals_10: "f32[384]", primals_11: "f32[1, 1, 384]", primals_12: "f32[1, 1, 384]", primals_13: "f32[384]", primals_14: "f32[1, 1, 384]", primals_15: "f32[1, 1, 384]", primals_16: "f32[384]", primals_17: "f32[1, 1, 384]", primals_18: "f32[1, 1, 384]", primals_19: "f32[384]", primals_20: "f32[1, 1, 384]", primals_21: "f32[1, 1, 384]", primals_22: "f32[384]", primals_23: "f32[1, 1, 384]", primals_24: "f32[1, 1, 384]", primals_25: "f32[384]", primals_26: "f32[1, 1, 384]", primals_27: "f32[1, 1, 384]", primals_28: "f32[384]", primals_29: "f32[1, 1, 384]", primals_30: "f32[1, 1, 384]", primals_31: "f32[384]", primals_32: "f32[1, 1, 384]", primals_33: "f32[1, 1, 384]", primals_34: "f32[384]", primals_35: "f32[1, 1, 384]", primals_36: "f32[1, 1, 384]", primals_37: "f32[384]", primals_38: "f32[1, 1, 384]", primals_39: "f32[1, 1, 384]", primals_40: "f32[384]", primals_41: "f32[1, 1, 384]", primals_42: "f32[1, 1, 384]", primals_43: "f32[384]", primals_44: "f32[1, 1, 384]", primals_45: "f32[1, 1, 384]", primals_46: "f32[384]", primals_47: "f32[1, 1, 384]", primals_48: "f32[1, 1, 384]", primals_49: "f32[384]", primals_50: "f32[1, 1, 384]", primals_51: "f32[1, 1, 384]", primals_52: "f32[384]", primals_53: "f32[1, 1, 384]", primals_54: "f32[1, 1, 384]", primals_55: "f32[384]", primals_56: "f32[1, 1, 384]", primals_57: "f32[1, 1, 384]", primals_58: "f32[384]", primals_59: "f32[1, 1, 384]", primals_60: "f32[1, 1, 384]", primals_61: "f32[384]", primals_62: "f32[1, 1, 384]", primals_63: "f32[1, 1, 384]", primals_64: "f32[384]", primals_65: "f32[1, 1, 384]", primals_66: "f32[1, 1, 384]", primals_67: "f32[384]", primals_68: "f32[1, 1, 384]", primals_69: "f32[1, 1, 384]", primals_70: "f32[384]", primals_71: "f32[1, 1, 384]", primals_72: "f32[1, 1, 384]", primals_73: "f32[1, 1, 384]", primals_74: "f32[1, 1, 384]", primals_75: "f32[384, 3, 16, 16]", primals_76: "f32[384]", primals_77: "f32[196, 196]", primals_78: "f32[196]", primals_79: "f32[1536, 384]", primals_80: "f32[1536]", primals_81: "f32[384, 1536]", primals_82: "f32[384]", primals_83: "f32[196, 196]", primals_84: "f32[196]", primals_85: "f32[1536, 384]", primals_86: "f32[1536]", primals_87: "f32[384, 1536]", primals_88: "f32[384]", primals_89: "f32[196, 196]", primals_90: "f32[196]", primals_91: "f32[1536, 384]", primals_92: "f32[1536]", primals_93: "f32[384, 1536]", primals_94: "f32[384]", primals_95: "f32[196, 196]", primals_96: "f32[196]", primals_97: "f32[1536, 384]", primals_98: "f32[1536]", primals_99: "f32[384, 1536]", primals_100: "f32[384]", primals_101: "f32[196, 196]", primals_102: "f32[196]", primals_103: "f32[1536, 384]", primals_104: "f32[1536]", primals_105: "f32[384, 1536]", primals_106: "f32[384]", primals_107: "f32[196, 196]", primals_108: "f32[196]", primals_109: "f32[1536, 384]", primals_110: "f32[1536]", primals_111: "f32[384, 1536]", primals_112: "f32[384]", primals_113: "f32[196, 196]", primals_114: "f32[196]", primals_115: "f32[1536, 384]", primals_116: "f32[1536]", primals_117: "f32[384, 1536]", primals_118: "f32[384]", primals_119: "f32[196, 196]", primals_120: "f32[196]", primals_121: "f32[1536, 384]", primals_122: "f32[1536]", primals_123: "f32[384, 1536]", primals_124: "f32[384]", primals_125: "f32[196, 196]", primals_126: "f32[196]", primals_127: "f32[1536, 384]", primals_128: "f32[1536]", primals_129: "f32[384, 1536]", primals_130: "f32[384]", primals_131: "f32[196, 196]", primals_132: "f32[196]", primals_133: "f32[1536, 384]", primals_134: "f32[1536]", primals_135: "f32[384, 1536]", primals_136: "f32[384]", primals_137: "f32[196, 196]", primals_138: "f32[196]", primals_139: "f32[1536, 384]", primals_140: "f32[1536]", primals_141: "f32[384, 1536]", primals_142: "f32[384]", primals_143: "f32[196, 196]", primals_144: "f32[196]", primals_145: "f32[1536, 384]", primals_146: "f32[1536]", primals_147: "f32[384, 1536]", primals_148: "f32[384]", primals_149: "f32[1000, 384]", primals_150: "f32[1000]", primals_151: "f32[8, 3, 224, 224]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(primals_151, primals_75, primals_76, [16, 16], [0, 0], [1, 1], False, [0, 0], 1);  primals_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:89, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    view: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(convolution, [8, 384, 196])
    permute: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(primals_3, 1)
    mul_1: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul, permute);  mul = None
    add: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(primals_2, mul_1);  primals_2 = mul_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_1: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add, [0, 2, 1]);  add = None
    view_1: "f32[3072, 196]" = torch.ops.aten.reshape.default(permute_1, [3072, 196]);  permute_1 = None
    permute_2: "f32[196, 196]" = torch.ops.aten.permute.default(primals_77, [1, 0]);  primals_77 = None
    addmm: "f32[3072, 196]" = torch.ops.aten.addmm.default(primals_78, view_1, permute_2);  primals_78 = None
    view_2: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(addmm, [8, 384, 196])
    permute_3: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_2, [0, 2, 1]);  view_2 = None
    mul_2: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(primals_1, permute_3);  permute_3 = None
    add_1: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(permute, mul_2);  permute = mul_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_3: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(primals_6, 1)
    mul_4: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_3, add_1);  mul_3 = None
    add_2: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(primals_5, mul_4);  primals_5 = mul_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_4: "f32[384, 1536]" = torch.ops.aten.permute.default(primals_79, [1, 0]);  primals_79 = None
    clone: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_2, memory_format = torch.contiguous_format);  add_2 = None
    view_3: "f32[1568, 384]" = torch.ops.aten.reshape.default(clone, [1568, 384]);  clone = None
    mm: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_3, permute_4)
    view_4: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(mm, [8, 196, 1536])
    add_3: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(view_4, primals_80);  view_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_5: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_3, 0.5)
    mul_6: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_3, 0.7071067811865476);  add_3 = None
    erf: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_6);  mul_6 = None
    add_4: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_7: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_5, add_4);  mul_5 = add_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_5: "f32[1568, 1536]" = torch.ops.aten.reshape.default(mul_7, [1568, 1536]);  mul_7 = None
    permute_5: "f32[1536, 384]" = torch.ops.aten.permute.default(primals_81, [1, 0]);  primals_81 = None
    addmm_1: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_82, view_5, permute_5);  primals_82 = None
    view_6: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(addmm_1, [8, 196, 384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    mul_8: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(primals_4, view_6);  view_6 = None
    add_5: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_1, mul_8);  add_1 = mul_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_9: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(primals_9, 1)
    mul_10: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_9, add_5);  mul_9 = None
    add_6: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(primals_8, mul_10);  primals_8 = mul_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_6: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_6, [0, 2, 1]);  add_6 = None
    view_7: "f32[3072, 196]" = torch.ops.aten.reshape.default(permute_6, [3072, 196]);  permute_6 = None
    permute_7: "f32[196, 196]" = torch.ops.aten.permute.default(primals_83, [1, 0]);  primals_83 = None
    addmm_2: "f32[3072, 196]" = torch.ops.aten.addmm.default(primals_84, view_7, permute_7);  primals_84 = None
    view_8: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(addmm_2, [8, 384, 196])
    permute_8: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_8, [0, 2, 1]);  view_8 = None
    mul_11: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(primals_7, permute_8);  permute_8 = None
    add_7: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_5, mul_11);  add_5 = mul_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_12: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(primals_12, 1)
    mul_13: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_12, add_7);  mul_12 = None
    add_8: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(primals_11, mul_13);  primals_11 = mul_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_9: "f32[384, 1536]" = torch.ops.aten.permute.default(primals_85, [1, 0]);  primals_85 = None
    clone_3: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_8, memory_format = torch.contiguous_format);  add_8 = None
    view_9: "f32[1568, 384]" = torch.ops.aten.reshape.default(clone_3, [1568, 384]);  clone_3 = None
    mm_1: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_9, permute_9)
    view_10: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(mm_1, [8, 196, 1536])
    add_9: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(view_10, primals_86);  view_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_14: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_9, 0.5)
    mul_15: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_9, 0.7071067811865476);  add_9 = None
    erf_1: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_15);  mul_15 = None
    add_10: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_16: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_14, add_10);  mul_14 = add_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_11: "f32[1568, 1536]" = torch.ops.aten.reshape.default(mul_16, [1568, 1536]);  mul_16 = None
    permute_10: "f32[1536, 384]" = torch.ops.aten.permute.default(primals_87, [1, 0]);  primals_87 = None
    addmm_3: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_88, view_11, permute_10);  primals_88 = None
    view_12: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(addmm_3, [8, 196, 384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    mul_17: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(primals_10, view_12);  view_12 = None
    add_11: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_7, mul_17);  add_7 = mul_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_18: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(primals_15, 1)
    mul_19: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_18, add_11);  mul_18 = None
    add_12: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(primals_14, mul_19);  primals_14 = mul_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_11: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_12, [0, 2, 1]);  add_12 = None
    view_13: "f32[3072, 196]" = torch.ops.aten.reshape.default(permute_11, [3072, 196]);  permute_11 = None
    permute_12: "f32[196, 196]" = torch.ops.aten.permute.default(primals_89, [1, 0]);  primals_89 = None
    addmm_4: "f32[3072, 196]" = torch.ops.aten.addmm.default(primals_90, view_13, permute_12);  primals_90 = None
    view_14: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(addmm_4, [8, 384, 196])
    permute_13: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_14, [0, 2, 1]);  view_14 = None
    mul_20: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(primals_13, permute_13);  permute_13 = None
    add_13: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_11, mul_20);  add_11 = mul_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_21: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(primals_18, 1)
    mul_22: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_21, add_13);  mul_21 = None
    add_14: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(primals_17, mul_22);  primals_17 = mul_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_14: "f32[384, 1536]" = torch.ops.aten.permute.default(primals_91, [1, 0]);  primals_91 = None
    clone_6: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_14, memory_format = torch.contiguous_format);  add_14 = None
    view_15: "f32[1568, 384]" = torch.ops.aten.reshape.default(clone_6, [1568, 384]);  clone_6 = None
    mm_2: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_15, permute_14)
    view_16: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(mm_2, [8, 196, 1536])
    add_15: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(view_16, primals_92);  view_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_23: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_15, 0.5)
    mul_24: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_15, 0.7071067811865476);  add_15 = None
    erf_2: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_24);  mul_24 = None
    add_16: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_25: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_23, add_16);  mul_23 = add_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_17: "f32[1568, 1536]" = torch.ops.aten.reshape.default(mul_25, [1568, 1536]);  mul_25 = None
    permute_15: "f32[1536, 384]" = torch.ops.aten.permute.default(primals_93, [1, 0]);  primals_93 = None
    addmm_5: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_94, view_17, permute_15);  primals_94 = None
    view_18: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(addmm_5, [8, 196, 384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    mul_26: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(primals_16, view_18);  view_18 = None
    add_17: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_13, mul_26);  add_13 = mul_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_27: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(primals_21, 1)
    mul_28: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_27, add_17);  mul_27 = None
    add_18: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(primals_20, mul_28);  primals_20 = mul_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_16: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_18, [0, 2, 1]);  add_18 = None
    view_19: "f32[3072, 196]" = torch.ops.aten.reshape.default(permute_16, [3072, 196]);  permute_16 = None
    permute_17: "f32[196, 196]" = torch.ops.aten.permute.default(primals_95, [1, 0]);  primals_95 = None
    addmm_6: "f32[3072, 196]" = torch.ops.aten.addmm.default(primals_96, view_19, permute_17);  primals_96 = None
    view_20: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(addmm_6, [8, 384, 196])
    permute_18: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_20, [0, 2, 1]);  view_20 = None
    mul_29: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(primals_19, permute_18);  permute_18 = None
    add_19: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_17, mul_29);  add_17 = mul_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_30: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(primals_24, 1)
    mul_31: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_30, add_19);  mul_30 = None
    add_20: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(primals_23, mul_31);  primals_23 = mul_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_19: "f32[384, 1536]" = torch.ops.aten.permute.default(primals_97, [1, 0]);  primals_97 = None
    clone_9: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_20, memory_format = torch.contiguous_format);  add_20 = None
    view_21: "f32[1568, 384]" = torch.ops.aten.reshape.default(clone_9, [1568, 384]);  clone_9 = None
    mm_3: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_21, permute_19)
    view_22: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(mm_3, [8, 196, 1536])
    add_21: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(view_22, primals_98);  view_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_32: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_21, 0.5)
    mul_33: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_21, 0.7071067811865476);  add_21 = None
    erf_3: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_33);  mul_33 = None
    add_22: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_34: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_32, add_22);  mul_32 = add_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_23: "f32[1568, 1536]" = torch.ops.aten.reshape.default(mul_34, [1568, 1536]);  mul_34 = None
    permute_20: "f32[1536, 384]" = torch.ops.aten.permute.default(primals_99, [1, 0]);  primals_99 = None
    addmm_7: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_100, view_23, permute_20);  primals_100 = None
    view_24: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(addmm_7, [8, 196, 384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    mul_35: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(primals_22, view_24);  view_24 = None
    add_23: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_19, mul_35);  add_19 = mul_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_36: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(primals_27, 1)
    mul_37: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_36, add_23);  mul_36 = None
    add_24: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(primals_26, mul_37);  primals_26 = mul_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_21: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_24, [0, 2, 1]);  add_24 = None
    view_25: "f32[3072, 196]" = torch.ops.aten.reshape.default(permute_21, [3072, 196]);  permute_21 = None
    permute_22: "f32[196, 196]" = torch.ops.aten.permute.default(primals_101, [1, 0]);  primals_101 = None
    addmm_8: "f32[3072, 196]" = torch.ops.aten.addmm.default(primals_102, view_25, permute_22);  primals_102 = None
    view_26: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(addmm_8, [8, 384, 196])
    permute_23: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_26, [0, 2, 1]);  view_26 = None
    mul_38: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(primals_25, permute_23);  permute_23 = None
    add_25: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_23, mul_38);  add_23 = mul_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_39: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(primals_30, 1)
    mul_40: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_39, add_25);  mul_39 = None
    add_26: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(primals_29, mul_40);  primals_29 = mul_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_24: "f32[384, 1536]" = torch.ops.aten.permute.default(primals_103, [1, 0]);  primals_103 = None
    clone_12: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_26, memory_format = torch.contiguous_format);  add_26 = None
    view_27: "f32[1568, 384]" = torch.ops.aten.reshape.default(clone_12, [1568, 384]);  clone_12 = None
    mm_4: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_27, permute_24)
    view_28: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(mm_4, [8, 196, 1536])
    add_27: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(view_28, primals_104);  view_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_41: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_27, 0.5)
    mul_42: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_27, 0.7071067811865476);  add_27 = None
    erf_4: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_42);  mul_42 = None
    add_28: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_43: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_41, add_28);  mul_41 = add_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_29: "f32[1568, 1536]" = torch.ops.aten.reshape.default(mul_43, [1568, 1536]);  mul_43 = None
    permute_25: "f32[1536, 384]" = torch.ops.aten.permute.default(primals_105, [1, 0]);  primals_105 = None
    addmm_9: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_106, view_29, permute_25);  primals_106 = None
    view_30: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(addmm_9, [8, 196, 384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    mul_44: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(primals_28, view_30);  view_30 = None
    add_29: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_25, mul_44);  add_25 = mul_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_45: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(primals_33, 1)
    mul_46: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_45, add_29);  mul_45 = None
    add_30: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(primals_32, mul_46);  primals_32 = mul_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_26: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_30, [0, 2, 1]);  add_30 = None
    view_31: "f32[3072, 196]" = torch.ops.aten.reshape.default(permute_26, [3072, 196]);  permute_26 = None
    permute_27: "f32[196, 196]" = torch.ops.aten.permute.default(primals_107, [1, 0]);  primals_107 = None
    addmm_10: "f32[3072, 196]" = torch.ops.aten.addmm.default(primals_108, view_31, permute_27);  primals_108 = None
    view_32: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(addmm_10, [8, 384, 196])
    permute_28: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_32, [0, 2, 1]);  view_32 = None
    mul_47: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(primals_31, permute_28);  permute_28 = None
    add_31: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_29, mul_47);  add_29 = mul_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_48: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(primals_36, 1)
    mul_49: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_48, add_31);  mul_48 = None
    add_32: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(primals_35, mul_49);  primals_35 = mul_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_29: "f32[384, 1536]" = torch.ops.aten.permute.default(primals_109, [1, 0]);  primals_109 = None
    clone_15: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_32, memory_format = torch.contiguous_format);  add_32 = None
    view_33: "f32[1568, 384]" = torch.ops.aten.reshape.default(clone_15, [1568, 384]);  clone_15 = None
    mm_5: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_33, permute_29)
    view_34: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(mm_5, [8, 196, 1536])
    add_33: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(view_34, primals_110);  view_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_50: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_33, 0.5)
    mul_51: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_33, 0.7071067811865476);  add_33 = None
    erf_5: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_51);  mul_51 = None
    add_34: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_52: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_50, add_34);  mul_50 = add_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_35: "f32[1568, 1536]" = torch.ops.aten.reshape.default(mul_52, [1568, 1536]);  mul_52 = None
    permute_30: "f32[1536, 384]" = torch.ops.aten.permute.default(primals_111, [1, 0]);  primals_111 = None
    addmm_11: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_112, view_35, permute_30);  primals_112 = None
    view_36: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(addmm_11, [8, 196, 384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    mul_53: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(primals_34, view_36);  view_36 = None
    add_35: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_31, mul_53);  add_31 = mul_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_54: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(primals_39, 1)
    mul_55: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_54, add_35);  mul_54 = None
    add_36: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(primals_38, mul_55);  primals_38 = mul_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_31: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_36, [0, 2, 1]);  add_36 = None
    view_37: "f32[3072, 196]" = torch.ops.aten.reshape.default(permute_31, [3072, 196]);  permute_31 = None
    permute_32: "f32[196, 196]" = torch.ops.aten.permute.default(primals_113, [1, 0]);  primals_113 = None
    addmm_12: "f32[3072, 196]" = torch.ops.aten.addmm.default(primals_114, view_37, permute_32);  primals_114 = None
    view_38: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(addmm_12, [8, 384, 196])
    permute_33: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_38, [0, 2, 1]);  view_38 = None
    mul_56: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(primals_37, permute_33);  permute_33 = None
    add_37: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_35, mul_56);  add_35 = mul_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_57: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(primals_42, 1)
    mul_58: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_57, add_37);  mul_57 = None
    add_38: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(primals_41, mul_58);  primals_41 = mul_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_34: "f32[384, 1536]" = torch.ops.aten.permute.default(primals_115, [1, 0]);  primals_115 = None
    clone_18: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_38, memory_format = torch.contiguous_format);  add_38 = None
    view_39: "f32[1568, 384]" = torch.ops.aten.reshape.default(clone_18, [1568, 384]);  clone_18 = None
    mm_6: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_39, permute_34)
    view_40: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(mm_6, [8, 196, 1536])
    add_39: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(view_40, primals_116);  view_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_59: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_39, 0.5)
    mul_60: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_39, 0.7071067811865476);  add_39 = None
    erf_6: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_60);  mul_60 = None
    add_40: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_61: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_59, add_40);  mul_59 = add_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_41: "f32[1568, 1536]" = torch.ops.aten.reshape.default(mul_61, [1568, 1536]);  mul_61 = None
    permute_35: "f32[1536, 384]" = torch.ops.aten.permute.default(primals_117, [1, 0]);  primals_117 = None
    addmm_13: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_118, view_41, permute_35);  primals_118 = None
    view_42: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(addmm_13, [8, 196, 384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    mul_62: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(primals_40, view_42);  view_42 = None
    add_41: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_37, mul_62);  add_37 = mul_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_63: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(primals_45, 1)
    mul_64: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_63, add_41);  mul_63 = None
    add_42: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(primals_44, mul_64);  primals_44 = mul_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_36: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_42, [0, 2, 1]);  add_42 = None
    view_43: "f32[3072, 196]" = torch.ops.aten.reshape.default(permute_36, [3072, 196]);  permute_36 = None
    permute_37: "f32[196, 196]" = torch.ops.aten.permute.default(primals_119, [1, 0]);  primals_119 = None
    addmm_14: "f32[3072, 196]" = torch.ops.aten.addmm.default(primals_120, view_43, permute_37);  primals_120 = None
    view_44: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(addmm_14, [8, 384, 196])
    permute_38: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_44, [0, 2, 1]);  view_44 = None
    mul_65: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(primals_43, permute_38);  permute_38 = None
    add_43: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_41, mul_65);  add_41 = mul_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_66: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(primals_48, 1)
    mul_67: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_66, add_43);  mul_66 = None
    add_44: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(primals_47, mul_67);  primals_47 = mul_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_39: "f32[384, 1536]" = torch.ops.aten.permute.default(primals_121, [1, 0]);  primals_121 = None
    clone_21: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_44, memory_format = torch.contiguous_format);  add_44 = None
    view_45: "f32[1568, 384]" = torch.ops.aten.reshape.default(clone_21, [1568, 384]);  clone_21 = None
    mm_7: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_45, permute_39)
    view_46: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(mm_7, [8, 196, 1536])
    add_45: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(view_46, primals_122);  view_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_68: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_45, 0.5)
    mul_69: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_45, 0.7071067811865476);  add_45 = None
    erf_7: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_69);  mul_69 = None
    add_46: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_70: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_68, add_46);  mul_68 = add_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_47: "f32[1568, 1536]" = torch.ops.aten.reshape.default(mul_70, [1568, 1536]);  mul_70 = None
    permute_40: "f32[1536, 384]" = torch.ops.aten.permute.default(primals_123, [1, 0]);  primals_123 = None
    addmm_15: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_124, view_47, permute_40);  primals_124 = None
    view_48: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(addmm_15, [8, 196, 384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    mul_71: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(primals_46, view_48);  view_48 = None
    add_47: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_43, mul_71);  add_43 = mul_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_72: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(primals_51, 1)
    mul_73: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_72, add_47);  mul_72 = None
    add_48: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(primals_50, mul_73);  primals_50 = mul_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_41: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_48, [0, 2, 1]);  add_48 = None
    view_49: "f32[3072, 196]" = torch.ops.aten.reshape.default(permute_41, [3072, 196]);  permute_41 = None
    permute_42: "f32[196, 196]" = torch.ops.aten.permute.default(primals_125, [1, 0]);  primals_125 = None
    addmm_16: "f32[3072, 196]" = torch.ops.aten.addmm.default(primals_126, view_49, permute_42);  primals_126 = None
    view_50: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(addmm_16, [8, 384, 196])
    permute_43: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_50, [0, 2, 1]);  view_50 = None
    mul_74: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(primals_49, permute_43);  permute_43 = None
    add_49: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_47, mul_74);  add_47 = mul_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_75: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(primals_54, 1)
    mul_76: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_75, add_49);  mul_75 = None
    add_50: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(primals_53, mul_76);  primals_53 = mul_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_44: "f32[384, 1536]" = torch.ops.aten.permute.default(primals_127, [1, 0]);  primals_127 = None
    clone_24: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_50, memory_format = torch.contiguous_format);  add_50 = None
    view_51: "f32[1568, 384]" = torch.ops.aten.reshape.default(clone_24, [1568, 384]);  clone_24 = None
    mm_8: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_51, permute_44)
    view_52: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(mm_8, [8, 196, 1536])
    add_51: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(view_52, primals_128);  view_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_77: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_51, 0.5)
    mul_78: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_51, 0.7071067811865476);  add_51 = None
    erf_8: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_78);  mul_78 = None
    add_52: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_79: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_77, add_52);  mul_77 = add_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_53: "f32[1568, 1536]" = torch.ops.aten.reshape.default(mul_79, [1568, 1536]);  mul_79 = None
    permute_45: "f32[1536, 384]" = torch.ops.aten.permute.default(primals_129, [1, 0]);  primals_129 = None
    addmm_17: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_130, view_53, permute_45);  primals_130 = None
    view_54: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(addmm_17, [8, 196, 384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    mul_80: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(primals_52, view_54);  view_54 = None
    add_53: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_49, mul_80);  add_49 = mul_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_81: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(primals_57, 1)
    mul_82: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_81, add_53);  mul_81 = None
    add_54: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(primals_56, mul_82);  primals_56 = mul_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_46: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_54, [0, 2, 1]);  add_54 = None
    view_55: "f32[3072, 196]" = torch.ops.aten.reshape.default(permute_46, [3072, 196]);  permute_46 = None
    permute_47: "f32[196, 196]" = torch.ops.aten.permute.default(primals_131, [1, 0]);  primals_131 = None
    addmm_18: "f32[3072, 196]" = torch.ops.aten.addmm.default(primals_132, view_55, permute_47);  primals_132 = None
    view_56: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(addmm_18, [8, 384, 196])
    permute_48: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_56, [0, 2, 1]);  view_56 = None
    mul_83: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(primals_55, permute_48);  permute_48 = None
    add_55: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_53, mul_83);  add_53 = mul_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_84: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(primals_60, 1)
    mul_85: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_84, add_55);  mul_84 = None
    add_56: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(primals_59, mul_85);  primals_59 = mul_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_49: "f32[384, 1536]" = torch.ops.aten.permute.default(primals_133, [1, 0]);  primals_133 = None
    clone_27: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_56, memory_format = torch.contiguous_format);  add_56 = None
    view_57: "f32[1568, 384]" = torch.ops.aten.reshape.default(clone_27, [1568, 384]);  clone_27 = None
    mm_9: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_57, permute_49)
    view_58: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(mm_9, [8, 196, 1536])
    add_57: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(view_58, primals_134);  view_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_86: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_57, 0.5)
    mul_87: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_57, 0.7071067811865476);  add_57 = None
    erf_9: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_87);  mul_87 = None
    add_58: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_88: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_86, add_58);  mul_86 = add_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_59: "f32[1568, 1536]" = torch.ops.aten.reshape.default(mul_88, [1568, 1536]);  mul_88 = None
    permute_50: "f32[1536, 384]" = torch.ops.aten.permute.default(primals_135, [1, 0]);  primals_135 = None
    addmm_19: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_136, view_59, permute_50);  primals_136 = None
    view_60: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(addmm_19, [8, 196, 384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    mul_89: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(primals_58, view_60);  view_60 = None
    add_59: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_55, mul_89);  add_55 = mul_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_90: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(primals_63, 1)
    mul_91: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_90, add_59);  mul_90 = None
    add_60: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(primals_62, mul_91);  primals_62 = mul_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_51: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_60, [0, 2, 1]);  add_60 = None
    view_61: "f32[3072, 196]" = torch.ops.aten.reshape.default(permute_51, [3072, 196]);  permute_51 = None
    permute_52: "f32[196, 196]" = torch.ops.aten.permute.default(primals_137, [1, 0]);  primals_137 = None
    addmm_20: "f32[3072, 196]" = torch.ops.aten.addmm.default(primals_138, view_61, permute_52);  primals_138 = None
    view_62: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(addmm_20, [8, 384, 196])
    permute_53: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_62, [0, 2, 1]);  view_62 = None
    mul_92: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(primals_61, permute_53);  permute_53 = None
    add_61: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_59, mul_92);  add_59 = mul_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_93: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(primals_66, 1)
    mul_94: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_93, add_61);  mul_93 = None
    add_62: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(primals_65, mul_94);  primals_65 = mul_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_54: "f32[384, 1536]" = torch.ops.aten.permute.default(primals_139, [1, 0]);  primals_139 = None
    clone_30: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_62, memory_format = torch.contiguous_format);  add_62 = None
    view_63: "f32[1568, 384]" = torch.ops.aten.reshape.default(clone_30, [1568, 384]);  clone_30 = None
    mm_10: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_63, permute_54)
    view_64: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(mm_10, [8, 196, 1536])
    add_63: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(view_64, primals_140);  view_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_95: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_63, 0.5)
    mul_96: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_63, 0.7071067811865476);  add_63 = None
    erf_10: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_96);  mul_96 = None
    add_64: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_97: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_95, add_64);  mul_95 = add_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_65: "f32[1568, 1536]" = torch.ops.aten.reshape.default(mul_97, [1568, 1536]);  mul_97 = None
    permute_55: "f32[1536, 384]" = torch.ops.aten.permute.default(primals_141, [1, 0]);  primals_141 = None
    addmm_21: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_142, view_65, permute_55);  primals_142 = None
    view_66: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(addmm_21, [8, 196, 384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    mul_98: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(primals_64, view_66);  view_66 = None
    add_65: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_61, mul_98);  add_61 = mul_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_99: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(primals_69, 1)
    mul_100: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_99, add_65);  mul_99 = None
    add_66: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(primals_68, mul_100);  primals_68 = mul_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_56: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_66, [0, 2, 1]);  add_66 = None
    view_67: "f32[3072, 196]" = torch.ops.aten.reshape.default(permute_56, [3072, 196]);  permute_56 = None
    permute_57: "f32[196, 196]" = torch.ops.aten.permute.default(primals_143, [1, 0]);  primals_143 = None
    addmm_22: "f32[3072, 196]" = torch.ops.aten.addmm.default(primals_144, view_67, permute_57);  primals_144 = None
    view_68: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(addmm_22, [8, 384, 196])
    permute_58: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_68, [0, 2, 1]);  view_68 = None
    mul_101: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(primals_67, permute_58);  permute_58 = None
    add_67: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_65, mul_101);  add_65 = mul_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_102: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(primals_72, 1)
    mul_103: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_102, add_67);  mul_102 = None
    add_68: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(primals_71, mul_103);  primals_71 = mul_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_59: "f32[384, 1536]" = torch.ops.aten.permute.default(primals_145, [1, 0]);  primals_145 = None
    clone_33: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_68, memory_format = torch.contiguous_format);  add_68 = None
    view_69: "f32[1568, 384]" = torch.ops.aten.reshape.default(clone_33, [1568, 384]);  clone_33 = None
    mm_11: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_69, permute_59)
    view_70: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(mm_11, [8, 196, 1536])
    add_69: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(view_70, primals_146);  view_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_104: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_69, 0.5)
    mul_105: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_69, 0.7071067811865476);  add_69 = None
    erf_11: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_105);  mul_105 = None
    add_70: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_106: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_104, add_70);  mul_104 = add_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_71: "f32[1568, 1536]" = torch.ops.aten.reshape.default(mul_106, [1568, 1536]);  mul_106 = None
    permute_60: "f32[1536, 384]" = torch.ops.aten.permute.default(primals_147, [1, 0]);  primals_147 = None
    addmm_23: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_148, view_71, permute_60);  primals_148 = None
    view_72: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(addmm_23, [8, 196, 384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    mul_107: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(primals_70, view_72);  view_72 = None
    add_71: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_67, mul_107);  add_67 = mul_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_108: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(primals_74, 1)
    mul_109: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_108, add_71);  mul_108 = add_71 = None
    add_72: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(primals_73, mul_109);  primals_73 = mul_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:271, code: x = x.mean(dim=1)
    mean: "f32[8, 384]" = torch.ops.aten.mean.dim(add_72, [1]);  add_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:273, code: return x if pre_logits else self.head(x)
    permute_61: "f32[384, 1000]" = torch.ops.aten.permute.default(primals_149, [1, 0]);  primals_149 = None
    addmm_24: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_150, mean, permute_61);  primals_150 = None
    permute_62: "f32[1000, 384]" = torch.ops.aten.permute.default(permute_61, [1, 0]);  permute_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_66: "f32[384, 1536]" = torch.ops.aten.permute.default(permute_60, [1, 0]);  permute_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_72: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_75: "f32[196, 196]" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_80: "f32[384, 1536]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_86: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_89: "f32[196, 196]" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_94: "f32[384, 1536]" = torch.ops.aten.permute.default(permute_50, [1, 0]);  permute_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_100: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_49, [1, 0]);  permute_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_103: "f32[196, 196]" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_108: "f32[384, 1536]" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_114: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_117: "f32[196, 196]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_122: "f32[384, 1536]" = torch.ops.aten.permute.default(permute_40, [1, 0]);  permute_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_128: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_39, [1, 0]);  permute_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_131: "f32[196, 196]" = torch.ops.aten.permute.default(permute_37, [1, 0]);  permute_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_136: "f32[384, 1536]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_142: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_145: "f32[196, 196]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_150: "f32[384, 1536]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_156: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_29, [1, 0]);  permute_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_159: "f32[196, 196]" = torch.ops.aten.permute.default(permute_27, [1, 0]);  permute_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_164: "f32[384, 1536]" = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_170: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_173: "f32[196, 196]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_178: "f32[384, 1536]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_184: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_187: "f32[196, 196]" = torch.ops.aten.permute.default(permute_17, [1, 0]);  permute_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_192: "f32[384, 1536]" = torch.ops.aten.permute.default(permute_15, [1, 0]);  permute_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_198: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_201: "f32[196, 196]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_206: "f32[384, 1536]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_212: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_215: "f32[196, 196]" = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_220: "f32[384, 1536]" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_226: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_229: "f32[196, 196]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
    return [addmm_24, primals_1, primals_3, primals_4, primals_6, primals_7, primals_9, primals_10, primals_12, primals_13, primals_15, primals_16, primals_18, primals_19, primals_21, primals_22, primals_24, primals_25, primals_27, primals_28, primals_30, primals_31, primals_33, primals_34, primals_36, primals_37, primals_39, primals_40, primals_42, primals_43, primals_45, primals_46, primals_48, primals_49, primals_51, primals_52, primals_54, primals_55, primals_57, primals_58, primals_60, primals_61, primals_63, primals_64, primals_66, primals_67, primals_69, primals_70, primals_72, primals_74, primals_75, primals_80, primals_86, primals_92, primals_98, primals_104, primals_110, primals_116, primals_122, primals_128, primals_134, primals_140, primals_146, primals_151, convolution, view_1, addmm, view_3, mm, view_5, addmm_1, view_7, addmm_2, view_9, mm_1, view_11, addmm_3, view_13, addmm_4, view_15, mm_2, view_17, addmm_5, view_19, addmm_6, view_21, mm_3, view_23, addmm_7, view_25, addmm_8, view_27, mm_4, view_29, addmm_9, view_31, addmm_10, view_33, mm_5, view_35, addmm_11, view_37, addmm_12, view_39, mm_6, view_41, addmm_13, view_43, addmm_14, view_45, mm_7, view_47, addmm_15, view_49, addmm_16, view_51, mm_8, view_53, addmm_17, view_55, addmm_18, view_57, mm_9, view_59, addmm_19, view_61, addmm_20, view_63, mm_10, view_65, addmm_21, view_67, addmm_22, view_69, mm_11, view_71, addmm_23, mean, permute_62, permute_66, permute_72, permute_75, permute_80, permute_86, permute_89, permute_94, permute_100, permute_103, permute_108, permute_114, permute_117, permute_122, permute_128, permute_131, permute_136, permute_142, permute_145, permute_150, permute_156, permute_159, permute_164, permute_170, permute_173, permute_178, permute_184, permute_187, permute_192, permute_198, permute_201, permute_206, permute_212, permute_215, permute_220, permute_226, permute_229]
    