from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[384]"; primals_2: "f32[1, 1, 384]"; primals_3: "f32[1, 1, 384]"; primals_4: "f32[384]"; primals_5: "f32[1, 1, 384]"; primals_6: "f32[1, 1, 384]"; primals_7: "f32[384]"; primals_8: "f32[1, 1, 384]"; primals_9: "f32[1, 1, 384]"; primals_10: "f32[384]"; primals_11: "f32[1, 1, 384]"; primals_12: "f32[1, 1, 384]"; primals_13: "f32[384]"; primals_14: "f32[1, 1, 384]"; primals_15: "f32[1, 1, 384]"; primals_16: "f32[384]"; primals_17: "f32[1, 1, 384]"; primals_18: "f32[1, 1, 384]"; primals_19: "f32[384]"; primals_20: "f32[1, 1, 384]"; primals_21: "f32[1, 1, 384]"; primals_22: "f32[384]"; primals_23: "f32[1, 1, 384]"; primals_24: "f32[1, 1, 384]"; primals_25: "f32[384]"; primals_26: "f32[1, 1, 384]"; primals_27: "f32[1, 1, 384]"; primals_28: "f32[384]"; primals_29: "f32[1, 1, 384]"; primals_30: "f32[1, 1, 384]"; primals_31: "f32[384]"; primals_32: "f32[1, 1, 384]"; primals_33: "f32[1, 1, 384]"; primals_34: "f32[384]"; primals_35: "f32[1, 1, 384]"; primals_36: "f32[1, 1, 384]"; primals_37: "f32[384]"; primals_38: "f32[1, 1, 384]"; primals_39: "f32[1, 1, 384]"; primals_40: "f32[384]"; primals_41: "f32[1, 1, 384]"; primals_42: "f32[1, 1, 384]"; primals_43: "f32[384]"; primals_44: "f32[1, 1, 384]"; primals_45: "f32[1, 1, 384]"; primals_46: "f32[384]"; primals_47: "f32[1, 1, 384]"; primals_48: "f32[1, 1, 384]"; primals_49: "f32[384]"; primals_50: "f32[1, 1, 384]"; primals_51: "f32[1, 1, 384]"; primals_52: "f32[384]"; primals_53: "f32[1, 1, 384]"; primals_54: "f32[1, 1, 384]"; primals_55: "f32[384]"; primals_56: "f32[1, 1, 384]"; primals_57: "f32[1, 1, 384]"; primals_58: "f32[384]"; primals_59: "f32[1, 1, 384]"; primals_60: "f32[1, 1, 384]"; primals_61: "f32[384]"; primals_62: "f32[1, 1, 384]"; primals_63: "f32[1, 1, 384]"; primals_64: "f32[384]"; primals_65: "f32[1, 1, 384]"; primals_66: "f32[1, 1, 384]"; primals_67: "f32[384]"; primals_68: "f32[1, 1, 384]"; primals_69: "f32[1, 1, 384]"; primals_70: "f32[384]"; primals_71: "f32[1, 1, 384]"; primals_72: "f32[1, 1, 384]"; primals_73: "f32[1, 1, 384]"; primals_74: "f32[1, 1, 384]"; primals_75: "f32[384, 3, 16, 16]"; primals_76: "f32[384]"; primals_77: "f32[196, 196]"; primals_78: "f32[196]"; primals_79: "f32[1536, 384]"; primals_80: "f32[1536]"; primals_81: "f32[384, 1536]"; primals_82: "f32[384]"; primals_83: "f32[196, 196]"; primals_84: "f32[196]"; primals_85: "f32[1536, 384]"; primals_86: "f32[1536]"; primals_87: "f32[384, 1536]"; primals_88: "f32[384]"; primals_89: "f32[196, 196]"; primals_90: "f32[196]"; primals_91: "f32[1536, 384]"; primals_92: "f32[1536]"; primals_93: "f32[384, 1536]"; primals_94: "f32[384]"; primals_95: "f32[196, 196]"; primals_96: "f32[196]"; primals_97: "f32[1536, 384]"; primals_98: "f32[1536]"; primals_99: "f32[384, 1536]"; primals_100: "f32[384]"; primals_101: "f32[196, 196]"; primals_102: "f32[196]"; primals_103: "f32[1536, 384]"; primals_104: "f32[1536]"; primals_105: "f32[384, 1536]"; primals_106: "f32[384]"; primals_107: "f32[196, 196]"; primals_108: "f32[196]"; primals_109: "f32[1536, 384]"; primals_110: "f32[1536]"; primals_111: "f32[384, 1536]"; primals_112: "f32[384]"; primals_113: "f32[196, 196]"; primals_114: "f32[196]"; primals_115: "f32[1536, 384]"; primals_116: "f32[1536]"; primals_117: "f32[384, 1536]"; primals_118: "f32[384]"; primals_119: "f32[196, 196]"; primals_120: "f32[196]"; primals_121: "f32[1536, 384]"; primals_122: "f32[1536]"; primals_123: "f32[384, 1536]"; primals_124: "f32[384]"; primals_125: "f32[196, 196]"; primals_126: "f32[196]"; primals_127: "f32[1536, 384]"; primals_128: "f32[1536]"; primals_129: "f32[384, 1536]"; primals_130: "f32[384]"; primals_131: "f32[196, 196]"; primals_132: "f32[196]"; primals_133: "f32[1536, 384]"; primals_134: "f32[1536]"; primals_135: "f32[384, 1536]"; primals_136: "f32[384]"; primals_137: "f32[196, 196]"; primals_138: "f32[196]"; primals_139: "f32[1536, 384]"; primals_140: "f32[1536]"; primals_141: "f32[384, 1536]"; primals_142: "f32[384]"; primals_143: "f32[196, 196]"; primals_144: "f32[196]"; primals_145: "f32[1536, 384]"; primals_146: "f32[1536]"; primals_147: "f32[384, 1536]"; primals_148: "f32[384]"; primals_149: "f32[1000, 384]"; primals_150: "f32[1000]"; primals_151: "f32[8, 3, 224, 224]"; tangents_1: "f32[8, 1000]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(primals_151, primals_75, primals_76, [16, 16], [0, 0], [1, 1], False, [0, 0], 1);  primals_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:89, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    view: "f32[8, 384, 196]" = torch.ops.aten.view.default(convolution, [8, 384, 196]);  convolution = None
    permute: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(primals_3, 1)
    mul_1: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul, permute);  mul = None
    add: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(primals_2, mul_1);  primals_2 = mul_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_1: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add, [0, 2, 1]);  add = None
    view_1: "f32[3072, 196]" = torch.ops.aten.view.default(permute_1, [3072, 196]);  permute_1 = None
    permute_2: "f32[196, 196]" = torch.ops.aten.permute.default(primals_77, [1, 0]);  primals_77 = None
    addmm: "f32[3072, 196]" = torch.ops.aten.addmm.default(primals_78, view_1, permute_2);  primals_78 = None
    view_2: "f32[8, 384, 196]" = torch.ops.aten.view.default(addmm, [8, 384, 196]);  addmm = None
    permute_3: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_2, [0, 2, 1]);  view_2 = None
    mul_2: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(primals_1, permute_3)
    add_1: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(permute, mul_2);  mul_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_3: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(primals_6, 1)
    mul_4: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_3, add_1);  mul_3 = None
    add_2: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(primals_5, mul_4);  primals_5 = mul_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_4: "f32[384, 1536]" = torch.ops.aten.permute.default(primals_79, [1, 0]);  primals_79 = None
    clone: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_2, memory_format = torch.contiguous_format);  add_2 = None
    view_3: "f32[1568, 384]" = torch.ops.aten.view.default(clone, [1568, 384]);  clone = None
    mm: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_3, permute_4)
    view_4: "f32[8, 196, 1536]" = torch.ops.aten.view.default(mm, [8, 196, 1536]);  mm = None
    add_3: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(view_4, primals_80);  view_4 = primals_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_5: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_3, 0.5)
    mul_6: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_3, 0.7071067811865476)
    erf: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_6);  mul_6 = None
    add_4: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_7: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_5, add_4);  mul_5 = add_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_1: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_7);  mul_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_5: "f32[1568, 1536]" = torch.ops.aten.view.default(clone_1, [1568, 1536]);  clone_1 = None
    permute_5: "f32[1536, 384]" = torch.ops.aten.permute.default(primals_81, [1, 0]);  primals_81 = None
    addmm_1: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_82, view_5, permute_5);  primals_82 = None
    view_6: "f32[8, 196, 384]" = torch.ops.aten.view.default(addmm_1, [8, 196, 384]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_2: "f32[8, 196, 384]" = torch.ops.aten.clone.default(view_6);  view_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    mul_8: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(primals_4, clone_2)
    add_5: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_1, mul_8);  mul_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_9: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(primals_9, 1)
    mul_10: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_9, add_5);  mul_9 = None
    add_6: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(primals_8, mul_10);  primals_8 = mul_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_6: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_6, [0, 2, 1]);  add_6 = None
    view_7: "f32[3072, 196]" = torch.ops.aten.view.default(permute_6, [3072, 196]);  permute_6 = None
    permute_7: "f32[196, 196]" = torch.ops.aten.permute.default(primals_83, [1, 0]);  primals_83 = None
    addmm_2: "f32[3072, 196]" = torch.ops.aten.addmm.default(primals_84, view_7, permute_7);  primals_84 = None
    view_8: "f32[8, 384, 196]" = torch.ops.aten.view.default(addmm_2, [8, 384, 196]);  addmm_2 = None
    permute_8: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_8, [0, 2, 1]);  view_8 = None
    mul_11: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(primals_7, permute_8)
    add_7: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_5, mul_11);  mul_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_12: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(primals_12, 1)
    mul_13: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_12, add_7);  mul_12 = None
    add_8: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(primals_11, mul_13);  primals_11 = mul_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_9: "f32[384, 1536]" = torch.ops.aten.permute.default(primals_85, [1, 0]);  primals_85 = None
    clone_3: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_8, memory_format = torch.contiguous_format);  add_8 = None
    view_9: "f32[1568, 384]" = torch.ops.aten.view.default(clone_3, [1568, 384]);  clone_3 = None
    mm_1: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_9, permute_9)
    view_10: "f32[8, 196, 1536]" = torch.ops.aten.view.default(mm_1, [8, 196, 1536]);  mm_1 = None
    add_9: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(view_10, primals_86);  view_10 = primals_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_14: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_9, 0.5)
    mul_15: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_9, 0.7071067811865476)
    erf_1: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_15);  mul_15 = None
    add_10: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_16: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_14, add_10);  mul_14 = add_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_4: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_16);  mul_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_11: "f32[1568, 1536]" = torch.ops.aten.view.default(clone_4, [1568, 1536]);  clone_4 = None
    permute_10: "f32[1536, 384]" = torch.ops.aten.permute.default(primals_87, [1, 0]);  primals_87 = None
    addmm_3: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_88, view_11, permute_10);  primals_88 = None
    view_12: "f32[8, 196, 384]" = torch.ops.aten.view.default(addmm_3, [8, 196, 384]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_5: "f32[8, 196, 384]" = torch.ops.aten.clone.default(view_12);  view_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    mul_17: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(primals_10, clone_5)
    add_11: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_7, mul_17);  mul_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_18: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(primals_15, 1)
    mul_19: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_18, add_11);  mul_18 = None
    add_12: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(primals_14, mul_19);  primals_14 = mul_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_11: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_12, [0, 2, 1]);  add_12 = None
    view_13: "f32[3072, 196]" = torch.ops.aten.view.default(permute_11, [3072, 196]);  permute_11 = None
    permute_12: "f32[196, 196]" = torch.ops.aten.permute.default(primals_89, [1, 0]);  primals_89 = None
    addmm_4: "f32[3072, 196]" = torch.ops.aten.addmm.default(primals_90, view_13, permute_12);  primals_90 = None
    view_14: "f32[8, 384, 196]" = torch.ops.aten.view.default(addmm_4, [8, 384, 196]);  addmm_4 = None
    permute_13: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_14, [0, 2, 1]);  view_14 = None
    mul_20: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(primals_13, permute_13)
    add_13: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_11, mul_20);  mul_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_21: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(primals_18, 1)
    mul_22: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_21, add_13);  mul_21 = None
    add_14: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(primals_17, mul_22);  primals_17 = mul_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_14: "f32[384, 1536]" = torch.ops.aten.permute.default(primals_91, [1, 0]);  primals_91 = None
    clone_6: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_14, memory_format = torch.contiguous_format);  add_14 = None
    view_15: "f32[1568, 384]" = torch.ops.aten.view.default(clone_6, [1568, 384]);  clone_6 = None
    mm_2: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_15, permute_14)
    view_16: "f32[8, 196, 1536]" = torch.ops.aten.view.default(mm_2, [8, 196, 1536]);  mm_2 = None
    add_15: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(view_16, primals_92);  view_16 = primals_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_23: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_15, 0.5)
    mul_24: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_15, 0.7071067811865476)
    erf_2: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_24);  mul_24 = None
    add_16: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_25: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_23, add_16);  mul_23 = add_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_7: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_25);  mul_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_17: "f32[1568, 1536]" = torch.ops.aten.view.default(clone_7, [1568, 1536]);  clone_7 = None
    permute_15: "f32[1536, 384]" = torch.ops.aten.permute.default(primals_93, [1, 0]);  primals_93 = None
    addmm_5: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_94, view_17, permute_15);  primals_94 = None
    view_18: "f32[8, 196, 384]" = torch.ops.aten.view.default(addmm_5, [8, 196, 384]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_8: "f32[8, 196, 384]" = torch.ops.aten.clone.default(view_18);  view_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    mul_26: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(primals_16, clone_8)
    add_17: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_13, mul_26);  mul_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_27: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(primals_21, 1)
    mul_28: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_27, add_17);  mul_27 = None
    add_18: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(primals_20, mul_28);  primals_20 = mul_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_16: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_18, [0, 2, 1]);  add_18 = None
    view_19: "f32[3072, 196]" = torch.ops.aten.view.default(permute_16, [3072, 196]);  permute_16 = None
    permute_17: "f32[196, 196]" = torch.ops.aten.permute.default(primals_95, [1, 0]);  primals_95 = None
    addmm_6: "f32[3072, 196]" = torch.ops.aten.addmm.default(primals_96, view_19, permute_17);  primals_96 = None
    view_20: "f32[8, 384, 196]" = torch.ops.aten.view.default(addmm_6, [8, 384, 196]);  addmm_6 = None
    permute_18: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_20, [0, 2, 1]);  view_20 = None
    mul_29: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(primals_19, permute_18)
    add_19: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_17, mul_29);  mul_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_30: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(primals_24, 1)
    mul_31: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_30, add_19);  mul_30 = None
    add_20: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(primals_23, mul_31);  primals_23 = mul_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_19: "f32[384, 1536]" = torch.ops.aten.permute.default(primals_97, [1, 0]);  primals_97 = None
    clone_9: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_20, memory_format = torch.contiguous_format);  add_20 = None
    view_21: "f32[1568, 384]" = torch.ops.aten.view.default(clone_9, [1568, 384]);  clone_9 = None
    mm_3: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_21, permute_19)
    view_22: "f32[8, 196, 1536]" = torch.ops.aten.view.default(mm_3, [8, 196, 1536]);  mm_3 = None
    add_21: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(view_22, primals_98);  view_22 = primals_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_32: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_21, 0.5)
    mul_33: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_21, 0.7071067811865476)
    erf_3: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_33);  mul_33 = None
    add_22: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_34: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_32, add_22);  mul_32 = add_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_10: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_34);  mul_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_23: "f32[1568, 1536]" = torch.ops.aten.view.default(clone_10, [1568, 1536]);  clone_10 = None
    permute_20: "f32[1536, 384]" = torch.ops.aten.permute.default(primals_99, [1, 0]);  primals_99 = None
    addmm_7: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_100, view_23, permute_20);  primals_100 = None
    view_24: "f32[8, 196, 384]" = torch.ops.aten.view.default(addmm_7, [8, 196, 384]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_11: "f32[8, 196, 384]" = torch.ops.aten.clone.default(view_24);  view_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    mul_35: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(primals_22, clone_11)
    add_23: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_19, mul_35);  mul_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_36: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(primals_27, 1)
    mul_37: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_36, add_23);  mul_36 = None
    add_24: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(primals_26, mul_37);  primals_26 = mul_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_21: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_24, [0, 2, 1]);  add_24 = None
    view_25: "f32[3072, 196]" = torch.ops.aten.view.default(permute_21, [3072, 196]);  permute_21 = None
    permute_22: "f32[196, 196]" = torch.ops.aten.permute.default(primals_101, [1, 0]);  primals_101 = None
    addmm_8: "f32[3072, 196]" = torch.ops.aten.addmm.default(primals_102, view_25, permute_22);  primals_102 = None
    view_26: "f32[8, 384, 196]" = torch.ops.aten.view.default(addmm_8, [8, 384, 196]);  addmm_8 = None
    permute_23: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_26, [0, 2, 1]);  view_26 = None
    mul_38: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(primals_25, permute_23)
    add_25: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_23, mul_38);  mul_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_39: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(primals_30, 1)
    mul_40: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_39, add_25);  mul_39 = None
    add_26: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(primals_29, mul_40);  primals_29 = mul_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_24: "f32[384, 1536]" = torch.ops.aten.permute.default(primals_103, [1, 0]);  primals_103 = None
    clone_12: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_26, memory_format = torch.contiguous_format);  add_26 = None
    view_27: "f32[1568, 384]" = torch.ops.aten.view.default(clone_12, [1568, 384]);  clone_12 = None
    mm_4: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_27, permute_24)
    view_28: "f32[8, 196, 1536]" = torch.ops.aten.view.default(mm_4, [8, 196, 1536]);  mm_4 = None
    add_27: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(view_28, primals_104);  view_28 = primals_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_41: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_27, 0.5)
    mul_42: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_27, 0.7071067811865476)
    erf_4: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_42);  mul_42 = None
    add_28: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_43: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_41, add_28);  mul_41 = add_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_13: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_43);  mul_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_29: "f32[1568, 1536]" = torch.ops.aten.view.default(clone_13, [1568, 1536]);  clone_13 = None
    permute_25: "f32[1536, 384]" = torch.ops.aten.permute.default(primals_105, [1, 0]);  primals_105 = None
    addmm_9: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_106, view_29, permute_25);  primals_106 = None
    view_30: "f32[8, 196, 384]" = torch.ops.aten.view.default(addmm_9, [8, 196, 384]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_14: "f32[8, 196, 384]" = torch.ops.aten.clone.default(view_30);  view_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    mul_44: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(primals_28, clone_14)
    add_29: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_25, mul_44);  mul_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_45: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(primals_33, 1)
    mul_46: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_45, add_29);  mul_45 = None
    add_30: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(primals_32, mul_46);  primals_32 = mul_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_26: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_30, [0, 2, 1]);  add_30 = None
    view_31: "f32[3072, 196]" = torch.ops.aten.view.default(permute_26, [3072, 196]);  permute_26 = None
    permute_27: "f32[196, 196]" = torch.ops.aten.permute.default(primals_107, [1, 0]);  primals_107 = None
    addmm_10: "f32[3072, 196]" = torch.ops.aten.addmm.default(primals_108, view_31, permute_27);  primals_108 = None
    view_32: "f32[8, 384, 196]" = torch.ops.aten.view.default(addmm_10, [8, 384, 196]);  addmm_10 = None
    permute_28: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_32, [0, 2, 1]);  view_32 = None
    mul_47: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(primals_31, permute_28)
    add_31: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_29, mul_47);  mul_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_48: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(primals_36, 1)
    mul_49: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_48, add_31);  mul_48 = None
    add_32: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(primals_35, mul_49);  primals_35 = mul_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_29: "f32[384, 1536]" = torch.ops.aten.permute.default(primals_109, [1, 0]);  primals_109 = None
    clone_15: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_32, memory_format = torch.contiguous_format);  add_32 = None
    view_33: "f32[1568, 384]" = torch.ops.aten.view.default(clone_15, [1568, 384]);  clone_15 = None
    mm_5: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_33, permute_29)
    view_34: "f32[8, 196, 1536]" = torch.ops.aten.view.default(mm_5, [8, 196, 1536]);  mm_5 = None
    add_33: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(view_34, primals_110);  view_34 = primals_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_50: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_33, 0.5)
    mul_51: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_33, 0.7071067811865476)
    erf_5: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_51);  mul_51 = None
    add_34: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_52: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_50, add_34);  mul_50 = add_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_16: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_52);  mul_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_35: "f32[1568, 1536]" = torch.ops.aten.view.default(clone_16, [1568, 1536]);  clone_16 = None
    permute_30: "f32[1536, 384]" = torch.ops.aten.permute.default(primals_111, [1, 0]);  primals_111 = None
    addmm_11: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_112, view_35, permute_30);  primals_112 = None
    view_36: "f32[8, 196, 384]" = torch.ops.aten.view.default(addmm_11, [8, 196, 384]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_17: "f32[8, 196, 384]" = torch.ops.aten.clone.default(view_36);  view_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    mul_53: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(primals_34, clone_17)
    add_35: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_31, mul_53);  mul_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_54: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(primals_39, 1)
    mul_55: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_54, add_35);  mul_54 = None
    add_36: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(primals_38, mul_55);  primals_38 = mul_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_31: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_36, [0, 2, 1]);  add_36 = None
    view_37: "f32[3072, 196]" = torch.ops.aten.view.default(permute_31, [3072, 196]);  permute_31 = None
    permute_32: "f32[196, 196]" = torch.ops.aten.permute.default(primals_113, [1, 0]);  primals_113 = None
    addmm_12: "f32[3072, 196]" = torch.ops.aten.addmm.default(primals_114, view_37, permute_32);  primals_114 = None
    view_38: "f32[8, 384, 196]" = torch.ops.aten.view.default(addmm_12, [8, 384, 196]);  addmm_12 = None
    permute_33: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_38, [0, 2, 1]);  view_38 = None
    mul_56: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(primals_37, permute_33)
    add_37: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_35, mul_56);  mul_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_57: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(primals_42, 1)
    mul_58: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_57, add_37);  mul_57 = None
    add_38: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(primals_41, mul_58);  primals_41 = mul_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_34: "f32[384, 1536]" = torch.ops.aten.permute.default(primals_115, [1, 0]);  primals_115 = None
    clone_18: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_38, memory_format = torch.contiguous_format);  add_38 = None
    view_39: "f32[1568, 384]" = torch.ops.aten.view.default(clone_18, [1568, 384]);  clone_18 = None
    mm_6: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_39, permute_34)
    view_40: "f32[8, 196, 1536]" = torch.ops.aten.view.default(mm_6, [8, 196, 1536]);  mm_6 = None
    add_39: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(view_40, primals_116);  view_40 = primals_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_59: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_39, 0.5)
    mul_60: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_39, 0.7071067811865476)
    erf_6: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_60);  mul_60 = None
    add_40: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_61: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_59, add_40);  mul_59 = add_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_19: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_61);  mul_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_41: "f32[1568, 1536]" = torch.ops.aten.view.default(clone_19, [1568, 1536]);  clone_19 = None
    permute_35: "f32[1536, 384]" = torch.ops.aten.permute.default(primals_117, [1, 0]);  primals_117 = None
    addmm_13: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_118, view_41, permute_35);  primals_118 = None
    view_42: "f32[8, 196, 384]" = torch.ops.aten.view.default(addmm_13, [8, 196, 384]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_20: "f32[8, 196, 384]" = torch.ops.aten.clone.default(view_42);  view_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    mul_62: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(primals_40, clone_20)
    add_41: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_37, mul_62);  mul_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_63: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(primals_45, 1)
    mul_64: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_63, add_41);  mul_63 = None
    add_42: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(primals_44, mul_64);  primals_44 = mul_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_36: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_42, [0, 2, 1]);  add_42 = None
    view_43: "f32[3072, 196]" = torch.ops.aten.view.default(permute_36, [3072, 196]);  permute_36 = None
    permute_37: "f32[196, 196]" = torch.ops.aten.permute.default(primals_119, [1, 0]);  primals_119 = None
    addmm_14: "f32[3072, 196]" = torch.ops.aten.addmm.default(primals_120, view_43, permute_37);  primals_120 = None
    view_44: "f32[8, 384, 196]" = torch.ops.aten.view.default(addmm_14, [8, 384, 196]);  addmm_14 = None
    permute_38: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_44, [0, 2, 1]);  view_44 = None
    mul_65: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(primals_43, permute_38)
    add_43: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_41, mul_65);  mul_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_66: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(primals_48, 1)
    mul_67: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_66, add_43);  mul_66 = None
    add_44: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(primals_47, mul_67);  primals_47 = mul_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_39: "f32[384, 1536]" = torch.ops.aten.permute.default(primals_121, [1, 0]);  primals_121 = None
    clone_21: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_44, memory_format = torch.contiguous_format);  add_44 = None
    view_45: "f32[1568, 384]" = torch.ops.aten.view.default(clone_21, [1568, 384]);  clone_21 = None
    mm_7: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_45, permute_39)
    view_46: "f32[8, 196, 1536]" = torch.ops.aten.view.default(mm_7, [8, 196, 1536]);  mm_7 = None
    add_45: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(view_46, primals_122);  view_46 = primals_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_68: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_45, 0.5)
    mul_69: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_45, 0.7071067811865476)
    erf_7: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_69);  mul_69 = None
    add_46: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_70: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_68, add_46);  mul_68 = add_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_22: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_70);  mul_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_47: "f32[1568, 1536]" = torch.ops.aten.view.default(clone_22, [1568, 1536]);  clone_22 = None
    permute_40: "f32[1536, 384]" = torch.ops.aten.permute.default(primals_123, [1, 0]);  primals_123 = None
    addmm_15: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_124, view_47, permute_40);  primals_124 = None
    view_48: "f32[8, 196, 384]" = torch.ops.aten.view.default(addmm_15, [8, 196, 384]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_23: "f32[8, 196, 384]" = torch.ops.aten.clone.default(view_48);  view_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    mul_71: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(primals_46, clone_23)
    add_47: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_43, mul_71);  mul_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_72: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(primals_51, 1)
    mul_73: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_72, add_47);  mul_72 = None
    add_48: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(primals_50, mul_73);  primals_50 = mul_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_41: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_48, [0, 2, 1]);  add_48 = None
    view_49: "f32[3072, 196]" = torch.ops.aten.view.default(permute_41, [3072, 196]);  permute_41 = None
    permute_42: "f32[196, 196]" = torch.ops.aten.permute.default(primals_125, [1, 0]);  primals_125 = None
    addmm_16: "f32[3072, 196]" = torch.ops.aten.addmm.default(primals_126, view_49, permute_42);  primals_126 = None
    view_50: "f32[8, 384, 196]" = torch.ops.aten.view.default(addmm_16, [8, 384, 196]);  addmm_16 = None
    permute_43: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_50, [0, 2, 1]);  view_50 = None
    mul_74: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(primals_49, permute_43)
    add_49: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_47, mul_74);  mul_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_75: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(primals_54, 1)
    mul_76: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_75, add_49);  mul_75 = None
    add_50: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(primals_53, mul_76);  primals_53 = mul_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_44: "f32[384, 1536]" = torch.ops.aten.permute.default(primals_127, [1, 0]);  primals_127 = None
    clone_24: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_50, memory_format = torch.contiguous_format);  add_50 = None
    view_51: "f32[1568, 384]" = torch.ops.aten.view.default(clone_24, [1568, 384]);  clone_24 = None
    mm_8: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_51, permute_44)
    view_52: "f32[8, 196, 1536]" = torch.ops.aten.view.default(mm_8, [8, 196, 1536]);  mm_8 = None
    add_51: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(view_52, primals_128);  view_52 = primals_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_77: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_51, 0.5)
    mul_78: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_51, 0.7071067811865476)
    erf_8: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_78);  mul_78 = None
    add_52: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_79: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_77, add_52);  mul_77 = add_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_25: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_79);  mul_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_53: "f32[1568, 1536]" = torch.ops.aten.view.default(clone_25, [1568, 1536]);  clone_25 = None
    permute_45: "f32[1536, 384]" = torch.ops.aten.permute.default(primals_129, [1, 0]);  primals_129 = None
    addmm_17: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_130, view_53, permute_45);  primals_130 = None
    view_54: "f32[8, 196, 384]" = torch.ops.aten.view.default(addmm_17, [8, 196, 384]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_26: "f32[8, 196, 384]" = torch.ops.aten.clone.default(view_54);  view_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    mul_80: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(primals_52, clone_26)
    add_53: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_49, mul_80);  mul_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_81: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(primals_57, 1)
    mul_82: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_81, add_53);  mul_81 = None
    add_54: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(primals_56, mul_82);  primals_56 = mul_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_46: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_54, [0, 2, 1]);  add_54 = None
    view_55: "f32[3072, 196]" = torch.ops.aten.view.default(permute_46, [3072, 196]);  permute_46 = None
    permute_47: "f32[196, 196]" = torch.ops.aten.permute.default(primals_131, [1, 0]);  primals_131 = None
    addmm_18: "f32[3072, 196]" = torch.ops.aten.addmm.default(primals_132, view_55, permute_47);  primals_132 = None
    view_56: "f32[8, 384, 196]" = torch.ops.aten.view.default(addmm_18, [8, 384, 196]);  addmm_18 = None
    permute_48: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_56, [0, 2, 1]);  view_56 = None
    mul_83: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(primals_55, permute_48)
    add_55: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_53, mul_83);  mul_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_84: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(primals_60, 1)
    mul_85: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_84, add_55);  mul_84 = None
    add_56: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(primals_59, mul_85);  primals_59 = mul_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_49: "f32[384, 1536]" = torch.ops.aten.permute.default(primals_133, [1, 0]);  primals_133 = None
    clone_27: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_56, memory_format = torch.contiguous_format);  add_56 = None
    view_57: "f32[1568, 384]" = torch.ops.aten.view.default(clone_27, [1568, 384]);  clone_27 = None
    mm_9: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_57, permute_49)
    view_58: "f32[8, 196, 1536]" = torch.ops.aten.view.default(mm_9, [8, 196, 1536]);  mm_9 = None
    add_57: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(view_58, primals_134);  view_58 = primals_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_86: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_57, 0.5)
    mul_87: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_57, 0.7071067811865476)
    erf_9: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_87);  mul_87 = None
    add_58: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_88: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_86, add_58);  mul_86 = add_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_28: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_88);  mul_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_59: "f32[1568, 1536]" = torch.ops.aten.view.default(clone_28, [1568, 1536]);  clone_28 = None
    permute_50: "f32[1536, 384]" = torch.ops.aten.permute.default(primals_135, [1, 0]);  primals_135 = None
    addmm_19: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_136, view_59, permute_50);  primals_136 = None
    view_60: "f32[8, 196, 384]" = torch.ops.aten.view.default(addmm_19, [8, 196, 384]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_29: "f32[8, 196, 384]" = torch.ops.aten.clone.default(view_60);  view_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    mul_89: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(primals_58, clone_29)
    add_59: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_55, mul_89);  mul_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_90: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(primals_63, 1)
    mul_91: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_90, add_59);  mul_90 = None
    add_60: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(primals_62, mul_91);  primals_62 = mul_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_51: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_60, [0, 2, 1]);  add_60 = None
    view_61: "f32[3072, 196]" = torch.ops.aten.view.default(permute_51, [3072, 196]);  permute_51 = None
    permute_52: "f32[196, 196]" = torch.ops.aten.permute.default(primals_137, [1, 0]);  primals_137 = None
    addmm_20: "f32[3072, 196]" = torch.ops.aten.addmm.default(primals_138, view_61, permute_52);  primals_138 = None
    view_62: "f32[8, 384, 196]" = torch.ops.aten.view.default(addmm_20, [8, 384, 196]);  addmm_20 = None
    permute_53: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_62, [0, 2, 1]);  view_62 = None
    mul_92: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(primals_61, permute_53)
    add_61: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_59, mul_92);  mul_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_93: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(primals_66, 1)
    mul_94: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_93, add_61);  mul_93 = None
    add_62: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(primals_65, mul_94);  primals_65 = mul_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_54: "f32[384, 1536]" = torch.ops.aten.permute.default(primals_139, [1, 0]);  primals_139 = None
    clone_30: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_62, memory_format = torch.contiguous_format);  add_62 = None
    view_63: "f32[1568, 384]" = torch.ops.aten.view.default(clone_30, [1568, 384]);  clone_30 = None
    mm_10: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_63, permute_54)
    view_64: "f32[8, 196, 1536]" = torch.ops.aten.view.default(mm_10, [8, 196, 1536]);  mm_10 = None
    add_63: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(view_64, primals_140);  view_64 = primals_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_95: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_63, 0.5)
    mul_96: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_63, 0.7071067811865476)
    erf_10: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_96);  mul_96 = None
    add_64: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_97: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_95, add_64);  mul_95 = add_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_31: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_97);  mul_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_65: "f32[1568, 1536]" = torch.ops.aten.view.default(clone_31, [1568, 1536]);  clone_31 = None
    permute_55: "f32[1536, 384]" = torch.ops.aten.permute.default(primals_141, [1, 0]);  primals_141 = None
    addmm_21: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_142, view_65, permute_55);  primals_142 = None
    view_66: "f32[8, 196, 384]" = torch.ops.aten.view.default(addmm_21, [8, 196, 384]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_32: "f32[8, 196, 384]" = torch.ops.aten.clone.default(view_66);  view_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    mul_98: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(primals_64, clone_32)
    add_65: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_61, mul_98);  mul_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_99: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(primals_69, 1)
    mul_100: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_99, add_65);  mul_99 = None
    add_66: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(primals_68, mul_100);  primals_68 = mul_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_56: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_66, [0, 2, 1]);  add_66 = None
    view_67: "f32[3072, 196]" = torch.ops.aten.view.default(permute_56, [3072, 196]);  permute_56 = None
    permute_57: "f32[196, 196]" = torch.ops.aten.permute.default(primals_143, [1, 0]);  primals_143 = None
    addmm_22: "f32[3072, 196]" = torch.ops.aten.addmm.default(primals_144, view_67, permute_57);  primals_144 = None
    view_68: "f32[8, 384, 196]" = torch.ops.aten.view.default(addmm_22, [8, 384, 196]);  addmm_22 = None
    permute_58: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_68, [0, 2, 1]);  view_68 = None
    mul_101: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(primals_67, permute_58)
    add_67: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_65, mul_101);  mul_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_102: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(primals_72, 1)
    mul_103: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_102, add_67);  mul_102 = None
    add_68: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(primals_71, mul_103);  primals_71 = mul_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_59: "f32[384, 1536]" = torch.ops.aten.permute.default(primals_145, [1, 0]);  primals_145 = None
    clone_33: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_68, memory_format = torch.contiguous_format);  add_68 = None
    view_69: "f32[1568, 384]" = torch.ops.aten.view.default(clone_33, [1568, 384]);  clone_33 = None
    mm_11: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_69, permute_59)
    view_70: "f32[8, 196, 1536]" = torch.ops.aten.view.default(mm_11, [8, 196, 1536]);  mm_11 = None
    add_69: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(view_70, primals_146);  view_70 = primals_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_104: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_69, 0.5)
    mul_105: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_69, 0.7071067811865476)
    erf_11: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_105);  mul_105 = None
    add_70: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_106: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_104, add_70);  mul_104 = add_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_34: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_106);  mul_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_71: "f32[1568, 1536]" = torch.ops.aten.view.default(clone_34, [1568, 1536]);  clone_34 = None
    permute_60: "f32[1536, 384]" = torch.ops.aten.permute.default(primals_147, [1, 0]);  primals_147 = None
    addmm_23: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_148, view_71, permute_60);  primals_148 = None
    view_72: "f32[8, 196, 384]" = torch.ops.aten.view.default(addmm_23, [8, 196, 384]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_35: "f32[8, 196, 384]" = torch.ops.aten.clone.default(view_72);  view_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    mul_107: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(primals_70, clone_35)
    add_71: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_67, mul_107);  mul_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_108: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(primals_74, 1)
    mul_109: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_108, add_71);  mul_108 = None
    add_72: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(primals_73, mul_109);  primals_73 = mul_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:271, code: x = x.mean(dim=1)
    mean: "f32[8, 384]" = torch.ops.aten.mean.dim(add_72, [1]);  add_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:272, code: x = self.head_drop(x)
    clone_36: "f32[8, 384]" = torch.ops.aten.clone.default(mean);  mean = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:273, code: return x if pre_logits else self.head(x)
    permute_61: "f32[384, 1000]" = torch.ops.aten.permute.default(primals_149, [1, 0]);  primals_149 = None
    addmm_24: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_150, clone_36, permute_61);  primals_150 = None
    permute_62: "f32[1000, 384]" = torch.ops.aten.permute.default(permute_61, [1, 0]);  permute_61 = None
    mm_12: "f32[8, 384]" = torch.ops.aten.mm.default(tangents_1, permute_62);  permute_62 = None
    permute_63: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_13: "f32[1000, 384]" = torch.ops.aten.mm.default(permute_63, clone_36);  permute_63 = clone_36 = None
    permute_64: "f32[384, 1000]" = torch.ops.aten.permute.default(mm_13, [1, 0]);  mm_13 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_73: "f32[1000]" = torch.ops.aten.view.default(sum_1, [1000]);  sum_1 = None
    permute_65: "f32[1000, 384]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:271, code: x = x.mean(dim=1)
    unsqueeze: "f32[8, 1, 384]" = torch.ops.aten.unsqueeze.default(mm_12, 1);  mm_12 = None
    expand: "f32[8, 196, 384]" = torch.ops.aten.expand.default(unsqueeze, [8, 196, 384]);  unsqueeze = None
    div: "f32[8, 196, 384]" = torch.ops.aten.div.Scalar(expand, 196);  expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_110: "f32[8, 196, 384]" = torch.ops.aten.mul.Scalar(add_71, 1);  add_71 = None
    mul_111: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(div, mul_110);  mul_110 = None
    mul_112: "f32[1, 1, 384]" = torch.ops.aten.mul.Scalar(primals_74, 1);  primals_74 = None
    mul_113: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(div, mul_112);  mul_112 = None
    sum_2: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(div, [0, 1], True);  div = None
    sum_3: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(mul_111, [0, 1], True);  mul_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    mul_114: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_113, primals_70);  primals_70 = None
    mul_115: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_113, clone_35);  clone_35 = None
    sum_4: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(mul_115, [0, 1], True);  mul_115 = None
    view_74: "f32[384]" = torch.ops.aten.view.default(sum_4, [384]);  sum_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_75: "f32[1568, 384]" = torch.ops.aten.view.default(mul_114, [1568, 384]);  mul_114 = None
    permute_66: "f32[384, 1536]" = torch.ops.aten.permute.default(permute_60, [1, 0]);  permute_60 = None
    mm_14: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_75, permute_66);  permute_66 = None
    permute_67: "f32[384, 1568]" = torch.ops.aten.permute.default(view_75, [1, 0])
    mm_15: "f32[384, 1536]" = torch.ops.aten.mm.default(permute_67, view_71);  permute_67 = view_71 = None
    permute_68: "f32[1536, 384]" = torch.ops.aten.permute.default(mm_15, [1, 0]);  mm_15 = None
    sum_5: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_75, [0], True);  view_75 = None
    view_76: "f32[384]" = torch.ops.aten.view.default(sum_5, [384]);  sum_5 = None
    permute_69: "f32[384, 1536]" = torch.ops.aten.permute.default(permute_68, [1, 0]);  permute_68 = None
    view_77: "f32[8, 196, 1536]" = torch.ops.aten.view.default(mm_14, [8, 196, 1536]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_116: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_69, 0.7071067811865476)
    erf_12: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_116);  mul_116 = None
    add_73: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_117: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_73, 0.5);  add_73 = None
    mul_118: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_69, add_69)
    mul_119: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_118, -0.5);  mul_118 = None
    exp: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_119);  mul_119 = None
    mul_120: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
    mul_121: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_69, mul_120);  add_69 = mul_120 = None
    add_74: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_117, mul_121);  mul_117 = mul_121 = None
    mul_122: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_77, add_74);  view_77 = add_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_6: "f32[1, 1, 1536]" = torch.ops.aten.sum.dim_IntList(mul_122, [0, 1], True)
    view_78: "f32[1536]" = torch.ops.aten.view.default(sum_6, [1536]);  sum_6 = None
    view_79: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_122, [1568, 1536]);  mul_122 = None
    permute_70: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_79, [1, 0])
    mm_16: "f32[1536, 384]" = torch.ops.aten.mm.default(permute_70, view_69);  permute_70 = view_69 = None
    permute_71: "f32[384, 1536]" = torch.ops.aten.permute.default(mm_16, [1, 0]);  mm_16 = None
    permute_72: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None
    mm_17: "f32[1568, 384]" = torch.ops.aten.mm.default(view_79, permute_72);  view_79 = permute_72 = None
    view_80: "f32[8, 196, 384]" = torch.ops.aten.view.default(mm_17, [8, 196, 384]);  mm_17 = None
    permute_73: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_71, [1, 0]);  permute_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_123: "f32[8, 196, 384]" = torch.ops.aten.mul.Scalar(add_67, 1);  add_67 = None
    mul_124: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_80, mul_123);  mul_123 = None
    mul_125: "f32[1, 1, 384]" = torch.ops.aten.mul.Scalar(primals_72, 1);  primals_72 = None
    mul_126: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_80, mul_125);  mul_125 = None
    sum_7: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(view_80, [0, 1], True);  view_80 = None
    sum_8: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(mul_124, [0, 1], True);  mul_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    add_75: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_113, mul_126);  mul_113 = mul_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    mul_127: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(add_75, primals_67);  primals_67 = None
    mul_128: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(add_75, permute_58);  permute_58 = None
    sum_9: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(mul_128, [0, 1], True);  mul_128 = None
    view_81: "f32[384]" = torch.ops.aten.view.default(sum_9, [384]);  sum_9 = None
    permute_74: "f32[8, 384, 196]" = torch.ops.aten.permute.default(mul_127, [0, 2, 1]);  mul_127 = None
    clone_37: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_74, memory_format = torch.contiguous_format);  permute_74 = None
    view_82: "f32[3072, 196]" = torch.ops.aten.view.default(clone_37, [3072, 196]);  clone_37 = None
    permute_75: "f32[196, 196]" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
    mm_18: "f32[3072, 196]" = torch.ops.aten.mm.default(view_82, permute_75);  permute_75 = None
    permute_76: "f32[196, 3072]" = torch.ops.aten.permute.default(view_82, [1, 0])
    mm_19: "f32[196, 196]" = torch.ops.aten.mm.default(permute_76, view_67);  permute_76 = view_67 = None
    permute_77: "f32[196, 196]" = torch.ops.aten.permute.default(mm_19, [1, 0]);  mm_19 = None
    sum_10: "f32[1, 196]" = torch.ops.aten.sum.dim_IntList(view_82, [0], True);  view_82 = None
    view_83: "f32[196]" = torch.ops.aten.view.default(sum_10, [196]);  sum_10 = None
    permute_78: "f32[196, 196]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
    view_84: "f32[8, 384, 196]" = torch.ops.aten.view.default(mm_18, [8, 384, 196]);  mm_18 = None
    permute_79: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_84, [0, 2, 1]);  view_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_129: "f32[8, 196, 384]" = torch.ops.aten.mul.Scalar(add_65, 1);  add_65 = None
    mul_130: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(permute_79, mul_129);  mul_129 = None
    mul_131: "f32[1, 1, 384]" = torch.ops.aten.mul.Scalar(primals_69, 1);  primals_69 = None
    mul_132: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(permute_79, mul_131);  mul_131 = None
    sum_11: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(permute_79, [0, 1], True);  permute_79 = None
    sum_12: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(mul_130, [0, 1], True);  mul_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    add_76: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_75, mul_132);  add_75 = mul_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    mul_133: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(add_76, primals_64);  primals_64 = None
    mul_134: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(add_76, clone_32);  clone_32 = None
    sum_13: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(mul_134, [0, 1], True);  mul_134 = None
    view_85: "f32[384]" = torch.ops.aten.view.default(sum_13, [384]);  sum_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_86: "f32[1568, 384]" = torch.ops.aten.view.default(mul_133, [1568, 384]);  mul_133 = None
    permute_80: "f32[384, 1536]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
    mm_20: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_86, permute_80);  permute_80 = None
    permute_81: "f32[384, 1568]" = torch.ops.aten.permute.default(view_86, [1, 0])
    mm_21: "f32[384, 1536]" = torch.ops.aten.mm.default(permute_81, view_65);  permute_81 = view_65 = None
    permute_82: "f32[1536, 384]" = torch.ops.aten.permute.default(mm_21, [1, 0]);  mm_21 = None
    sum_14: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_86, [0], True);  view_86 = None
    view_87: "f32[384]" = torch.ops.aten.view.default(sum_14, [384]);  sum_14 = None
    permute_83: "f32[384, 1536]" = torch.ops.aten.permute.default(permute_82, [1, 0]);  permute_82 = None
    view_88: "f32[8, 196, 1536]" = torch.ops.aten.view.default(mm_20, [8, 196, 1536]);  mm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_135: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_63, 0.7071067811865476)
    erf_13: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_135);  mul_135 = None
    add_77: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    mul_136: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_77, 0.5);  add_77 = None
    mul_137: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_63, add_63)
    mul_138: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_137, -0.5);  mul_137 = None
    exp_1: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_138);  mul_138 = None
    mul_139: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_1, 0.3989422804014327);  exp_1 = None
    mul_140: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_63, mul_139);  add_63 = mul_139 = None
    add_78: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_136, mul_140);  mul_136 = mul_140 = None
    mul_141: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_88, add_78);  view_88 = add_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_15: "f32[1, 1, 1536]" = torch.ops.aten.sum.dim_IntList(mul_141, [0, 1], True)
    view_89: "f32[1536]" = torch.ops.aten.view.default(sum_15, [1536]);  sum_15 = None
    view_90: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_141, [1568, 1536]);  mul_141 = None
    permute_84: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_90, [1, 0])
    mm_22: "f32[1536, 384]" = torch.ops.aten.mm.default(permute_84, view_63);  permute_84 = view_63 = None
    permute_85: "f32[384, 1536]" = torch.ops.aten.permute.default(mm_22, [1, 0]);  mm_22 = None
    permute_86: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
    mm_23: "f32[1568, 384]" = torch.ops.aten.mm.default(view_90, permute_86);  view_90 = permute_86 = None
    view_91: "f32[8, 196, 384]" = torch.ops.aten.view.default(mm_23, [8, 196, 384]);  mm_23 = None
    permute_87: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_142: "f32[8, 196, 384]" = torch.ops.aten.mul.Scalar(add_61, 1);  add_61 = None
    mul_143: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_91, mul_142);  mul_142 = None
    mul_144: "f32[1, 1, 384]" = torch.ops.aten.mul.Scalar(primals_66, 1);  primals_66 = None
    mul_145: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_91, mul_144);  mul_144 = None
    sum_16: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(view_91, [0, 1], True);  view_91 = None
    sum_17: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(mul_143, [0, 1], True);  mul_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    add_79: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_76, mul_145);  add_76 = mul_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    mul_146: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(add_79, primals_61);  primals_61 = None
    mul_147: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(add_79, permute_53);  permute_53 = None
    sum_18: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(mul_147, [0, 1], True);  mul_147 = None
    view_92: "f32[384]" = torch.ops.aten.view.default(sum_18, [384]);  sum_18 = None
    permute_88: "f32[8, 384, 196]" = torch.ops.aten.permute.default(mul_146, [0, 2, 1]);  mul_146 = None
    clone_38: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_88, memory_format = torch.contiguous_format);  permute_88 = None
    view_93: "f32[3072, 196]" = torch.ops.aten.view.default(clone_38, [3072, 196]);  clone_38 = None
    permute_89: "f32[196, 196]" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
    mm_24: "f32[3072, 196]" = torch.ops.aten.mm.default(view_93, permute_89);  permute_89 = None
    permute_90: "f32[196, 3072]" = torch.ops.aten.permute.default(view_93, [1, 0])
    mm_25: "f32[196, 196]" = torch.ops.aten.mm.default(permute_90, view_61);  permute_90 = view_61 = None
    permute_91: "f32[196, 196]" = torch.ops.aten.permute.default(mm_25, [1, 0]);  mm_25 = None
    sum_19: "f32[1, 196]" = torch.ops.aten.sum.dim_IntList(view_93, [0], True);  view_93 = None
    view_94: "f32[196]" = torch.ops.aten.view.default(sum_19, [196]);  sum_19 = None
    permute_92: "f32[196, 196]" = torch.ops.aten.permute.default(permute_91, [1, 0]);  permute_91 = None
    view_95: "f32[8, 384, 196]" = torch.ops.aten.view.default(mm_24, [8, 384, 196]);  mm_24 = None
    permute_93: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_95, [0, 2, 1]);  view_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_148: "f32[8, 196, 384]" = torch.ops.aten.mul.Scalar(add_59, 1);  add_59 = None
    mul_149: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(permute_93, mul_148);  mul_148 = None
    mul_150: "f32[1, 1, 384]" = torch.ops.aten.mul.Scalar(primals_63, 1);  primals_63 = None
    mul_151: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(permute_93, mul_150);  mul_150 = None
    sum_20: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(permute_93, [0, 1], True);  permute_93 = None
    sum_21: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(mul_149, [0, 1], True);  mul_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    add_80: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_79, mul_151);  add_79 = mul_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    mul_152: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(add_80, primals_58);  primals_58 = None
    mul_153: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(add_80, clone_29);  clone_29 = None
    sum_22: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(mul_153, [0, 1], True);  mul_153 = None
    view_96: "f32[384]" = torch.ops.aten.view.default(sum_22, [384]);  sum_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_97: "f32[1568, 384]" = torch.ops.aten.view.default(mul_152, [1568, 384]);  mul_152 = None
    permute_94: "f32[384, 1536]" = torch.ops.aten.permute.default(permute_50, [1, 0]);  permute_50 = None
    mm_26: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_97, permute_94);  permute_94 = None
    permute_95: "f32[384, 1568]" = torch.ops.aten.permute.default(view_97, [1, 0])
    mm_27: "f32[384, 1536]" = torch.ops.aten.mm.default(permute_95, view_59);  permute_95 = view_59 = None
    permute_96: "f32[1536, 384]" = torch.ops.aten.permute.default(mm_27, [1, 0]);  mm_27 = None
    sum_23: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_97, [0], True);  view_97 = None
    view_98: "f32[384]" = torch.ops.aten.view.default(sum_23, [384]);  sum_23 = None
    permute_97: "f32[384, 1536]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
    view_99: "f32[8, 196, 1536]" = torch.ops.aten.view.default(mm_26, [8, 196, 1536]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_154: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_57, 0.7071067811865476)
    erf_14: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_154);  mul_154 = None
    add_81: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    mul_155: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_81, 0.5);  add_81 = None
    mul_156: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_57, add_57)
    mul_157: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_156, -0.5);  mul_156 = None
    exp_2: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_157);  mul_157 = None
    mul_158: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_2, 0.3989422804014327);  exp_2 = None
    mul_159: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_57, mul_158);  add_57 = mul_158 = None
    add_82: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_155, mul_159);  mul_155 = mul_159 = None
    mul_160: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_99, add_82);  view_99 = add_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_24: "f32[1, 1, 1536]" = torch.ops.aten.sum.dim_IntList(mul_160, [0, 1], True)
    view_100: "f32[1536]" = torch.ops.aten.view.default(sum_24, [1536]);  sum_24 = None
    view_101: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_160, [1568, 1536]);  mul_160 = None
    permute_98: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_101, [1, 0])
    mm_28: "f32[1536, 384]" = torch.ops.aten.mm.default(permute_98, view_57);  permute_98 = view_57 = None
    permute_99: "f32[384, 1536]" = torch.ops.aten.permute.default(mm_28, [1, 0]);  mm_28 = None
    permute_100: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_49, [1, 0]);  permute_49 = None
    mm_29: "f32[1568, 384]" = torch.ops.aten.mm.default(view_101, permute_100);  view_101 = permute_100 = None
    view_102: "f32[8, 196, 384]" = torch.ops.aten.view.default(mm_29, [8, 196, 384]);  mm_29 = None
    permute_101: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_161: "f32[8, 196, 384]" = torch.ops.aten.mul.Scalar(add_55, 1);  add_55 = None
    mul_162: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_102, mul_161);  mul_161 = None
    mul_163: "f32[1, 1, 384]" = torch.ops.aten.mul.Scalar(primals_60, 1);  primals_60 = None
    mul_164: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_102, mul_163);  mul_163 = None
    sum_25: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(view_102, [0, 1], True);  view_102 = None
    sum_26: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(mul_162, [0, 1], True);  mul_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    add_83: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_80, mul_164);  add_80 = mul_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    mul_165: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(add_83, primals_55);  primals_55 = None
    mul_166: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(add_83, permute_48);  permute_48 = None
    sum_27: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(mul_166, [0, 1], True);  mul_166 = None
    view_103: "f32[384]" = torch.ops.aten.view.default(sum_27, [384]);  sum_27 = None
    permute_102: "f32[8, 384, 196]" = torch.ops.aten.permute.default(mul_165, [0, 2, 1]);  mul_165 = None
    clone_39: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_102, memory_format = torch.contiguous_format);  permute_102 = None
    view_104: "f32[3072, 196]" = torch.ops.aten.view.default(clone_39, [3072, 196]);  clone_39 = None
    permute_103: "f32[196, 196]" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
    mm_30: "f32[3072, 196]" = torch.ops.aten.mm.default(view_104, permute_103);  permute_103 = None
    permute_104: "f32[196, 3072]" = torch.ops.aten.permute.default(view_104, [1, 0])
    mm_31: "f32[196, 196]" = torch.ops.aten.mm.default(permute_104, view_55);  permute_104 = view_55 = None
    permute_105: "f32[196, 196]" = torch.ops.aten.permute.default(mm_31, [1, 0]);  mm_31 = None
    sum_28: "f32[1, 196]" = torch.ops.aten.sum.dim_IntList(view_104, [0], True);  view_104 = None
    view_105: "f32[196]" = torch.ops.aten.view.default(sum_28, [196]);  sum_28 = None
    permute_106: "f32[196, 196]" = torch.ops.aten.permute.default(permute_105, [1, 0]);  permute_105 = None
    view_106: "f32[8, 384, 196]" = torch.ops.aten.view.default(mm_30, [8, 384, 196]);  mm_30 = None
    permute_107: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_106, [0, 2, 1]);  view_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_167: "f32[8, 196, 384]" = torch.ops.aten.mul.Scalar(add_53, 1);  add_53 = None
    mul_168: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(permute_107, mul_167);  mul_167 = None
    mul_169: "f32[1, 1, 384]" = torch.ops.aten.mul.Scalar(primals_57, 1);  primals_57 = None
    mul_170: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(permute_107, mul_169);  mul_169 = None
    sum_29: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(permute_107, [0, 1], True);  permute_107 = None
    sum_30: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(mul_168, [0, 1], True);  mul_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    add_84: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_83, mul_170);  add_83 = mul_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    mul_171: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(add_84, primals_52);  primals_52 = None
    mul_172: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(add_84, clone_26);  clone_26 = None
    sum_31: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(mul_172, [0, 1], True);  mul_172 = None
    view_107: "f32[384]" = torch.ops.aten.view.default(sum_31, [384]);  sum_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_108: "f32[1568, 384]" = torch.ops.aten.view.default(mul_171, [1568, 384]);  mul_171 = None
    permute_108: "f32[384, 1536]" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
    mm_32: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_108, permute_108);  permute_108 = None
    permute_109: "f32[384, 1568]" = torch.ops.aten.permute.default(view_108, [1, 0])
    mm_33: "f32[384, 1536]" = torch.ops.aten.mm.default(permute_109, view_53);  permute_109 = view_53 = None
    permute_110: "f32[1536, 384]" = torch.ops.aten.permute.default(mm_33, [1, 0]);  mm_33 = None
    sum_32: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_108, [0], True);  view_108 = None
    view_109: "f32[384]" = torch.ops.aten.view.default(sum_32, [384]);  sum_32 = None
    permute_111: "f32[384, 1536]" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
    view_110: "f32[8, 196, 1536]" = torch.ops.aten.view.default(mm_32, [8, 196, 1536]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_173: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_51, 0.7071067811865476)
    erf_15: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_173);  mul_173 = None
    add_85: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    mul_174: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_85, 0.5);  add_85 = None
    mul_175: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_51, add_51)
    mul_176: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_175, -0.5);  mul_175 = None
    exp_3: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_176);  mul_176 = None
    mul_177: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_3, 0.3989422804014327);  exp_3 = None
    mul_178: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_51, mul_177);  add_51 = mul_177 = None
    add_86: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_174, mul_178);  mul_174 = mul_178 = None
    mul_179: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_110, add_86);  view_110 = add_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_33: "f32[1, 1, 1536]" = torch.ops.aten.sum.dim_IntList(mul_179, [0, 1], True)
    view_111: "f32[1536]" = torch.ops.aten.view.default(sum_33, [1536]);  sum_33 = None
    view_112: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_179, [1568, 1536]);  mul_179 = None
    permute_112: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_112, [1, 0])
    mm_34: "f32[1536, 384]" = torch.ops.aten.mm.default(permute_112, view_51);  permute_112 = view_51 = None
    permute_113: "f32[384, 1536]" = torch.ops.aten.permute.default(mm_34, [1, 0]);  mm_34 = None
    permute_114: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
    mm_35: "f32[1568, 384]" = torch.ops.aten.mm.default(view_112, permute_114);  view_112 = permute_114 = None
    view_113: "f32[8, 196, 384]" = torch.ops.aten.view.default(mm_35, [8, 196, 384]);  mm_35 = None
    permute_115: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_113, [1, 0]);  permute_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_180: "f32[8, 196, 384]" = torch.ops.aten.mul.Scalar(add_49, 1);  add_49 = None
    mul_181: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_113, mul_180);  mul_180 = None
    mul_182: "f32[1, 1, 384]" = torch.ops.aten.mul.Scalar(primals_54, 1);  primals_54 = None
    mul_183: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_113, mul_182);  mul_182 = None
    sum_34: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(view_113, [0, 1], True);  view_113 = None
    sum_35: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(mul_181, [0, 1], True);  mul_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    add_87: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_84, mul_183);  add_84 = mul_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    mul_184: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(add_87, primals_49);  primals_49 = None
    mul_185: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(add_87, permute_43);  permute_43 = None
    sum_36: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(mul_185, [0, 1], True);  mul_185 = None
    view_114: "f32[384]" = torch.ops.aten.view.default(sum_36, [384]);  sum_36 = None
    permute_116: "f32[8, 384, 196]" = torch.ops.aten.permute.default(mul_184, [0, 2, 1]);  mul_184 = None
    clone_40: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_116, memory_format = torch.contiguous_format);  permute_116 = None
    view_115: "f32[3072, 196]" = torch.ops.aten.view.default(clone_40, [3072, 196]);  clone_40 = None
    permute_117: "f32[196, 196]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
    mm_36: "f32[3072, 196]" = torch.ops.aten.mm.default(view_115, permute_117);  permute_117 = None
    permute_118: "f32[196, 3072]" = torch.ops.aten.permute.default(view_115, [1, 0])
    mm_37: "f32[196, 196]" = torch.ops.aten.mm.default(permute_118, view_49);  permute_118 = view_49 = None
    permute_119: "f32[196, 196]" = torch.ops.aten.permute.default(mm_37, [1, 0]);  mm_37 = None
    sum_37: "f32[1, 196]" = torch.ops.aten.sum.dim_IntList(view_115, [0], True);  view_115 = None
    view_116: "f32[196]" = torch.ops.aten.view.default(sum_37, [196]);  sum_37 = None
    permute_120: "f32[196, 196]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
    view_117: "f32[8, 384, 196]" = torch.ops.aten.view.default(mm_36, [8, 384, 196]);  mm_36 = None
    permute_121: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_117, [0, 2, 1]);  view_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_186: "f32[8, 196, 384]" = torch.ops.aten.mul.Scalar(add_47, 1);  add_47 = None
    mul_187: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(permute_121, mul_186);  mul_186 = None
    mul_188: "f32[1, 1, 384]" = torch.ops.aten.mul.Scalar(primals_51, 1);  primals_51 = None
    mul_189: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(permute_121, mul_188);  mul_188 = None
    sum_38: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(permute_121, [0, 1], True);  permute_121 = None
    sum_39: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(mul_187, [0, 1], True);  mul_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    add_88: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_87, mul_189);  add_87 = mul_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    mul_190: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(add_88, primals_46);  primals_46 = None
    mul_191: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(add_88, clone_23);  clone_23 = None
    sum_40: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(mul_191, [0, 1], True);  mul_191 = None
    view_118: "f32[384]" = torch.ops.aten.view.default(sum_40, [384]);  sum_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_119: "f32[1568, 384]" = torch.ops.aten.view.default(mul_190, [1568, 384]);  mul_190 = None
    permute_122: "f32[384, 1536]" = torch.ops.aten.permute.default(permute_40, [1, 0]);  permute_40 = None
    mm_38: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_119, permute_122);  permute_122 = None
    permute_123: "f32[384, 1568]" = torch.ops.aten.permute.default(view_119, [1, 0])
    mm_39: "f32[384, 1536]" = torch.ops.aten.mm.default(permute_123, view_47);  permute_123 = view_47 = None
    permute_124: "f32[1536, 384]" = torch.ops.aten.permute.default(mm_39, [1, 0]);  mm_39 = None
    sum_41: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_119, [0], True);  view_119 = None
    view_120: "f32[384]" = torch.ops.aten.view.default(sum_41, [384]);  sum_41 = None
    permute_125: "f32[384, 1536]" = torch.ops.aten.permute.default(permute_124, [1, 0]);  permute_124 = None
    view_121: "f32[8, 196, 1536]" = torch.ops.aten.view.default(mm_38, [8, 196, 1536]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_192: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_45, 0.7071067811865476)
    erf_16: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_192);  mul_192 = None
    add_89: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    mul_193: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_89, 0.5);  add_89 = None
    mul_194: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_45, add_45)
    mul_195: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_194, -0.5);  mul_194 = None
    exp_4: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_195);  mul_195 = None
    mul_196: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_4, 0.3989422804014327);  exp_4 = None
    mul_197: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_45, mul_196);  add_45 = mul_196 = None
    add_90: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_193, mul_197);  mul_193 = mul_197 = None
    mul_198: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_121, add_90);  view_121 = add_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_42: "f32[1, 1, 1536]" = torch.ops.aten.sum.dim_IntList(mul_198, [0, 1], True)
    view_122: "f32[1536]" = torch.ops.aten.view.default(sum_42, [1536]);  sum_42 = None
    view_123: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_198, [1568, 1536]);  mul_198 = None
    permute_126: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_123, [1, 0])
    mm_40: "f32[1536, 384]" = torch.ops.aten.mm.default(permute_126, view_45);  permute_126 = view_45 = None
    permute_127: "f32[384, 1536]" = torch.ops.aten.permute.default(mm_40, [1, 0]);  mm_40 = None
    permute_128: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_39, [1, 0]);  permute_39 = None
    mm_41: "f32[1568, 384]" = torch.ops.aten.mm.default(view_123, permute_128);  view_123 = permute_128 = None
    view_124: "f32[8, 196, 384]" = torch.ops.aten.view.default(mm_41, [8, 196, 384]);  mm_41 = None
    permute_129: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_127, [1, 0]);  permute_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_199: "f32[8, 196, 384]" = torch.ops.aten.mul.Scalar(add_43, 1);  add_43 = None
    mul_200: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_124, mul_199);  mul_199 = None
    mul_201: "f32[1, 1, 384]" = torch.ops.aten.mul.Scalar(primals_48, 1);  primals_48 = None
    mul_202: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_124, mul_201);  mul_201 = None
    sum_43: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(view_124, [0, 1], True);  view_124 = None
    sum_44: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(mul_200, [0, 1], True);  mul_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    add_91: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_88, mul_202);  add_88 = mul_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    mul_203: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(add_91, primals_43);  primals_43 = None
    mul_204: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(add_91, permute_38);  permute_38 = None
    sum_45: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(mul_204, [0, 1], True);  mul_204 = None
    view_125: "f32[384]" = torch.ops.aten.view.default(sum_45, [384]);  sum_45 = None
    permute_130: "f32[8, 384, 196]" = torch.ops.aten.permute.default(mul_203, [0, 2, 1]);  mul_203 = None
    clone_41: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_130, memory_format = torch.contiguous_format);  permute_130 = None
    view_126: "f32[3072, 196]" = torch.ops.aten.view.default(clone_41, [3072, 196]);  clone_41 = None
    permute_131: "f32[196, 196]" = torch.ops.aten.permute.default(permute_37, [1, 0]);  permute_37 = None
    mm_42: "f32[3072, 196]" = torch.ops.aten.mm.default(view_126, permute_131);  permute_131 = None
    permute_132: "f32[196, 3072]" = torch.ops.aten.permute.default(view_126, [1, 0])
    mm_43: "f32[196, 196]" = torch.ops.aten.mm.default(permute_132, view_43);  permute_132 = view_43 = None
    permute_133: "f32[196, 196]" = torch.ops.aten.permute.default(mm_43, [1, 0]);  mm_43 = None
    sum_46: "f32[1, 196]" = torch.ops.aten.sum.dim_IntList(view_126, [0], True);  view_126 = None
    view_127: "f32[196]" = torch.ops.aten.view.default(sum_46, [196]);  sum_46 = None
    permute_134: "f32[196, 196]" = torch.ops.aten.permute.default(permute_133, [1, 0]);  permute_133 = None
    view_128: "f32[8, 384, 196]" = torch.ops.aten.view.default(mm_42, [8, 384, 196]);  mm_42 = None
    permute_135: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_128, [0, 2, 1]);  view_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_205: "f32[8, 196, 384]" = torch.ops.aten.mul.Scalar(add_41, 1);  add_41 = None
    mul_206: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(permute_135, mul_205);  mul_205 = None
    mul_207: "f32[1, 1, 384]" = torch.ops.aten.mul.Scalar(primals_45, 1);  primals_45 = None
    mul_208: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(permute_135, mul_207);  mul_207 = None
    sum_47: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(permute_135, [0, 1], True);  permute_135 = None
    sum_48: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(mul_206, [0, 1], True);  mul_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    add_92: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_91, mul_208);  add_91 = mul_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    mul_209: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(add_92, primals_40);  primals_40 = None
    mul_210: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(add_92, clone_20);  clone_20 = None
    sum_49: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(mul_210, [0, 1], True);  mul_210 = None
    view_129: "f32[384]" = torch.ops.aten.view.default(sum_49, [384]);  sum_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_130: "f32[1568, 384]" = torch.ops.aten.view.default(mul_209, [1568, 384]);  mul_209 = None
    permute_136: "f32[384, 1536]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
    mm_44: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_130, permute_136);  permute_136 = None
    permute_137: "f32[384, 1568]" = torch.ops.aten.permute.default(view_130, [1, 0])
    mm_45: "f32[384, 1536]" = torch.ops.aten.mm.default(permute_137, view_41);  permute_137 = view_41 = None
    permute_138: "f32[1536, 384]" = torch.ops.aten.permute.default(mm_45, [1, 0]);  mm_45 = None
    sum_50: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_130, [0], True);  view_130 = None
    view_131: "f32[384]" = torch.ops.aten.view.default(sum_50, [384]);  sum_50 = None
    permute_139: "f32[384, 1536]" = torch.ops.aten.permute.default(permute_138, [1, 0]);  permute_138 = None
    view_132: "f32[8, 196, 1536]" = torch.ops.aten.view.default(mm_44, [8, 196, 1536]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_211: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_39, 0.7071067811865476)
    erf_17: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_211);  mul_211 = None
    add_93: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    mul_212: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_93, 0.5);  add_93 = None
    mul_213: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_39, add_39)
    mul_214: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_213, -0.5);  mul_213 = None
    exp_5: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_214);  mul_214 = None
    mul_215: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_5, 0.3989422804014327);  exp_5 = None
    mul_216: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_39, mul_215);  add_39 = mul_215 = None
    add_94: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_212, mul_216);  mul_212 = mul_216 = None
    mul_217: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_132, add_94);  view_132 = add_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_51: "f32[1, 1, 1536]" = torch.ops.aten.sum.dim_IntList(mul_217, [0, 1], True)
    view_133: "f32[1536]" = torch.ops.aten.view.default(sum_51, [1536]);  sum_51 = None
    view_134: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_217, [1568, 1536]);  mul_217 = None
    permute_140: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_134, [1, 0])
    mm_46: "f32[1536, 384]" = torch.ops.aten.mm.default(permute_140, view_39);  permute_140 = view_39 = None
    permute_141: "f32[384, 1536]" = torch.ops.aten.permute.default(mm_46, [1, 0]);  mm_46 = None
    permute_142: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
    mm_47: "f32[1568, 384]" = torch.ops.aten.mm.default(view_134, permute_142);  view_134 = permute_142 = None
    view_135: "f32[8, 196, 384]" = torch.ops.aten.view.default(mm_47, [8, 196, 384]);  mm_47 = None
    permute_143: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_141, [1, 0]);  permute_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_218: "f32[8, 196, 384]" = torch.ops.aten.mul.Scalar(add_37, 1);  add_37 = None
    mul_219: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_135, mul_218);  mul_218 = None
    mul_220: "f32[1, 1, 384]" = torch.ops.aten.mul.Scalar(primals_42, 1);  primals_42 = None
    mul_221: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_135, mul_220);  mul_220 = None
    sum_52: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(view_135, [0, 1], True);  view_135 = None
    sum_53: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(mul_219, [0, 1], True);  mul_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    add_95: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_92, mul_221);  add_92 = mul_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    mul_222: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(add_95, primals_37);  primals_37 = None
    mul_223: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(add_95, permute_33);  permute_33 = None
    sum_54: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(mul_223, [0, 1], True);  mul_223 = None
    view_136: "f32[384]" = torch.ops.aten.view.default(sum_54, [384]);  sum_54 = None
    permute_144: "f32[8, 384, 196]" = torch.ops.aten.permute.default(mul_222, [0, 2, 1]);  mul_222 = None
    clone_42: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_144, memory_format = torch.contiguous_format);  permute_144 = None
    view_137: "f32[3072, 196]" = torch.ops.aten.view.default(clone_42, [3072, 196]);  clone_42 = None
    permute_145: "f32[196, 196]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
    mm_48: "f32[3072, 196]" = torch.ops.aten.mm.default(view_137, permute_145);  permute_145 = None
    permute_146: "f32[196, 3072]" = torch.ops.aten.permute.default(view_137, [1, 0])
    mm_49: "f32[196, 196]" = torch.ops.aten.mm.default(permute_146, view_37);  permute_146 = view_37 = None
    permute_147: "f32[196, 196]" = torch.ops.aten.permute.default(mm_49, [1, 0]);  mm_49 = None
    sum_55: "f32[1, 196]" = torch.ops.aten.sum.dim_IntList(view_137, [0], True);  view_137 = None
    view_138: "f32[196]" = torch.ops.aten.view.default(sum_55, [196]);  sum_55 = None
    permute_148: "f32[196, 196]" = torch.ops.aten.permute.default(permute_147, [1, 0]);  permute_147 = None
    view_139: "f32[8, 384, 196]" = torch.ops.aten.view.default(mm_48, [8, 384, 196]);  mm_48 = None
    permute_149: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_139, [0, 2, 1]);  view_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_224: "f32[8, 196, 384]" = torch.ops.aten.mul.Scalar(add_35, 1);  add_35 = None
    mul_225: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(permute_149, mul_224);  mul_224 = None
    mul_226: "f32[1, 1, 384]" = torch.ops.aten.mul.Scalar(primals_39, 1);  primals_39 = None
    mul_227: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(permute_149, mul_226);  mul_226 = None
    sum_56: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(permute_149, [0, 1], True);  permute_149 = None
    sum_57: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(mul_225, [0, 1], True);  mul_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    add_96: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_95, mul_227);  add_95 = mul_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    mul_228: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(add_96, primals_34);  primals_34 = None
    mul_229: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(add_96, clone_17);  clone_17 = None
    sum_58: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(mul_229, [0, 1], True);  mul_229 = None
    view_140: "f32[384]" = torch.ops.aten.view.default(sum_58, [384]);  sum_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_141: "f32[1568, 384]" = torch.ops.aten.view.default(mul_228, [1568, 384]);  mul_228 = None
    permute_150: "f32[384, 1536]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
    mm_50: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_141, permute_150);  permute_150 = None
    permute_151: "f32[384, 1568]" = torch.ops.aten.permute.default(view_141, [1, 0])
    mm_51: "f32[384, 1536]" = torch.ops.aten.mm.default(permute_151, view_35);  permute_151 = view_35 = None
    permute_152: "f32[1536, 384]" = torch.ops.aten.permute.default(mm_51, [1, 0]);  mm_51 = None
    sum_59: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_141, [0], True);  view_141 = None
    view_142: "f32[384]" = torch.ops.aten.view.default(sum_59, [384]);  sum_59 = None
    permute_153: "f32[384, 1536]" = torch.ops.aten.permute.default(permute_152, [1, 0]);  permute_152 = None
    view_143: "f32[8, 196, 1536]" = torch.ops.aten.view.default(mm_50, [8, 196, 1536]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_230: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_33, 0.7071067811865476)
    erf_18: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_230);  mul_230 = None
    add_97: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    mul_231: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_97, 0.5);  add_97 = None
    mul_232: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_33, add_33)
    mul_233: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_232, -0.5);  mul_232 = None
    exp_6: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_233);  mul_233 = None
    mul_234: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_6, 0.3989422804014327);  exp_6 = None
    mul_235: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_33, mul_234);  add_33 = mul_234 = None
    add_98: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_231, mul_235);  mul_231 = mul_235 = None
    mul_236: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_143, add_98);  view_143 = add_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_60: "f32[1, 1, 1536]" = torch.ops.aten.sum.dim_IntList(mul_236, [0, 1], True)
    view_144: "f32[1536]" = torch.ops.aten.view.default(sum_60, [1536]);  sum_60 = None
    view_145: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_236, [1568, 1536]);  mul_236 = None
    permute_154: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_145, [1, 0])
    mm_52: "f32[1536, 384]" = torch.ops.aten.mm.default(permute_154, view_33);  permute_154 = view_33 = None
    permute_155: "f32[384, 1536]" = torch.ops.aten.permute.default(mm_52, [1, 0]);  mm_52 = None
    permute_156: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_29, [1, 0]);  permute_29 = None
    mm_53: "f32[1568, 384]" = torch.ops.aten.mm.default(view_145, permute_156);  view_145 = permute_156 = None
    view_146: "f32[8, 196, 384]" = torch.ops.aten.view.default(mm_53, [8, 196, 384]);  mm_53 = None
    permute_157: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_155, [1, 0]);  permute_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_237: "f32[8, 196, 384]" = torch.ops.aten.mul.Scalar(add_31, 1);  add_31 = None
    mul_238: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_146, mul_237);  mul_237 = None
    mul_239: "f32[1, 1, 384]" = torch.ops.aten.mul.Scalar(primals_36, 1);  primals_36 = None
    mul_240: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_146, mul_239);  mul_239 = None
    sum_61: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(view_146, [0, 1], True);  view_146 = None
    sum_62: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(mul_238, [0, 1], True);  mul_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    add_99: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_96, mul_240);  add_96 = mul_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    mul_241: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(add_99, primals_31);  primals_31 = None
    mul_242: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(add_99, permute_28);  permute_28 = None
    sum_63: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(mul_242, [0, 1], True);  mul_242 = None
    view_147: "f32[384]" = torch.ops.aten.view.default(sum_63, [384]);  sum_63 = None
    permute_158: "f32[8, 384, 196]" = torch.ops.aten.permute.default(mul_241, [0, 2, 1]);  mul_241 = None
    clone_43: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_158, memory_format = torch.contiguous_format);  permute_158 = None
    view_148: "f32[3072, 196]" = torch.ops.aten.view.default(clone_43, [3072, 196]);  clone_43 = None
    permute_159: "f32[196, 196]" = torch.ops.aten.permute.default(permute_27, [1, 0]);  permute_27 = None
    mm_54: "f32[3072, 196]" = torch.ops.aten.mm.default(view_148, permute_159);  permute_159 = None
    permute_160: "f32[196, 3072]" = torch.ops.aten.permute.default(view_148, [1, 0])
    mm_55: "f32[196, 196]" = torch.ops.aten.mm.default(permute_160, view_31);  permute_160 = view_31 = None
    permute_161: "f32[196, 196]" = torch.ops.aten.permute.default(mm_55, [1, 0]);  mm_55 = None
    sum_64: "f32[1, 196]" = torch.ops.aten.sum.dim_IntList(view_148, [0], True);  view_148 = None
    view_149: "f32[196]" = torch.ops.aten.view.default(sum_64, [196]);  sum_64 = None
    permute_162: "f32[196, 196]" = torch.ops.aten.permute.default(permute_161, [1, 0]);  permute_161 = None
    view_150: "f32[8, 384, 196]" = torch.ops.aten.view.default(mm_54, [8, 384, 196]);  mm_54 = None
    permute_163: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_150, [0, 2, 1]);  view_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_243: "f32[8, 196, 384]" = torch.ops.aten.mul.Scalar(add_29, 1);  add_29 = None
    mul_244: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(permute_163, mul_243);  mul_243 = None
    mul_245: "f32[1, 1, 384]" = torch.ops.aten.mul.Scalar(primals_33, 1);  primals_33 = None
    mul_246: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(permute_163, mul_245);  mul_245 = None
    sum_65: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(permute_163, [0, 1], True);  permute_163 = None
    sum_66: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(mul_244, [0, 1], True);  mul_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    add_100: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_99, mul_246);  add_99 = mul_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    mul_247: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(add_100, primals_28);  primals_28 = None
    mul_248: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(add_100, clone_14);  clone_14 = None
    sum_67: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(mul_248, [0, 1], True);  mul_248 = None
    view_151: "f32[384]" = torch.ops.aten.view.default(sum_67, [384]);  sum_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_152: "f32[1568, 384]" = torch.ops.aten.view.default(mul_247, [1568, 384]);  mul_247 = None
    permute_164: "f32[384, 1536]" = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None
    mm_56: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_152, permute_164);  permute_164 = None
    permute_165: "f32[384, 1568]" = torch.ops.aten.permute.default(view_152, [1, 0])
    mm_57: "f32[384, 1536]" = torch.ops.aten.mm.default(permute_165, view_29);  permute_165 = view_29 = None
    permute_166: "f32[1536, 384]" = torch.ops.aten.permute.default(mm_57, [1, 0]);  mm_57 = None
    sum_68: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_152, [0], True);  view_152 = None
    view_153: "f32[384]" = torch.ops.aten.view.default(sum_68, [384]);  sum_68 = None
    permute_167: "f32[384, 1536]" = torch.ops.aten.permute.default(permute_166, [1, 0]);  permute_166 = None
    view_154: "f32[8, 196, 1536]" = torch.ops.aten.view.default(mm_56, [8, 196, 1536]);  mm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_249: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_27, 0.7071067811865476)
    erf_19: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_249);  mul_249 = None
    add_101: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    mul_250: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_101, 0.5);  add_101 = None
    mul_251: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_27, add_27)
    mul_252: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_251, -0.5);  mul_251 = None
    exp_7: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_252);  mul_252 = None
    mul_253: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_7, 0.3989422804014327);  exp_7 = None
    mul_254: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_27, mul_253);  add_27 = mul_253 = None
    add_102: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_250, mul_254);  mul_250 = mul_254 = None
    mul_255: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_154, add_102);  view_154 = add_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_69: "f32[1, 1, 1536]" = torch.ops.aten.sum.dim_IntList(mul_255, [0, 1], True)
    view_155: "f32[1536]" = torch.ops.aten.view.default(sum_69, [1536]);  sum_69 = None
    view_156: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_255, [1568, 1536]);  mul_255 = None
    permute_168: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_156, [1, 0])
    mm_58: "f32[1536, 384]" = torch.ops.aten.mm.default(permute_168, view_27);  permute_168 = view_27 = None
    permute_169: "f32[384, 1536]" = torch.ops.aten.permute.default(mm_58, [1, 0]);  mm_58 = None
    permute_170: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
    mm_59: "f32[1568, 384]" = torch.ops.aten.mm.default(view_156, permute_170);  view_156 = permute_170 = None
    view_157: "f32[8, 196, 384]" = torch.ops.aten.view.default(mm_59, [8, 196, 384]);  mm_59 = None
    permute_171: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_169, [1, 0]);  permute_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_256: "f32[8, 196, 384]" = torch.ops.aten.mul.Scalar(add_25, 1);  add_25 = None
    mul_257: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_157, mul_256);  mul_256 = None
    mul_258: "f32[1, 1, 384]" = torch.ops.aten.mul.Scalar(primals_30, 1);  primals_30 = None
    mul_259: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_157, mul_258);  mul_258 = None
    sum_70: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(view_157, [0, 1], True);  view_157 = None
    sum_71: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(mul_257, [0, 1], True);  mul_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    add_103: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_100, mul_259);  add_100 = mul_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    mul_260: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(add_103, primals_25);  primals_25 = None
    mul_261: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(add_103, permute_23);  permute_23 = None
    sum_72: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(mul_261, [0, 1], True);  mul_261 = None
    view_158: "f32[384]" = torch.ops.aten.view.default(sum_72, [384]);  sum_72 = None
    permute_172: "f32[8, 384, 196]" = torch.ops.aten.permute.default(mul_260, [0, 2, 1]);  mul_260 = None
    clone_44: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_172, memory_format = torch.contiguous_format);  permute_172 = None
    view_159: "f32[3072, 196]" = torch.ops.aten.view.default(clone_44, [3072, 196]);  clone_44 = None
    permute_173: "f32[196, 196]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
    mm_60: "f32[3072, 196]" = torch.ops.aten.mm.default(view_159, permute_173);  permute_173 = None
    permute_174: "f32[196, 3072]" = torch.ops.aten.permute.default(view_159, [1, 0])
    mm_61: "f32[196, 196]" = torch.ops.aten.mm.default(permute_174, view_25);  permute_174 = view_25 = None
    permute_175: "f32[196, 196]" = torch.ops.aten.permute.default(mm_61, [1, 0]);  mm_61 = None
    sum_73: "f32[1, 196]" = torch.ops.aten.sum.dim_IntList(view_159, [0], True);  view_159 = None
    view_160: "f32[196]" = torch.ops.aten.view.default(sum_73, [196]);  sum_73 = None
    permute_176: "f32[196, 196]" = torch.ops.aten.permute.default(permute_175, [1, 0]);  permute_175 = None
    view_161: "f32[8, 384, 196]" = torch.ops.aten.view.default(mm_60, [8, 384, 196]);  mm_60 = None
    permute_177: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_161, [0, 2, 1]);  view_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_262: "f32[8, 196, 384]" = torch.ops.aten.mul.Scalar(add_23, 1);  add_23 = None
    mul_263: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(permute_177, mul_262);  mul_262 = None
    mul_264: "f32[1, 1, 384]" = torch.ops.aten.mul.Scalar(primals_27, 1);  primals_27 = None
    mul_265: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(permute_177, mul_264);  mul_264 = None
    sum_74: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(permute_177, [0, 1], True);  permute_177 = None
    sum_75: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(mul_263, [0, 1], True);  mul_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    add_104: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_103, mul_265);  add_103 = mul_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    mul_266: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(add_104, primals_22);  primals_22 = None
    mul_267: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(add_104, clone_11);  clone_11 = None
    sum_76: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(mul_267, [0, 1], True);  mul_267 = None
    view_162: "f32[384]" = torch.ops.aten.view.default(sum_76, [384]);  sum_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_163: "f32[1568, 384]" = torch.ops.aten.view.default(mul_266, [1568, 384]);  mul_266 = None
    permute_178: "f32[384, 1536]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
    mm_62: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_163, permute_178);  permute_178 = None
    permute_179: "f32[384, 1568]" = torch.ops.aten.permute.default(view_163, [1, 0])
    mm_63: "f32[384, 1536]" = torch.ops.aten.mm.default(permute_179, view_23);  permute_179 = view_23 = None
    permute_180: "f32[1536, 384]" = torch.ops.aten.permute.default(mm_63, [1, 0]);  mm_63 = None
    sum_77: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_163, [0], True);  view_163 = None
    view_164: "f32[384]" = torch.ops.aten.view.default(sum_77, [384]);  sum_77 = None
    permute_181: "f32[384, 1536]" = torch.ops.aten.permute.default(permute_180, [1, 0]);  permute_180 = None
    view_165: "f32[8, 196, 1536]" = torch.ops.aten.view.default(mm_62, [8, 196, 1536]);  mm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_268: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_21, 0.7071067811865476)
    erf_20: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_268);  mul_268 = None
    add_105: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    mul_269: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_105, 0.5);  add_105 = None
    mul_270: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_21, add_21)
    mul_271: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_270, -0.5);  mul_270 = None
    exp_8: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_271);  mul_271 = None
    mul_272: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_8, 0.3989422804014327);  exp_8 = None
    mul_273: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_21, mul_272);  add_21 = mul_272 = None
    add_106: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_269, mul_273);  mul_269 = mul_273 = None
    mul_274: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_165, add_106);  view_165 = add_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_78: "f32[1, 1, 1536]" = torch.ops.aten.sum.dim_IntList(mul_274, [0, 1], True)
    view_166: "f32[1536]" = torch.ops.aten.view.default(sum_78, [1536]);  sum_78 = None
    view_167: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_274, [1568, 1536]);  mul_274 = None
    permute_182: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_167, [1, 0])
    mm_64: "f32[1536, 384]" = torch.ops.aten.mm.default(permute_182, view_21);  permute_182 = view_21 = None
    permute_183: "f32[384, 1536]" = torch.ops.aten.permute.default(mm_64, [1, 0]);  mm_64 = None
    permute_184: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
    mm_65: "f32[1568, 384]" = torch.ops.aten.mm.default(view_167, permute_184);  view_167 = permute_184 = None
    view_168: "f32[8, 196, 384]" = torch.ops.aten.view.default(mm_65, [8, 196, 384]);  mm_65 = None
    permute_185: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_183, [1, 0]);  permute_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_275: "f32[8, 196, 384]" = torch.ops.aten.mul.Scalar(add_19, 1);  add_19 = None
    mul_276: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_168, mul_275);  mul_275 = None
    mul_277: "f32[1, 1, 384]" = torch.ops.aten.mul.Scalar(primals_24, 1);  primals_24 = None
    mul_278: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_168, mul_277);  mul_277 = None
    sum_79: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(view_168, [0, 1], True);  view_168 = None
    sum_80: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(mul_276, [0, 1], True);  mul_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    add_107: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_104, mul_278);  add_104 = mul_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    mul_279: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(add_107, primals_19);  primals_19 = None
    mul_280: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(add_107, permute_18);  permute_18 = None
    sum_81: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(mul_280, [0, 1], True);  mul_280 = None
    view_169: "f32[384]" = torch.ops.aten.view.default(sum_81, [384]);  sum_81 = None
    permute_186: "f32[8, 384, 196]" = torch.ops.aten.permute.default(mul_279, [0, 2, 1]);  mul_279 = None
    clone_45: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_186, memory_format = torch.contiguous_format);  permute_186 = None
    view_170: "f32[3072, 196]" = torch.ops.aten.view.default(clone_45, [3072, 196]);  clone_45 = None
    permute_187: "f32[196, 196]" = torch.ops.aten.permute.default(permute_17, [1, 0]);  permute_17 = None
    mm_66: "f32[3072, 196]" = torch.ops.aten.mm.default(view_170, permute_187);  permute_187 = None
    permute_188: "f32[196, 3072]" = torch.ops.aten.permute.default(view_170, [1, 0])
    mm_67: "f32[196, 196]" = torch.ops.aten.mm.default(permute_188, view_19);  permute_188 = view_19 = None
    permute_189: "f32[196, 196]" = torch.ops.aten.permute.default(mm_67, [1, 0]);  mm_67 = None
    sum_82: "f32[1, 196]" = torch.ops.aten.sum.dim_IntList(view_170, [0], True);  view_170 = None
    view_171: "f32[196]" = torch.ops.aten.view.default(sum_82, [196]);  sum_82 = None
    permute_190: "f32[196, 196]" = torch.ops.aten.permute.default(permute_189, [1, 0]);  permute_189 = None
    view_172: "f32[8, 384, 196]" = torch.ops.aten.view.default(mm_66, [8, 384, 196]);  mm_66 = None
    permute_191: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_172, [0, 2, 1]);  view_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_281: "f32[8, 196, 384]" = torch.ops.aten.mul.Scalar(add_17, 1);  add_17 = None
    mul_282: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(permute_191, mul_281);  mul_281 = None
    mul_283: "f32[1, 1, 384]" = torch.ops.aten.mul.Scalar(primals_21, 1);  primals_21 = None
    mul_284: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(permute_191, mul_283);  mul_283 = None
    sum_83: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(permute_191, [0, 1], True);  permute_191 = None
    sum_84: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(mul_282, [0, 1], True);  mul_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    add_108: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_107, mul_284);  add_107 = mul_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    mul_285: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(add_108, primals_16);  primals_16 = None
    mul_286: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(add_108, clone_8);  clone_8 = None
    sum_85: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(mul_286, [0, 1], True);  mul_286 = None
    view_173: "f32[384]" = torch.ops.aten.view.default(sum_85, [384]);  sum_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_174: "f32[1568, 384]" = torch.ops.aten.view.default(mul_285, [1568, 384]);  mul_285 = None
    permute_192: "f32[384, 1536]" = torch.ops.aten.permute.default(permute_15, [1, 0]);  permute_15 = None
    mm_68: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_174, permute_192);  permute_192 = None
    permute_193: "f32[384, 1568]" = torch.ops.aten.permute.default(view_174, [1, 0])
    mm_69: "f32[384, 1536]" = torch.ops.aten.mm.default(permute_193, view_17);  permute_193 = view_17 = None
    permute_194: "f32[1536, 384]" = torch.ops.aten.permute.default(mm_69, [1, 0]);  mm_69 = None
    sum_86: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_174, [0], True);  view_174 = None
    view_175: "f32[384]" = torch.ops.aten.view.default(sum_86, [384]);  sum_86 = None
    permute_195: "f32[384, 1536]" = torch.ops.aten.permute.default(permute_194, [1, 0]);  permute_194 = None
    view_176: "f32[8, 196, 1536]" = torch.ops.aten.view.default(mm_68, [8, 196, 1536]);  mm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_287: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_15, 0.7071067811865476)
    erf_21: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_287);  mul_287 = None
    add_109: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    mul_288: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_109, 0.5);  add_109 = None
    mul_289: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_15, add_15)
    mul_290: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_289, -0.5);  mul_289 = None
    exp_9: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_290);  mul_290 = None
    mul_291: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_9, 0.3989422804014327);  exp_9 = None
    mul_292: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_15, mul_291);  add_15 = mul_291 = None
    add_110: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_288, mul_292);  mul_288 = mul_292 = None
    mul_293: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_176, add_110);  view_176 = add_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_87: "f32[1, 1, 1536]" = torch.ops.aten.sum.dim_IntList(mul_293, [0, 1], True)
    view_177: "f32[1536]" = torch.ops.aten.view.default(sum_87, [1536]);  sum_87 = None
    view_178: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_293, [1568, 1536]);  mul_293 = None
    permute_196: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_178, [1, 0])
    mm_70: "f32[1536, 384]" = torch.ops.aten.mm.default(permute_196, view_15);  permute_196 = view_15 = None
    permute_197: "f32[384, 1536]" = torch.ops.aten.permute.default(mm_70, [1, 0]);  mm_70 = None
    permute_198: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None
    mm_71: "f32[1568, 384]" = torch.ops.aten.mm.default(view_178, permute_198);  view_178 = permute_198 = None
    view_179: "f32[8, 196, 384]" = torch.ops.aten.view.default(mm_71, [8, 196, 384]);  mm_71 = None
    permute_199: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_197, [1, 0]);  permute_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_294: "f32[8, 196, 384]" = torch.ops.aten.mul.Scalar(add_13, 1);  add_13 = None
    mul_295: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_179, mul_294);  mul_294 = None
    mul_296: "f32[1, 1, 384]" = torch.ops.aten.mul.Scalar(primals_18, 1);  primals_18 = None
    mul_297: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_179, mul_296);  mul_296 = None
    sum_88: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(view_179, [0, 1], True);  view_179 = None
    sum_89: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(mul_295, [0, 1], True);  mul_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    add_111: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_108, mul_297);  add_108 = mul_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    mul_298: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(add_111, primals_13);  primals_13 = None
    mul_299: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(add_111, permute_13);  permute_13 = None
    sum_90: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(mul_299, [0, 1], True);  mul_299 = None
    view_180: "f32[384]" = torch.ops.aten.view.default(sum_90, [384]);  sum_90 = None
    permute_200: "f32[8, 384, 196]" = torch.ops.aten.permute.default(mul_298, [0, 2, 1]);  mul_298 = None
    clone_46: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_200, memory_format = torch.contiguous_format);  permute_200 = None
    view_181: "f32[3072, 196]" = torch.ops.aten.view.default(clone_46, [3072, 196]);  clone_46 = None
    permute_201: "f32[196, 196]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
    mm_72: "f32[3072, 196]" = torch.ops.aten.mm.default(view_181, permute_201);  permute_201 = None
    permute_202: "f32[196, 3072]" = torch.ops.aten.permute.default(view_181, [1, 0])
    mm_73: "f32[196, 196]" = torch.ops.aten.mm.default(permute_202, view_13);  permute_202 = view_13 = None
    permute_203: "f32[196, 196]" = torch.ops.aten.permute.default(mm_73, [1, 0]);  mm_73 = None
    sum_91: "f32[1, 196]" = torch.ops.aten.sum.dim_IntList(view_181, [0], True);  view_181 = None
    view_182: "f32[196]" = torch.ops.aten.view.default(sum_91, [196]);  sum_91 = None
    permute_204: "f32[196, 196]" = torch.ops.aten.permute.default(permute_203, [1, 0]);  permute_203 = None
    view_183: "f32[8, 384, 196]" = torch.ops.aten.view.default(mm_72, [8, 384, 196]);  mm_72 = None
    permute_205: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_183, [0, 2, 1]);  view_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_300: "f32[8, 196, 384]" = torch.ops.aten.mul.Scalar(add_11, 1);  add_11 = None
    mul_301: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(permute_205, mul_300);  mul_300 = None
    mul_302: "f32[1, 1, 384]" = torch.ops.aten.mul.Scalar(primals_15, 1);  primals_15 = None
    mul_303: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(permute_205, mul_302);  mul_302 = None
    sum_92: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(permute_205, [0, 1], True);  permute_205 = None
    sum_93: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(mul_301, [0, 1], True);  mul_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    add_112: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_111, mul_303);  add_111 = mul_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    mul_304: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(add_112, primals_10);  primals_10 = None
    mul_305: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(add_112, clone_5);  clone_5 = None
    sum_94: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(mul_305, [0, 1], True);  mul_305 = None
    view_184: "f32[384]" = torch.ops.aten.view.default(sum_94, [384]);  sum_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_185: "f32[1568, 384]" = torch.ops.aten.view.default(mul_304, [1568, 384]);  mul_304 = None
    permute_206: "f32[384, 1536]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    mm_74: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_185, permute_206);  permute_206 = None
    permute_207: "f32[384, 1568]" = torch.ops.aten.permute.default(view_185, [1, 0])
    mm_75: "f32[384, 1536]" = torch.ops.aten.mm.default(permute_207, view_11);  permute_207 = view_11 = None
    permute_208: "f32[1536, 384]" = torch.ops.aten.permute.default(mm_75, [1, 0]);  mm_75 = None
    sum_95: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_185, [0], True);  view_185 = None
    view_186: "f32[384]" = torch.ops.aten.view.default(sum_95, [384]);  sum_95 = None
    permute_209: "f32[384, 1536]" = torch.ops.aten.permute.default(permute_208, [1, 0]);  permute_208 = None
    view_187: "f32[8, 196, 1536]" = torch.ops.aten.view.default(mm_74, [8, 196, 1536]);  mm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_306: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_9, 0.7071067811865476)
    erf_22: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_306);  mul_306 = None
    add_113: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    mul_307: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_113, 0.5);  add_113 = None
    mul_308: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_9, add_9)
    mul_309: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_308, -0.5);  mul_308 = None
    exp_10: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_309);  mul_309 = None
    mul_310: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_10, 0.3989422804014327);  exp_10 = None
    mul_311: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_9, mul_310);  add_9 = mul_310 = None
    add_114: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_307, mul_311);  mul_307 = mul_311 = None
    mul_312: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_187, add_114);  view_187 = add_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_96: "f32[1, 1, 1536]" = torch.ops.aten.sum.dim_IntList(mul_312, [0, 1], True)
    view_188: "f32[1536]" = torch.ops.aten.view.default(sum_96, [1536]);  sum_96 = None
    view_189: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_312, [1568, 1536]);  mul_312 = None
    permute_210: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_189, [1, 0])
    mm_76: "f32[1536, 384]" = torch.ops.aten.mm.default(permute_210, view_9);  permute_210 = view_9 = None
    permute_211: "f32[384, 1536]" = torch.ops.aten.permute.default(mm_76, [1, 0]);  mm_76 = None
    permute_212: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
    mm_77: "f32[1568, 384]" = torch.ops.aten.mm.default(view_189, permute_212);  view_189 = permute_212 = None
    view_190: "f32[8, 196, 384]" = torch.ops.aten.view.default(mm_77, [8, 196, 384]);  mm_77 = None
    permute_213: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_211, [1, 0]);  permute_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_313: "f32[8, 196, 384]" = torch.ops.aten.mul.Scalar(add_7, 1);  add_7 = None
    mul_314: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_190, mul_313);  mul_313 = None
    mul_315: "f32[1, 1, 384]" = torch.ops.aten.mul.Scalar(primals_12, 1);  primals_12 = None
    mul_316: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_190, mul_315);  mul_315 = None
    sum_97: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(view_190, [0, 1], True);  view_190 = None
    sum_98: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(mul_314, [0, 1], True);  mul_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    add_115: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_112, mul_316);  add_112 = mul_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    mul_317: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(add_115, primals_7);  primals_7 = None
    mul_318: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(add_115, permute_8);  permute_8 = None
    sum_99: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(mul_318, [0, 1], True);  mul_318 = None
    view_191: "f32[384]" = torch.ops.aten.view.default(sum_99, [384]);  sum_99 = None
    permute_214: "f32[8, 384, 196]" = torch.ops.aten.permute.default(mul_317, [0, 2, 1]);  mul_317 = None
    clone_47: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_214, memory_format = torch.contiguous_format);  permute_214 = None
    view_192: "f32[3072, 196]" = torch.ops.aten.view.default(clone_47, [3072, 196]);  clone_47 = None
    permute_215: "f32[196, 196]" = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None
    mm_78: "f32[3072, 196]" = torch.ops.aten.mm.default(view_192, permute_215);  permute_215 = None
    permute_216: "f32[196, 3072]" = torch.ops.aten.permute.default(view_192, [1, 0])
    mm_79: "f32[196, 196]" = torch.ops.aten.mm.default(permute_216, view_7);  permute_216 = view_7 = None
    permute_217: "f32[196, 196]" = torch.ops.aten.permute.default(mm_79, [1, 0]);  mm_79 = None
    sum_100: "f32[1, 196]" = torch.ops.aten.sum.dim_IntList(view_192, [0], True);  view_192 = None
    view_193: "f32[196]" = torch.ops.aten.view.default(sum_100, [196]);  sum_100 = None
    permute_218: "f32[196, 196]" = torch.ops.aten.permute.default(permute_217, [1, 0]);  permute_217 = None
    view_194: "f32[8, 384, 196]" = torch.ops.aten.view.default(mm_78, [8, 384, 196]);  mm_78 = None
    permute_219: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_194, [0, 2, 1]);  view_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_319: "f32[8, 196, 384]" = torch.ops.aten.mul.Scalar(add_5, 1);  add_5 = None
    mul_320: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(permute_219, mul_319);  mul_319 = None
    mul_321: "f32[1, 1, 384]" = torch.ops.aten.mul.Scalar(primals_9, 1);  primals_9 = None
    mul_322: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(permute_219, mul_321);  mul_321 = None
    sum_101: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(permute_219, [0, 1], True);  permute_219 = None
    sum_102: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(mul_320, [0, 1], True);  mul_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    add_116: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_115, mul_322);  add_115 = mul_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:124, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
    mul_323: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(add_116, primals_4);  primals_4 = None
    mul_324: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(add_116, clone_2);  clone_2 = None
    sum_103: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(mul_324, [0, 1], True);  mul_324 = None
    view_195: "f32[384]" = torch.ops.aten.view.default(sum_103, [384]);  sum_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_196: "f32[1568, 384]" = torch.ops.aten.view.default(mul_323, [1568, 384]);  mul_323 = None
    permute_220: "f32[384, 1536]" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
    mm_80: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_196, permute_220);  permute_220 = None
    permute_221: "f32[384, 1568]" = torch.ops.aten.permute.default(view_196, [1, 0])
    mm_81: "f32[384, 1536]" = torch.ops.aten.mm.default(permute_221, view_5);  permute_221 = view_5 = None
    permute_222: "f32[1536, 384]" = torch.ops.aten.permute.default(mm_81, [1, 0]);  mm_81 = None
    sum_104: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_196, [0], True);  view_196 = None
    view_197: "f32[384]" = torch.ops.aten.view.default(sum_104, [384]);  sum_104 = None
    permute_223: "f32[384, 1536]" = torch.ops.aten.permute.default(permute_222, [1, 0]);  permute_222 = None
    view_198: "f32[8, 196, 1536]" = torch.ops.aten.view.default(mm_80, [8, 196, 1536]);  mm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_325: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_3, 0.7071067811865476)
    erf_23: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_325);  mul_325 = None
    add_117: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    mul_326: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_117, 0.5);  add_117 = None
    mul_327: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_3, add_3)
    mul_328: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_327, -0.5);  mul_327 = None
    exp_11: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_328);  mul_328 = None
    mul_329: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_11, 0.3989422804014327);  exp_11 = None
    mul_330: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_3, mul_329);  add_3 = mul_329 = None
    add_118: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_326, mul_330);  mul_326 = mul_330 = None
    mul_331: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_198, add_118);  view_198 = add_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_105: "f32[1, 1, 1536]" = torch.ops.aten.sum.dim_IntList(mul_331, [0, 1], True)
    view_199: "f32[1536]" = torch.ops.aten.view.default(sum_105, [1536]);  sum_105 = None
    view_200: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_331, [1568, 1536]);  mul_331 = None
    permute_224: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_200, [1, 0])
    mm_82: "f32[1536, 384]" = torch.ops.aten.mm.default(permute_224, view_3);  permute_224 = view_3 = None
    permute_225: "f32[384, 1536]" = torch.ops.aten.permute.default(mm_82, [1, 0]);  mm_82 = None
    permute_226: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
    mm_83: "f32[1568, 384]" = torch.ops.aten.mm.default(view_200, permute_226);  view_200 = permute_226 = None
    view_201: "f32[8, 196, 384]" = torch.ops.aten.view.default(mm_83, [8, 196, 384]);  mm_83 = None
    permute_227: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_225, [1, 0]);  permute_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_332: "f32[8, 196, 384]" = torch.ops.aten.mul.Scalar(add_1, 1);  add_1 = None
    mul_333: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_201, mul_332);  mul_332 = None
    mul_334: "f32[1, 1, 384]" = torch.ops.aten.mul.Scalar(primals_6, 1);  primals_6 = None
    mul_335: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_201, mul_334);  mul_334 = None
    sum_106: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(view_201, [0, 1], True);  view_201 = None
    sum_107: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(mul_333, [0, 1], True);  mul_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    add_119: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_116, mul_335);  add_116 = mul_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:123, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    mul_336: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(add_119, primals_1);  primals_1 = None
    mul_337: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(add_119, permute_3);  permute_3 = None
    sum_108: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(mul_337, [0, 1], True);  mul_337 = None
    view_202: "f32[384]" = torch.ops.aten.view.default(sum_108, [384]);  sum_108 = None
    permute_228: "f32[8, 384, 196]" = torch.ops.aten.permute.default(mul_336, [0, 2, 1]);  mul_336 = None
    clone_48: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_228, memory_format = torch.contiguous_format);  permute_228 = None
    view_203: "f32[3072, 196]" = torch.ops.aten.view.default(clone_48, [3072, 196]);  clone_48 = None
    permute_229: "f32[196, 196]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
    mm_84: "f32[3072, 196]" = torch.ops.aten.mm.default(view_203, permute_229);  permute_229 = None
    permute_230: "f32[196, 3072]" = torch.ops.aten.permute.default(view_203, [1, 0])
    mm_85: "f32[196, 196]" = torch.ops.aten.mm.default(permute_230, view_1);  permute_230 = view_1 = None
    permute_231: "f32[196, 196]" = torch.ops.aten.permute.default(mm_85, [1, 0]);  mm_85 = None
    sum_109: "f32[1, 196]" = torch.ops.aten.sum.dim_IntList(view_203, [0], True);  view_203 = None
    view_204: "f32[196]" = torch.ops.aten.view.default(sum_109, [196]);  sum_109 = None
    permute_232: "f32[196, 196]" = torch.ops.aten.permute.default(permute_231, [1, 0]);  permute_231 = None
    view_205: "f32[8, 384, 196]" = torch.ops.aten.view.default(mm_84, [8, 384, 196]);  mm_84 = None
    permute_233: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_205, [0, 2, 1]);  view_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    mul_338: "f32[8, 196, 384]" = torch.ops.aten.mul.Scalar(permute, 1);  permute = None
    mul_339: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(permute_233, mul_338);  mul_338 = None
    mul_340: "f32[1, 1, 384]" = torch.ops.aten.mul.Scalar(primals_3, 1);  primals_3 = None
    mul_341: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(permute_233, mul_340);  mul_340 = None
    sum_110: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(permute_233, [0, 1], True);  permute_233 = None
    sum_111: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(mul_339, [0, 1], True);  mul_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:92, code: return torch.addcmul(self.beta, self.alpha, x)
    add_120: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_119, mul_341);  add_119 = mul_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:89, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    permute_234: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_120, [0, 2, 1]);  add_120 = None
    view_206: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(permute_234, [8, 384, 14, 14]);  permute_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution_backward = torch.ops.aten.convolution_backward.default(view_206, primals_151, primals_75, [384], [16, 16], [0, 0], [1, 1], False, [0, 0], 1, [False, True, True]);  view_206 = primals_151 = primals_75 = None
    getitem_1: "f32[384, 3, 16, 16]" = convolution_backward[1]
    getitem_2: "f32[384]" = convolution_backward[2];  convolution_backward = None
    return pytree.tree_unflatten([addmm_24, view_202, sum_110, sum_111, view_195, sum_106, sum_107, view_191, sum_101, sum_102, view_184, sum_97, sum_98, view_180, sum_92, sum_93, view_173, sum_88, sum_89, view_169, sum_83, sum_84, view_162, sum_79, sum_80, view_158, sum_74, sum_75, view_151, sum_70, sum_71, view_147, sum_65, sum_66, view_140, sum_61, sum_62, view_136, sum_56, sum_57, view_129, sum_52, sum_53, view_125, sum_47, sum_48, view_118, sum_43, sum_44, view_114, sum_38, sum_39, view_107, sum_34, sum_35, view_103, sum_29, sum_30, view_96, sum_25, sum_26, view_92, sum_20, sum_21, view_85, sum_16, sum_17, view_81, sum_11, sum_12, view_74, sum_7, sum_8, sum_2, sum_3, getitem_1, getitem_2, permute_232, view_204, permute_227, view_199, permute_223, view_197, permute_218, view_193, permute_213, view_188, permute_209, view_186, permute_204, view_182, permute_199, view_177, permute_195, view_175, permute_190, view_171, permute_185, view_166, permute_181, view_164, permute_176, view_160, permute_171, view_155, permute_167, view_153, permute_162, view_149, permute_157, view_144, permute_153, view_142, permute_148, view_138, permute_143, view_133, permute_139, view_131, permute_134, view_127, permute_129, view_122, permute_125, view_120, permute_120, view_116, permute_115, view_111, permute_111, view_109, permute_106, view_105, permute_101, view_100, permute_97, view_98, permute_92, view_94, permute_87, view_89, permute_83, view_87, permute_78, view_83, permute_73, view_78, permute_69, view_76, permute_65, view_73, None], self._out_spec)
    