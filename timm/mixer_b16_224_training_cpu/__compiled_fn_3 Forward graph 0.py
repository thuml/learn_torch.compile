from __future__ import annotations



def forward(self, primals_1: "f32[768, 3, 16, 16]", primals_2: "f32[768]", primals_3: "f32[768]", primals_4: "f32[768]", primals_5: "f32[384, 196]", primals_6: "f32[384]", primals_7: "f32[196, 384]", primals_8: "f32[196]", primals_9: "f32[768]", primals_10: "f32[768]", primals_11: "f32[3072, 768]", primals_12: "f32[3072]", primals_13: "f32[768, 3072]", primals_14: "f32[768]", primals_15: "f32[768]", primals_16: "f32[768]", primals_17: "f32[384, 196]", primals_18: "f32[384]", primals_19: "f32[196, 384]", primals_20: "f32[196]", primals_21: "f32[768]", primals_22: "f32[768]", primals_23: "f32[3072, 768]", primals_24: "f32[3072]", primals_25: "f32[768, 3072]", primals_26: "f32[768]", primals_27: "f32[768]", primals_28: "f32[768]", primals_29: "f32[384, 196]", primals_30: "f32[384]", primals_31: "f32[196, 384]", primals_32: "f32[196]", primals_33: "f32[768]", primals_34: "f32[768]", primals_35: "f32[3072, 768]", primals_36: "f32[3072]", primals_37: "f32[768, 3072]", primals_38: "f32[768]", primals_39: "f32[768]", primals_40: "f32[768]", primals_41: "f32[384, 196]", primals_42: "f32[384]", primals_43: "f32[196, 384]", primals_44: "f32[196]", primals_45: "f32[768]", primals_46: "f32[768]", primals_47: "f32[3072, 768]", primals_48: "f32[3072]", primals_49: "f32[768, 3072]", primals_50: "f32[768]", primals_51: "f32[768]", primals_52: "f32[768]", primals_53: "f32[384, 196]", primals_54: "f32[384]", primals_55: "f32[196, 384]", primals_56: "f32[196]", primals_57: "f32[768]", primals_58: "f32[768]", primals_59: "f32[3072, 768]", primals_60: "f32[3072]", primals_61: "f32[768, 3072]", primals_62: "f32[768]", primals_63: "f32[768]", primals_64: "f32[768]", primals_65: "f32[384, 196]", primals_66: "f32[384]", primals_67: "f32[196, 384]", primals_68: "f32[196]", primals_69: "f32[768]", primals_70: "f32[768]", primals_71: "f32[3072, 768]", primals_72: "f32[3072]", primals_73: "f32[768, 3072]", primals_74: "f32[768]", primals_75: "f32[768]", primals_76: "f32[768]", primals_77: "f32[384, 196]", primals_78: "f32[384]", primals_79: "f32[196, 384]", primals_80: "f32[196]", primals_81: "f32[768]", primals_82: "f32[768]", primals_83: "f32[3072, 768]", primals_84: "f32[3072]", primals_85: "f32[768, 3072]", primals_86: "f32[768]", primals_87: "f32[768]", primals_88: "f32[768]", primals_89: "f32[384, 196]", primals_90: "f32[384]", primals_91: "f32[196, 384]", primals_92: "f32[196]", primals_93: "f32[768]", primals_94: "f32[768]", primals_95: "f32[3072, 768]", primals_96: "f32[3072]", primals_97: "f32[768, 3072]", primals_98: "f32[768]", primals_99: "f32[768]", primals_100: "f32[768]", primals_101: "f32[384, 196]", primals_102: "f32[384]", primals_103: "f32[196, 384]", primals_104: "f32[196]", primals_105: "f32[768]", primals_106: "f32[768]", primals_107: "f32[3072, 768]", primals_108: "f32[3072]", primals_109: "f32[768, 3072]", primals_110: "f32[768]", primals_111: "f32[768]", primals_112: "f32[768]", primals_113: "f32[384, 196]", primals_114: "f32[384]", primals_115: "f32[196, 384]", primals_116: "f32[196]", primals_117: "f32[768]", primals_118: "f32[768]", primals_119: "f32[3072, 768]", primals_120: "f32[3072]", primals_121: "f32[768, 3072]", primals_122: "f32[768]", primals_123: "f32[768]", primals_124: "f32[768]", primals_125: "f32[384, 196]", primals_126: "f32[384]", primals_127: "f32[196, 384]", primals_128: "f32[196]", primals_129: "f32[768]", primals_130: "f32[768]", primals_131: "f32[3072, 768]", primals_132: "f32[3072]", primals_133: "f32[768, 3072]", primals_134: "f32[768]", primals_135: "f32[768]", primals_136: "f32[768]", primals_137: "f32[384, 196]", primals_138: "f32[384]", primals_139: "f32[196, 384]", primals_140: "f32[196]", primals_141: "f32[768]", primals_142: "f32[768]", primals_143: "f32[3072, 768]", primals_144: "f32[3072]", primals_145: "f32[768, 3072]", primals_146: "f32[768]", primals_147: "f32[768]", primals_148: "f32[768]", primals_149: "f32[1000, 768]", primals_150: "f32[1000]", primals_151: "f32[8, 3, 224, 224]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution: "f32[8, 768, 14, 14]" = torch.ops.aten.convolution.default(primals_151, primals_1, primals_2, [16, 16], [0, 0], [1, 1], False, [0, 0], 1);  primals_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:89, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    view: "f32[8, 768, 196]" = torch.ops.aten.view.default(convolution, [8, 768, 196]);  convolution = None
    permute: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format)
    var_mean = torch.ops.aten.var_mean.correction(clone, [2], correction = 0, keepdim = True)
    getitem: "f32[8, 196, 1]" = var_mean[0]
    getitem_1: "f32[8, 196, 1]" = var_mean[1];  var_mean = None
    add: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
    rsqrt: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
    sub: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone, getitem_1);  clone = getitem_1 = None
    mul: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    mul_1: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul, primals_3)
    add_1: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_1, primals_4);  mul_1 = primals_4 = None
    permute_1: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_1, [0, 2, 1]);  add_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_2: "f32[196, 384]" = torch.ops.aten.permute.default(primals_5, [1, 0]);  primals_5 = None
    clone_1: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
    view_1: "f32[6144, 196]" = torch.ops.aten.view.default(clone_1, [6144, 196]);  clone_1 = None
    mm: "f32[6144, 384]" = torch.ops.aten.mm.default(view_1, permute_2)
    view_2: "f32[8, 768, 384]" = torch.ops.aten.view.default(mm, [8, 768, 384])
    add_2: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(view_2, primals_6);  view_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_2: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_2, 0.5)
    mul_3: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_2, 0.7071067811865476);  add_2 = None
    erf: "f32[8, 768, 384]" = torch.ops.aten.erf.default(mul_3);  mul_3 = None
    add_3: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_4: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(mul_2, add_3);  mul_2 = add_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_2: "f32[8, 768, 384]" = torch.ops.aten.clone.default(mul_4);  mul_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_3: "f32[6144, 384]" = torch.ops.aten.view.default(clone_2, [6144, 384]);  clone_2 = None
    permute_3: "f32[384, 196]" = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
    addmm: "f32[6144, 196]" = torch.ops.aten.addmm.default(primals_8, view_3, permute_3);  primals_8 = None
    view_4: "f32[8, 768, 196]" = torch.ops.aten.view.default(addmm, [8, 768, 196]);  addmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_3: "f32[8, 768, 196]" = torch.ops.aten.clone.default(view_4);  view_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_4: "f32[8, 196, 768]" = torch.ops.aten.permute.default(clone_3, [0, 2, 1]);  clone_3 = None
    add_4: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(permute, permute_4);  permute = permute_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_4: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_4, memory_format = torch.contiguous_format)
    var_mean_1 = torch.ops.aten.var_mean.correction(clone_4, [2], correction = 0, keepdim = True)
    getitem_2: "f32[8, 196, 1]" = var_mean_1[0]
    getitem_3: "f32[8, 196, 1]" = var_mean_1[1];  var_mean_1 = None
    add_5: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-06);  getitem_2 = None
    rsqrt_1: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
    sub_1: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_4, getitem_3);  clone_4 = getitem_3 = None
    mul_5: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
    mul_6: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_5, primals_9)
    add_6: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_6, primals_10);  mul_6 = primals_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_5: "f32[1568, 768]" = torch.ops.aten.view.default(add_6, [1568, 768]);  add_6 = None
    permute_5: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_11, [1, 0]);  primals_11 = None
    addmm_1: "f32[1568, 3072]" = torch.ops.aten.addmm.default(primals_12, view_5, permute_5);  primals_12 = None
    view_6: "f32[8, 196, 3072]" = torch.ops.aten.view.default(addmm_1, [8, 196, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_7: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_6, 0.5)
    mul_8: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_6, 0.7071067811865476);  view_6 = None
    erf_1: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_8);  mul_8 = None
    add_7: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_9: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_7, add_7);  mul_7 = add_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_5: "f32[8, 196, 3072]" = torch.ops.aten.clone.default(mul_9);  mul_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_7: "f32[1568, 3072]" = torch.ops.aten.view.default(clone_5, [1568, 3072]);  clone_5 = None
    permute_6: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_13, [1, 0]);  primals_13 = None
    addmm_2: "f32[1568, 768]" = torch.ops.aten.addmm.default(primals_14, view_7, permute_6);  primals_14 = None
    view_8: "f32[8, 196, 768]" = torch.ops.aten.view.default(addmm_2, [8, 196, 768]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_6: "f32[8, 196, 768]" = torch.ops.aten.clone.default(view_8);  view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_8: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_4, clone_6);  add_4 = clone_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_7: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_8, memory_format = torch.contiguous_format)
    var_mean_2 = torch.ops.aten.var_mean.correction(clone_7, [2], correction = 0, keepdim = True)
    getitem_4: "f32[8, 196, 1]" = var_mean_2[0]
    getitem_5: "f32[8, 196, 1]" = var_mean_2[1];  var_mean_2 = None
    add_9: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-06);  getitem_4 = None
    rsqrt_2: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
    sub_2: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_7, getitem_5);  clone_7 = getitem_5 = None
    mul_10: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
    mul_11: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_10, primals_15)
    add_10: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_11, primals_16);  mul_11 = primals_16 = None
    permute_7: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_10, [0, 2, 1]);  add_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_8: "f32[196, 384]" = torch.ops.aten.permute.default(primals_17, [1, 0]);  primals_17 = None
    clone_8: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    view_9: "f32[6144, 196]" = torch.ops.aten.view.default(clone_8, [6144, 196]);  clone_8 = None
    mm_1: "f32[6144, 384]" = torch.ops.aten.mm.default(view_9, permute_8)
    view_10: "f32[8, 768, 384]" = torch.ops.aten.view.default(mm_1, [8, 768, 384])
    add_11: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(view_10, primals_18);  view_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_12: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_11, 0.5)
    mul_13: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_11, 0.7071067811865476);  add_11 = None
    erf_2: "f32[8, 768, 384]" = torch.ops.aten.erf.default(mul_13);  mul_13 = None
    add_12: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_14: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(mul_12, add_12);  mul_12 = add_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_9: "f32[8, 768, 384]" = torch.ops.aten.clone.default(mul_14);  mul_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_11: "f32[6144, 384]" = torch.ops.aten.view.default(clone_9, [6144, 384]);  clone_9 = None
    permute_9: "f32[384, 196]" = torch.ops.aten.permute.default(primals_19, [1, 0]);  primals_19 = None
    addmm_3: "f32[6144, 196]" = torch.ops.aten.addmm.default(primals_20, view_11, permute_9);  primals_20 = None
    view_12: "f32[8, 768, 196]" = torch.ops.aten.view.default(addmm_3, [8, 768, 196]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_10: "f32[8, 768, 196]" = torch.ops.aten.clone.default(view_12);  view_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_10: "f32[8, 196, 768]" = torch.ops.aten.permute.default(clone_10, [0, 2, 1]);  clone_10 = None
    add_13: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_8, permute_10);  add_8 = permute_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_11: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_13, memory_format = torch.contiguous_format)
    var_mean_3 = torch.ops.aten.var_mean.correction(clone_11, [2], correction = 0, keepdim = True)
    getitem_6: "f32[8, 196, 1]" = var_mean_3[0]
    getitem_7: "f32[8, 196, 1]" = var_mean_3[1];  var_mean_3 = None
    add_14: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-06);  getitem_6 = None
    rsqrt_3: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
    sub_3: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_11, getitem_7);  clone_11 = getitem_7 = None
    mul_15: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
    mul_16: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_15, primals_21)
    add_15: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_16, primals_22);  mul_16 = primals_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_13: "f32[1568, 768]" = torch.ops.aten.view.default(add_15, [1568, 768]);  add_15 = None
    permute_11: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_23, [1, 0]);  primals_23 = None
    addmm_4: "f32[1568, 3072]" = torch.ops.aten.addmm.default(primals_24, view_13, permute_11);  primals_24 = None
    view_14: "f32[8, 196, 3072]" = torch.ops.aten.view.default(addmm_4, [8, 196, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_17: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_14, 0.5)
    mul_18: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_14, 0.7071067811865476);  view_14 = None
    erf_3: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_18);  mul_18 = None
    add_16: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_19: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_17, add_16);  mul_17 = add_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_12: "f32[8, 196, 3072]" = torch.ops.aten.clone.default(mul_19);  mul_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_15: "f32[1568, 3072]" = torch.ops.aten.view.default(clone_12, [1568, 3072]);  clone_12 = None
    permute_12: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_25, [1, 0]);  primals_25 = None
    addmm_5: "f32[1568, 768]" = torch.ops.aten.addmm.default(primals_26, view_15, permute_12);  primals_26 = None
    view_16: "f32[8, 196, 768]" = torch.ops.aten.view.default(addmm_5, [8, 196, 768]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_13: "f32[8, 196, 768]" = torch.ops.aten.clone.default(view_16);  view_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_17: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_13, clone_13);  add_13 = clone_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_14: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_17, memory_format = torch.contiguous_format)
    var_mean_4 = torch.ops.aten.var_mean.correction(clone_14, [2], correction = 0, keepdim = True)
    getitem_8: "f32[8, 196, 1]" = var_mean_4[0]
    getitem_9: "f32[8, 196, 1]" = var_mean_4[1];  var_mean_4 = None
    add_18: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-06);  getitem_8 = None
    rsqrt_4: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
    sub_4: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_14, getitem_9);  clone_14 = getitem_9 = None
    mul_20: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
    mul_21: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_20, primals_27)
    add_19: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_21, primals_28);  mul_21 = primals_28 = None
    permute_13: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_19, [0, 2, 1]);  add_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_14: "f32[196, 384]" = torch.ops.aten.permute.default(primals_29, [1, 0]);  primals_29 = None
    clone_15: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_13, memory_format = torch.contiguous_format);  permute_13 = None
    view_17: "f32[6144, 196]" = torch.ops.aten.view.default(clone_15, [6144, 196]);  clone_15 = None
    mm_2: "f32[6144, 384]" = torch.ops.aten.mm.default(view_17, permute_14)
    view_18: "f32[8, 768, 384]" = torch.ops.aten.view.default(mm_2, [8, 768, 384])
    add_20: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(view_18, primals_30);  view_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_22: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_20, 0.5)
    mul_23: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_20, 0.7071067811865476);  add_20 = None
    erf_4: "f32[8, 768, 384]" = torch.ops.aten.erf.default(mul_23);  mul_23 = None
    add_21: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_24: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(mul_22, add_21);  mul_22 = add_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_16: "f32[8, 768, 384]" = torch.ops.aten.clone.default(mul_24);  mul_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_19: "f32[6144, 384]" = torch.ops.aten.view.default(clone_16, [6144, 384]);  clone_16 = None
    permute_15: "f32[384, 196]" = torch.ops.aten.permute.default(primals_31, [1, 0]);  primals_31 = None
    addmm_6: "f32[6144, 196]" = torch.ops.aten.addmm.default(primals_32, view_19, permute_15);  primals_32 = None
    view_20: "f32[8, 768, 196]" = torch.ops.aten.view.default(addmm_6, [8, 768, 196]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_17: "f32[8, 768, 196]" = torch.ops.aten.clone.default(view_20);  view_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_16: "f32[8, 196, 768]" = torch.ops.aten.permute.default(clone_17, [0, 2, 1]);  clone_17 = None
    add_22: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_17, permute_16);  add_17 = permute_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_18: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_22, memory_format = torch.contiguous_format)
    var_mean_5 = torch.ops.aten.var_mean.correction(clone_18, [2], correction = 0, keepdim = True)
    getitem_10: "f32[8, 196, 1]" = var_mean_5[0]
    getitem_11: "f32[8, 196, 1]" = var_mean_5[1];  var_mean_5 = None
    add_23: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-06);  getitem_10 = None
    rsqrt_5: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
    sub_5: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_18, getitem_11);  clone_18 = getitem_11 = None
    mul_25: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
    mul_26: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_25, primals_33)
    add_24: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_26, primals_34);  mul_26 = primals_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_21: "f32[1568, 768]" = torch.ops.aten.view.default(add_24, [1568, 768]);  add_24 = None
    permute_17: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_35, [1, 0]);  primals_35 = None
    addmm_7: "f32[1568, 3072]" = torch.ops.aten.addmm.default(primals_36, view_21, permute_17);  primals_36 = None
    view_22: "f32[8, 196, 3072]" = torch.ops.aten.view.default(addmm_7, [8, 196, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_27: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_22, 0.5)
    mul_28: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_22, 0.7071067811865476);  view_22 = None
    erf_5: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_28);  mul_28 = None
    add_25: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_29: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_27, add_25);  mul_27 = add_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_19: "f32[8, 196, 3072]" = torch.ops.aten.clone.default(mul_29);  mul_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_23: "f32[1568, 3072]" = torch.ops.aten.view.default(clone_19, [1568, 3072]);  clone_19 = None
    permute_18: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_37, [1, 0]);  primals_37 = None
    addmm_8: "f32[1568, 768]" = torch.ops.aten.addmm.default(primals_38, view_23, permute_18);  primals_38 = None
    view_24: "f32[8, 196, 768]" = torch.ops.aten.view.default(addmm_8, [8, 196, 768]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_20: "f32[8, 196, 768]" = torch.ops.aten.clone.default(view_24);  view_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_26: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_22, clone_20);  add_22 = clone_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_21: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_26, memory_format = torch.contiguous_format)
    var_mean_6 = torch.ops.aten.var_mean.correction(clone_21, [2], correction = 0, keepdim = True)
    getitem_12: "f32[8, 196, 1]" = var_mean_6[0]
    getitem_13: "f32[8, 196, 1]" = var_mean_6[1];  var_mean_6 = None
    add_27: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-06);  getitem_12 = None
    rsqrt_6: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
    sub_6: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_21, getitem_13);  clone_21 = getitem_13 = None
    mul_30: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
    mul_31: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_30, primals_39)
    add_28: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_31, primals_40);  mul_31 = primals_40 = None
    permute_19: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_28, [0, 2, 1]);  add_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_20: "f32[196, 384]" = torch.ops.aten.permute.default(primals_41, [1, 0]);  primals_41 = None
    clone_22: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_19, memory_format = torch.contiguous_format);  permute_19 = None
    view_25: "f32[6144, 196]" = torch.ops.aten.view.default(clone_22, [6144, 196]);  clone_22 = None
    mm_3: "f32[6144, 384]" = torch.ops.aten.mm.default(view_25, permute_20)
    view_26: "f32[8, 768, 384]" = torch.ops.aten.view.default(mm_3, [8, 768, 384])
    add_29: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(view_26, primals_42);  view_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_32: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_29, 0.5)
    mul_33: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_29, 0.7071067811865476);  add_29 = None
    erf_6: "f32[8, 768, 384]" = torch.ops.aten.erf.default(mul_33);  mul_33 = None
    add_30: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_34: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(mul_32, add_30);  mul_32 = add_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_23: "f32[8, 768, 384]" = torch.ops.aten.clone.default(mul_34);  mul_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_27: "f32[6144, 384]" = torch.ops.aten.view.default(clone_23, [6144, 384]);  clone_23 = None
    permute_21: "f32[384, 196]" = torch.ops.aten.permute.default(primals_43, [1, 0]);  primals_43 = None
    addmm_9: "f32[6144, 196]" = torch.ops.aten.addmm.default(primals_44, view_27, permute_21);  primals_44 = None
    view_28: "f32[8, 768, 196]" = torch.ops.aten.view.default(addmm_9, [8, 768, 196]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_24: "f32[8, 768, 196]" = torch.ops.aten.clone.default(view_28);  view_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_22: "f32[8, 196, 768]" = torch.ops.aten.permute.default(clone_24, [0, 2, 1]);  clone_24 = None
    add_31: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_26, permute_22);  add_26 = permute_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_25: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_31, memory_format = torch.contiguous_format)
    var_mean_7 = torch.ops.aten.var_mean.correction(clone_25, [2], correction = 0, keepdim = True)
    getitem_14: "f32[8, 196, 1]" = var_mean_7[0]
    getitem_15: "f32[8, 196, 1]" = var_mean_7[1];  var_mean_7 = None
    add_32: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-06);  getitem_14 = None
    rsqrt_7: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
    sub_7: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_25, getitem_15);  clone_25 = getitem_15 = None
    mul_35: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
    mul_36: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_35, primals_45)
    add_33: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_36, primals_46);  mul_36 = primals_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_29: "f32[1568, 768]" = torch.ops.aten.view.default(add_33, [1568, 768]);  add_33 = None
    permute_23: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_47, [1, 0]);  primals_47 = None
    addmm_10: "f32[1568, 3072]" = torch.ops.aten.addmm.default(primals_48, view_29, permute_23);  primals_48 = None
    view_30: "f32[8, 196, 3072]" = torch.ops.aten.view.default(addmm_10, [8, 196, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_37: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_30, 0.5)
    mul_38: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_30, 0.7071067811865476);  view_30 = None
    erf_7: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_38);  mul_38 = None
    add_34: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_39: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_37, add_34);  mul_37 = add_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_26: "f32[8, 196, 3072]" = torch.ops.aten.clone.default(mul_39);  mul_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_31: "f32[1568, 3072]" = torch.ops.aten.view.default(clone_26, [1568, 3072]);  clone_26 = None
    permute_24: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_49, [1, 0]);  primals_49 = None
    addmm_11: "f32[1568, 768]" = torch.ops.aten.addmm.default(primals_50, view_31, permute_24);  primals_50 = None
    view_32: "f32[8, 196, 768]" = torch.ops.aten.view.default(addmm_11, [8, 196, 768]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_27: "f32[8, 196, 768]" = torch.ops.aten.clone.default(view_32);  view_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_35: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_31, clone_27);  add_31 = clone_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_28: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_35, memory_format = torch.contiguous_format)
    var_mean_8 = torch.ops.aten.var_mean.correction(clone_28, [2], correction = 0, keepdim = True)
    getitem_16: "f32[8, 196, 1]" = var_mean_8[0]
    getitem_17: "f32[8, 196, 1]" = var_mean_8[1];  var_mean_8 = None
    add_36: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-06);  getitem_16 = None
    rsqrt_8: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
    sub_8: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_28, getitem_17);  clone_28 = getitem_17 = None
    mul_40: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
    mul_41: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_40, primals_51)
    add_37: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_41, primals_52);  mul_41 = primals_52 = None
    permute_25: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_37, [0, 2, 1]);  add_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_26: "f32[196, 384]" = torch.ops.aten.permute.default(primals_53, [1, 0]);  primals_53 = None
    clone_29: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_25, memory_format = torch.contiguous_format);  permute_25 = None
    view_33: "f32[6144, 196]" = torch.ops.aten.view.default(clone_29, [6144, 196]);  clone_29 = None
    mm_4: "f32[6144, 384]" = torch.ops.aten.mm.default(view_33, permute_26)
    view_34: "f32[8, 768, 384]" = torch.ops.aten.view.default(mm_4, [8, 768, 384])
    add_38: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(view_34, primals_54);  view_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_42: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_38, 0.5)
    mul_43: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_38, 0.7071067811865476);  add_38 = None
    erf_8: "f32[8, 768, 384]" = torch.ops.aten.erf.default(mul_43);  mul_43 = None
    add_39: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_44: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(mul_42, add_39);  mul_42 = add_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_30: "f32[8, 768, 384]" = torch.ops.aten.clone.default(mul_44);  mul_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_35: "f32[6144, 384]" = torch.ops.aten.view.default(clone_30, [6144, 384]);  clone_30 = None
    permute_27: "f32[384, 196]" = torch.ops.aten.permute.default(primals_55, [1, 0]);  primals_55 = None
    addmm_12: "f32[6144, 196]" = torch.ops.aten.addmm.default(primals_56, view_35, permute_27);  primals_56 = None
    view_36: "f32[8, 768, 196]" = torch.ops.aten.view.default(addmm_12, [8, 768, 196]);  addmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_31: "f32[8, 768, 196]" = torch.ops.aten.clone.default(view_36);  view_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_28: "f32[8, 196, 768]" = torch.ops.aten.permute.default(clone_31, [0, 2, 1]);  clone_31 = None
    add_40: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_35, permute_28);  add_35 = permute_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_32: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_40, memory_format = torch.contiguous_format)
    var_mean_9 = torch.ops.aten.var_mean.correction(clone_32, [2], correction = 0, keepdim = True)
    getitem_18: "f32[8, 196, 1]" = var_mean_9[0]
    getitem_19: "f32[8, 196, 1]" = var_mean_9[1];  var_mean_9 = None
    add_41: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-06);  getitem_18 = None
    rsqrt_9: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_41);  add_41 = None
    sub_9: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_32, getitem_19);  clone_32 = getitem_19 = None
    mul_45: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
    mul_46: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_45, primals_57)
    add_42: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_46, primals_58);  mul_46 = primals_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_37: "f32[1568, 768]" = torch.ops.aten.view.default(add_42, [1568, 768]);  add_42 = None
    permute_29: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_59, [1, 0]);  primals_59 = None
    addmm_13: "f32[1568, 3072]" = torch.ops.aten.addmm.default(primals_60, view_37, permute_29);  primals_60 = None
    view_38: "f32[8, 196, 3072]" = torch.ops.aten.view.default(addmm_13, [8, 196, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_47: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_38, 0.5)
    mul_48: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_38, 0.7071067811865476);  view_38 = None
    erf_9: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_48);  mul_48 = None
    add_43: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_49: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_47, add_43);  mul_47 = add_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_33: "f32[8, 196, 3072]" = torch.ops.aten.clone.default(mul_49);  mul_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_39: "f32[1568, 3072]" = torch.ops.aten.view.default(clone_33, [1568, 3072]);  clone_33 = None
    permute_30: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_61, [1, 0]);  primals_61 = None
    addmm_14: "f32[1568, 768]" = torch.ops.aten.addmm.default(primals_62, view_39, permute_30);  primals_62 = None
    view_40: "f32[8, 196, 768]" = torch.ops.aten.view.default(addmm_14, [8, 196, 768]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_34: "f32[8, 196, 768]" = torch.ops.aten.clone.default(view_40);  view_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_44: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_40, clone_34);  add_40 = clone_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_35: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_44, memory_format = torch.contiguous_format)
    var_mean_10 = torch.ops.aten.var_mean.correction(clone_35, [2], correction = 0, keepdim = True)
    getitem_20: "f32[8, 196, 1]" = var_mean_10[0]
    getitem_21: "f32[8, 196, 1]" = var_mean_10[1];  var_mean_10 = None
    add_45: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-06);  getitem_20 = None
    rsqrt_10: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
    sub_10: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_35, getitem_21);  clone_35 = getitem_21 = None
    mul_50: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
    mul_51: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_50, primals_63)
    add_46: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_51, primals_64);  mul_51 = primals_64 = None
    permute_31: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_46, [0, 2, 1]);  add_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_32: "f32[196, 384]" = torch.ops.aten.permute.default(primals_65, [1, 0]);  primals_65 = None
    clone_36: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_31, memory_format = torch.contiguous_format);  permute_31 = None
    view_41: "f32[6144, 196]" = torch.ops.aten.view.default(clone_36, [6144, 196]);  clone_36 = None
    mm_5: "f32[6144, 384]" = torch.ops.aten.mm.default(view_41, permute_32)
    view_42: "f32[8, 768, 384]" = torch.ops.aten.view.default(mm_5, [8, 768, 384])
    add_47: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(view_42, primals_66);  view_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_52: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_47, 0.5)
    mul_53: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_47, 0.7071067811865476);  add_47 = None
    erf_10: "f32[8, 768, 384]" = torch.ops.aten.erf.default(mul_53);  mul_53 = None
    add_48: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_54: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(mul_52, add_48);  mul_52 = add_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_37: "f32[8, 768, 384]" = torch.ops.aten.clone.default(mul_54);  mul_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_43: "f32[6144, 384]" = torch.ops.aten.view.default(clone_37, [6144, 384]);  clone_37 = None
    permute_33: "f32[384, 196]" = torch.ops.aten.permute.default(primals_67, [1, 0]);  primals_67 = None
    addmm_15: "f32[6144, 196]" = torch.ops.aten.addmm.default(primals_68, view_43, permute_33);  primals_68 = None
    view_44: "f32[8, 768, 196]" = torch.ops.aten.view.default(addmm_15, [8, 768, 196]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_38: "f32[8, 768, 196]" = torch.ops.aten.clone.default(view_44);  view_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_34: "f32[8, 196, 768]" = torch.ops.aten.permute.default(clone_38, [0, 2, 1]);  clone_38 = None
    add_49: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_44, permute_34);  add_44 = permute_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_39: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_49, memory_format = torch.contiguous_format)
    var_mean_11 = torch.ops.aten.var_mean.correction(clone_39, [2], correction = 0, keepdim = True)
    getitem_22: "f32[8, 196, 1]" = var_mean_11[0]
    getitem_23: "f32[8, 196, 1]" = var_mean_11[1];  var_mean_11 = None
    add_50: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-06);  getitem_22 = None
    rsqrt_11: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
    sub_11: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_39, getitem_23);  clone_39 = getitem_23 = None
    mul_55: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
    mul_56: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_55, primals_69)
    add_51: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_56, primals_70);  mul_56 = primals_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_45: "f32[1568, 768]" = torch.ops.aten.view.default(add_51, [1568, 768]);  add_51 = None
    permute_35: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_71, [1, 0]);  primals_71 = None
    addmm_16: "f32[1568, 3072]" = torch.ops.aten.addmm.default(primals_72, view_45, permute_35);  primals_72 = None
    view_46: "f32[8, 196, 3072]" = torch.ops.aten.view.default(addmm_16, [8, 196, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_57: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_46, 0.5)
    mul_58: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_46, 0.7071067811865476);  view_46 = None
    erf_11: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_58);  mul_58 = None
    add_52: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_59: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_57, add_52);  mul_57 = add_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_40: "f32[8, 196, 3072]" = torch.ops.aten.clone.default(mul_59);  mul_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_47: "f32[1568, 3072]" = torch.ops.aten.view.default(clone_40, [1568, 3072]);  clone_40 = None
    permute_36: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_73, [1, 0]);  primals_73 = None
    addmm_17: "f32[1568, 768]" = torch.ops.aten.addmm.default(primals_74, view_47, permute_36);  primals_74 = None
    view_48: "f32[8, 196, 768]" = torch.ops.aten.view.default(addmm_17, [8, 196, 768]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_41: "f32[8, 196, 768]" = torch.ops.aten.clone.default(view_48);  view_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_53: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_49, clone_41);  add_49 = clone_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_42: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_53, memory_format = torch.contiguous_format)
    var_mean_12 = torch.ops.aten.var_mean.correction(clone_42, [2], correction = 0, keepdim = True)
    getitem_24: "f32[8, 196, 1]" = var_mean_12[0]
    getitem_25: "f32[8, 196, 1]" = var_mean_12[1];  var_mean_12 = None
    add_54: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-06);  getitem_24 = None
    rsqrt_12: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
    sub_12: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_42, getitem_25);  clone_42 = getitem_25 = None
    mul_60: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
    mul_61: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_60, primals_75)
    add_55: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_61, primals_76);  mul_61 = primals_76 = None
    permute_37: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_55, [0, 2, 1]);  add_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_38: "f32[196, 384]" = torch.ops.aten.permute.default(primals_77, [1, 0]);  primals_77 = None
    clone_43: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
    view_49: "f32[6144, 196]" = torch.ops.aten.view.default(clone_43, [6144, 196]);  clone_43 = None
    mm_6: "f32[6144, 384]" = torch.ops.aten.mm.default(view_49, permute_38)
    view_50: "f32[8, 768, 384]" = torch.ops.aten.view.default(mm_6, [8, 768, 384])
    add_56: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(view_50, primals_78);  view_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_62: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_56, 0.5)
    mul_63: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_56, 0.7071067811865476);  add_56 = None
    erf_12: "f32[8, 768, 384]" = torch.ops.aten.erf.default(mul_63);  mul_63 = None
    add_57: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_64: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(mul_62, add_57);  mul_62 = add_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_44: "f32[8, 768, 384]" = torch.ops.aten.clone.default(mul_64);  mul_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_51: "f32[6144, 384]" = torch.ops.aten.view.default(clone_44, [6144, 384]);  clone_44 = None
    permute_39: "f32[384, 196]" = torch.ops.aten.permute.default(primals_79, [1, 0]);  primals_79 = None
    addmm_18: "f32[6144, 196]" = torch.ops.aten.addmm.default(primals_80, view_51, permute_39);  primals_80 = None
    view_52: "f32[8, 768, 196]" = torch.ops.aten.view.default(addmm_18, [8, 768, 196]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_45: "f32[8, 768, 196]" = torch.ops.aten.clone.default(view_52);  view_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_40: "f32[8, 196, 768]" = torch.ops.aten.permute.default(clone_45, [0, 2, 1]);  clone_45 = None
    add_58: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_53, permute_40);  add_53 = permute_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_46: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_58, memory_format = torch.contiguous_format)
    var_mean_13 = torch.ops.aten.var_mean.correction(clone_46, [2], correction = 0, keepdim = True)
    getitem_26: "f32[8, 196, 1]" = var_mean_13[0]
    getitem_27: "f32[8, 196, 1]" = var_mean_13[1];  var_mean_13 = None
    add_59: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-06);  getitem_26 = None
    rsqrt_13: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_59);  add_59 = None
    sub_13: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_46, getitem_27);  clone_46 = getitem_27 = None
    mul_65: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
    mul_66: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_65, primals_81)
    add_60: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_66, primals_82);  mul_66 = primals_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_53: "f32[1568, 768]" = torch.ops.aten.view.default(add_60, [1568, 768]);  add_60 = None
    permute_41: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_83, [1, 0]);  primals_83 = None
    addmm_19: "f32[1568, 3072]" = torch.ops.aten.addmm.default(primals_84, view_53, permute_41);  primals_84 = None
    view_54: "f32[8, 196, 3072]" = torch.ops.aten.view.default(addmm_19, [8, 196, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_67: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_54, 0.5)
    mul_68: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_54, 0.7071067811865476);  view_54 = None
    erf_13: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_68);  mul_68 = None
    add_61: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    mul_69: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_67, add_61);  mul_67 = add_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_47: "f32[8, 196, 3072]" = torch.ops.aten.clone.default(mul_69);  mul_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_55: "f32[1568, 3072]" = torch.ops.aten.view.default(clone_47, [1568, 3072]);  clone_47 = None
    permute_42: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_85, [1, 0]);  primals_85 = None
    addmm_20: "f32[1568, 768]" = torch.ops.aten.addmm.default(primals_86, view_55, permute_42);  primals_86 = None
    view_56: "f32[8, 196, 768]" = torch.ops.aten.view.default(addmm_20, [8, 196, 768]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_48: "f32[8, 196, 768]" = torch.ops.aten.clone.default(view_56);  view_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_62: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_58, clone_48);  add_58 = clone_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_49: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_62, memory_format = torch.contiguous_format)
    var_mean_14 = torch.ops.aten.var_mean.correction(clone_49, [2], correction = 0, keepdim = True)
    getitem_28: "f32[8, 196, 1]" = var_mean_14[0]
    getitem_29: "f32[8, 196, 1]" = var_mean_14[1];  var_mean_14 = None
    add_63: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-06);  getitem_28 = None
    rsqrt_14: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
    sub_14: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_49, getitem_29);  clone_49 = getitem_29 = None
    mul_70: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
    mul_71: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_70, primals_87)
    add_64: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_71, primals_88);  mul_71 = primals_88 = None
    permute_43: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_64, [0, 2, 1]);  add_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_44: "f32[196, 384]" = torch.ops.aten.permute.default(primals_89, [1, 0]);  primals_89 = None
    clone_50: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_43, memory_format = torch.contiguous_format);  permute_43 = None
    view_57: "f32[6144, 196]" = torch.ops.aten.view.default(clone_50, [6144, 196]);  clone_50 = None
    mm_7: "f32[6144, 384]" = torch.ops.aten.mm.default(view_57, permute_44)
    view_58: "f32[8, 768, 384]" = torch.ops.aten.view.default(mm_7, [8, 768, 384])
    add_65: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(view_58, primals_90);  view_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_72: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_65, 0.5)
    mul_73: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_65, 0.7071067811865476);  add_65 = None
    erf_14: "f32[8, 768, 384]" = torch.ops.aten.erf.default(mul_73);  mul_73 = None
    add_66: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    mul_74: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(mul_72, add_66);  mul_72 = add_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_51: "f32[8, 768, 384]" = torch.ops.aten.clone.default(mul_74);  mul_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_59: "f32[6144, 384]" = torch.ops.aten.view.default(clone_51, [6144, 384]);  clone_51 = None
    permute_45: "f32[384, 196]" = torch.ops.aten.permute.default(primals_91, [1, 0]);  primals_91 = None
    addmm_21: "f32[6144, 196]" = torch.ops.aten.addmm.default(primals_92, view_59, permute_45);  primals_92 = None
    view_60: "f32[8, 768, 196]" = torch.ops.aten.view.default(addmm_21, [8, 768, 196]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_52: "f32[8, 768, 196]" = torch.ops.aten.clone.default(view_60);  view_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_46: "f32[8, 196, 768]" = torch.ops.aten.permute.default(clone_52, [0, 2, 1]);  clone_52 = None
    add_67: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_62, permute_46);  add_62 = permute_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_53: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_67, memory_format = torch.contiguous_format)
    var_mean_15 = torch.ops.aten.var_mean.correction(clone_53, [2], correction = 0, keepdim = True)
    getitem_30: "f32[8, 196, 1]" = var_mean_15[0]
    getitem_31: "f32[8, 196, 1]" = var_mean_15[1];  var_mean_15 = None
    add_68: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-06);  getitem_30 = None
    rsqrt_15: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
    sub_15: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_53, getitem_31);  clone_53 = getitem_31 = None
    mul_75: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
    mul_76: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_75, primals_93)
    add_69: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_76, primals_94);  mul_76 = primals_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_61: "f32[1568, 768]" = torch.ops.aten.view.default(add_69, [1568, 768]);  add_69 = None
    permute_47: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_95, [1, 0]);  primals_95 = None
    addmm_22: "f32[1568, 3072]" = torch.ops.aten.addmm.default(primals_96, view_61, permute_47);  primals_96 = None
    view_62: "f32[8, 196, 3072]" = torch.ops.aten.view.default(addmm_22, [8, 196, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_77: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_62, 0.5)
    mul_78: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_62, 0.7071067811865476);  view_62 = None
    erf_15: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_78);  mul_78 = None
    add_70: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    mul_79: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_77, add_70);  mul_77 = add_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_54: "f32[8, 196, 3072]" = torch.ops.aten.clone.default(mul_79);  mul_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_63: "f32[1568, 3072]" = torch.ops.aten.view.default(clone_54, [1568, 3072]);  clone_54 = None
    permute_48: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_97, [1, 0]);  primals_97 = None
    addmm_23: "f32[1568, 768]" = torch.ops.aten.addmm.default(primals_98, view_63, permute_48);  primals_98 = None
    view_64: "f32[8, 196, 768]" = torch.ops.aten.view.default(addmm_23, [8, 196, 768]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_55: "f32[8, 196, 768]" = torch.ops.aten.clone.default(view_64);  view_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_71: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_67, clone_55);  add_67 = clone_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_56: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_71, memory_format = torch.contiguous_format)
    var_mean_16 = torch.ops.aten.var_mean.correction(clone_56, [2], correction = 0, keepdim = True)
    getitem_32: "f32[8, 196, 1]" = var_mean_16[0]
    getitem_33: "f32[8, 196, 1]" = var_mean_16[1];  var_mean_16 = None
    add_72: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-06);  getitem_32 = None
    rsqrt_16: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
    sub_16: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_56, getitem_33);  clone_56 = getitem_33 = None
    mul_80: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
    mul_81: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_80, primals_99)
    add_73: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_81, primals_100);  mul_81 = primals_100 = None
    permute_49: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_73, [0, 2, 1]);  add_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_50: "f32[196, 384]" = torch.ops.aten.permute.default(primals_101, [1, 0]);  primals_101 = None
    clone_57: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
    view_65: "f32[6144, 196]" = torch.ops.aten.view.default(clone_57, [6144, 196]);  clone_57 = None
    mm_8: "f32[6144, 384]" = torch.ops.aten.mm.default(view_65, permute_50)
    view_66: "f32[8, 768, 384]" = torch.ops.aten.view.default(mm_8, [8, 768, 384])
    add_74: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(view_66, primals_102);  view_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_82: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_74, 0.5)
    mul_83: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_74, 0.7071067811865476);  add_74 = None
    erf_16: "f32[8, 768, 384]" = torch.ops.aten.erf.default(mul_83);  mul_83 = None
    add_75: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    mul_84: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(mul_82, add_75);  mul_82 = add_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_58: "f32[8, 768, 384]" = torch.ops.aten.clone.default(mul_84);  mul_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_67: "f32[6144, 384]" = torch.ops.aten.view.default(clone_58, [6144, 384]);  clone_58 = None
    permute_51: "f32[384, 196]" = torch.ops.aten.permute.default(primals_103, [1, 0]);  primals_103 = None
    addmm_24: "f32[6144, 196]" = torch.ops.aten.addmm.default(primals_104, view_67, permute_51);  primals_104 = None
    view_68: "f32[8, 768, 196]" = torch.ops.aten.view.default(addmm_24, [8, 768, 196]);  addmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_59: "f32[8, 768, 196]" = torch.ops.aten.clone.default(view_68);  view_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_52: "f32[8, 196, 768]" = torch.ops.aten.permute.default(clone_59, [0, 2, 1]);  clone_59 = None
    add_76: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_71, permute_52);  add_71 = permute_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_60: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_76, memory_format = torch.contiguous_format)
    var_mean_17 = torch.ops.aten.var_mean.correction(clone_60, [2], correction = 0, keepdim = True)
    getitem_34: "f32[8, 196, 1]" = var_mean_17[0]
    getitem_35: "f32[8, 196, 1]" = var_mean_17[1];  var_mean_17 = None
    add_77: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-06);  getitem_34 = None
    rsqrt_17: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
    sub_17: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_60, getitem_35);  clone_60 = getitem_35 = None
    mul_85: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
    mul_86: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_85, primals_105)
    add_78: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_86, primals_106);  mul_86 = primals_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_69: "f32[1568, 768]" = torch.ops.aten.view.default(add_78, [1568, 768]);  add_78 = None
    permute_53: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_107, [1, 0]);  primals_107 = None
    addmm_25: "f32[1568, 3072]" = torch.ops.aten.addmm.default(primals_108, view_69, permute_53);  primals_108 = None
    view_70: "f32[8, 196, 3072]" = torch.ops.aten.view.default(addmm_25, [8, 196, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_87: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_70, 0.5)
    mul_88: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_70, 0.7071067811865476);  view_70 = None
    erf_17: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_88);  mul_88 = None
    add_79: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    mul_89: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_87, add_79);  mul_87 = add_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_61: "f32[8, 196, 3072]" = torch.ops.aten.clone.default(mul_89);  mul_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_71: "f32[1568, 3072]" = torch.ops.aten.view.default(clone_61, [1568, 3072]);  clone_61 = None
    permute_54: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_109, [1, 0]);  primals_109 = None
    addmm_26: "f32[1568, 768]" = torch.ops.aten.addmm.default(primals_110, view_71, permute_54);  primals_110 = None
    view_72: "f32[8, 196, 768]" = torch.ops.aten.view.default(addmm_26, [8, 196, 768]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_62: "f32[8, 196, 768]" = torch.ops.aten.clone.default(view_72);  view_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_80: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_76, clone_62);  add_76 = clone_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_63: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_80, memory_format = torch.contiguous_format)
    var_mean_18 = torch.ops.aten.var_mean.correction(clone_63, [2], correction = 0, keepdim = True)
    getitem_36: "f32[8, 196, 1]" = var_mean_18[0]
    getitem_37: "f32[8, 196, 1]" = var_mean_18[1];  var_mean_18 = None
    add_81: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-06);  getitem_36 = None
    rsqrt_18: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
    sub_18: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_63, getitem_37);  clone_63 = getitem_37 = None
    mul_90: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
    mul_91: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_90, primals_111)
    add_82: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_91, primals_112);  mul_91 = primals_112 = None
    permute_55: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_82, [0, 2, 1]);  add_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_56: "f32[196, 384]" = torch.ops.aten.permute.default(primals_113, [1, 0]);  primals_113 = None
    clone_64: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_55, memory_format = torch.contiguous_format);  permute_55 = None
    view_73: "f32[6144, 196]" = torch.ops.aten.view.default(clone_64, [6144, 196]);  clone_64 = None
    mm_9: "f32[6144, 384]" = torch.ops.aten.mm.default(view_73, permute_56)
    view_74: "f32[8, 768, 384]" = torch.ops.aten.view.default(mm_9, [8, 768, 384])
    add_83: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(view_74, primals_114);  view_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_92: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_83, 0.5)
    mul_93: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_83, 0.7071067811865476);  add_83 = None
    erf_18: "f32[8, 768, 384]" = torch.ops.aten.erf.default(mul_93);  mul_93 = None
    add_84: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    mul_94: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(mul_92, add_84);  mul_92 = add_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_65: "f32[8, 768, 384]" = torch.ops.aten.clone.default(mul_94);  mul_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_75: "f32[6144, 384]" = torch.ops.aten.view.default(clone_65, [6144, 384]);  clone_65 = None
    permute_57: "f32[384, 196]" = torch.ops.aten.permute.default(primals_115, [1, 0]);  primals_115 = None
    addmm_27: "f32[6144, 196]" = torch.ops.aten.addmm.default(primals_116, view_75, permute_57);  primals_116 = None
    view_76: "f32[8, 768, 196]" = torch.ops.aten.view.default(addmm_27, [8, 768, 196]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_66: "f32[8, 768, 196]" = torch.ops.aten.clone.default(view_76);  view_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_58: "f32[8, 196, 768]" = torch.ops.aten.permute.default(clone_66, [0, 2, 1]);  clone_66 = None
    add_85: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_80, permute_58);  add_80 = permute_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_67: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_85, memory_format = torch.contiguous_format)
    var_mean_19 = torch.ops.aten.var_mean.correction(clone_67, [2], correction = 0, keepdim = True)
    getitem_38: "f32[8, 196, 1]" = var_mean_19[0]
    getitem_39: "f32[8, 196, 1]" = var_mean_19[1];  var_mean_19 = None
    add_86: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-06);  getitem_38 = None
    rsqrt_19: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
    sub_19: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_67, getitem_39);  clone_67 = getitem_39 = None
    mul_95: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
    mul_96: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_95, primals_117)
    add_87: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_96, primals_118);  mul_96 = primals_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_77: "f32[1568, 768]" = torch.ops.aten.view.default(add_87, [1568, 768]);  add_87 = None
    permute_59: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_119, [1, 0]);  primals_119 = None
    addmm_28: "f32[1568, 3072]" = torch.ops.aten.addmm.default(primals_120, view_77, permute_59);  primals_120 = None
    view_78: "f32[8, 196, 3072]" = torch.ops.aten.view.default(addmm_28, [8, 196, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_97: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_78, 0.5)
    mul_98: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_78, 0.7071067811865476);  view_78 = None
    erf_19: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_98);  mul_98 = None
    add_88: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    mul_99: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_97, add_88);  mul_97 = add_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_68: "f32[8, 196, 3072]" = torch.ops.aten.clone.default(mul_99);  mul_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_79: "f32[1568, 3072]" = torch.ops.aten.view.default(clone_68, [1568, 3072]);  clone_68 = None
    permute_60: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_121, [1, 0]);  primals_121 = None
    addmm_29: "f32[1568, 768]" = torch.ops.aten.addmm.default(primals_122, view_79, permute_60);  primals_122 = None
    view_80: "f32[8, 196, 768]" = torch.ops.aten.view.default(addmm_29, [8, 196, 768]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_69: "f32[8, 196, 768]" = torch.ops.aten.clone.default(view_80);  view_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_89: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_85, clone_69);  add_85 = clone_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_70: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_89, memory_format = torch.contiguous_format)
    var_mean_20 = torch.ops.aten.var_mean.correction(clone_70, [2], correction = 0, keepdim = True)
    getitem_40: "f32[8, 196, 1]" = var_mean_20[0]
    getitem_41: "f32[8, 196, 1]" = var_mean_20[1];  var_mean_20 = None
    add_90: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-06);  getitem_40 = None
    rsqrt_20: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
    sub_20: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_70, getitem_41);  clone_70 = getitem_41 = None
    mul_100: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
    mul_101: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_100, primals_123)
    add_91: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_101, primals_124);  mul_101 = primals_124 = None
    permute_61: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_91, [0, 2, 1]);  add_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_62: "f32[196, 384]" = torch.ops.aten.permute.default(primals_125, [1, 0]);  primals_125 = None
    clone_71: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_61, memory_format = torch.contiguous_format);  permute_61 = None
    view_81: "f32[6144, 196]" = torch.ops.aten.view.default(clone_71, [6144, 196]);  clone_71 = None
    mm_10: "f32[6144, 384]" = torch.ops.aten.mm.default(view_81, permute_62)
    view_82: "f32[8, 768, 384]" = torch.ops.aten.view.default(mm_10, [8, 768, 384])
    add_92: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(view_82, primals_126);  view_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_102: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_92, 0.5)
    mul_103: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_92, 0.7071067811865476);  add_92 = None
    erf_20: "f32[8, 768, 384]" = torch.ops.aten.erf.default(mul_103);  mul_103 = None
    add_93: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    mul_104: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(mul_102, add_93);  mul_102 = add_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_72: "f32[8, 768, 384]" = torch.ops.aten.clone.default(mul_104);  mul_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_83: "f32[6144, 384]" = torch.ops.aten.view.default(clone_72, [6144, 384]);  clone_72 = None
    permute_63: "f32[384, 196]" = torch.ops.aten.permute.default(primals_127, [1, 0]);  primals_127 = None
    addmm_30: "f32[6144, 196]" = torch.ops.aten.addmm.default(primals_128, view_83, permute_63);  primals_128 = None
    view_84: "f32[8, 768, 196]" = torch.ops.aten.view.default(addmm_30, [8, 768, 196]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_73: "f32[8, 768, 196]" = torch.ops.aten.clone.default(view_84);  view_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_64: "f32[8, 196, 768]" = torch.ops.aten.permute.default(clone_73, [0, 2, 1]);  clone_73 = None
    add_94: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_89, permute_64);  add_89 = permute_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_74: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_94, memory_format = torch.contiguous_format)
    var_mean_21 = torch.ops.aten.var_mean.correction(clone_74, [2], correction = 0, keepdim = True)
    getitem_42: "f32[8, 196, 1]" = var_mean_21[0]
    getitem_43: "f32[8, 196, 1]" = var_mean_21[1];  var_mean_21 = None
    add_95: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-06);  getitem_42 = None
    rsqrt_21: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_95);  add_95 = None
    sub_21: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_74, getitem_43);  clone_74 = getitem_43 = None
    mul_105: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
    mul_106: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_105, primals_129)
    add_96: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_106, primals_130);  mul_106 = primals_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_85: "f32[1568, 768]" = torch.ops.aten.view.default(add_96, [1568, 768]);  add_96 = None
    permute_65: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_131, [1, 0]);  primals_131 = None
    addmm_31: "f32[1568, 3072]" = torch.ops.aten.addmm.default(primals_132, view_85, permute_65);  primals_132 = None
    view_86: "f32[8, 196, 3072]" = torch.ops.aten.view.default(addmm_31, [8, 196, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_107: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_86, 0.5)
    mul_108: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_86, 0.7071067811865476);  view_86 = None
    erf_21: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_108);  mul_108 = None
    add_97: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    mul_109: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_107, add_97);  mul_107 = add_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_75: "f32[8, 196, 3072]" = torch.ops.aten.clone.default(mul_109);  mul_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_87: "f32[1568, 3072]" = torch.ops.aten.view.default(clone_75, [1568, 3072]);  clone_75 = None
    permute_66: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_133, [1, 0]);  primals_133 = None
    addmm_32: "f32[1568, 768]" = torch.ops.aten.addmm.default(primals_134, view_87, permute_66);  primals_134 = None
    view_88: "f32[8, 196, 768]" = torch.ops.aten.view.default(addmm_32, [8, 196, 768]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_76: "f32[8, 196, 768]" = torch.ops.aten.clone.default(view_88);  view_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_98: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_94, clone_76);  add_94 = clone_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_77: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_98, memory_format = torch.contiguous_format)
    var_mean_22 = torch.ops.aten.var_mean.correction(clone_77, [2], correction = 0, keepdim = True)
    getitem_44: "f32[8, 196, 1]" = var_mean_22[0]
    getitem_45: "f32[8, 196, 1]" = var_mean_22[1];  var_mean_22 = None
    add_99: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-06);  getitem_44 = None
    rsqrt_22: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_99);  add_99 = None
    sub_22: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_77, getitem_45);  clone_77 = getitem_45 = None
    mul_110: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
    mul_111: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_110, primals_135)
    add_100: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_111, primals_136);  mul_111 = primals_136 = None
    permute_67: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_100, [0, 2, 1]);  add_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_68: "f32[196, 384]" = torch.ops.aten.permute.default(primals_137, [1, 0]);  primals_137 = None
    clone_78: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_67, memory_format = torch.contiguous_format);  permute_67 = None
    view_89: "f32[6144, 196]" = torch.ops.aten.view.default(clone_78, [6144, 196]);  clone_78 = None
    mm_11: "f32[6144, 384]" = torch.ops.aten.mm.default(view_89, permute_68)
    view_90: "f32[8, 768, 384]" = torch.ops.aten.view.default(mm_11, [8, 768, 384])
    add_101: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(view_90, primals_138);  view_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_112: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_101, 0.5)
    mul_113: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_101, 0.7071067811865476);  add_101 = None
    erf_22: "f32[8, 768, 384]" = torch.ops.aten.erf.default(mul_113);  mul_113 = None
    add_102: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    mul_114: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(mul_112, add_102);  mul_112 = add_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_79: "f32[8, 768, 384]" = torch.ops.aten.clone.default(mul_114);  mul_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_91: "f32[6144, 384]" = torch.ops.aten.view.default(clone_79, [6144, 384]);  clone_79 = None
    permute_69: "f32[384, 196]" = torch.ops.aten.permute.default(primals_139, [1, 0]);  primals_139 = None
    addmm_33: "f32[6144, 196]" = torch.ops.aten.addmm.default(primals_140, view_91, permute_69);  primals_140 = None
    view_92: "f32[8, 768, 196]" = torch.ops.aten.view.default(addmm_33, [8, 768, 196]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_80: "f32[8, 768, 196]" = torch.ops.aten.clone.default(view_92);  view_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_70: "f32[8, 196, 768]" = torch.ops.aten.permute.default(clone_80, [0, 2, 1]);  clone_80 = None
    add_103: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_98, permute_70);  add_98 = permute_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_81: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_103, memory_format = torch.contiguous_format)
    var_mean_23 = torch.ops.aten.var_mean.correction(clone_81, [2], correction = 0, keepdim = True)
    getitem_46: "f32[8, 196, 1]" = var_mean_23[0]
    getitem_47: "f32[8, 196, 1]" = var_mean_23[1];  var_mean_23 = None
    add_104: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-06);  getitem_46 = None
    rsqrt_23: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_104);  add_104 = None
    sub_23: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_81, getitem_47);  clone_81 = getitem_47 = None
    mul_115: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
    mul_116: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_115, primals_141)
    add_105: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_116, primals_142);  mul_116 = primals_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_93: "f32[1568, 768]" = torch.ops.aten.view.default(add_105, [1568, 768]);  add_105 = None
    permute_71: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_143, [1, 0]);  primals_143 = None
    addmm_34: "f32[1568, 3072]" = torch.ops.aten.addmm.default(primals_144, view_93, permute_71);  primals_144 = None
    view_94: "f32[8, 196, 3072]" = torch.ops.aten.view.default(addmm_34, [8, 196, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_117: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_94, 0.5)
    mul_118: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_94, 0.7071067811865476);  view_94 = None
    erf_23: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_118);  mul_118 = None
    add_106: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    mul_119: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_117, add_106);  mul_117 = add_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_82: "f32[8, 196, 3072]" = torch.ops.aten.clone.default(mul_119);  mul_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_95: "f32[1568, 3072]" = torch.ops.aten.view.default(clone_82, [1568, 3072]);  clone_82 = None
    permute_72: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_145, [1, 0]);  primals_145 = None
    addmm_35: "f32[1568, 768]" = torch.ops.aten.addmm.default(primals_146, view_95, permute_72);  primals_146 = None
    view_96: "f32[8, 196, 768]" = torch.ops.aten.view.default(addmm_35, [8, 196, 768]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_83: "f32[8, 196, 768]" = torch.ops.aten.clone.default(view_96);  view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_107: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_103, clone_83);  add_103 = clone_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:266, code: x = self.norm(x)
    clone_84: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_107, memory_format = torch.contiguous_format);  add_107 = None
    var_mean_24 = torch.ops.aten.var_mean.correction(clone_84, [2], correction = 0, keepdim = True)
    getitem_48: "f32[8, 196, 1]" = var_mean_24[0]
    getitem_49: "f32[8, 196, 1]" = var_mean_24[1];  var_mean_24 = None
    add_108: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-06);  getitem_48 = None
    rsqrt_24: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
    sub_24: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_84, getitem_49);  clone_84 = getitem_49 = None
    mul_120: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
    mul_121: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_120, primals_147)
    add_109: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_121, primals_148);  mul_121 = primals_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:271, code: x = x.mean(dim=1)
    mean: "f32[8, 768]" = torch.ops.aten.mean.dim(add_109, [1]);  add_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:272, code: x = self.head_drop(x)
    clone_85: "f32[8, 768]" = torch.ops.aten.clone.default(mean);  mean = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:273, code: return x if pre_logits else self.head(x)
    permute_73: "f32[768, 1000]" = torch.ops.aten.permute.default(primals_149, [1, 0]);  primals_149 = None
    addmm_36: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_150, clone_85, permute_73);  primals_150 = None
    permute_74: "f32[1000, 768]" = torch.ops.aten.permute.default(permute_73, [1, 0]);  permute_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:266, code: x = self.norm(x)
    div_1: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 768);  rsqrt_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_78: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_72, [1, 0]);  permute_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_82: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_71, [1, 0]);  permute_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    div_2: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_87: "f32[196, 384]" = torch.ops.aten.permute.default(permute_69, [1, 0]);  permute_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_93: "f32[384, 196]" = torch.ops.aten.permute.default(permute_68, [1, 0]);  permute_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    div_3: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_96: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_100: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    div_4: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_105: "f32[196, 384]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_111: "f32[384, 196]" = torch.ops.aten.permute.default(permute_62, [1, 0]);  permute_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    div_5: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_114: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_60, [1, 0]);  permute_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_118: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    div_6: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_123: "f32[196, 384]" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_129: "f32[384, 196]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    div_7: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_132: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_136: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    div_8: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_141: "f32[196, 384]" = torch.ops.aten.permute.default(permute_51, [1, 0]);  permute_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_147: "f32[384, 196]" = torch.ops.aten.permute.default(permute_50, [1, 0]);  permute_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    div_9: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_150: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_154: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    div_10: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_159: "f32[196, 384]" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_165: "f32[384, 196]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    div_11: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_168: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_172: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    div_12: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_177: "f32[196, 384]" = torch.ops.aten.permute.default(permute_39, [1, 0]);  permute_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_183: "f32[384, 196]" = torch.ops.aten.permute.default(permute_38, [1, 0]);  permute_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    div_13: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_186: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_190: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    div_14: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_195: "f32[196, 384]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_201: "f32[384, 196]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    div_15: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_204: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_208: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_29, [1, 0]);  permute_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    div_16: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_213: "f32[196, 384]" = torch.ops.aten.permute.default(permute_27, [1, 0]);  permute_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_219: "f32[384, 196]" = torch.ops.aten.permute.default(permute_26, [1, 0]);  permute_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    div_17: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_222: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_226: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    div_18: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_231: "f32[196, 384]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_237: "f32[384, 196]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    div_19: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_240: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_18, [1, 0]);  permute_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_244: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_17, [1, 0]);  permute_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    div_20: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_249: "f32[196, 384]" = torch.ops.aten.permute.default(permute_15, [1, 0]);  permute_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_255: "f32[384, 196]" = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    div_21: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_258: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_262: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    div_22: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_267: "f32[196, 384]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_273: "f32[384, 196]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    div_23: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_276: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_6, [1, 0]);  permute_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_280: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    div_24: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_285: "f32[196, 384]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_291: "f32[384, 196]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    div_25: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
    return [addmm_36, primals_1, primals_3, primals_6, primals_9, primals_15, primals_18, primals_21, primals_27, primals_30, primals_33, primals_39, primals_42, primals_45, primals_51, primals_54, primals_57, primals_63, primals_66, primals_69, primals_75, primals_78, primals_81, primals_87, primals_90, primals_93, primals_99, primals_102, primals_105, primals_111, primals_114, primals_117, primals_123, primals_126, primals_129, primals_135, primals_138, primals_141, primals_147, primals_151, mul, view_1, mm, view_3, mul_5, view_5, addmm_1, view_7, mul_10, view_9, mm_1, view_11, mul_15, view_13, addmm_4, view_15, mul_20, view_17, mm_2, view_19, mul_25, view_21, addmm_7, view_23, mul_30, view_25, mm_3, view_27, mul_35, view_29, addmm_10, view_31, mul_40, view_33, mm_4, view_35, mul_45, view_37, addmm_13, view_39, mul_50, view_41, mm_5, view_43, mul_55, view_45, addmm_16, view_47, mul_60, view_49, mm_6, view_51, mul_65, view_53, addmm_19, view_55, mul_70, view_57, mm_7, view_59, mul_75, view_61, addmm_22, view_63, mul_80, view_65, mm_8, view_67, mul_85, view_69, addmm_25, view_71, mul_90, view_73, mm_9, view_75, mul_95, view_77, addmm_28, view_79, mul_100, view_81, mm_10, view_83, mul_105, view_85, addmm_31, view_87, mul_110, view_89, mm_11, view_91, mul_115, view_93, addmm_34, view_95, mul_120, clone_85, permute_74, div_1, permute_78, permute_82, div_2, permute_87, permute_93, div_3, permute_96, permute_100, div_4, permute_105, permute_111, div_5, permute_114, permute_118, div_6, permute_123, permute_129, div_7, permute_132, permute_136, div_8, permute_141, permute_147, div_9, permute_150, permute_154, div_10, permute_159, permute_165, div_11, permute_168, permute_172, div_12, permute_177, permute_183, div_13, permute_186, permute_190, div_14, permute_195, permute_201, div_15, permute_204, permute_208, div_16, permute_213, permute_219, div_17, permute_222, permute_226, div_18, permute_231, permute_237, div_19, permute_240, permute_244, div_20, permute_249, permute_255, div_21, permute_258, permute_262, div_22, permute_267, permute_273, div_23, permute_276, permute_280, div_24, permute_285, permute_291, div_25]
    