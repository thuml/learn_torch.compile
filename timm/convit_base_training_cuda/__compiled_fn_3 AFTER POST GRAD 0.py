from __future__ import annotations



def forward(self, primals_1: "f32[1, 196, 768]", primals_2: "f32[1, 1, 768]", primals_3: "f32[768]", primals_4: "f32[768]", primals_5: "f32[16]", primals_6: "f32[768]", primals_7: "f32[768]", primals_8: "f32[768]", primals_9: "f32[768]", primals_10: "f32[16]", primals_11: "f32[768]", primals_12: "f32[768]", primals_13: "f32[768]", primals_14: "f32[768]", primals_15: "f32[16]", primals_16: "f32[768]", primals_17: "f32[768]", primals_18: "f32[768]", primals_19: "f32[768]", primals_20: "f32[16]", primals_21: "f32[768]", primals_22: "f32[768]", primals_23: "f32[768]", primals_24: "f32[768]", primals_25: "f32[16]", primals_26: "f32[768]", primals_27: "f32[768]", primals_28: "f32[768]", primals_29: "f32[768]", primals_30: "f32[16]", primals_31: "f32[768]", primals_32: "f32[768]", primals_33: "f32[768]", primals_34: "f32[768]", primals_35: "f32[16]", primals_36: "f32[768]", primals_37: "f32[768]", primals_38: "f32[768]", primals_39: "f32[768]", primals_40: "f32[16]", primals_41: "f32[768]", primals_42: "f32[768]", primals_43: "f32[768]", primals_44: "f32[768]", primals_45: "f32[16]", primals_46: "f32[768]", primals_47: "f32[768]", primals_48: "f32[768]", primals_49: "f32[768]", primals_50: "f32[16]", primals_51: "f32[768]", primals_52: "f32[768]", primals_53: "f32[768]", primals_54: "f32[768]", primals_55: "f32[768]", primals_56: "f32[768]", primals_57: "f32[768]", primals_58: "f32[768]", primals_59: "f32[768]", primals_60: "f32[768]", primals_61: "f32[768]", primals_62: "f32[768]", primals_63: "f32[768, 3, 16, 16]", primals_64: "f32[768]", primals_65: "f32[1536, 768]", primals_66: "f32[16, 3]", primals_67: "f32[16]", primals_68: "f32[768, 768]", primals_69: "f32[768, 768]", primals_70: "f32[768]", primals_71: "f32[3072, 768]", primals_72: "f32[3072]", primals_73: "f32[768, 3072]", primals_74: "f32[768]", primals_75: "f32[1536, 768]", primals_76: "f32[16, 3]", primals_77: "f32[16]", primals_78: "f32[768, 768]", primals_79: "f32[768, 768]", primals_80: "f32[768]", primals_81: "f32[3072, 768]", primals_82: "f32[3072]", primals_83: "f32[768, 3072]", primals_84: "f32[768]", primals_85: "f32[1536, 768]", primals_86: "f32[16, 3]", primals_87: "f32[16]", primals_88: "f32[768, 768]", primals_89: "f32[768, 768]", primals_90: "f32[768]", primals_91: "f32[3072, 768]", primals_92: "f32[3072]", primals_93: "f32[768, 3072]", primals_94: "f32[768]", primals_95: "f32[1536, 768]", primals_96: "f32[16, 3]", primals_97: "f32[16]", primals_98: "f32[768, 768]", primals_99: "f32[768, 768]", primals_100: "f32[768]", primals_101: "f32[3072, 768]", primals_102: "f32[3072]", primals_103: "f32[768, 3072]", primals_104: "f32[768]", primals_105: "f32[1536, 768]", primals_106: "f32[16, 3]", primals_107: "f32[16]", primals_108: "f32[768, 768]", primals_109: "f32[768, 768]", primals_110: "f32[768]", primals_111: "f32[3072, 768]", primals_112: "f32[3072]", primals_113: "f32[768, 3072]", primals_114: "f32[768]", primals_115: "f32[1536, 768]", primals_116: "f32[16, 3]", primals_117: "f32[16]", primals_118: "f32[768, 768]", primals_119: "f32[768, 768]", primals_120: "f32[768]", primals_121: "f32[3072, 768]", primals_122: "f32[3072]", primals_123: "f32[768, 3072]", primals_124: "f32[768]", primals_125: "f32[1536, 768]", primals_126: "f32[16, 3]", primals_127: "f32[16]", primals_128: "f32[768, 768]", primals_129: "f32[768, 768]", primals_130: "f32[768]", primals_131: "f32[3072, 768]", primals_132: "f32[3072]", primals_133: "f32[768, 3072]", primals_134: "f32[768]", primals_135: "f32[1536, 768]", primals_136: "f32[16, 3]", primals_137: "f32[16]", primals_138: "f32[768, 768]", primals_139: "f32[768, 768]", primals_140: "f32[768]", primals_141: "f32[3072, 768]", primals_142: "f32[3072]", primals_143: "f32[768, 3072]", primals_144: "f32[768]", primals_145: "f32[1536, 768]", primals_146: "f32[16, 3]", primals_147: "f32[16]", primals_148: "f32[768, 768]", primals_149: "f32[768, 768]", primals_150: "f32[768]", primals_151: "f32[3072, 768]", primals_152: "f32[3072]", primals_153: "f32[768, 3072]", primals_154: "f32[768]", primals_155: "f32[1536, 768]", primals_156: "f32[16, 3]", primals_157: "f32[16]", primals_158: "f32[768, 768]", primals_159: "f32[768, 768]", primals_160: "f32[768]", primals_161: "f32[3072, 768]", primals_162: "f32[3072]", primals_163: "f32[768, 3072]", primals_164: "f32[768]", primals_165: "f32[2304, 768]", primals_166: "f32[768, 768]", primals_167: "f32[768]", primals_168: "f32[3072, 768]", primals_169: "f32[3072]", primals_170: "f32[768, 3072]", primals_171: "f32[768]", primals_172: "f32[2304, 768]", primals_173: "f32[768, 768]", primals_174: "f32[768]", primals_175: "f32[3072, 768]", primals_176: "f32[3072]", primals_177: "f32[768, 3072]", primals_178: "f32[768]", primals_179: "f32[1000, 768]", primals_180: "f32[1000]", primals_181: "f32[8, 3, 224, 224]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution: "f32[8, 768, 14, 14]" = torch.ops.aten.convolution.default(primals_181, primals_63, primals_64, [16, 16], [0, 0], [1, 1], False, [0, 0], 1);  primals_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:89, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    view: "f32[8, 768, 196]" = torch.ops.aten.reshape.default(convolution, [8, 768, 196]);  convolution = None
    permute: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:362, code: x = x + self.pos_embed
    add: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(permute, primals_1);  permute = primals_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:364, code: cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
    expand: "f32[8, 1, 768]" = torch.ops.aten.expand.default(primals_2, [8, -1, -1]);  primals_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_1: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add, memory_format = torch.contiguous_format)
    var_mean = torch.ops.aten.var_mean.correction(clone_1, [2], correction = 0, keepdim = True)
    getitem: "f32[8, 196, 1]" = var_mean[0]
    getitem_1: "f32[8, 196, 1]" = var_mean[1];  var_mean = None
    add_1: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
    rsqrt: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    sub: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_1, getitem_1);  clone_1 = getitem_1 = None
    mul: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    mul_1: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul, primals_3)
    add_2: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_1, primals_4);  mul_1 = primals_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:121, code: rel_indices = torch.zeros(1, num_patches, num_patches, 3)
    full: "f32[1, 196, 196, 3]" = torch.ops.aten.full.default([1, 196, 196, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:122, code: ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
    iota: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_1: "i64[1, 14]" = torch.ops.aten.reshape.default(iota, [1, -1])
    view_2: "i64[14, 1]" = torch.ops.aten.reshape.default(iota, [-1, 1]);  iota = None
    sub_1: "i64[14, 14]" = torch.ops.aten.sub.Tensor(view_1, view_2);  view_1 = view_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:123, code: indx = ind.repeat(img_size, img_size)
    repeat: "i64[196, 196]" = torch.ops.aten.repeat.default(sub_1, [14, 14])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:124, code: indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
    unsqueeze: "i64[14, 1, 14]" = torch.ops.aten.unsqueeze.default(sub_1, 1);  sub_1 = None
    expand_1: "i64[14, 14, 14]" = torch.ops.aten.expand.default(unsqueeze, [14, 14, 14]);  unsqueeze = None
    clone_2: "i64[14, 14, 14]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
    view_3: "i64[196, 14]" = torch.ops.aten.reshape.default(clone_2, [196, 14]);  clone_2 = None
    unsqueeze_1: "i64[196, 14, 1]" = torch.ops.aten.unsqueeze.default(view_3, 2);  view_3 = None
    expand_2: "i64[196, 14, 14]" = torch.ops.aten.expand.default(unsqueeze_1, [196, 14, 14]);  unsqueeze_1 = None
    clone_3: "i64[196, 14, 14]" = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
    view_4: "i64[196, 196]" = torch.ops.aten.reshape.default(clone_3, [196, 196]);  clone_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:125, code: indd = indx ** 2 + indy ** 2
    pow_1: "i64[196, 196]" = torch.ops.aten.pow.Tensor_Scalar(repeat, 2)
    pow_2: "i64[196, 196]" = torch.ops.aten.pow.Tensor_Scalar(view_4, 2)
    add_3: "i64[196, 196]" = torch.ops.aten.add.Tensor(pow_1, pow_2);  pow_1 = pow_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:126, code: rel_indices[:, :, :, 2] = indd.unsqueeze(0)
    unsqueeze_2: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(add_3, 0);  add_3 = None
    select: "f32[1, 196, 196]" = torch.ops.aten.select.int(full, 3, 2)
    copy: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select, unsqueeze_2);  select = unsqueeze_2 = None
    select_scatter: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(full, copy, 3, 2);  full = copy = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:127, code: rel_indices[:, :, :, 1] = indy.unsqueeze(0)
    unsqueeze_3: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(view_4, 0);  view_4 = None
    select_3: "f32[1, 196, 196]" = torch.ops.aten.select.int(select_scatter, 3, 1)
    copy_1: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_3, unsqueeze_3);  select_3 = unsqueeze_3 = None
    select_scatter_1: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(select_scatter, copy_1, 3, 1);  select_scatter = copy_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:128, code: rel_indices[:, :, :, 0] = indx.unsqueeze(0)
    unsqueeze_4: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(repeat, 0);  repeat = None
    select_6: "f32[1, 196, 196]" = torch.ops.aten.select.int(select_scatter_1, 3, 0)
    copy_2: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_6, unsqueeze_4);  select_6 = unsqueeze_4 = None
    select_scatter_2: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(select_scatter_1, copy_2, 3, 0);  select_scatter_1 = copy_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:130, code: return rel_indices.to(device)
    device_put: "f32[1, 196, 196, 3]" = torch.ops.prims.device_put.default(select_scatter_2, device(type='cuda', index=0));  select_scatter_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_1: "f32[768, 1536]" = torch.ops.aten.permute.default(primals_65, [1, 0]);  primals_65 = None
    view_5: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_2, [1568, 768]);  add_2 = None
    mm: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_5, permute_1)
    view_6: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(mm, [8, 196, 1536]);  mm = None
    view_7: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.reshape.default(view_6, [8, 196, 2, 16, 48]);  view_6 = None
    permute_2: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.permute.default(view_7, [2, 0, 3, 1, 4]);  view_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    select_8: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_2, 0, 0)
    select_9: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_2, 0, 1);  permute_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:84, code: pos_score = self.rel_indices.expand(B, -1, -1, -1)
    expand_3: "f32[8, 196, 196, 3]" = torch.ops.aten.expand.default(device_put, [8, -1, -1, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    permute_3: "f32[3, 16]" = torch.ops.aten.permute.default(primals_66, [1, 0]);  primals_66 = None
    clone_4: "f32[8, 196, 196, 3]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
    view_8: "f32[307328, 3]" = torch.ops.aten.reshape.default(clone_4, [307328, 3]);  clone_4 = None
    mm_1: "f32[307328, 16]" = torch.ops.aten.mm.default(view_8, permute_3);  permute_3 = None
    view_9: "f32[8, 196, 196, 16]" = torch.ops.aten.reshape.default(mm_1, [8, 196, 196, 16]);  mm_1 = None
    add_4: "f32[8, 196, 196, 16]" = torch.ops.aten.add.Tensor(view_9, primals_67);  view_9 = primals_67 = None
    permute_4: "f32[8, 16, 196, 196]" = torch.ops.aten.permute.default(add_4, [0, 3, 1, 2]);  add_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    permute_5: "f32[8, 16, 48, 196]" = torch.ops.aten.permute.default(select_9, [0, 1, 3, 2]);  select_9 = None
    expand_4: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(select_8, [8, 16, 196, 48]);  select_8 = None
    clone_5: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
    view_10: "f32[128, 196, 48]" = torch.ops.aten.reshape.default(clone_5, [128, 196, 48]);  clone_5 = None
    expand_5: "f32[8, 16, 48, 196]" = torch.ops.aten.expand.default(permute_5, [8, 16, 48, 196]);  permute_5 = None
    clone_6: "f32[8, 16, 48, 196]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
    view_11: "f32[128, 48, 196]" = torch.ops.aten.reshape.default(clone_6, [128, 48, 196]);  clone_6 = None
    bmm: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_10, view_11)
    view_12: "f32[8, 16, 196, 196]" = torch.ops.aten.reshape.default(bmm, [8, 16, 196, 196]);  bmm = None
    mul_2: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(view_12, 0.14433756729740643);  view_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    amax: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(mul_2, [-1], True)
    sub_2: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_2, amax);  mul_2 = amax = None
    exp: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
    sum_1: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    clone_7: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
    amax_1: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(clone_7, [-1], True)
    sub_3: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(clone_7, amax_1);  clone_7 = amax_1 = None
    exp_1: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_3);  sub_3 = None
    sum_2: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_1: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_13: "f32[1, 16, 1, 1]" = torch.ops.aten.reshape.default(primals_5, [1, -1, 1, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    sigmoid: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_13);  view_13 = None
    sub_4: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1.0, sigmoid)
    mul_3: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_4, div);  sub_4 = None
    mul_4: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sigmoid, div_1);  sigmoid = None
    add_5: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(mul_3, mul_4);  mul_3 = mul_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    sum_3: "f32[8, 16, 196]" = torch.ops.aten.sum.dim_IntList(add_5, [-1])
    unsqueeze_5: "f32[8, 16, 196, 1]" = torch.ops.aten.unsqueeze.default(sum_3, -1);  sum_3 = None
    div_2: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(add_5, unsqueeze_5);  add_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_6: "f32[768, 768]" = torch.ops.aten.permute.default(primals_68, [1, 0]);  primals_68 = None
    mm_2: "f32[1568, 768]" = torch.ops.aten.mm.default(view_5, permute_6)
    view_15: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_2, [8, 196, 768]);  mm_2 = None
    view_16: "f32[8, 196, 16, 48]" = torch.ops.aten.reshape.default(view_15, [8, 196, 16, 48]);  view_15 = None
    permute_7: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_6: "f32[8, 16, 196, 196]" = torch.ops.aten.expand.default(div_2, [8, 16, 196, 196]);  div_2 = None
    view_17: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(expand_6, [128, 196, 196]);  expand_6 = None
    expand_7: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(permute_7, [8, 16, 196, 48]);  permute_7 = None
    clone_10: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
    view_18: "f32[128, 196, 48]" = torch.ops.aten.reshape.default(clone_10, [128, 196, 48]);  clone_10 = None
    bmm_1: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(view_17, view_18)
    view_19: "f32[8, 16, 196, 48]" = torch.ops.aten.reshape.default(bmm_1, [8, 16, 196, 48]);  bmm_1 = None
    permute_8: "f32[8, 196, 16, 48]" = torch.ops.aten.permute.default(view_19, [0, 2, 1, 3]);  view_19 = None
    clone_11: "f32[8, 196, 16, 48]" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None
    view_20: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(clone_11, [8, 196, 768]);  clone_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_21: "f32[1568, 768]" = torch.ops.aten.reshape.default(view_20, [1568, 768]);  view_20 = None
    permute_9: "f32[768, 768]" = torch.ops.aten.permute.default(primals_69, [1, 0]);  primals_69 = None
    
    # No stacktrace found for following nodes
    mm_default_23: "f32[1568, 768]" = torch.ops.aten.mm.default(view_21, permute_9)
    add_tensor_23: "f32[1568, 768]" = torch.ops.aten.add.Tensor(mm_default_23, primals_70);  mm_default_23 = primals_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_22: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(add_tensor_23, [8, 196, 768]);  add_tensor_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:235, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_6: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add, view_22);  add = view_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_13: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_6, memory_format = torch.contiguous_format)
    var_mean_1 = torch.ops.aten.var_mean.correction(clone_13, [2], correction = 0, keepdim = True)
    getitem_2: "f32[8, 196, 1]" = var_mean_1[0]
    getitem_3: "f32[8, 196, 1]" = var_mean_1[1];  var_mean_1 = None
    add_7: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-06);  getitem_2 = None
    rsqrt_1: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
    sub_5: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_13, getitem_3);  clone_13 = getitem_3 = None
    mul_5: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_1);  sub_5 = None
    mul_6: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_5, primals_6)
    add_8: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_6, primals_7);  mul_6 = primals_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_23: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_8, [1568, 768]);  add_8 = None
    permute_10: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_71, [1, 0]);  primals_71 = None
    addmm_1: "f32[1568, 3072]" = torch.ops.aten.addmm.default(primals_72, view_23, permute_10);  primals_72 = None
    view_24: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(addmm_1, [8, 196, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_7: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_24, 0.5)
    mul_8: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_24, 0.7071067811865476);  view_24 = None
    erf: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_8);  mul_8 = None
    add_9: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_9: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_7, add_9);  mul_7 = add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_25: "f32[1568, 3072]" = torch.ops.aten.reshape.default(mul_9, [1568, 3072]);  mul_9 = None
    permute_11: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_73, [1, 0]);  primals_73 = None
    
    # No stacktrace found for following nodes
    mm_default_22: "f32[1568, 768]" = torch.ops.aten.mm.default(view_25, permute_11)
    add_tensor_22: "f32[1568, 768]" = torch.ops.aten.add.Tensor(mm_default_22, primals_74);  mm_default_22 = primals_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_26: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(add_tensor_22, [8, 196, 768]);  add_tensor_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:236, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_10: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_6, view_26);  add_6 = view_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_16: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_10, memory_format = torch.contiguous_format)
    var_mean_2 = torch.ops.aten.var_mean.correction(clone_16, [2], correction = 0, keepdim = True)
    getitem_4: "f32[8, 196, 1]" = var_mean_2[0]
    getitem_5: "f32[8, 196, 1]" = var_mean_2[1];  var_mean_2 = None
    add_11: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-06);  getitem_4 = None
    rsqrt_2: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    sub_6: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_16, getitem_5);  clone_16 = getitem_5 = None
    mul_10: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_2);  sub_6 = None
    mul_11: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_10, primals_8)
    add_12: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_11, primals_9);  mul_11 = primals_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_12: "f32[768, 1536]" = torch.ops.aten.permute.default(primals_75, [1, 0]);  primals_75 = None
    view_31: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_12, [1568, 768]);  add_12 = None
    mm_3: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_31, permute_12)
    view_32: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(mm_3, [8, 196, 1536]);  mm_3 = None
    view_33: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.reshape.default(view_32, [8, 196, 2, 16, 48]);  view_32 = None
    permute_13: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.permute.default(view_33, [2, 0, 3, 1, 4]);  view_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    select_18: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_13, 0, 0)
    select_19: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_13, 0, 1);  permute_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    permute_14: "f32[3, 16]" = torch.ops.aten.permute.default(primals_76, [1, 0]);  primals_76 = None
    mm_4: "f32[307328, 16]" = torch.ops.aten.mm.default(view_8, permute_14);  permute_14 = None
    view_35: "f32[8, 196, 196, 16]" = torch.ops.aten.reshape.default(mm_4, [8, 196, 196, 16]);  mm_4 = None
    add_14: "f32[8, 196, 196, 16]" = torch.ops.aten.add.Tensor(view_35, primals_77);  view_35 = primals_77 = None
    permute_15: "f32[8, 16, 196, 196]" = torch.ops.aten.permute.default(add_14, [0, 3, 1, 2]);  add_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    permute_16: "f32[8, 16, 48, 196]" = torch.ops.aten.permute.default(select_19, [0, 1, 3, 2]);  select_19 = None
    expand_11: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(select_18, [8, 16, 196, 48]);  select_18 = None
    clone_20: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
    view_36: "f32[128, 196, 48]" = torch.ops.aten.reshape.default(clone_20, [128, 196, 48]);  clone_20 = None
    expand_12: "f32[8, 16, 48, 196]" = torch.ops.aten.expand.default(permute_16, [8, 16, 48, 196]);  permute_16 = None
    clone_21: "f32[8, 16, 48, 196]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
    view_37: "f32[128, 48, 196]" = torch.ops.aten.reshape.default(clone_21, [128, 48, 196]);  clone_21 = None
    bmm_2: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_36, view_37)
    view_38: "f32[8, 16, 196, 196]" = torch.ops.aten.reshape.default(bmm_2, [8, 16, 196, 196]);  bmm_2 = None
    mul_12: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(view_38, 0.14433756729740643);  view_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    amax_2: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(mul_12, [-1], True)
    sub_8: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_12, amax_2);  mul_12 = amax_2 = None
    exp_2: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_8);  sub_8 = None
    sum_4: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_3: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_2, sum_4);  exp_2 = sum_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    clone_22: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
    amax_3: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(clone_22, [-1], True)
    sub_9: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(clone_22, amax_3);  clone_22 = amax_3 = None
    exp_3: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_9);  sub_9 = None
    sum_5: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_4: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_3, sum_5);  exp_3 = sum_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_39: "f32[1, 16, 1, 1]" = torch.ops.aten.reshape.default(primals_10, [1, -1, 1, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    sigmoid_2: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_39);  view_39 = None
    sub_10: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1.0, sigmoid_2)
    mul_13: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_10, div_3);  sub_10 = None
    mul_14: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sigmoid_2, div_4);  sigmoid_2 = None
    add_15: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(mul_13, mul_14);  mul_13 = mul_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    sum_6: "f32[8, 16, 196]" = torch.ops.aten.sum.dim_IntList(add_15, [-1])
    unsqueeze_11: "f32[8, 16, 196, 1]" = torch.ops.aten.unsqueeze.default(sum_6, -1);  sum_6 = None
    div_5: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(add_15, unsqueeze_11);  add_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_17: "f32[768, 768]" = torch.ops.aten.permute.default(primals_78, [1, 0]);  primals_78 = None
    mm_5: "f32[1568, 768]" = torch.ops.aten.mm.default(view_31, permute_17)
    view_41: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_5, [8, 196, 768]);  mm_5 = None
    view_42: "f32[8, 196, 16, 48]" = torch.ops.aten.reshape.default(view_41, [8, 196, 16, 48]);  view_41 = None
    permute_18: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_42, [0, 2, 1, 3]);  view_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_13: "f32[8, 16, 196, 196]" = torch.ops.aten.expand.default(div_5, [8, 16, 196, 196]);  div_5 = None
    view_43: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(expand_13, [128, 196, 196]);  expand_13 = None
    expand_14: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(permute_18, [8, 16, 196, 48]);  permute_18 = None
    clone_25: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_14, memory_format = torch.contiguous_format);  expand_14 = None
    view_44: "f32[128, 196, 48]" = torch.ops.aten.reshape.default(clone_25, [128, 196, 48]);  clone_25 = None
    bmm_3: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(view_43, view_44)
    view_45: "f32[8, 16, 196, 48]" = torch.ops.aten.reshape.default(bmm_3, [8, 16, 196, 48]);  bmm_3 = None
    permute_19: "f32[8, 196, 16, 48]" = torch.ops.aten.permute.default(view_45, [0, 2, 1, 3]);  view_45 = None
    clone_26: "f32[8, 196, 16, 48]" = torch.ops.aten.clone.default(permute_19, memory_format = torch.contiguous_format);  permute_19 = None
    view_46: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(clone_26, [8, 196, 768]);  clone_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_47: "f32[1568, 768]" = torch.ops.aten.reshape.default(view_46, [1568, 768]);  view_46 = None
    permute_20: "f32[768, 768]" = torch.ops.aten.permute.default(primals_79, [1, 0]);  primals_79 = None
    
    # No stacktrace found for following nodes
    mm_default_21: "f32[1568, 768]" = torch.ops.aten.mm.default(view_47, permute_20)
    add_tensor_21: "f32[1568, 768]" = torch.ops.aten.add.Tensor(mm_default_21, primals_80);  mm_default_21 = primals_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_48: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(add_tensor_21, [8, 196, 768]);  add_tensor_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:235, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_16: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_10, view_48);  add_10 = view_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_28: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_16, memory_format = torch.contiguous_format)
    var_mean_3 = torch.ops.aten.var_mean.correction(clone_28, [2], correction = 0, keepdim = True)
    getitem_6: "f32[8, 196, 1]" = var_mean_3[0]
    getitem_7: "f32[8, 196, 1]" = var_mean_3[1];  var_mean_3 = None
    add_17: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-06);  getitem_6 = None
    rsqrt_3: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
    sub_11: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_28, getitem_7);  clone_28 = getitem_7 = None
    mul_15: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_3);  sub_11 = None
    mul_16: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_15, primals_11)
    add_18: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_16, primals_12);  mul_16 = primals_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_49: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_18, [1568, 768]);  add_18 = None
    permute_21: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_81, [1, 0]);  primals_81 = None
    addmm_4: "f32[1568, 3072]" = torch.ops.aten.addmm.default(primals_82, view_49, permute_21);  primals_82 = None
    view_50: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(addmm_4, [8, 196, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_17: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_50, 0.5)
    mul_18: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_50, 0.7071067811865476);  view_50 = None
    erf_1: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_18);  mul_18 = None
    add_19: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_19: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_17, add_19);  mul_17 = add_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_51: "f32[1568, 3072]" = torch.ops.aten.reshape.default(mul_19, [1568, 3072]);  mul_19 = None
    permute_22: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_83, [1, 0]);  primals_83 = None
    
    # No stacktrace found for following nodes
    mm_default_20: "f32[1568, 768]" = torch.ops.aten.mm.default(view_51, permute_22)
    add_tensor_20: "f32[1568, 768]" = torch.ops.aten.add.Tensor(mm_default_20, primals_84);  mm_default_20 = primals_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_52: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(add_tensor_20, [8, 196, 768]);  add_tensor_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:236, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_20: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_16, view_52);  add_16 = view_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_31: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_20, memory_format = torch.contiguous_format)
    var_mean_4 = torch.ops.aten.var_mean.correction(clone_31, [2], correction = 0, keepdim = True)
    getitem_8: "f32[8, 196, 1]" = var_mean_4[0]
    getitem_9: "f32[8, 196, 1]" = var_mean_4[1];  var_mean_4 = None
    add_21: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-06);  getitem_8 = None
    rsqrt_4: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
    sub_12: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_31, getitem_9);  clone_31 = getitem_9 = None
    mul_20: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_4);  sub_12 = None
    mul_21: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_20, primals_13)
    add_22: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_21, primals_14);  mul_21 = primals_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_23: "f32[768, 1536]" = torch.ops.aten.permute.default(primals_85, [1, 0]);  primals_85 = None
    view_57: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_22, [1568, 768]);  add_22 = None
    mm_6: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_57, permute_23)
    view_58: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(mm_6, [8, 196, 1536]);  mm_6 = None
    view_59: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.reshape.default(view_58, [8, 196, 2, 16, 48]);  view_58 = None
    permute_24: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.permute.default(view_59, [2, 0, 3, 1, 4]);  view_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    select_28: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_24, 0, 0)
    select_29: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_24, 0, 1);  permute_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    permute_25: "f32[3, 16]" = torch.ops.aten.permute.default(primals_86, [1, 0]);  primals_86 = None
    mm_7: "f32[307328, 16]" = torch.ops.aten.mm.default(view_8, permute_25);  permute_25 = None
    view_61: "f32[8, 196, 196, 16]" = torch.ops.aten.reshape.default(mm_7, [8, 196, 196, 16]);  mm_7 = None
    add_24: "f32[8, 196, 196, 16]" = torch.ops.aten.add.Tensor(view_61, primals_87);  view_61 = primals_87 = None
    permute_26: "f32[8, 16, 196, 196]" = torch.ops.aten.permute.default(add_24, [0, 3, 1, 2]);  add_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    permute_27: "f32[8, 16, 48, 196]" = torch.ops.aten.permute.default(select_29, [0, 1, 3, 2]);  select_29 = None
    expand_18: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(select_28, [8, 16, 196, 48]);  select_28 = None
    clone_35: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
    view_62: "f32[128, 196, 48]" = torch.ops.aten.reshape.default(clone_35, [128, 196, 48]);  clone_35 = None
    expand_19: "f32[8, 16, 48, 196]" = torch.ops.aten.expand.default(permute_27, [8, 16, 48, 196]);  permute_27 = None
    clone_36: "f32[8, 16, 48, 196]" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
    view_63: "f32[128, 48, 196]" = torch.ops.aten.reshape.default(clone_36, [128, 48, 196]);  clone_36 = None
    bmm_4: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_62, view_63)
    view_64: "f32[8, 16, 196, 196]" = torch.ops.aten.reshape.default(bmm_4, [8, 16, 196, 196]);  bmm_4 = None
    mul_22: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(view_64, 0.14433756729740643);  view_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    amax_4: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(mul_22, [-1], True)
    sub_14: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_22, amax_4);  mul_22 = amax_4 = None
    exp_4: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_14);  sub_14 = None
    sum_7: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_6: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_4, sum_7);  exp_4 = sum_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    clone_37: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(permute_26, memory_format = torch.contiguous_format);  permute_26 = None
    amax_5: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(clone_37, [-1], True)
    sub_15: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(clone_37, amax_5);  clone_37 = amax_5 = None
    exp_5: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_15);  sub_15 = None
    sum_8: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_7: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_5, sum_8);  exp_5 = sum_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_65: "f32[1, 16, 1, 1]" = torch.ops.aten.reshape.default(primals_15, [1, -1, 1, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    sigmoid_4: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_65);  view_65 = None
    sub_16: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1.0, sigmoid_4)
    mul_23: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_16, div_6);  sub_16 = None
    mul_24: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sigmoid_4, div_7);  sigmoid_4 = None
    add_25: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(mul_23, mul_24);  mul_23 = mul_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    sum_9: "f32[8, 16, 196]" = torch.ops.aten.sum.dim_IntList(add_25, [-1])
    unsqueeze_17: "f32[8, 16, 196, 1]" = torch.ops.aten.unsqueeze.default(sum_9, -1);  sum_9 = None
    div_8: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(add_25, unsqueeze_17);  add_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_28: "f32[768, 768]" = torch.ops.aten.permute.default(primals_88, [1, 0]);  primals_88 = None
    mm_8: "f32[1568, 768]" = torch.ops.aten.mm.default(view_57, permute_28)
    view_67: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_8, [8, 196, 768]);  mm_8 = None
    view_68: "f32[8, 196, 16, 48]" = torch.ops.aten.reshape.default(view_67, [8, 196, 16, 48]);  view_67 = None
    permute_29: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_68, [0, 2, 1, 3]);  view_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_20: "f32[8, 16, 196, 196]" = torch.ops.aten.expand.default(div_8, [8, 16, 196, 196]);  div_8 = None
    view_69: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(expand_20, [128, 196, 196]);  expand_20 = None
    expand_21: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(permute_29, [8, 16, 196, 48]);  permute_29 = None
    clone_40: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
    view_70: "f32[128, 196, 48]" = torch.ops.aten.reshape.default(clone_40, [128, 196, 48]);  clone_40 = None
    bmm_5: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(view_69, view_70)
    view_71: "f32[8, 16, 196, 48]" = torch.ops.aten.reshape.default(bmm_5, [8, 16, 196, 48]);  bmm_5 = None
    permute_30: "f32[8, 196, 16, 48]" = torch.ops.aten.permute.default(view_71, [0, 2, 1, 3]);  view_71 = None
    clone_41: "f32[8, 196, 16, 48]" = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None
    view_72: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(clone_41, [8, 196, 768]);  clone_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_73: "f32[1568, 768]" = torch.ops.aten.reshape.default(view_72, [1568, 768]);  view_72 = None
    permute_31: "f32[768, 768]" = torch.ops.aten.permute.default(primals_89, [1, 0]);  primals_89 = None
    
    # No stacktrace found for following nodes
    mm_default_19: "f32[1568, 768]" = torch.ops.aten.mm.default(view_73, permute_31)
    add_tensor_19: "f32[1568, 768]" = torch.ops.aten.add.Tensor(mm_default_19, primals_90);  mm_default_19 = primals_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_74: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(add_tensor_19, [8, 196, 768]);  add_tensor_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:235, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_26: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_20, view_74);  add_20 = view_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_43: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_26, memory_format = torch.contiguous_format)
    var_mean_5 = torch.ops.aten.var_mean.correction(clone_43, [2], correction = 0, keepdim = True)
    getitem_10: "f32[8, 196, 1]" = var_mean_5[0]
    getitem_11: "f32[8, 196, 1]" = var_mean_5[1];  var_mean_5 = None
    add_27: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-06);  getitem_10 = None
    rsqrt_5: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
    sub_17: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_43, getitem_11);  clone_43 = getitem_11 = None
    mul_25: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_5);  sub_17 = None
    mul_26: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_25, primals_16)
    add_28: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_26, primals_17);  mul_26 = primals_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_75: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_28, [1568, 768]);  add_28 = None
    permute_32: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_91, [1, 0]);  primals_91 = None
    addmm_7: "f32[1568, 3072]" = torch.ops.aten.addmm.default(primals_92, view_75, permute_32);  primals_92 = None
    view_76: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(addmm_7, [8, 196, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_27: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_76, 0.5)
    mul_28: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_76, 0.7071067811865476);  view_76 = None
    erf_2: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_28);  mul_28 = None
    add_29: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_29: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_27, add_29);  mul_27 = add_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_77: "f32[1568, 3072]" = torch.ops.aten.reshape.default(mul_29, [1568, 3072]);  mul_29 = None
    permute_33: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_93, [1, 0]);  primals_93 = None
    
    # No stacktrace found for following nodes
    mm_default_18: "f32[1568, 768]" = torch.ops.aten.mm.default(view_77, permute_33)
    add_tensor_18: "f32[1568, 768]" = torch.ops.aten.add.Tensor(mm_default_18, primals_94);  mm_default_18 = primals_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_78: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(add_tensor_18, [8, 196, 768]);  add_tensor_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:236, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_30: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_26, view_78);  add_26 = view_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_46: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_30, memory_format = torch.contiguous_format)
    var_mean_6 = torch.ops.aten.var_mean.correction(clone_46, [2], correction = 0, keepdim = True)
    getitem_12: "f32[8, 196, 1]" = var_mean_6[0]
    getitem_13: "f32[8, 196, 1]" = var_mean_6[1];  var_mean_6 = None
    add_31: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-06);  getitem_12 = None
    rsqrt_6: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
    sub_18: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_46, getitem_13);  clone_46 = getitem_13 = None
    mul_30: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_6);  sub_18 = None
    mul_31: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_30, primals_18)
    add_32: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_31, primals_19);  mul_31 = primals_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_34: "f32[768, 1536]" = torch.ops.aten.permute.default(primals_95, [1, 0]);  primals_95 = None
    view_83: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_32, [1568, 768]);  add_32 = None
    mm_9: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_83, permute_34)
    view_84: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(mm_9, [8, 196, 1536]);  mm_9 = None
    view_85: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.reshape.default(view_84, [8, 196, 2, 16, 48]);  view_84 = None
    permute_35: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.permute.default(view_85, [2, 0, 3, 1, 4]);  view_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    select_38: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_35, 0, 0)
    select_39: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_35, 0, 1);  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    permute_36: "f32[3, 16]" = torch.ops.aten.permute.default(primals_96, [1, 0]);  primals_96 = None
    mm_10: "f32[307328, 16]" = torch.ops.aten.mm.default(view_8, permute_36);  permute_36 = None
    view_87: "f32[8, 196, 196, 16]" = torch.ops.aten.reshape.default(mm_10, [8, 196, 196, 16]);  mm_10 = None
    add_34: "f32[8, 196, 196, 16]" = torch.ops.aten.add.Tensor(view_87, primals_97);  view_87 = primals_97 = None
    permute_37: "f32[8, 16, 196, 196]" = torch.ops.aten.permute.default(add_34, [0, 3, 1, 2]);  add_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    permute_38: "f32[8, 16, 48, 196]" = torch.ops.aten.permute.default(select_39, [0, 1, 3, 2]);  select_39 = None
    expand_25: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(select_38, [8, 16, 196, 48]);  select_38 = None
    clone_50: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
    view_88: "f32[128, 196, 48]" = torch.ops.aten.reshape.default(clone_50, [128, 196, 48]);  clone_50 = None
    expand_26: "f32[8, 16, 48, 196]" = torch.ops.aten.expand.default(permute_38, [8, 16, 48, 196]);  permute_38 = None
    clone_51: "f32[8, 16, 48, 196]" = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
    view_89: "f32[128, 48, 196]" = torch.ops.aten.reshape.default(clone_51, [128, 48, 196]);  clone_51 = None
    bmm_6: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_88, view_89)
    view_90: "f32[8, 16, 196, 196]" = torch.ops.aten.reshape.default(bmm_6, [8, 16, 196, 196]);  bmm_6 = None
    mul_32: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(view_90, 0.14433756729740643);  view_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    amax_6: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(mul_32, [-1], True)
    sub_20: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_32, amax_6);  mul_32 = amax_6 = None
    exp_6: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_20);  sub_20 = None
    sum_10: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_9: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_6, sum_10);  exp_6 = sum_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    clone_52: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
    amax_7: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(clone_52, [-1], True)
    sub_21: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(clone_52, amax_7);  clone_52 = amax_7 = None
    exp_7: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_21);  sub_21 = None
    sum_11: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_10: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_7, sum_11);  exp_7 = sum_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_91: "f32[1, 16, 1, 1]" = torch.ops.aten.reshape.default(primals_20, [1, -1, 1, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    sigmoid_6: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_91);  view_91 = None
    sub_22: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1.0, sigmoid_6)
    mul_33: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_22, div_9);  sub_22 = None
    mul_34: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sigmoid_6, div_10);  sigmoid_6 = None
    add_35: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(mul_33, mul_34);  mul_33 = mul_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    sum_12: "f32[8, 16, 196]" = torch.ops.aten.sum.dim_IntList(add_35, [-1])
    unsqueeze_23: "f32[8, 16, 196, 1]" = torch.ops.aten.unsqueeze.default(sum_12, -1);  sum_12 = None
    div_11: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(add_35, unsqueeze_23);  add_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_39: "f32[768, 768]" = torch.ops.aten.permute.default(primals_98, [1, 0]);  primals_98 = None
    mm_11: "f32[1568, 768]" = torch.ops.aten.mm.default(view_83, permute_39)
    view_93: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_11, [8, 196, 768]);  mm_11 = None
    view_94: "f32[8, 196, 16, 48]" = torch.ops.aten.reshape.default(view_93, [8, 196, 16, 48]);  view_93 = None
    permute_40: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_27: "f32[8, 16, 196, 196]" = torch.ops.aten.expand.default(div_11, [8, 16, 196, 196]);  div_11 = None
    view_95: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(expand_27, [128, 196, 196]);  expand_27 = None
    expand_28: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(permute_40, [8, 16, 196, 48]);  permute_40 = None
    clone_55: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
    view_96: "f32[128, 196, 48]" = torch.ops.aten.reshape.default(clone_55, [128, 196, 48]);  clone_55 = None
    bmm_7: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(view_95, view_96)
    view_97: "f32[8, 16, 196, 48]" = torch.ops.aten.reshape.default(bmm_7, [8, 16, 196, 48]);  bmm_7 = None
    permute_41: "f32[8, 196, 16, 48]" = torch.ops.aten.permute.default(view_97, [0, 2, 1, 3]);  view_97 = None
    clone_56: "f32[8, 196, 16, 48]" = torch.ops.aten.clone.default(permute_41, memory_format = torch.contiguous_format);  permute_41 = None
    view_98: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(clone_56, [8, 196, 768]);  clone_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_99: "f32[1568, 768]" = torch.ops.aten.reshape.default(view_98, [1568, 768]);  view_98 = None
    permute_42: "f32[768, 768]" = torch.ops.aten.permute.default(primals_99, [1, 0]);  primals_99 = None
    
    # No stacktrace found for following nodes
    mm_default_17: "f32[1568, 768]" = torch.ops.aten.mm.default(view_99, permute_42)
    add_tensor_17: "f32[1568, 768]" = torch.ops.aten.add.Tensor(mm_default_17, primals_100);  mm_default_17 = primals_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_100: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(add_tensor_17, [8, 196, 768]);  add_tensor_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:235, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_36: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_30, view_100);  add_30 = view_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_58: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_36, memory_format = torch.contiguous_format)
    var_mean_7 = torch.ops.aten.var_mean.correction(clone_58, [2], correction = 0, keepdim = True)
    getitem_14: "f32[8, 196, 1]" = var_mean_7[0]
    getitem_15: "f32[8, 196, 1]" = var_mean_7[1];  var_mean_7 = None
    add_37: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-06);  getitem_14 = None
    rsqrt_7: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
    sub_23: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_58, getitem_15);  clone_58 = getitem_15 = None
    mul_35: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_7);  sub_23 = None
    mul_36: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_35, primals_21)
    add_38: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_36, primals_22);  mul_36 = primals_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_101: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_38, [1568, 768]);  add_38 = None
    permute_43: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_101, [1, 0]);  primals_101 = None
    addmm_10: "f32[1568, 3072]" = torch.ops.aten.addmm.default(primals_102, view_101, permute_43);  primals_102 = None
    view_102: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(addmm_10, [8, 196, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_37: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_102, 0.5)
    mul_38: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_102, 0.7071067811865476);  view_102 = None
    erf_3: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_38);  mul_38 = None
    add_39: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_39: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_37, add_39);  mul_37 = add_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_103: "f32[1568, 3072]" = torch.ops.aten.reshape.default(mul_39, [1568, 3072]);  mul_39 = None
    permute_44: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_103, [1, 0]);  primals_103 = None
    
    # No stacktrace found for following nodes
    mm_default_16: "f32[1568, 768]" = torch.ops.aten.mm.default(view_103, permute_44)
    add_tensor_16: "f32[1568, 768]" = torch.ops.aten.add.Tensor(mm_default_16, primals_104);  mm_default_16 = primals_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_104: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(add_tensor_16, [8, 196, 768]);  add_tensor_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:236, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_40: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_36, view_104);  add_36 = view_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_61: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_40, memory_format = torch.contiguous_format)
    var_mean_8 = torch.ops.aten.var_mean.correction(clone_61, [2], correction = 0, keepdim = True)
    getitem_16: "f32[8, 196, 1]" = var_mean_8[0]
    getitem_17: "f32[8, 196, 1]" = var_mean_8[1];  var_mean_8 = None
    add_41: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-06);  getitem_16 = None
    rsqrt_8: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_41);  add_41 = None
    sub_24: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_61, getitem_17);  clone_61 = getitem_17 = None
    mul_40: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_8);  sub_24 = None
    mul_41: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_40, primals_23)
    add_42: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_41, primals_24);  mul_41 = primals_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_45: "f32[768, 1536]" = torch.ops.aten.permute.default(primals_105, [1, 0]);  primals_105 = None
    view_109: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_42, [1568, 768]);  add_42 = None
    mm_12: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_109, permute_45)
    view_110: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(mm_12, [8, 196, 1536]);  mm_12 = None
    view_111: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.reshape.default(view_110, [8, 196, 2, 16, 48]);  view_110 = None
    permute_46: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.permute.default(view_111, [2, 0, 3, 1, 4]);  view_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    select_48: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_46, 0, 0)
    select_49: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_46, 0, 1);  permute_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    permute_47: "f32[3, 16]" = torch.ops.aten.permute.default(primals_106, [1, 0]);  primals_106 = None
    mm_13: "f32[307328, 16]" = torch.ops.aten.mm.default(view_8, permute_47);  permute_47 = None
    view_113: "f32[8, 196, 196, 16]" = torch.ops.aten.reshape.default(mm_13, [8, 196, 196, 16]);  mm_13 = None
    add_44: "f32[8, 196, 196, 16]" = torch.ops.aten.add.Tensor(view_113, primals_107);  view_113 = primals_107 = None
    permute_48: "f32[8, 16, 196, 196]" = torch.ops.aten.permute.default(add_44, [0, 3, 1, 2]);  add_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    permute_49: "f32[8, 16, 48, 196]" = torch.ops.aten.permute.default(select_49, [0, 1, 3, 2]);  select_49 = None
    expand_32: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(select_48, [8, 16, 196, 48]);  select_48 = None
    clone_65: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
    view_114: "f32[128, 196, 48]" = torch.ops.aten.reshape.default(clone_65, [128, 196, 48]);  clone_65 = None
    expand_33: "f32[8, 16, 48, 196]" = torch.ops.aten.expand.default(permute_49, [8, 16, 48, 196]);  permute_49 = None
    clone_66: "f32[8, 16, 48, 196]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
    view_115: "f32[128, 48, 196]" = torch.ops.aten.reshape.default(clone_66, [128, 48, 196]);  clone_66 = None
    bmm_8: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_114, view_115)
    view_116: "f32[8, 16, 196, 196]" = torch.ops.aten.reshape.default(bmm_8, [8, 16, 196, 196]);  bmm_8 = None
    mul_42: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(view_116, 0.14433756729740643);  view_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    amax_8: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(mul_42, [-1], True)
    sub_26: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_42, amax_8);  mul_42 = amax_8 = None
    exp_8: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_26);  sub_26 = None
    sum_13: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_12: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_8, sum_13);  exp_8 = sum_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    clone_67: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(permute_48, memory_format = torch.contiguous_format);  permute_48 = None
    amax_9: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(clone_67, [-1], True)
    sub_27: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(clone_67, amax_9);  clone_67 = amax_9 = None
    exp_9: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_27);  sub_27 = None
    sum_14: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_13: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_9, sum_14);  exp_9 = sum_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_117: "f32[1, 16, 1, 1]" = torch.ops.aten.reshape.default(primals_25, [1, -1, 1, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    sigmoid_8: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_117);  view_117 = None
    sub_28: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1.0, sigmoid_8)
    mul_43: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_28, div_12);  sub_28 = None
    mul_44: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sigmoid_8, div_13);  sigmoid_8 = None
    add_45: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(mul_43, mul_44);  mul_43 = mul_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    sum_15: "f32[8, 16, 196]" = torch.ops.aten.sum.dim_IntList(add_45, [-1])
    unsqueeze_29: "f32[8, 16, 196, 1]" = torch.ops.aten.unsqueeze.default(sum_15, -1);  sum_15 = None
    div_14: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(add_45, unsqueeze_29);  add_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_50: "f32[768, 768]" = torch.ops.aten.permute.default(primals_108, [1, 0]);  primals_108 = None
    mm_14: "f32[1568, 768]" = torch.ops.aten.mm.default(view_109, permute_50)
    view_119: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_14, [8, 196, 768]);  mm_14 = None
    view_120: "f32[8, 196, 16, 48]" = torch.ops.aten.reshape.default(view_119, [8, 196, 16, 48]);  view_119 = None
    permute_51: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_34: "f32[8, 16, 196, 196]" = torch.ops.aten.expand.default(div_14, [8, 16, 196, 196]);  div_14 = None
    view_121: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(expand_34, [128, 196, 196]);  expand_34 = None
    expand_35: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(permute_51, [8, 16, 196, 48]);  permute_51 = None
    clone_70: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
    view_122: "f32[128, 196, 48]" = torch.ops.aten.reshape.default(clone_70, [128, 196, 48]);  clone_70 = None
    bmm_9: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(view_121, view_122)
    view_123: "f32[8, 16, 196, 48]" = torch.ops.aten.reshape.default(bmm_9, [8, 16, 196, 48]);  bmm_9 = None
    permute_52: "f32[8, 196, 16, 48]" = torch.ops.aten.permute.default(view_123, [0, 2, 1, 3]);  view_123 = None
    clone_71: "f32[8, 196, 16, 48]" = torch.ops.aten.clone.default(permute_52, memory_format = torch.contiguous_format);  permute_52 = None
    view_124: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(clone_71, [8, 196, 768]);  clone_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_125: "f32[1568, 768]" = torch.ops.aten.reshape.default(view_124, [1568, 768]);  view_124 = None
    permute_53: "f32[768, 768]" = torch.ops.aten.permute.default(primals_109, [1, 0]);  primals_109 = None
    
    # No stacktrace found for following nodes
    mm_default_15: "f32[1568, 768]" = torch.ops.aten.mm.default(view_125, permute_53)
    add_tensor_15: "f32[1568, 768]" = torch.ops.aten.add.Tensor(mm_default_15, primals_110);  mm_default_15 = primals_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_126: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(add_tensor_15, [8, 196, 768]);  add_tensor_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:235, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_46: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_40, view_126);  add_40 = view_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_73: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_46, memory_format = torch.contiguous_format)
    var_mean_9 = torch.ops.aten.var_mean.correction(clone_73, [2], correction = 0, keepdim = True)
    getitem_18: "f32[8, 196, 1]" = var_mean_9[0]
    getitem_19: "f32[8, 196, 1]" = var_mean_9[1];  var_mean_9 = None
    add_47: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-06);  getitem_18 = None
    rsqrt_9: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
    sub_29: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_73, getitem_19);  clone_73 = getitem_19 = None
    mul_45: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_9);  sub_29 = None
    mul_46: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_45, primals_26)
    add_48: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_46, primals_27);  mul_46 = primals_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_127: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_48, [1568, 768]);  add_48 = None
    permute_54: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_111, [1, 0]);  primals_111 = None
    addmm_13: "f32[1568, 3072]" = torch.ops.aten.addmm.default(primals_112, view_127, permute_54);  primals_112 = None
    view_128: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(addmm_13, [8, 196, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_47: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_128, 0.5)
    mul_48: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_128, 0.7071067811865476);  view_128 = None
    erf_4: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_48);  mul_48 = None
    add_49: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_49: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_47, add_49);  mul_47 = add_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_129: "f32[1568, 3072]" = torch.ops.aten.reshape.default(mul_49, [1568, 3072]);  mul_49 = None
    permute_55: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_113, [1, 0]);  primals_113 = None
    
    # No stacktrace found for following nodes
    mm_default_14: "f32[1568, 768]" = torch.ops.aten.mm.default(view_129, permute_55)
    add_tensor_14: "f32[1568, 768]" = torch.ops.aten.add.Tensor(mm_default_14, primals_114);  mm_default_14 = primals_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_130: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(add_tensor_14, [8, 196, 768]);  add_tensor_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:236, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_50: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_46, view_130);  add_46 = view_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_76: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_50, memory_format = torch.contiguous_format)
    var_mean_10 = torch.ops.aten.var_mean.correction(clone_76, [2], correction = 0, keepdim = True)
    getitem_20: "f32[8, 196, 1]" = var_mean_10[0]
    getitem_21: "f32[8, 196, 1]" = var_mean_10[1];  var_mean_10 = None
    add_51: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-06);  getitem_20 = None
    rsqrt_10: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
    sub_30: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_76, getitem_21);  clone_76 = getitem_21 = None
    mul_50: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_10);  sub_30 = None
    mul_51: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_50, primals_28)
    add_52: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_51, primals_29);  mul_51 = primals_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_56: "f32[768, 1536]" = torch.ops.aten.permute.default(primals_115, [1, 0]);  primals_115 = None
    view_135: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_52, [1568, 768]);  add_52 = None
    mm_15: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_135, permute_56)
    view_136: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(mm_15, [8, 196, 1536]);  mm_15 = None
    view_137: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.reshape.default(view_136, [8, 196, 2, 16, 48]);  view_136 = None
    permute_57: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.permute.default(view_137, [2, 0, 3, 1, 4]);  view_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    select_58: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_57, 0, 0)
    select_59: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_57, 0, 1);  permute_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    permute_58: "f32[3, 16]" = torch.ops.aten.permute.default(primals_116, [1, 0]);  primals_116 = None
    mm_16: "f32[307328, 16]" = torch.ops.aten.mm.default(view_8, permute_58);  permute_58 = None
    view_139: "f32[8, 196, 196, 16]" = torch.ops.aten.reshape.default(mm_16, [8, 196, 196, 16]);  mm_16 = None
    add_54: "f32[8, 196, 196, 16]" = torch.ops.aten.add.Tensor(view_139, primals_117);  view_139 = primals_117 = None
    permute_59: "f32[8, 16, 196, 196]" = torch.ops.aten.permute.default(add_54, [0, 3, 1, 2]);  add_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    permute_60: "f32[8, 16, 48, 196]" = torch.ops.aten.permute.default(select_59, [0, 1, 3, 2]);  select_59 = None
    expand_39: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(select_58, [8, 16, 196, 48]);  select_58 = None
    clone_80: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
    view_140: "f32[128, 196, 48]" = torch.ops.aten.reshape.default(clone_80, [128, 196, 48]);  clone_80 = None
    expand_40: "f32[8, 16, 48, 196]" = torch.ops.aten.expand.default(permute_60, [8, 16, 48, 196]);  permute_60 = None
    clone_81: "f32[8, 16, 48, 196]" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
    view_141: "f32[128, 48, 196]" = torch.ops.aten.reshape.default(clone_81, [128, 48, 196]);  clone_81 = None
    bmm_10: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_140, view_141)
    view_142: "f32[8, 16, 196, 196]" = torch.ops.aten.reshape.default(bmm_10, [8, 16, 196, 196]);  bmm_10 = None
    mul_52: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(view_142, 0.14433756729740643);  view_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    amax_10: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(mul_52, [-1], True)
    sub_32: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_52, amax_10);  mul_52 = amax_10 = None
    exp_10: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_32);  sub_32 = None
    sum_16: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_15: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_10, sum_16);  exp_10 = sum_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    clone_82: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None
    amax_11: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(clone_82, [-1], True)
    sub_33: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(clone_82, amax_11);  clone_82 = amax_11 = None
    exp_11: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_33);  sub_33 = None
    sum_17: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_16: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_11, sum_17);  exp_11 = sum_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_143: "f32[1, 16, 1, 1]" = torch.ops.aten.reshape.default(primals_30, [1, -1, 1, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    sigmoid_10: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_143);  view_143 = None
    sub_34: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1.0, sigmoid_10)
    mul_53: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_34, div_15);  sub_34 = None
    mul_54: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sigmoid_10, div_16);  sigmoid_10 = None
    add_55: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(mul_53, mul_54);  mul_53 = mul_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    sum_18: "f32[8, 16, 196]" = torch.ops.aten.sum.dim_IntList(add_55, [-1])
    unsqueeze_35: "f32[8, 16, 196, 1]" = torch.ops.aten.unsqueeze.default(sum_18, -1);  sum_18 = None
    div_17: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(add_55, unsqueeze_35);  add_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_61: "f32[768, 768]" = torch.ops.aten.permute.default(primals_118, [1, 0]);  primals_118 = None
    mm_17: "f32[1568, 768]" = torch.ops.aten.mm.default(view_135, permute_61)
    view_145: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_17, [8, 196, 768]);  mm_17 = None
    view_146: "f32[8, 196, 16, 48]" = torch.ops.aten.reshape.default(view_145, [8, 196, 16, 48]);  view_145 = None
    permute_62: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_146, [0, 2, 1, 3]);  view_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_41: "f32[8, 16, 196, 196]" = torch.ops.aten.expand.default(div_17, [8, 16, 196, 196]);  div_17 = None
    view_147: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(expand_41, [128, 196, 196]);  expand_41 = None
    expand_42: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(permute_62, [8, 16, 196, 48]);  permute_62 = None
    clone_85: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
    view_148: "f32[128, 196, 48]" = torch.ops.aten.reshape.default(clone_85, [128, 196, 48]);  clone_85 = None
    bmm_11: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(view_147, view_148)
    view_149: "f32[8, 16, 196, 48]" = torch.ops.aten.reshape.default(bmm_11, [8, 16, 196, 48]);  bmm_11 = None
    permute_63: "f32[8, 196, 16, 48]" = torch.ops.aten.permute.default(view_149, [0, 2, 1, 3]);  view_149 = None
    clone_86: "f32[8, 196, 16, 48]" = torch.ops.aten.clone.default(permute_63, memory_format = torch.contiguous_format);  permute_63 = None
    view_150: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(clone_86, [8, 196, 768]);  clone_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_151: "f32[1568, 768]" = torch.ops.aten.reshape.default(view_150, [1568, 768]);  view_150 = None
    permute_64: "f32[768, 768]" = torch.ops.aten.permute.default(primals_119, [1, 0]);  primals_119 = None
    
    # No stacktrace found for following nodes
    mm_default_13: "f32[1568, 768]" = torch.ops.aten.mm.default(view_151, permute_64)
    add_tensor_13: "f32[1568, 768]" = torch.ops.aten.add.Tensor(mm_default_13, primals_120);  mm_default_13 = primals_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_152: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(add_tensor_13, [8, 196, 768]);  add_tensor_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:235, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_56: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_50, view_152);  add_50 = view_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_88: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_56, memory_format = torch.contiguous_format)
    var_mean_11 = torch.ops.aten.var_mean.correction(clone_88, [2], correction = 0, keepdim = True)
    getitem_22: "f32[8, 196, 1]" = var_mean_11[0]
    getitem_23: "f32[8, 196, 1]" = var_mean_11[1];  var_mean_11 = None
    add_57: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-06);  getitem_22 = None
    rsqrt_11: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
    sub_35: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_88, getitem_23);  clone_88 = getitem_23 = None
    mul_55: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_11);  sub_35 = None
    mul_56: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_55, primals_31)
    add_58: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_56, primals_32);  mul_56 = primals_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_153: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_58, [1568, 768]);  add_58 = None
    permute_65: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_121, [1, 0]);  primals_121 = None
    addmm_16: "f32[1568, 3072]" = torch.ops.aten.addmm.default(primals_122, view_153, permute_65);  primals_122 = None
    view_154: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(addmm_16, [8, 196, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_57: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_154, 0.5)
    mul_58: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_154, 0.7071067811865476);  view_154 = None
    erf_5: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_58);  mul_58 = None
    add_59: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_59: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_57, add_59);  mul_57 = add_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_155: "f32[1568, 3072]" = torch.ops.aten.reshape.default(mul_59, [1568, 3072]);  mul_59 = None
    permute_66: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_123, [1, 0]);  primals_123 = None
    
    # No stacktrace found for following nodes
    mm_default_12: "f32[1568, 768]" = torch.ops.aten.mm.default(view_155, permute_66)
    add_tensor_12: "f32[1568, 768]" = torch.ops.aten.add.Tensor(mm_default_12, primals_124);  mm_default_12 = primals_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_156: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(add_tensor_12, [8, 196, 768]);  add_tensor_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:236, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_60: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_56, view_156);  add_56 = view_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_91: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_60, memory_format = torch.contiguous_format)
    var_mean_12 = torch.ops.aten.var_mean.correction(clone_91, [2], correction = 0, keepdim = True)
    getitem_24: "f32[8, 196, 1]" = var_mean_12[0]
    getitem_25: "f32[8, 196, 1]" = var_mean_12[1];  var_mean_12 = None
    add_61: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-06);  getitem_24 = None
    rsqrt_12: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
    sub_36: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_91, getitem_25);  clone_91 = getitem_25 = None
    mul_60: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_12);  sub_36 = None
    mul_61: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_60, primals_33)
    add_62: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_61, primals_34);  mul_61 = primals_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_67: "f32[768, 1536]" = torch.ops.aten.permute.default(primals_125, [1, 0]);  primals_125 = None
    view_161: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_62, [1568, 768]);  add_62 = None
    mm_18: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_161, permute_67)
    view_162: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(mm_18, [8, 196, 1536]);  mm_18 = None
    view_163: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.reshape.default(view_162, [8, 196, 2, 16, 48]);  view_162 = None
    permute_68: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.permute.default(view_163, [2, 0, 3, 1, 4]);  view_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    select_68: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_68, 0, 0)
    select_69: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_68, 0, 1);  permute_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    permute_69: "f32[3, 16]" = torch.ops.aten.permute.default(primals_126, [1, 0]);  primals_126 = None
    mm_19: "f32[307328, 16]" = torch.ops.aten.mm.default(view_8, permute_69);  permute_69 = None
    view_165: "f32[8, 196, 196, 16]" = torch.ops.aten.reshape.default(mm_19, [8, 196, 196, 16]);  mm_19 = None
    add_64: "f32[8, 196, 196, 16]" = torch.ops.aten.add.Tensor(view_165, primals_127);  view_165 = primals_127 = None
    permute_70: "f32[8, 16, 196, 196]" = torch.ops.aten.permute.default(add_64, [0, 3, 1, 2]);  add_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    permute_71: "f32[8, 16, 48, 196]" = torch.ops.aten.permute.default(select_69, [0, 1, 3, 2]);  select_69 = None
    expand_46: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(select_68, [8, 16, 196, 48]);  select_68 = None
    clone_95: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_46, memory_format = torch.contiguous_format);  expand_46 = None
    view_166: "f32[128, 196, 48]" = torch.ops.aten.reshape.default(clone_95, [128, 196, 48]);  clone_95 = None
    expand_47: "f32[8, 16, 48, 196]" = torch.ops.aten.expand.default(permute_71, [8, 16, 48, 196]);  permute_71 = None
    clone_96: "f32[8, 16, 48, 196]" = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
    view_167: "f32[128, 48, 196]" = torch.ops.aten.reshape.default(clone_96, [128, 48, 196]);  clone_96 = None
    bmm_12: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_166, view_167)
    view_168: "f32[8, 16, 196, 196]" = torch.ops.aten.reshape.default(bmm_12, [8, 16, 196, 196]);  bmm_12 = None
    mul_62: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(view_168, 0.14433756729740643);  view_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    amax_12: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(mul_62, [-1], True)
    sub_38: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_62, amax_12);  mul_62 = amax_12 = None
    exp_12: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_38);  sub_38 = None
    sum_19: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
    div_18: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_12, sum_19);  exp_12 = sum_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    clone_97: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(permute_70, memory_format = torch.contiguous_format);  permute_70 = None
    amax_13: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(clone_97, [-1], True)
    sub_39: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(clone_97, amax_13);  clone_97 = amax_13 = None
    exp_13: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_39);  sub_39 = None
    sum_20: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
    div_19: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_13, sum_20);  exp_13 = sum_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_169: "f32[1, 16, 1, 1]" = torch.ops.aten.reshape.default(primals_35, [1, -1, 1, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    sigmoid_12: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_169);  view_169 = None
    sub_40: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1.0, sigmoid_12)
    mul_63: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_40, div_18);  sub_40 = None
    mul_64: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sigmoid_12, div_19);  sigmoid_12 = None
    add_65: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(mul_63, mul_64);  mul_63 = mul_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    sum_21: "f32[8, 16, 196]" = torch.ops.aten.sum.dim_IntList(add_65, [-1])
    unsqueeze_41: "f32[8, 16, 196, 1]" = torch.ops.aten.unsqueeze.default(sum_21, -1);  sum_21 = None
    div_20: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(add_65, unsqueeze_41);  add_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_72: "f32[768, 768]" = torch.ops.aten.permute.default(primals_128, [1, 0]);  primals_128 = None
    mm_20: "f32[1568, 768]" = torch.ops.aten.mm.default(view_161, permute_72)
    view_171: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_20, [8, 196, 768]);  mm_20 = None
    view_172: "f32[8, 196, 16, 48]" = torch.ops.aten.reshape.default(view_171, [8, 196, 16, 48]);  view_171 = None
    permute_73: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_172, [0, 2, 1, 3]);  view_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_48: "f32[8, 16, 196, 196]" = torch.ops.aten.expand.default(div_20, [8, 16, 196, 196]);  div_20 = None
    view_173: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(expand_48, [128, 196, 196]);  expand_48 = None
    expand_49: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(permute_73, [8, 16, 196, 48]);  permute_73 = None
    clone_100: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
    view_174: "f32[128, 196, 48]" = torch.ops.aten.reshape.default(clone_100, [128, 196, 48]);  clone_100 = None
    bmm_13: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(view_173, view_174)
    view_175: "f32[8, 16, 196, 48]" = torch.ops.aten.reshape.default(bmm_13, [8, 16, 196, 48]);  bmm_13 = None
    permute_74: "f32[8, 196, 16, 48]" = torch.ops.aten.permute.default(view_175, [0, 2, 1, 3]);  view_175 = None
    clone_101: "f32[8, 196, 16, 48]" = torch.ops.aten.clone.default(permute_74, memory_format = torch.contiguous_format);  permute_74 = None
    view_176: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(clone_101, [8, 196, 768]);  clone_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_177: "f32[1568, 768]" = torch.ops.aten.reshape.default(view_176, [1568, 768]);  view_176 = None
    permute_75: "f32[768, 768]" = torch.ops.aten.permute.default(primals_129, [1, 0]);  primals_129 = None
    
    # No stacktrace found for following nodes
    mm_default_11: "f32[1568, 768]" = torch.ops.aten.mm.default(view_177, permute_75)
    add_tensor_11: "f32[1568, 768]" = torch.ops.aten.add.Tensor(mm_default_11, primals_130);  mm_default_11 = primals_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_178: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(add_tensor_11, [8, 196, 768]);  add_tensor_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:235, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_66: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_60, view_178);  add_60 = view_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_103: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_66, memory_format = torch.contiguous_format)
    var_mean_13 = torch.ops.aten.var_mean.correction(clone_103, [2], correction = 0, keepdim = True)
    getitem_26: "f32[8, 196, 1]" = var_mean_13[0]
    getitem_27: "f32[8, 196, 1]" = var_mean_13[1];  var_mean_13 = None
    add_67: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-06);  getitem_26 = None
    rsqrt_13: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
    sub_41: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_103, getitem_27);  clone_103 = getitem_27 = None
    mul_65: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_13);  sub_41 = None
    mul_66: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_65, primals_36)
    add_68: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_66, primals_37);  mul_66 = primals_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_179: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_68, [1568, 768]);  add_68 = None
    permute_76: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_131, [1, 0]);  primals_131 = None
    addmm_19: "f32[1568, 3072]" = torch.ops.aten.addmm.default(primals_132, view_179, permute_76);  primals_132 = None
    view_180: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(addmm_19, [8, 196, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_67: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_180, 0.5)
    mul_68: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_180, 0.7071067811865476);  view_180 = None
    erf_6: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_68);  mul_68 = None
    add_69: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_69: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_67, add_69);  mul_67 = add_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_181: "f32[1568, 3072]" = torch.ops.aten.reshape.default(mul_69, [1568, 3072]);  mul_69 = None
    permute_77: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_133, [1, 0]);  primals_133 = None
    
    # No stacktrace found for following nodes
    mm_default_10: "f32[1568, 768]" = torch.ops.aten.mm.default(view_181, permute_77)
    add_tensor_10: "f32[1568, 768]" = torch.ops.aten.add.Tensor(mm_default_10, primals_134);  mm_default_10 = primals_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_182: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(add_tensor_10, [8, 196, 768]);  add_tensor_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:236, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_70: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_66, view_182);  add_66 = view_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_106: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_70, memory_format = torch.contiguous_format)
    var_mean_14 = torch.ops.aten.var_mean.correction(clone_106, [2], correction = 0, keepdim = True)
    getitem_28: "f32[8, 196, 1]" = var_mean_14[0]
    getitem_29: "f32[8, 196, 1]" = var_mean_14[1];  var_mean_14 = None
    add_71: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-06);  getitem_28 = None
    rsqrt_14: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
    sub_42: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_106, getitem_29);  clone_106 = getitem_29 = None
    mul_70: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_14);  sub_42 = None
    mul_71: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_70, primals_38)
    add_72: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_71, primals_39);  mul_71 = primals_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_78: "f32[768, 1536]" = torch.ops.aten.permute.default(primals_135, [1, 0]);  primals_135 = None
    view_187: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_72, [1568, 768]);  add_72 = None
    mm_21: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_187, permute_78)
    view_188: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(mm_21, [8, 196, 1536]);  mm_21 = None
    view_189: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.reshape.default(view_188, [8, 196, 2, 16, 48]);  view_188 = None
    permute_79: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.permute.default(view_189, [2, 0, 3, 1, 4]);  view_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    select_78: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_79, 0, 0)
    select_79: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_79, 0, 1);  permute_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    permute_80: "f32[3, 16]" = torch.ops.aten.permute.default(primals_136, [1, 0]);  primals_136 = None
    mm_22: "f32[307328, 16]" = torch.ops.aten.mm.default(view_8, permute_80);  permute_80 = None
    view_191: "f32[8, 196, 196, 16]" = torch.ops.aten.reshape.default(mm_22, [8, 196, 196, 16]);  mm_22 = None
    add_74: "f32[8, 196, 196, 16]" = torch.ops.aten.add.Tensor(view_191, primals_137);  view_191 = primals_137 = None
    permute_81: "f32[8, 16, 196, 196]" = torch.ops.aten.permute.default(add_74, [0, 3, 1, 2]);  add_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    permute_82: "f32[8, 16, 48, 196]" = torch.ops.aten.permute.default(select_79, [0, 1, 3, 2]);  select_79 = None
    expand_53: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(select_78, [8, 16, 196, 48]);  select_78 = None
    clone_110: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_53, memory_format = torch.contiguous_format);  expand_53 = None
    view_192: "f32[128, 196, 48]" = torch.ops.aten.reshape.default(clone_110, [128, 196, 48]);  clone_110 = None
    expand_54: "f32[8, 16, 48, 196]" = torch.ops.aten.expand.default(permute_82, [8, 16, 48, 196]);  permute_82 = None
    clone_111: "f32[8, 16, 48, 196]" = torch.ops.aten.clone.default(expand_54, memory_format = torch.contiguous_format);  expand_54 = None
    view_193: "f32[128, 48, 196]" = torch.ops.aten.reshape.default(clone_111, [128, 48, 196]);  clone_111 = None
    bmm_14: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_192, view_193)
    view_194: "f32[8, 16, 196, 196]" = torch.ops.aten.reshape.default(bmm_14, [8, 16, 196, 196]);  bmm_14 = None
    mul_72: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(view_194, 0.14433756729740643);  view_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    amax_14: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(mul_72, [-1], True)
    sub_44: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_72, amax_14);  mul_72 = amax_14 = None
    exp_14: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_44);  sub_44 = None
    sum_22: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
    div_21: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_14, sum_22);  exp_14 = sum_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    clone_112: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(permute_81, memory_format = torch.contiguous_format);  permute_81 = None
    amax_15: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(clone_112, [-1], True)
    sub_45: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(clone_112, amax_15);  clone_112 = amax_15 = None
    exp_15: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_45);  sub_45 = None
    sum_23: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
    div_22: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_15, sum_23);  exp_15 = sum_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_195: "f32[1, 16, 1, 1]" = torch.ops.aten.reshape.default(primals_40, [1, -1, 1, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    sigmoid_14: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_195);  view_195 = None
    sub_46: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1.0, sigmoid_14)
    mul_73: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_46, div_21);  sub_46 = None
    mul_74: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sigmoid_14, div_22);  sigmoid_14 = None
    add_75: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(mul_73, mul_74);  mul_73 = mul_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    sum_24: "f32[8, 16, 196]" = torch.ops.aten.sum.dim_IntList(add_75, [-1])
    unsqueeze_47: "f32[8, 16, 196, 1]" = torch.ops.aten.unsqueeze.default(sum_24, -1);  sum_24 = None
    div_23: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(add_75, unsqueeze_47);  add_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_83: "f32[768, 768]" = torch.ops.aten.permute.default(primals_138, [1, 0]);  primals_138 = None
    mm_23: "f32[1568, 768]" = torch.ops.aten.mm.default(view_187, permute_83)
    view_197: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_23, [8, 196, 768]);  mm_23 = None
    view_198: "f32[8, 196, 16, 48]" = torch.ops.aten.reshape.default(view_197, [8, 196, 16, 48]);  view_197 = None
    permute_84: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_198, [0, 2, 1, 3]);  view_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_55: "f32[8, 16, 196, 196]" = torch.ops.aten.expand.default(div_23, [8, 16, 196, 196]);  div_23 = None
    view_199: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(expand_55, [128, 196, 196]);  expand_55 = None
    expand_56: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(permute_84, [8, 16, 196, 48]);  permute_84 = None
    clone_115: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
    view_200: "f32[128, 196, 48]" = torch.ops.aten.reshape.default(clone_115, [128, 196, 48]);  clone_115 = None
    bmm_15: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(view_199, view_200)
    view_201: "f32[8, 16, 196, 48]" = torch.ops.aten.reshape.default(bmm_15, [8, 16, 196, 48]);  bmm_15 = None
    permute_85: "f32[8, 196, 16, 48]" = torch.ops.aten.permute.default(view_201, [0, 2, 1, 3]);  view_201 = None
    clone_116: "f32[8, 196, 16, 48]" = torch.ops.aten.clone.default(permute_85, memory_format = torch.contiguous_format);  permute_85 = None
    view_202: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(clone_116, [8, 196, 768]);  clone_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_203: "f32[1568, 768]" = torch.ops.aten.reshape.default(view_202, [1568, 768]);  view_202 = None
    permute_86: "f32[768, 768]" = torch.ops.aten.permute.default(primals_139, [1, 0]);  primals_139 = None
    
    # No stacktrace found for following nodes
    mm_default_9: "f32[1568, 768]" = torch.ops.aten.mm.default(view_203, permute_86)
    add_tensor_9: "f32[1568, 768]" = torch.ops.aten.add.Tensor(mm_default_9, primals_140);  mm_default_9 = primals_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_204: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(add_tensor_9, [8, 196, 768]);  add_tensor_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:235, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_76: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_70, view_204);  add_70 = view_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_118: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_76, memory_format = torch.contiguous_format)
    var_mean_15 = torch.ops.aten.var_mean.correction(clone_118, [2], correction = 0, keepdim = True)
    getitem_30: "f32[8, 196, 1]" = var_mean_15[0]
    getitem_31: "f32[8, 196, 1]" = var_mean_15[1];  var_mean_15 = None
    add_77: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-06);  getitem_30 = None
    rsqrt_15: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
    sub_47: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_118, getitem_31);  clone_118 = getitem_31 = None
    mul_75: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_15);  sub_47 = None
    mul_76: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_75, primals_41)
    add_78: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_76, primals_42);  mul_76 = primals_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_205: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_78, [1568, 768]);  add_78 = None
    permute_87: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_141, [1, 0]);  primals_141 = None
    addmm_22: "f32[1568, 3072]" = torch.ops.aten.addmm.default(primals_142, view_205, permute_87);  primals_142 = None
    view_206: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(addmm_22, [8, 196, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_77: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_206, 0.5)
    mul_78: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_206, 0.7071067811865476);  view_206 = None
    erf_7: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_78);  mul_78 = None
    add_79: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_79: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_77, add_79);  mul_77 = add_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_207: "f32[1568, 3072]" = torch.ops.aten.reshape.default(mul_79, [1568, 3072]);  mul_79 = None
    permute_88: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_143, [1, 0]);  primals_143 = None
    
    # No stacktrace found for following nodes
    mm_default_8: "f32[1568, 768]" = torch.ops.aten.mm.default(view_207, permute_88)
    add_tensor_8: "f32[1568, 768]" = torch.ops.aten.add.Tensor(mm_default_8, primals_144);  mm_default_8 = primals_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_208: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(add_tensor_8, [8, 196, 768]);  add_tensor_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:236, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_80: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_76, view_208);  add_76 = view_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_121: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_80, memory_format = torch.contiguous_format)
    var_mean_16 = torch.ops.aten.var_mean.correction(clone_121, [2], correction = 0, keepdim = True)
    getitem_32: "f32[8, 196, 1]" = var_mean_16[0]
    getitem_33: "f32[8, 196, 1]" = var_mean_16[1];  var_mean_16 = None
    add_81: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-06);  getitem_32 = None
    rsqrt_16: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
    sub_48: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_121, getitem_33);  clone_121 = getitem_33 = None
    mul_80: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_16);  sub_48 = None
    mul_81: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_80, primals_43)
    add_82: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_81, primals_44);  mul_81 = primals_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_89: "f32[768, 1536]" = torch.ops.aten.permute.default(primals_145, [1, 0]);  primals_145 = None
    view_213: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_82, [1568, 768]);  add_82 = None
    mm_24: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_213, permute_89)
    view_214: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(mm_24, [8, 196, 1536]);  mm_24 = None
    view_215: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.reshape.default(view_214, [8, 196, 2, 16, 48]);  view_214 = None
    permute_90: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.permute.default(view_215, [2, 0, 3, 1, 4]);  view_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    select_88: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_90, 0, 0)
    select_89: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_90, 0, 1);  permute_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    permute_91: "f32[3, 16]" = torch.ops.aten.permute.default(primals_146, [1, 0]);  primals_146 = None
    mm_25: "f32[307328, 16]" = torch.ops.aten.mm.default(view_8, permute_91);  permute_91 = None
    view_217: "f32[8, 196, 196, 16]" = torch.ops.aten.reshape.default(mm_25, [8, 196, 196, 16]);  mm_25 = None
    add_84: "f32[8, 196, 196, 16]" = torch.ops.aten.add.Tensor(view_217, primals_147);  view_217 = primals_147 = None
    permute_92: "f32[8, 16, 196, 196]" = torch.ops.aten.permute.default(add_84, [0, 3, 1, 2]);  add_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    permute_93: "f32[8, 16, 48, 196]" = torch.ops.aten.permute.default(select_89, [0, 1, 3, 2]);  select_89 = None
    expand_60: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(select_88, [8, 16, 196, 48]);  select_88 = None
    clone_125: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_60, memory_format = torch.contiguous_format);  expand_60 = None
    view_218: "f32[128, 196, 48]" = torch.ops.aten.reshape.default(clone_125, [128, 196, 48]);  clone_125 = None
    expand_61: "f32[8, 16, 48, 196]" = torch.ops.aten.expand.default(permute_93, [8, 16, 48, 196]);  permute_93 = None
    clone_126: "f32[8, 16, 48, 196]" = torch.ops.aten.clone.default(expand_61, memory_format = torch.contiguous_format);  expand_61 = None
    view_219: "f32[128, 48, 196]" = torch.ops.aten.reshape.default(clone_126, [128, 48, 196]);  clone_126 = None
    bmm_16: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_218, view_219)
    view_220: "f32[8, 16, 196, 196]" = torch.ops.aten.reshape.default(bmm_16, [8, 16, 196, 196]);  bmm_16 = None
    mul_82: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(view_220, 0.14433756729740643);  view_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    amax_16: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(mul_82, [-1], True)
    sub_50: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_82, amax_16);  mul_82 = amax_16 = None
    exp_16: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_50);  sub_50 = None
    sum_25: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
    div_24: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_16, sum_25);  exp_16 = sum_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    clone_127: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(permute_92, memory_format = torch.contiguous_format);  permute_92 = None
    amax_17: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(clone_127, [-1], True)
    sub_51: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(clone_127, amax_17);  clone_127 = amax_17 = None
    exp_17: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_51);  sub_51 = None
    sum_26: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
    div_25: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_17, sum_26);  exp_17 = sum_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_221: "f32[1, 16, 1, 1]" = torch.ops.aten.reshape.default(primals_45, [1, -1, 1, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    sigmoid_16: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_221);  view_221 = None
    sub_52: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1.0, sigmoid_16)
    mul_83: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_52, div_24);  sub_52 = None
    mul_84: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sigmoid_16, div_25);  sigmoid_16 = None
    add_85: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(mul_83, mul_84);  mul_83 = mul_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    sum_27: "f32[8, 16, 196]" = torch.ops.aten.sum.dim_IntList(add_85, [-1])
    unsqueeze_53: "f32[8, 16, 196, 1]" = torch.ops.aten.unsqueeze.default(sum_27, -1);  sum_27 = None
    div_26: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(add_85, unsqueeze_53);  add_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_94: "f32[768, 768]" = torch.ops.aten.permute.default(primals_148, [1, 0]);  primals_148 = None
    mm_26: "f32[1568, 768]" = torch.ops.aten.mm.default(view_213, permute_94)
    view_223: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_26, [8, 196, 768]);  mm_26 = None
    view_224: "f32[8, 196, 16, 48]" = torch.ops.aten.reshape.default(view_223, [8, 196, 16, 48]);  view_223 = None
    permute_95: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_224, [0, 2, 1, 3]);  view_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_62: "f32[8, 16, 196, 196]" = torch.ops.aten.expand.default(div_26, [8, 16, 196, 196]);  div_26 = None
    view_225: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(expand_62, [128, 196, 196]);  expand_62 = None
    expand_63: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(permute_95, [8, 16, 196, 48]);  permute_95 = None
    clone_130: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_63, memory_format = torch.contiguous_format);  expand_63 = None
    view_226: "f32[128, 196, 48]" = torch.ops.aten.reshape.default(clone_130, [128, 196, 48]);  clone_130 = None
    bmm_17: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(view_225, view_226)
    view_227: "f32[8, 16, 196, 48]" = torch.ops.aten.reshape.default(bmm_17, [8, 16, 196, 48]);  bmm_17 = None
    permute_96: "f32[8, 196, 16, 48]" = torch.ops.aten.permute.default(view_227, [0, 2, 1, 3]);  view_227 = None
    clone_131: "f32[8, 196, 16, 48]" = torch.ops.aten.clone.default(permute_96, memory_format = torch.contiguous_format);  permute_96 = None
    view_228: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(clone_131, [8, 196, 768]);  clone_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_229: "f32[1568, 768]" = torch.ops.aten.reshape.default(view_228, [1568, 768]);  view_228 = None
    permute_97: "f32[768, 768]" = torch.ops.aten.permute.default(primals_149, [1, 0]);  primals_149 = None
    
    # No stacktrace found for following nodes
    mm_default_7: "f32[1568, 768]" = torch.ops.aten.mm.default(view_229, permute_97)
    add_tensor_7: "f32[1568, 768]" = torch.ops.aten.add.Tensor(mm_default_7, primals_150);  mm_default_7 = primals_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_230: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(add_tensor_7, [8, 196, 768]);  add_tensor_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:235, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_86: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_80, view_230);  add_80 = view_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_133: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_86, memory_format = torch.contiguous_format)
    var_mean_17 = torch.ops.aten.var_mean.correction(clone_133, [2], correction = 0, keepdim = True)
    getitem_34: "f32[8, 196, 1]" = var_mean_17[0]
    getitem_35: "f32[8, 196, 1]" = var_mean_17[1];  var_mean_17 = None
    add_87: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-06);  getitem_34 = None
    rsqrt_17: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
    sub_53: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_133, getitem_35);  clone_133 = getitem_35 = None
    mul_85: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_17);  sub_53 = None
    mul_86: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_85, primals_46)
    add_88: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_86, primals_47);  mul_86 = primals_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_231: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_88, [1568, 768]);  add_88 = None
    permute_98: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_151, [1, 0]);  primals_151 = None
    addmm_25: "f32[1568, 3072]" = torch.ops.aten.addmm.default(primals_152, view_231, permute_98);  primals_152 = None
    view_232: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(addmm_25, [8, 196, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_87: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_232, 0.5)
    mul_88: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_232, 0.7071067811865476);  view_232 = None
    erf_8: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_88);  mul_88 = None
    add_89: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_89: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_87, add_89);  mul_87 = add_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_233: "f32[1568, 3072]" = torch.ops.aten.reshape.default(mul_89, [1568, 3072]);  mul_89 = None
    permute_99: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_153, [1, 0]);  primals_153 = None
    
    # No stacktrace found for following nodes
    mm_default_6: "f32[1568, 768]" = torch.ops.aten.mm.default(view_233, permute_99)
    add_tensor_6: "f32[1568, 768]" = torch.ops.aten.add.Tensor(mm_default_6, primals_154);  mm_default_6 = primals_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_234: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(add_tensor_6, [8, 196, 768]);  add_tensor_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:236, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_90: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_86, view_234);  add_86 = view_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_136: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_90, memory_format = torch.contiguous_format)
    var_mean_18 = torch.ops.aten.var_mean.correction(clone_136, [2], correction = 0, keepdim = True)
    getitem_36: "f32[8, 196, 1]" = var_mean_18[0]
    getitem_37: "f32[8, 196, 1]" = var_mean_18[1];  var_mean_18 = None
    add_91: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-06);  getitem_36 = None
    rsqrt_18: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_91);  add_91 = None
    sub_54: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_136, getitem_37);  clone_136 = getitem_37 = None
    mul_90: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_18);  sub_54 = None
    mul_91: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_90, primals_48)
    add_92: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_91, primals_49);  mul_91 = primals_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_100: "f32[768, 1536]" = torch.ops.aten.permute.default(primals_155, [1, 0]);  primals_155 = None
    view_239: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_92, [1568, 768]);  add_92 = None
    mm_27: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_239, permute_100)
    view_240: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(mm_27, [8, 196, 1536]);  mm_27 = None
    view_241: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.reshape.default(view_240, [8, 196, 2, 16, 48]);  view_240 = None
    permute_101: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.permute.default(view_241, [2, 0, 3, 1, 4]);  view_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    select_98: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_101, 0, 0)
    select_99: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_101, 0, 1);  permute_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    permute_102: "f32[3, 16]" = torch.ops.aten.permute.default(primals_156, [1, 0]);  primals_156 = None
    mm_28: "f32[307328, 16]" = torch.ops.aten.mm.default(view_8, permute_102);  permute_102 = None
    view_243: "f32[8, 196, 196, 16]" = torch.ops.aten.reshape.default(mm_28, [8, 196, 196, 16]);  mm_28 = None
    add_94: "f32[8, 196, 196, 16]" = torch.ops.aten.add.Tensor(view_243, primals_157);  view_243 = primals_157 = None
    permute_103: "f32[8, 16, 196, 196]" = torch.ops.aten.permute.default(add_94, [0, 3, 1, 2]);  add_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    permute_104: "f32[8, 16, 48, 196]" = torch.ops.aten.permute.default(select_99, [0, 1, 3, 2]);  select_99 = None
    expand_67: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(select_98, [8, 16, 196, 48]);  select_98 = None
    clone_140: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_67, memory_format = torch.contiguous_format);  expand_67 = None
    view_244: "f32[128, 196, 48]" = torch.ops.aten.reshape.default(clone_140, [128, 196, 48]);  clone_140 = None
    expand_68: "f32[8, 16, 48, 196]" = torch.ops.aten.expand.default(permute_104, [8, 16, 48, 196]);  permute_104 = None
    clone_141: "f32[8, 16, 48, 196]" = torch.ops.aten.clone.default(expand_68, memory_format = torch.contiguous_format);  expand_68 = None
    view_245: "f32[128, 48, 196]" = torch.ops.aten.reshape.default(clone_141, [128, 48, 196]);  clone_141 = None
    bmm_18: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_244, view_245)
    view_246: "f32[8, 16, 196, 196]" = torch.ops.aten.reshape.default(bmm_18, [8, 16, 196, 196]);  bmm_18 = None
    mul_92: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(view_246, 0.14433756729740643);  view_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    amax_18: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(mul_92, [-1], True)
    sub_56: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_92, amax_18);  mul_92 = amax_18 = None
    exp_18: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_56);  sub_56 = None
    sum_28: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
    div_27: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_18, sum_28);  exp_18 = sum_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    clone_142: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(permute_103, memory_format = torch.contiguous_format);  permute_103 = None
    amax_19: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(clone_142, [-1], True)
    sub_57: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(clone_142, amax_19);  clone_142 = amax_19 = None
    exp_19: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_57);  sub_57 = None
    sum_29: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
    div_28: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_19, sum_29);  exp_19 = sum_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_247: "f32[1, 16, 1, 1]" = torch.ops.aten.reshape.default(primals_50, [1, -1, 1, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    sigmoid_18: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_247);  view_247 = None
    sub_58: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1.0, sigmoid_18)
    mul_93: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_58, div_27);  sub_58 = None
    mul_94: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sigmoid_18, div_28);  sigmoid_18 = None
    add_95: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(mul_93, mul_94);  mul_93 = mul_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    sum_30: "f32[8, 16, 196]" = torch.ops.aten.sum.dim_IntList(add_95, [-1])
    unsqueeze_59: "f32[8, 16, 196, 1]" = torch.ops.aten.unsqueeze.default(sum_30, -1);  sum_30 = None
    div_29: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(add_95, unsqueeze_59);  add_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_105: "f32[768, 768]" = torch.ops.aten.permute.default(primals_158, [1, 0]);  primals_158 = None
    mm_29: "f32[1568, 768]" = torch.ops.aten.mm.default(view_239, permute_105)
    view_249: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_29, [8, 196, 768]);  mm_29 = None
    view_250: "f32[8, 196, 16, 48]" = torch.ops.aten.reshape.default(view_249, [8, 196, 16, 48]);  view_249 = None
    permute_106: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_250, [0, 2, 1, 3]);  view_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_69: "f32[8, 16, 196, 196]" = torch.ops.aten.expand.default(div_29, [8, 16, 196, 196]);  div_29 = None
    view_251: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(expand_69, [128, 196, 196]);  expand_69 = None
    expand_70: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(permute_106, [8, 16, 196, 48]);  permute_106 = None
    clone_145: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_70, memory_format = torch.contiguous_format);  expand_70 = None
    view_252: "f32[128, 196, 48]" = torch.ops.aten.reshape.default(clone_145, [128, 196, 48]);  clone_145 = None
    bmm_19: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(view_251, view_252)
    view_253: "f32[8, 16, 196, 48]" = torch.ops.aten.reshape.default(bmm_19, [8, 16, 196, 48]);  bmm_19 = None
    permute_107: "f32[8, 196, 16, 48]" = torch.ops.aten.permute.default(view_253, [0, 2, 1, 3]);  view_253 = None
    clone_146: "f32[8, 196, 16, 48]" = torch.ops.aten.clone.default(permute_107, memory_format = torch.contiguous_format);  permute_107 = None
    view_254: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(clone_146, [8, 196, 768]);  clone_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_255: "f32[1568, 768]" = torch.ops.aten.reshape.default(view_254, [1568, 768]);  view_254 = None
    permute_108: "f32[768, 768]" = torch.ops.aten.permute.default(primals_159, [1, 0]);  primals_159 = None
    
    # No stacktrace found for following nodes
    mm_default_5: "f32[1568, 768]" = torch.ops.aten.mm.default(view_255, permute_108)
    add_tensor_5: "f32[1568, 768]" = torch.ops.aten.add.Tensor(mm_default_5, primals_160);  mm_default_5 = primals_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_256: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(add_tensor_5, [8, 196, 768]);  add_tensor_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:235, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_96: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_90, view_256);  add_90 = view_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_148: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_96, memory_format = torch.contiguous_format)
    var_mean_19 = torch.ops.aten.var_mean.correction(clone_148, [2], correction = 0, keepdim = True)
    getitem_38: "f32[8, 196, 1]" = var_mean_19[0]
    getitem_39: "f32[8, 196, 1]" = var_mean_19[1];  var_mean_19 = None
    add_97: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-06);  getitem_38 = None
    rsqrt_19: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_97);  add_97 = None
    sub_59: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_148, getitem_39);  clone_148 = getitem_39 = None
    mul_95: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_19);  sub_59 = None
    mul_96: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_95, primals_51)
    add_98: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_96, primals_52);  mul_96 = primals_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_257: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_98, [1568, 768]);  add_98 = None
    permute_109: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_161, [1, 0]);  primals_161 = None
    addmm_28: "f32[1568, 3072]" = torch.ops.aten.addmm.default(primals_162, view_257, permute_109);  primals_162 = None
    view_258: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(addmm_28, [8, 196, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_97: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_258, 0.5)
    mul_98: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_258, 0.7071067811865476);  view_258 = None
    erf_9: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_98);  mul_98 = None
    add_99: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_99: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_97, add_99);  mul_97 = add_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_259: "f32[1568, 3072]" = torch.ops.aten.reshape.default(mul_99, [1568, 3072]);  mul_99 = None
    permute_110: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_163, [1, 0]);  primals_163 = None
    
    # No stacktrace found for following nodes
    mm_default_4: "f32[1568, 768]" = torch.ops.aten.mm.default(view_259, permute_110)
    add_tensor_4: "f32[1568, 768]" = torch.ops.aten.add.Tensor(mm_default_4, primals_164);  mm_default_4 = primals_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_260: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(add_tensor_4, [8, 196, 768]);  add_tensor_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:236, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_100: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_96, view_260);  add_96 = view_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:367, code: x = torch.cat((cls_tokens, x), dim=1)
    cat: "f32[8, 197, 768]" = torch.ops.aten.cat.default([expand, add_100], 1);  expand = add_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_20 = torch.ops.aten.var_mean.correction(cat, [2], correction = 0, keepdim = True)
    getitem_40: "f32[8, 197, 1]" = var_mean_20[0]
    getitem_41: "f32[8, 197, 1]" = var_mean_20[1];  var_mean_20 = None
    add_101: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-06);  getitem_40 = None
    rsqrt_20: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_101);  add_101 = None
    sub_60: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(cat, getitem_41)
    mul_100: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_20);  sub_60 = None
    mul_101: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_100, primals_53);  mul_100 = None
    add_102: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_101, primals_54);  mul_101 = primals_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:175, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_111: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_165, [1, 0]);  primals_165 = None
    view_261: "f32[1576, 768]" = torch.ops.aten.reshape.default(add_102, [1576, 768]);  add_102 = None
    mm_30: "f32[1576, 2304]" = torch.ops.aten.mm.default(view_261, permute_111)
    view_262: "f32[8, 197, 2304]" = torch.ops.aten.reshape.default(mm_30, [8, 197, 2304]);  mm_30 = None
    view_263: "f32[8, 197, 3, 16, 48]" = torch.ops.aten.reshape.default(view_262, [8, 197, 3, 16, 48]);  view_262 = None
    permute_112: "f32[3, 8, 16, 197, 48]" = torch.ops.aten.permute.default(view_263, [2, 0, 3, 1, 4]);  view_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:176, code: q, k, v = qkv.unbind(0)
    unbind = torch.ops.aten.unbind.int(permute_112);  permute_112 = None
    getitem_42: "f32[8, 16, 197, 48]" = unbind[0]
    getitem_43: "f32[8, 16, 197, 48]" = unbind[1]
    getitem_44: "f32[8, 16, 197, 48]" = unbind[2];  unbind = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:178, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_113: "f32[8, 16, 48, 197]" = torch.ops.aten.permute.default(getitem_43, [0, 1, 3, 2]);  getitem_43 = None
    expand_71: "f32[8, 16, 197, 48]" = torch.ops.aten.expand.default(getitem_42, [8, 16, 197, 48]);  getitem_42 = None
    clone_151: "f32[8, 16, 197, 48]" = torch.ops.aten.clone.default(expand_71, memory_format = torch.contiguous_format);  expand_71 = None
    view_264: "f32[128, 197, 48]" = torch.ops.aten.reshape.default(clone_151, [128, 197, 48]);  clone_151 = None
    expand_72: "f32[8, 16, 48, 197]" = torch.ops.aten.expand.default(permute_113, [8, 16, 48, 197]);  permute_113 = None
    clone_152: "f32[8, 16, 48, 197]" = torch.ops.aten.clone.default(expand_72, memory_format = torch.contiguous_format);  expand_72 = None
    view_265: "f32[128, 48, 197]" = torch.ops.aten.reshape.default(clone_152, [128, 48, 197]);  clone_152 = None
    bmm_20: "f32[128, 197, 197]" = torch.ops.aten.bmm.default(view_264, view_265)
    view_266: "f32[8, 16, 197, 197]" = torch.ops.aten.reshape.default(bmm_20, [8, 16, 197, 197]);  bmm_20 = None
    mul_102: "f32[8, 16, 197, 197]" = torch.ops.aten.mul.Tensor(view_266, 0.14433756729740643);  view_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:179, code: attn = attn.softmax(dim=-1)
    amax_20: "f32[8, 16, 197, 1]" = torch.ops.aten.amax.default(mul_102, [-1], True)
    sub_61: "f32[8, 16, 197, 197]" = torch.ops.aten.sub.Tensor(mul_102, amax_20);  mul_102 = amax_20 = None
    exp_20: "f32[8, 16, 197, 197]" = torch.ops.aten.exp.default(sub_61);  sub_61 = None
    sum_31: "f32[8, 16, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
    div_30: "f32[8, 16, 197, 197]" = torch.ops.aten.div.Tensor(exp_20, sum_31);  exp_20 = sum_31 = None
    alias_40: "f32[8, 16, 197, 197]" = torch.ops.aten.alias.default(div_30)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:180, code: attn = self.attn_drop(attn)
    clone_153: "f32[8, 16, 197, 197]" = torch.ops.aten.clone.default(div_30);  div_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:182, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_73: "f32[8, 16, 197, 197]" = torch.ops.aten.expand.default(clone_153, [8, 16, 197, 197]);  clone_153 = None
    view_267: "f32[128, 197, 197]" = torch.ops.aten.reshape.default(expand_73, [128, 197, 197]);  expand_73 = None
    expand_74: "f32[8, 16, 197, 48]" = torch.ops.aten.expand.default(getitem_44, [8, 16, 197, 48]);  getitem_44 = None
    clone_154: "f32[8, 16, 197, 48]" = torch.ops.aten.clone.default(expand_74, memory_format = torch.contiguous_format);  expand_74 = None
    view_268: "f32[128, 197, 48]" = torch.ops.aten.reshape.default(clone_154, [128, 197, 48]);  clone_154 = None
    bmm_21: "f32[128, 197, 48]" = torch.ops.aten.bmm.default(view_267, view_268)
    view_269: "f32[8, 16, 197, 48]" = torch.ops.aten.reshape.default(bmm_21, [8, 16, 197, 48]);  bmm_21 = None
    permute_114: "f32[8, 197, 16, 48]" = torch.ops.aten.permute.default(view_269, [0, 2, 1, 3]);  view_269 = None
    clone_155: "f32[8, 197, 16, 48]" = torch.ops.aten.clone.default(permute_114, memory_format = torch.contiguous_format);  permute_114 = None
    view_270: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(clone_155, [8, 197, 768]);  clone_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:183, code: x = self.proj(x)
    view_271: "f32[1576, 768]" = torch.ops.aten.reshape.default(view_270, [1576, 768]);  view_270 = None
    permute_115: "f32[768, 768]" = torch.ops.aten.permute.default(primals_166, [1, 0]);  primals_166 = None
    
    # No stacktrace found for following nodes
    mm_default_3: "f32[1576, 768]" = torch.ops.aten.mm.default(view_271, permute_115)
    add_tensor_3: "f32[1576, 768]" = torch.ops.aten.add.Tensor(mm_default_3, primals_167);  mm_default_3 = primals_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:183, code: x = self.proj(x)
    view_272: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(add_tensor_3, [8, 197, 768]);  add_tensor_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:235, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_103: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(cat, view_272);  view_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_21 = torch.ops.aten.var_mean.correction(add_103, [2], correction = 0, keepdim = True)
    getitem_45: "f32[8, 197, 1]" = var_mean_21[0]
    getitem_46: "f32[8, 197, 1]" = var_mean_21[1];  var_mean_21 = None
    add_104: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_45, 1e-06);  getitem_45 = None
    rsqrt_21: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_104);  add_104 = None
    sub_62: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_103, getitem_46);  getitem_46 = None
    mul_103: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_21);  sub_62 = None
    mul_104: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_103, primals_55)
    add_105: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_104, primals_56);  mul_104 = primals_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_273: "f32[1576, 768]" = torch.ops.aten.reshape.default(add_105, [1576, 768]);  add_105 = None
    permute_116: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_168, [1, 0]);  primals_168 = None
    addmm_31: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_169, view_273, permute_116);  primals_169 = None
    view_274: "f32[8, 197, 3072]" = torch.ops.aten.reshape.default(addmm_31, [8, 197, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_105: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_274, 0.5)
    mul_106: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_274, 0.7071067811865476);  view_274 = None
    erf_10: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_106);  mul_106 = None
    add_106: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_107: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_105, add_106);  mul_105 = add_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_275: "f32[1576, 3072]" = torch.ops.aten.reshape.default(mul_107, [1576, 3072]);  mul_107 = None
    permute_117: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_170, [1, 0]);  primals_170 = None
    
    # No stacktrace found for following nodes
    mm_default_2: "f32[1576, 768]" = torch.ops.aten.mm.default(view_275, permute_117)
    add_tensor_2: "f32[1576, 768]" = torch.ops.aten.add.Tensor(mm_default_2, primals_171);  mm_default_2 = primals_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_276: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(add_tensor_2, [8, 197, 768]);  add_tensor_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:236, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_107: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_103, view_276);  add_103 = view_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_22 = torch.ops.aten.var_mean.correction(add_107, [2], correction = 0, keepdim = True)
    getitem_47: "f32[8, 197, 1]" = var_mean_22[0]
    getitem_48: "f32[8, 197, 1]" = var_mean_22[1];  var_mean_22 = None
    add_108: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_47, 1e-06);  getitem_47 = None
    rsqrt_22: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
    sub_63: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_107, getitem_48);  getitem_48 = None
    mul_108: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_22);  sub_63 = None
    mul_109: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_108, primals_57)
    add_109: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_109, primals_58);  mul_109 = primals_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:175, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_118: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_172, [1, 0]);  primals_172 = None
    view_277: "f32[1576, 768]" = torch.ops.aten.reshape.default(add_109, [1576, 768]);  add_109 = None
    mm_31: "f32[1576, 2304]" = torch.ops.aten.mm.default(view_277, permute_118)
    view_278: "f32[8, 197, 2304]" = torch.ops.aten.reshape.default(mm_31, [8, 197, 2304]);  mm_31 = None
    view_279: "f32[8, 197, 3, 16, 48]" = torch.ops.aten.reshape.default(view_278, [8, 197, 3, 16, 48]);  view_278 = None
    permute_119: "f32[3, 8, 16, 197, 48]" = torch.ops.aten.permute.default(view_279, [2, 0, 3, 1, 4]);  view_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:176, code: q, k, v = qkv.unbind(0)
    unbind_1 = torch.ops.aten.unbind.int(permute_119);  permute_119 = None
    getitem_49: "f32[8, 16, 197, 48]" = unbind_1[0]
    getitem_50: "f32[8, 16, 197, 48]" = unbind_1[1]
    getitem_51: "f32[8, 16, 197, 48]" = unbind_1[2];  unbind_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:178, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_120: "f32[8, 16, 48, 197]" = torch.ops.aten.permute.default(getitem_50, [0, 1, 3, 2]);  getitem_50 = None
    expand_75: "f32[8, 16, 197, 48]" = torch.ops.aten.expand.default(getitem_49, [8, 16, 197, 48]);  getitem_49 = None
    clone_159: "f32[8, 16, 197, 48]" = torch.ops.aten.clone.default(expand_75, memory_format = torch.contiguous_format);  expand_75 = None
    view_280: "f32[128, 197, 48]" = torch.ops.aten.reshape.default(clone_159, [128, 197, 48]);  clone_159 = None
    expand_76: "f32[8, 16, 48, 197]" = torch.ops.aten.expand.default(permute_120, [8, 16, 48, 197]);  permute_120 = None
    clone_160: "f32[8, 16, 48, 197]" = torch.ops.aten.clone.default(expand_76, memory_format = torch.contiguous_format);  expand_76 = None
    view_281: "f32[128, 48, 197]" = torch.ops.aten.reshape.default(clone_160, [128, 48, 197]);  clone_160 = None
    bmm_22: "f32[128, 197, 197]" = torch.ops.aten.bmm.default(view_280, view_281)
    view_282: "f32[8, 16, 197, 197]" = torch.ops.aten.reshape.default(bmm_22, [8, 16, 197, 197]);  bmm_22 = None
    mul_110: "f32[8, 16, 197, 197]" = torch.ops.aten.mul.Tensor(view_282, 0.14433756729740643);  view_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:179, code: attn = attn.softmax(dim=-1)
    amax_21: "f32[8, 16, 197, 1]" = torch.ops.aten.amax.default(mul_110, [-1], True)
    sub_64: "f32[8, 16, 197, 197]" = torch.ops.aten.sub.Tensor(mul_110, amax_21);  mul_110 = amax_21 = None
    exp_21: "f32[8, 16, 197, 197]" = torch.ops.aten.exp.default(sub_64);  sub_64 = None
    sum_32: "f32[8, 16, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
    div_31: "f32[8, 16, 197, 197]" = torch.ops.aten.div.Tensor(exp_21, sum_32);  exp_21 = sum_32 = None
    alias_41: "f32[8, 16, 197, 197]" = torch.ops.aten.alias.default(div_31)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:180, code: attn = self.attn_drop(attn)
    clone_161: "f32[8, 16, 197, 197]" = torch.ops.aten.clone.default(div_31);  div_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:182, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_77: "f32[8, 16, 197, 197]" = torch.ops.aten.expand.default(clone_161, [8, 16, 197, 197]);  clone_161 = None
    view_283: "f32[128, 197, 197]" = torch.ops.aten.reshape.default(expand_77, [128, 197, 197]);  expand_77 = None
    expand_78: "f32[8, 16, 197, 48]" = torch.ops.aten.expand.default(getitem_51, [8, 16, 197, 48]);  getitem_51 = None
    clone_162: "f32[8, 16, 197, 48]" = torch.ops.aten.clone.default(expand_78, memory_format = torch.contiguous_format);  expand_78 = None
    view_284: "f32[128, 197, 48]" = torch.ops.aten.reshape.default(clone_162, [128, 197, 48]);  clone_162 = None
    bmm_23: "f32[128, 197, 48]" = torch.ops.aten.bmm.default(view_283, view_284)
    view_285: "f32[8, 16, 197, 48]" = torch.ops.aten.reshape.default(bmm_23, [8, 16, 197, 48]);  bmm_23 = None
    permute_121: "f32[8, 197, 16, 48]" = torch.ops.aten.permute.default(view_285, [0, 2, 1, 3]);  view_285 = None
    clone_163: "f32[8, 197, 16, 48]" = torch.ops.aten.clone.default(permute_121, memory_format = torch.contiguous_format);  permute_121 = None
    view_286: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(clone_163, [8, 197, 768]);  clone_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:183, code: x = self.proj(x)
    view_287: "f32[1576, 768]" = torch.ops.aten.reshape.default(view_286, [1576, 768]);  view_286 = None
    permute_122: "f32[768, 768]" = torch.ops.aten.permute.default(primals_173, [1, 0]);  primals_173 = None
    
    # No stacktrace found for following nodes
    mm_default_1: "f32[1576, 768]" = torch.ops.aten.mm.default(view_287, permute_122)
    add_tensor_1: "f32[1576, 768]" = torch.ops.aten.add.Tensor(mm_default_1, primals_174);  mm_default_1 = primals_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:183, code: x = self.proj(x)
    view_288: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(add_tensor_1, [8, 197, 768]);  add_tensor_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:235, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_110: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_107, view_288);  add_107 = view_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_23 = torch.ops.aten.var_mean.correction(add_110, [2], correction = 0, keepdim = True)
    getitem_52: "f32[8, 197, 1]" = var_mean_23[0]
    getitem_53: "f32[8, 197, 1]" = var_mean_23[1];  var_mean_23 = None
    add_111: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-06);  getitem_52 = None
    rsqrt_23: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
    sub_65: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_110, getitem_53);  getitem_53 = None
    mul_111: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_23);  sub_65 = None
    mul_112: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_111, primals_59)
    add_112: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_112, primals_60);  mul_112 = primals_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_289: "f32[1576, 768]" = torch.ops.aten.reshape.default(add_112, [1576, 768]);  add_112 = None
    permute_123: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_175, [1, 0]);  primals_175 = None
    addmm_34: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_176, view_289, permute_123);  primals_176 = None
    view_290: "f32[8, 197, 3072]" = torch.ops.aten.reshape.default(addmm_34, [8, 197, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_113: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_290, 0.5)
    mul_114: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_290, 0.7071067811865476);  view_290 = None
    erf_11: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_114);  mul_114 = None
    add_113: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_115: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_113, add_113);  mul_113 = add_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_291: "f32[1576, 3072]" = torch.ops.aten.reshape.default(mul_115, [1576, 3072]);  mul_115 = None
    permute_124: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_177, [1, 0]);  primals_177 = None
    
    # No stacktrace found for following nodes
    mm_default: "f32[1576, 768]" = torch.ops.aten.mm.default(view_291, permute_124)
    add_tensor: "f32[1576, 768]" = torch.ops.aten.add.Tensor(mm_default, primals_178);  mm_default = primals_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_292: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(add_tensor, [8, 197, 768]);  add_tensor = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:236, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_114: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_110, view_292);  add_110 = view_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_24 = torch.ops.aten.var_mean.correction(add_114, [2], correction = 0, keepdim = True)
    getitem_54: "f32[8, 197, 1]" = var_mean_24[0]
    getitem_55: "f32[8, 197, 1]" = var_mean_24[1];  var_mean_24 = None
    add_115: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-06);  getitem_54 = None
    rsqrt_24: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
    sub_66: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_114, getitem_55);  add_114 = getitem_55 = None
    mul_116: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_24);  sub_66 = None
    mul_117: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_116, primals_61)
    add_116: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_117, primals_62);  mul_117 = primals_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:374, code: x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
    select_100: "f32[8, 768]" = torch.ops.aten.select.int(add_116, 1, 0);  add_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:375, code: x = self.head_drop(x)
    clone_167: "f32[8, 768]" = torch.ops.aten.clone.default(select_100);  select_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:376, code: return x if pre_logits else self.head(x)
    permute_125: "f32[768, 1000]" = torch.ops.aten.permute.default(primals_179, [1, 0]);  primals_179 = None
    addmm_36: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_180, clone_167, permute_125);  primals_180 = None
    permute_126: "f32[1000, 768]" = torch.ops.aten.permute.default(permute_125, [1, 0]);  permute_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_32: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 768);  rsqrt_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_130: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_124, [1, 0]);  permute_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_134: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_33: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:183, code: x = self.proj(x)
    permute_138: "f32[768, 768]" = torch.ops.aten.permute.default(permute_122, [1, 0]);  permute_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:182, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    permute_143: "f32[128, 197, 197]" = torch.ops.aten.permute.default(view_283, [0, 2, 1]);  view_283 = None
    permute_144: "f32[128, 48, 197]" = torch.ops.aten.permute.default(view_284, [0, 2, 1]);  view_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:179, code: attn = attn.softmax(dim=-1)
    alias_42: "f32[8, 16, 197, 197]" = torch.ops.aten.alias.default(alias_41);  alias_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:178, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_145: "f32[128, 48, 197]" = torch.ops.aten.permute.default(view_280, [0, 2, 1]);  view_280 = None
    permute_146: "f32[128, 197, 48]" = torch.ops.aten.permute.default(view_281, [0, 2, 1]);  view_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:175, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_151: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_34: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_153: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_117, [1, 0]);  permute_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_157: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_116, [1, 0]);  permute_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_35: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:183, code: x = self.proj(x)
    permute_161: "f32[768, 768]" = torch.ops.aten.permute.default(permute_115, [1, 0]);  permute_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:182, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    permute_166: "f32[128, 197, 197]" = torch.ops.aten.permute.default(view_267, [0, 2, 1]);  view_267 = None
    permute_167: "f32[128, 48, 197]" = torch.ops.aten.permute.default(view_268, [0, 2, 1]);  view_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:179, code: attn = attn.softmax(dim=-1)
    alias_43: "f32[8, 16, 197, 197]" = torch.ops.aten.alias.default(alias_40);  alias_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:178, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_168: "f32[128, 48, 197]" = torch.ops.aten.permute.default(view_264, [0, 2, 1]);  view_264 = None
    permute_169: "f32[128, 197, 48]" = torch.ops.aten.permute.default(view_265, [0, 2, 1]);  view_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:175, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_174: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_176: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_180: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_37: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    permute_184: "f32[768, 768]" = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    permute_189: "f32[128, 196, 196]" = torch.ops.aten.permute.default(view_251, [0, 2, 1]);  view_251 = None
    permute_190: "f32[128, 48, 196]" = torch.ops.aten.permute.default(view_252, [0, 2, 1]);  view_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_194: "f32[768, 768]" = torch.ops.aten.permute.default(permute_105, [1, 0]);  permute_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    permute_196: "f32[128, 48, 196]" = torch.ops.aten.permute.default(view_244, [0, 2, 1]);  view_244 = None
    permute_197: "f32[128, 196, 48]" = torch.ops.aten.permute.default(view_245, [0, 2, 1]);  view_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_206: "f32[1536, 768]" = torch.ops.aten.permute.default(permute_100, [1, 0]);  permute_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_41: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_208: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_212: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_42: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    permute_216: "f32[768, 768]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    permute_221: "f32[128, 196, 196]" = torch.ops.aten.permute.default(view_225, [0, 2, 1]);  view_225 = None
    permute_222: "f32[128, 48, 196]" = torch.ops.aten.permute.default(view_226, [0, 2, 1]);  view_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_226: "f32[768, 768]" = torch.ops.aten.permute.default(permute_94, [1, 0]);  permute_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    permute_228: "f32[128, 48, 196]" = torch.ops.aten.permute.default(view_218, [0, 2, 1]);  view_218 = None
    permute_229: "f32[128, 196, 48]" = torch.ops.aten.permute.default(view_219, [0, 2, 1]);  view_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_238: "f32[1536, 768]" = torch.ops.aten.permute.default(permute_89, [1, 0]);  permute_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_46: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_240: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_244: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_47: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    permute_248: "f32[768, 768]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    permute_253: "f32[128, 196, 196]" = torch.ops.aten.permute.default(view_199, [0, 2, 1]);  view_199 = None
    permute_254: "f32[128, 48, 196]" = torch.ops.aten.permute.default(view_200, [0, 2, 1]);  view_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_258: "f32[768, 768]" = torch.ops.aten.permute.default(permute_83, [1, 0]);  permute_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    permute_260: "f32[128, 48, 196]" = torch.ops.aten.permute.default(view_192, [0, 2, 1]);  view_192 = None
    permute_261: "f32[128, 196, 48]" = torch.ops.aten.permute.default(view_193, [0, 2, 1]);  view_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_270: "f32[1536, 768]" = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_51: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_272: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_276: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_52: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    permute_280: "f32[768, 768]" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    permute_285: "f32[128, 196, 196]" = torch.ops.aten.permute.default(view_173, [0, 2, 1]);  view_173 = None
    permute_286: "f32[128, 48, 196]" = torch.ops.aten.permute.default(view_174, [0, 2, 1]);  view_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_290: "f32[768, 768]" = torch.ops.aten.permute.default(permute_72, [1, 0]);  permute_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    permute_292: "f32[128, 48, 196]" = torch.ops.aten.permute.default(view_166, [0, 2, 1]);  view_166 = None
    permute_293: "f32[128, 196, 48]" = torch.ops.aten.permute.default(view_167, [0, 2, 1]);  view_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_302: "f32[1536, 768]" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_56: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_304: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_308: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_57: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    permute_312: "f32[768, 768]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    permute_317: "f32[128, 196, 196]" = torch.ops.aten.permute.default(view_147, [0, 2, 1]);  view_147 = None
    permute_318: "f32[128, 48, 196]" = torch.ops.aten.permute.default(view_148, [0, 2, 1]);  view_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_322: "f32[768, 768]" = torch.ops.aten.permute.default(permute_61, [1, 0]);  permute_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    permute_324: "f32[128, 48, 196]" = torch.ops.aten.permute.default(view_140, [0, 2, 1]);  view_140 = None
    permute_325: "f32[128, 196, 48]" = torch.ops.aten.permute.default(view_141, [0, 2, 1]);  view_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_334: "f32[1536, 768]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_61: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_336: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_340: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_62: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    permute_344: "f32[768, 768]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    permute_349: "f32[128, 196, 196]" = torch.ops.aten.permute.default(view_121, [0, 2, 1]);  view_121 = None
    permute_350: "f32[128, 48, 196]" = torch.ops.aten.permute.default(view_122, [0, 2, 1]);  view_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_354: "f32[768, 768]" = torch.ops.aten.permute.default(permute_50, [1, 0]);  permute_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    permute_356: "f32[128, 48, 196]" = torch.ops.aten.permute.default(view_114, [0, 2, 1]);  view_114 = None
    permute_357: "f32[128, 196, 48]" = torch.ops.aten.permute.default(view_115, [0, 2, 1]);  view_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_366: "f32[1536, 768]" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_66: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_368: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_372: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_67: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    permute_376: "f32[768, 768]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    permute_381: "f32[128, 196, 196]" = torch.ops.aten.permute.default(view_95, [0, 2, 1]);  view_95 = None
    permute_382: "f32[128, 48, 196]" = torch.ops.aten.permute.default(view_96, [0, 2, 1]);  view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_386: "f32[768, 768]" = torch.ops.aten.permute.default(permute_39, [1, 0]);  permute_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    permute_388: "f32[128, 48, 196]" = torch.ops.aten.permute.default(view_88, [0, 2, 1]);  view_88 = None
    permute_389: "f32[128, 196, 48]" = torch.ops.aten.permute.default(view_89, [0, 2, 1]);  view_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_398: "f32[1536, 768]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_71: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_400: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_404: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_72: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    permute_408: "f32[768, 768]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    permute_413: "f32[128, 196, 196]" = torch.ops.aten.permute.default(view_69, [0, 2, 1]);  view_69 = None
    permute_414: "f32[128, 48, 196]" = torch.ops.aten.permute.default(view_70, [0, 2, 1]);  view_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_418: "f32[768, 768]" = torch.ops.aten.permute.default(permute_28, [1, 0]);  permute_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    permute_420: "f32[128, 48, 196]" = torch.ops.aten.permute.default(view_62, [0, 2, 1]);  view_62 = None
    permute_421: "f32[128, 196, 48]" = torch.ops.aten.permute.default(view_63, [0, 2, 1]);  view_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_430: "f32[1536, 768]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_76: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_432: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_436: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_77: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    permute_440: "f32[768, 768]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    permute_445: "f32[128, 196, 196]" = torch.ops.aten.permute.default(view_43, [0, 2, 1]);  view_43 = None
    permute_446: "f32[128, 48, 196]" = torch.ops.aten.permute.default(view_44, [0, 2, 1]);  view_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_450: "f32[768, 768]" = torch.ops.aten.permute.default(permute_17, [1, 0]);  permute_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    permute_452: "f32[128, 48, 196]" = torch.ops.aten.permute.default(view_36, [0, 2, 1]);  view_36 = None
    permute_453: "f32[128, 196, 48]" = torch.ops.aten.permute.default(view_37, [0, 2, 1]);  view_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_462: "f32[1536, 768]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_81: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_464: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_468: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_82: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    permute_472: "f32[768, 768]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    permute_477: "f32[128, 196, 196]" = torch.ops.aten.permute.default(view_17, [0, 2, 1]);  view_17 = None
    permute_478: "f32[128, 48, 196]" = torch.ops.aten.permute.default(view_18, [0, 2, 1]);  view_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_482: "f32[768, 768]" = torch.ops.aten.permute.default(permute_6, [1, 0]);  permute_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    permute_484: "f32[128, 48, 196]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
    permute_485: "f32[128, 196, 48]" = torch.ops.aten.permute.default(view_11, [0, 2, 1]);  view_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_494: "f32[1536, 768]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_86: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
    return [addmm_36, device_put, device_put, device_put, device_put, device_put, device_put, device_put, device_put, device_put, device_put, primals_3, primals_5, primals_6, primals_8, primals_10, primals_11, primals_13, primals_15, primals_16, primals_18, primals_20, primals_21, primals_23, primals_25, primals_26, primals_28, primals_30, primals_31, primals_33, primals_35, primals_36, primals_38, primals_40, primals_41, primals_43, primals_45, primals_46, primals_48, primals_50, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_181, mul, view_5, view_8, div, div_1, unsqueeze_5, view_21, mul_5, view_23, addmm_1, view_25, mul_10, view_31, div_3, div_4, unsqueeze_11, view_47, mul_15, view_49, addmm_4, view_51, mul_20, view_57, div_6, div_7, unsqueeze_17, view_73, mul_25, view_75, addmm_7, view_77, mul_30, view_83, div_9, div_10, unsqueeze_23, view_99, mul_35, view_101, addmm_10, view_103, mul_40, view_109, div_12, div_13, unsqueeze_29, view_125, mul_45, view_127, addmm_13, view_129, mul_50, view_135, div_15, div_16, unsqueeze_35, view_151, mul_55, view_153, addmm_16, view_155, mul_60, view_161, div_18, div_19, unsqueeze_41, view_177, mul_65, view_179, addmm_19, view_181, mul_70, view_187, div_21, div_22, unsqueeze_47, view_203, mul_75, view_205, addmm_22, view_207, mul_80, view_213, div_24, div_25, unsqueeze_53, view_229, mul_85, view_231, addmm_25, view_233, mul_90, view_239, div_27, div_28, unsqueeze_59, view_255, mul_95, view_257, addmm_28, view_259, cat, getitem_41, rsqrt_20, view_261, view_271, mul_103, view_273, addmm_31, view_275, mul_108, view_277, view_287, mul_111, view_289, addmm_34, view_291, mul_116, clone_167, permute_126, div_32, permute_130, permute_134, div_33, permute_138, permute_143, permute_144, alias_42, permute_145, permute_146, permute_151, div_34, permute_153, permute_157, div_35, permute_161, permute_166, permute_167, alias_43, permute_168, permute_169, permute_174, permute_176, permute_180, div_37, permute_184, permute_189, permute_190, permute_194, permute_196, permute_197, permute_206, div_41, permute_208, permute_212, div_42, permute_216, permute_221, permute_222, permute_226, permute_228, permute_229, permute_238, div_46, permute_240, permute_244, div_47, permute_248, permute_253, permute_254, permute_258, permute_260, permute_261, permute_270, div_51, permute_272, permute_276, div_52, permute_280, permute_285, permute_286, permute_290, permute_292, permute_293, permute_302, div_56, permute_304, permute_308, div_57, permute_312, permute_317, permute_318, permute_322, permute_324, permute_325, permute_334, div_61, permute_336, permute_340, div_62, permute_344, permute_349, permute_350, permute_354, permute_356, permute_357, permute_366, div_66, permute_368, permute_372, div_67, permute_376, permute_381, permute_382, permute_386, permute_388, permute_389, permute_398, div_71, permute_400, permute_404, div_72, permute_408, permute_413, permute_414, permute_418, permute_420, permute_421, permute_430, div_76, permute_432, permute_436, div_77, permute_440, permute_445, permute_446, permute_450, permute_452, permute_453, permute_462, div_81, permute_464, permute_468, div_82, permute_472, permute_477, permute_478, permute_482, permute_484, permute_485, permute_494, div_86]
    