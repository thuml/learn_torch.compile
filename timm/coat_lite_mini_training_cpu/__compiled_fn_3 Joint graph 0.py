from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[1, 1, 64]"; primals_2: "f32[64]"; primals_3: "f32[64]"; primals_4: "f32[64]"; primals_5: "f32[64]"; primals_6: "f32[64]"; primals_7: "f32[64]"; primals_8: "f32[64]"; primals_9: "f32[64]"; primals_10: "f32[1, 1, 128]"; primals_11: "f32[128]"; primals_12: "f32[128]"; primals_13: "f32[128]"; primals_14: "f32[128]"; primals_15: "f32[128]"; primals_16: "f32[128]"; primals_17: "f32[128]"; primals_18: "f32[128]"; primals_19: "f32[1, 1, 320]"; primals_20: "f32[320]"; primals_21: "f32[320]"; primals_22: "f32[320]"; primals_23: "f32[320]"; primals_24: "f32[320]"; primals_25: "f32[320]"; primals_26: "f32[320]"; primals_27: "f32[320]"; primals_28: "f32[1, 1, 512]"; primals_29: "f32[512]"; primals_30: "f32[512]"; primals_31: "f32[512]"; primals_32: "f32[512]"; primals_33: "f32[512]"; primals_34: "f32[512]"; primals_35: "f32[512]"; primals_36: "f32[512]"; primals_37: "f32[512]"; primals_38: "f32[512]"; primals_39: "f32[64, 3, 4, 4]"; primals_40: "f32[64]"; primals_41: "f32[64]"; primals_42: "f32[64]"; primals_43: "f32[64, 1, 3, 3]"; primals_44: "f32[64]"; primals_45: "f32[192, 64]"; primals_46: "f32[192]"; primals_47: "f32[16, 1, 3, 3]"; primals_48: "f32[16]"; primals_49: "f32[24, 1, 5, 5]"; primals_50: "f32[24]"; primals_51: "f32[24, 1, 7, 7]"; primals_52: "f32[24]"; primals_53: "f32[64, 64]"; primals_54: "f32[64]"; primals_55: "f32[512, 64]"; primals_56: "f32[512]"; primals_57: "f32[64, 512]"; primals_58: "f32[64]"; primals_59: "f32[192, 64]"; primals_60: "f32[192]"; primals_61: "f32[64, 64]"; primals_62: "f32[64]"; primals_63: "f32[512, 64]"; primals_64: "f32[512]"; primals_65: "f32[64, 512]"; primals_66: "f32[64]"; primals_67: "f32[128, 64, 2, 2]"; primals_68: "f32[128]"; primals_69: "f32[128]"; primals_70: "f32[128]"; primals_71: "f32[128, 1, 3, 3]"; primals_72: "f32[128]"; primals_73: "f32[384, 128]"; primals_74: "f32[384]"; primals_75: "f32[32, 1, 3, 3]"; primals_76: "f32[32]"; primals_77: "f32[48, 1, 5, 5]"; primals_78: "f32[48]"; primals_79: "f32[48, 1, 7, 7]"; primals_80: "f32[48]"; primals_81: "f32[128, 128]"; primals_82: "f32[128]"; primals_83: "f32[1024, 128]"; primals_84: "f32[1024]"; primals_85: "f32[128, 1024]"; primals_86: "f32[128]"; primals_87: "f32[384, 128]"; primals_88: "f32[384]"; primals_89: "f32[128, 128]"; primals_90: "f32[128]"; primals_91: "f32[1024, 128]"; primals_92: "f32[1024]"; primals_93: "f32[128, 1024]"; primals_94: "f32[128]"; primals_95: "f32[320, 128, 2, 2]"; primals_96: "f32[320]"; primals_97: "f32[320]"; primals_98: "f32[320]"; primals_99: "f32[320, 1, 3, 3]"; primals_100: "f32[320]"; primals_101: "f32[960, 320]"; primals_102: "f32[960]"; primals_103: "f32[80, 1, 3, 3]"; primals_104: "f32[80]"; primals_105: "f32[120, 1, 5, 5]"; primals_106: "f32[120]"; primals_107: "f32[120, 1, 7, 7]"; primals_108: "f32[120]"; primals_109: "f32[320, 320]"; primals_110: "f32[320]"; primals_111: "f32[1280, 320]"; primals_112: "f32[1280]"; primals_113: "f32[320, 1280]"; primals_114: "f32[320]"; primals_115: "f32[960, 320]"; primals_116: "f32[960]"; primals_117: "f32[320, 320]"; primals_118: "f32[320]"; primals_119: "f32[1280, 320]"; primals_120: "f32[1280]"; primals_121: "f32[320, 1280]"; primals_122: "f32[320]"; primals_123: "f32[512, 320, 2, 2]"; primals_124: "f32[512]"; primals_125: "f32[512]"; primals_126: "f32[512]"; primals_127: "f32[512, 1, 3, 3]"; primals_128: "f32[512]"; primals_129: "f32[1536, 512]"; primals_130: "f32[1536]"; primals_131: "f32[128, 1, 3, 3]"; primals_132: "f32[128]"; primals_133: "f32[192, 1, 5, 5]"; primals_134: "f32[192]"; primals_135: "f32[192, 1, 7, 7]"; primals_136: "f32[192]"; primals_137: "f32[512, 512]"; primals_138: "f32[512]"; primals_139: "f32[2048, 512]"; primals_140: "f32[2048]"; primals_141: "f32[512, 2048]"; primals_142: "f32[512]"; primals_143: "f32[1536, 512]"; primals_144: "f32[1536]"; primals_145: "f32[512, 512]"; primals_146: "f32[512]"; primals_147: "f32[2048, 512]"; primals_148: "f32[2048]"; primals_149: "f32[512, 2048]"; primals_150: "f32[512]"; primals_151: "f32[1000, 512]"; primals_152: "f32[1000]"; primals_153: "f32[8, 3, 224, 224]"; tangents_1: "f32[8, 1000]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution: "f32[8, 64, 56, 56]" = torch.ops.aten.convolution.default(primals_153, primals_39, primals_40, [4, 4], [0, 0], [1, 1], False, [0, 0], 1);  primals_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:89, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    view: "f32[8, 64, 3136]" = torch.ops.aten.view.default(convolution, [8, 64, 3136]);  convolution = None
    permute: "f32[8, 3136, 64]" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:92, code: x = self.norm(x)
    clone: "f32[8, 3136, 64]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format)
    var_mean = torch.ops.aten.var_mean.correction(clone, [2], correction = 0, keepdim = True)
    getitem: "f32[8, 3136, 1]" = var_mean[0]
    getitem_1: "f32[8, 3136, 1]" = var_mean[1];  var_mean = None
    add: "f32[8, 3136, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
    rsqrt: "f32[8, 3136, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
    sub: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(clone, getitem_1);  clone = None
    mul: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    mul_1: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul, primals_41);  mul = None
    add_1: "f32[8, 3136, 64]" = torch.ops.aten.add.Tensor(mul_1, primals_42);  mul_1 = primals_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:677, code: cls_tokens = cls_token.expand(x.shape[0], -1, -1)
    expand: "f32[8, 1, 64]" = torch.ops.aten.expand.default(primals_1, [8, -1, -1]);  primals_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:678, code: x = torch.cat((cls_tokens, x), dim=1)
    cat: "f32[8, 3137, 64]" = torch.ops.aten.cat.default([expand, add_1], 1);  expand = add_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    slice_1: "f32[8, 3137, 64]" = torch.ops.aten.slice.Tensor(cat, 0, 0, 9223372036854775807)
    slice_2: "f32[8, 1, 64]" = torch.ops.aten.slice.Tensor(slice_1, 1, 0, 1);  slice_1 = None
    slice_3: "f32[8, 3137, 64]" = torch.ops.aten.slice.Tensor(cat, 0, 0, 9223372036854775807);  cat = None
    slice_4: "f32[8, 3136, 64]" = torch.ops.aten.slice.Tensor(slice_3, 1, 1, 9223372036854775807);  slice_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:158, code: feat = img_tokens.transpose(1, 2).view(B, C, H, W)
    permute_1: "f32[8, 64, 3136]" = torch.ops.aten.permute.default(slice_4, [0, 2, 1]);  slice_4 = None
    view_1: "f32[8, 64, 56, 56]" = torch.ops.aten.view.default(permute_1, [8, 64, 56, 56]);  permute_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:159, code: x = self.proj(feat) + feat
    convolution_1: "f32[8, 64, 56, 56]" = torch.ops.aten.convolution.default(view_1, primals_43, primals_44, [1, 1], [1, 1], [1, 1], False, [0, 0], 64)
    add_2: "f32[8, 64, 56, 56]" = torch.ops.aten.add.Tensor(convolution_1, view_1);  convolution_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:160, code: x = x.flatten(2).transpose(1, 2)
    view_2: "f32[8, 64, 3136]" = torch.ops.aten.view.default(add_2, [8, 64, 3136]);  add_2 = None
    permute_2: "f32[8, 3136, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1]);  view_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:163, code: x = torch.cat((cls_token, x), dim=1)
    cat_1: "f32[8, 3137, 64]" = torch.ops.aten.cat.default([slice_2, permute_2], 1);  slice_2 = permute_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_1 = torch.ops.aten.var_mean.correction(cat_1, [2], correction = 0, keepdim = True)
    getitem_2: "f32[8, 3137, 1]" = var_mean_1[0]
    getitem_3: "f32[8, 3137, 1]" = var_mean_1[1];  var_mean_1 = None
    add_3: "f32[8, 3137, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-06);  getitem_2 = None
    rsqrt_1: "f32[8, 3137, 1]" = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
    sub_1: "f32[8, 3137, 64]" = torch.ops.aten.sub.Tensor(cat_1, getitem_3)
    mul_2: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
    mul_3: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(mul_2, primals_2);  mul_2 = None
    add_4: "f32[8, 3137, 64]" = torch.ops.aten.add.Tensor(mul_3, primals_3);  mul_3 = primals_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:119, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_3: "f32[25096, 64]" = torch.ops.aten.view.default(add_4, [25096, 64]);  add_4 = None
    permute_3: "f32[64, 192]" = torch.ops.aten.permute.default(primals_45, [1, 0]);  primals_45 = None
    addmm: "f32[25096, 192]" = torch.ops.aten.addmm.default(primals_46, view_3, permute_3);  primals_46 = None
    view_4: "f32[8, 3137, 192]" = torch.ops.aten.view.default(addmm, [8, 3137, 192]);  addmm = None
    view_5: "f32[8, 3137, 3, 8, 8]" = torch.ops.aten.view.default(view_4, [8, 3137, 3, 8, 8]);  view_4 = None
    permute_4: "f32[3, 8, 8, 3137, 8]" = torch.ops.aten.permute.default(view_5, [2, 0, 3, 1, 4]);  view_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:120, code: q, k, v = qkv.unbind(0)  # [B, h, N, Ch]
    unbind = torch.ops.aten.unbind.int(permute_4);  permute_4 = None
    getitem_4: "f32[8, 8, 3137, 8]" = unbind[0]
    getitem_5: "f32[8, 8, 3137, 8]" = unbind[1]
    getitem_6: "f32[8, 8, 3137, 8]" = unbind[2];  unbind = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:123, code: k_softmax = k.softmax(dim=2)
    clone_1: "f32[8, 8, 3137, 8]" = torch.ops.aten.clone.default(getitem_5, memory_format = torch.contiguous_format);  getitem_5 = None
    amax: "f32[8, 8, 1, 8]" = torch.ops.aten.amax.default(clone_1, [2], True)
    sub_2: "f32[8, 8, 3137, 8]" = torch.ops.aten.sub.Tensor(clone_1, amax);  clone_1 = amax = None
    exp: "f32[8, 8, 3137, 8]" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
    sum_1: "f32[8, 8, 1, 8]" = torch.ops.aten.sum.dim_IntList(exp, [2], True)
    div: "f32[8, 8, 3137, 8]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    alias: "f32[8, 8, 3137, 8]" = torch.ops.aten.alias.default(div)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    permute_5: "f32[8, 8, 8, 3137]" = torch.ops.aten.permute.default(div, [0, 1, 3, 2]);  div = None
    expand_1: "f32[8, 8, 8, 3137]" = torch.ops.aten.expand.default(permute_5, [8, 8, 8, 3137]);  permute_5 = None
    view_6: "f32[64, 8, 3137]" = torch.ops.aten.view.default(expand_1, [64, 8, 3137]);  expand_1 = None
    expand_2: "f32[8, 8, 3137, 8]" = torch.ops.aten.expand.default(getitem_6, [8, 8, 3137, 8])
    clone_2: "f32[8, 8, 3137, 8]" = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
    view_7: "f32[64, 3137, 8]" = torch.ops.aten.view.default(clone_2, [64, 3137, 8]);  clone_2 = None
    bmm: "f32[64, 8, 8]" = torch.ops.aten.bmm.default(view_6, view_7)
    view_8: "f32[8, 8, 8, 8]" = torch.ops.aten.view.default(bmm, [8, 8, 8, 8]);  bmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    expand_3: "f32[8, 8, 3137, 8]" = torch.ops.aten.expand.default(getitem_4, [8, 8, 3137, 8])
    clone_3: "f32[8, 8, 3137, 8]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
    view_9: "f32[64, 3137, 8]" = torch.ops.aten.view.default(clone_3, [64, 3137, 8]);  clone_3 = None
    expand_4: "f32[8, 8, 8, 8]" = torch.ops.aten.expand.default(view_8, [8, 8, 8, 8]);  view_8 = None
    view_10: "f32[64, 8, 8]" = torch.ops.aten.view.default(expand_4, [64, 8, 8]);  expand_4 = None
    bmm_1: "f32[64, 3137, 8]" = torch.ops.aten.bmm.default(view_9, view_10)
    view_11: "f32[8, 8, 3137, 8]" = torch.ops.aten.view.default(bmm_1, [8, 8, 3137, 8]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:75, code: q_img = q[:, :, 1:, :]  # [B, h, H*W, Ch]
    slice_5: "f32[8, 8, 3137, 8]" = torch.ops.aten.slice.Tensor(getitem_4, 0, 0, 9223372036854775807);  getitem_4 = None
    slice_6: "f32[8, 8, 3137, 8]" = torch.ops.aten.slice.Tensor(slice_5, 1, 0, 9223372036854775807);  slice_5 = None
    slice_7: "f32[8, 8, 3136, 8]" = torch.ops.aten.slice.Tensor(slice_6, 2, 1, 9223372036854775807);  slice_6 = None
    slice_8: "f32[8, 8, 3136, 8]" = torch.ops.aten.slice.Tensor(slice_7, 3, 0, 9223372036854775807);  slice_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:76, code: v_img = v[:, :, 1:, :]  # [B, h, H*W, Ch]
    slice_9: "f32[8, 8, 3137, 8]" = torch.ops.aten.slice.Tensor(getitem_6, 0, 0, 9223372036854775807);  getitem_6 = None
    slice_10: "f32[8, 8, 3137, 8]" = torch.ops.aten.slice.Tensor(slice_9, 1, 0, 9223372036854775807);  slice_9 = None
    slice_11: "f32[8, 8, 3136, 8]" = torch.ops.aten.slice.Tensor(slice_10, 2, 1, 9223372036854775807);  slice_10 = None
    slice_12: "f32[8, 8, 3136, 8]" = torch.ops.aten.slice.Tensor(slice_11, 3, 0, 9223372036854775807);  slice_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:78, code: v_img = v_img.transpose(-1, -2).reshape(B, num_heads * C, H, W)
    permute_6: "f32[8, 8, 8, 3136]" = torch.ops.aten.permute.default(slice_12, [0, 1, 3, 2]);  slice_12 = None
    view_12: "f32[8, 64, 56, 56]" = torch.ops.aten.view.default(permute_6, [8, 64, 56, 56]);  permute_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:79, code: v_img_list = torch.split(v_img, self.channel_splits, dim=1)  # Split according to channels
    split_with_sizes = torch.ops.aten.split_with_sizes.default(view_12, [16, 24, 24], 1);  view_12 = None
    getitem_7: "f32[8, 16, 56, 56]" = split_with_sizes[0]
    getitem_8: "f32[8, 24, 56, 56]" = split_with_sizes[1]
    getitem_9: "f32[8, 24, 56, 56]" = split_with_sizes[2];  split_with_sizes = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    convolution_2: "f32[8, 16, 56, 56]" = torch.ops.aten.convolution.default(getitem_7, primals_47, primals_48, [1, 1], [1, 1], [1, 1], False, [0, 0], 16)
    convolution_3: "f32[8, 24, 56, 56]" = torch.ops.aten.convolution.default(getitem_8, primals_49, primals_50, [1, 1], [2, 2], [1, 1], False, [0, 0], 24)
    convolution_4: "f32[8, 24, 56, 56]" = torch.ops.aten.convolution.default(getitem_9, primals_51, primals_52, [1, 1], [3, 3], [1, 1], False, [0, 0], 24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:83, code: conv_v_img = torch.cat(conv_v_img_list, dim=1)
    cat_2: "f32[8, 64, 56, 56]" = torch.ops.aten.cat.default([convolution_2, convolution_3, convolution_4], 1);  convolution_2 = convolution_3 = convolution_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:84, code: conv_v_img = conv_v_img.reshape(B, num_heads, C, H * W).transpose(-1, -2)
    view_13: "f32[8, 8, 8, 3136]" = torch.ops.aten.view.default(cat_2, [8, 8, 8, 3136]);  cat_2 = None
    permute_7: "f32[8, 8, 3136, 8]" = torch.ops.aten.permute.default(view_13, [0, 1, 3, 2]);  view_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:86, code: EV_hat = q_img * conv_v_img
    mul_4: "f32[8, 8, 3136, 8]" = torch.ops.aten.mul.Tensor(slice_8, permute_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:87, code: EV_hat = F.pad(EV_hat, (0, 0, 1, 0, 0, 0))  # [B, h, N, Ch].
    constant_pad_nd: "f32[8, 8, 3137, 8]" = torch.ops.aten.constant_pad_nd.default(mul_4, [0, 0, 1, 0, 0, 0], 0.0);  mul_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:131, code: x = self.scale * factor_att + crpe
    mul_5: "f32[8, 8, 3137, 8]" = torch.ops.aten.mul.Tensor(view_11, 0.3535533905932738);  view_11 = None
    add_5: "f32[8, 8, 3137, 8]" = torch.ops.aten.add.Tensor(mul_5, constant_pad_nd);  mul_5 = constant_pad_nd = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:132, code: x = x.transpose(1, 2).reshape(B, N, C)  # [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C]
    permute_8: "f32[8, 3137, 8, 8]" = torch.ops.aten.permute.default(add_5, [0, 2, 1, 3]);  add_5 = None
    clone_4: "f32[8, 3137, 8, 8]" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None
    view_14: "f32[8, 3137, 64]" = torch.ops.aten.view.default(clone_4, [8, 3137, 64]);  clone_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:135, code: x = self.proj(x)
    view_15: "f32[25096, 64]" = torch.ops.aten.view.default(view_14, [25096, 64]);  view_14 = None
    permute_9: "f32[64, 64]" = torch.ops.aten.permute.default(primals_53, [1, 0]);  primals_53 = None
    addmm_1: "f32[25096, 64]" = torch.ops.aten.addmm.default(primals_54, view_15, permute_9);  primals_54 = None
    view_16: "f32[8, 3137, 64]" = torch.ops.aten.view.default(addmm_1, [8, 3137, 64]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:136, code: x = self.proj_drop(x)
    clone_5: "f32[8, 3137, 64]" = torch.ops.aten.clone.default(view_16);  view_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:216, code: x = x + self.drop_path(cur)
    add_6: "f32[8, 3137, 64]" = torch.ops.aten.add.Tensor(cat_1, clone_5);  clone_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_2 = torch.ops.aten.var_mean.correction(add_6, [2], correction = 0, keepdim = True)
    getitem_10: "f32[8, 3137, 1]" = var_mean_2[0]
    getitem_11: "f32[8, 3137, 1]" = var_mean_2[1];  var_mean_2 = None
    add_7: "f32[8, 3137, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-06);  getitem_10 = None
    rsqrt_2: "f32[8, 3137, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
    sub_3: "f32[8, 3137, 64]" = torch.ops.aten.sub.Tensor(add_6, getitem_11)
    mul_6: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = None
    mul_7: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(mul_6, primals_4);  mul_6 = None
    add_8: "f32[8, 3137, 64]" = torch.ops.aten.add.Tensor(mul_7, primals_5);  mul_7 = primals_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_17: "f32[25096, 64]" = torch.ops.aten.view.default(add_8, [25096, 64]);  add_8 = None
    permute_10: "f32[64, 512]" = torch.ops.aten.permute.default(primals_55, [1, 0]);  primals_55 = None
    addmm_2: "f32[25096, 512]" = torch.ops.aten.addmm.default(primals_56, view_17, permute_10);  primals_56 = None
    view_18: "f32[8, 3137, 512]" = torch.ops.aten.view.default(addmm_2, [8, 3137, 512]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_8: "f32[8, 3137, 512]" = torch.ops.aten.mul.Tensor(view_18, 0.5)
    mul_9: "f32[8, 3137, 512]" = torch.ops.aten.mul.Tensor(view_18, 0.7071067811865476)
    erf: "f32[8, 3137, 512]" = torch.ops.aten.erf.default(mul_9);  mul_9 = None
    add_9: "f32[8, 3137, 512]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_10: "f32[8, 3137, 512]" = torch.ops.aten.mul.Tensor(mul_8, add_9);  mul_8 = add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_6: "f32[8, 3137, 512]" = torch.ops.aten.clone.default(mul_10);  mul_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_19: "f32[25096, 512]" = torch.ops.aten.view.default(clone_6, [25096, 512]);  clone_6 = None
    permute_11: "f32[512, 64]" = torch.ops.aten.permute.default(primals_57, [1, 0]);  primals_57 = None
    addmm_3: "f32[25096, 64]" = torch.ops.aten.addmm.default(primals_58, view_19, permute_11);  primals_58 = None
    view_20: "f32[8, 3137, 64]" = torch.ops.aten.view.default(addmm_3, [8, 3137, 64]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_7: "f32[8, 3137, 64]" = torch.ops.aten.clone.default(view_20);  view_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:221, code: x = x + self.drop_path(cur)
    add_10: "f32[8, 3137, 64]" = torch.ops.aten.add.Tensor(add_6, clone_7);  clone_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    slice_13: "f32[8, 3137, 64]" = torch.ops.aten.slice.Tensor(add_10, 0, 0, 9223372036854775807)
    slice_14: "f32[8, 1, 64]" = torch.ops.aten.slice.Tensor(slice_13, 1, 0, 1);  slice_13 = None
    slice_15: "f32[8, 3137, 64]" = torch.ops.aten.slice.Tensor(add_10, 0, 0, 9223372036854775807);  add_10 = None
    slice_16: "f32[8, 3136, 64]" = torch.ops.aten.slice.Tensor(slice_15, 1, 1, 9223372036854775807);  slice_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:158, code: feat = img_tokens.transpose(1, 2).view(B, C, H, W)
    permute_12: "f32[8, 64, 3136]" = torch.ops.aten.permute.default(slice_16, [0, 2, 1]);  slice_16 = None
    view_21: "f32[8, 64, 56, 56]" = torch.ops.aten.view.default(permute_12, [8, 64, 56, 56]);  permute_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:159, code: x = self.proj(feat) + feat
    convolution_5: "f32[8, 64, 56, 56]" = torch.ops.aten.convolution.default(view_21, primals_43, primals_44, [1, 1], [1, 1], [1, 1], False, [0, 0], 64);  primals_44 = None
    add_11: "f32[8, 64, 56, 56]" = torch.ops.aten.add.Tensor(convolution_5, view_21);  convolution_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:160, code: x = x.flatten(2).transpose(1, 2)
    view_22: "f32[8, 64, 3136]" = torch.ops.aten.view.default(add_11, [8, 64, 3136]);  add_11 = None
    permute_13: "f32[8, 3136, 64]" = torch.ops.aten.permute.default(view_22, [0, 2, 1]);  view_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:163, code: x = torch.cat((cls_token, x), dim=1)
    cat_3: "f32[8, 3137, 64]" = torch.ops.aten.cat.default([slice_14, permute_13], 1);  slice_14 = permute_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_3 = torch.ops.aten.var_mean.correction(cat_3, [2], correction = 0, keepdim = True)
    getitem_12: "f32[8, 3137, 1]" = var_mean_3[0]
    getitem_13: "f32[8, 3137, 1]" = var_mean_3[1];  var_mean_3 = None
    add_12: "f32[8, 3137, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-06);  getitem_12 = None
    rsqrt_3: "f32[8, 3137, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
    sub_4: "f32[8, 3137, 64]" = torch.ops.aten.sub.Tensor(cat_3, getitem_13)
    mul_11: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_3);  sub_4 = None
    mul_12: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(mul_11, primals_6);  mul_11 = None
    add_13: "f32[8, 3137, 64]" = torch.ops.aten.add.Tensor(mul_12, primals_7);  mul_12 = primals_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:119, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_23: "f32[25096, 64]" = torch.ops.aten.view.default(add_13, [25096, 64]);  add_13 = None
    permute_14: "f32[64, 192]" = torch.ops.aten.permute.default(primals_59, [1, 0]);  primals_59 = None
    addmm_4: "f32[25096, 192]" = torch.ops.aten.addmm.default(primals_60, view_23, permute_14);  primals_60 = None
    view_24: "f32[8, 3137, 192]" = torch.ops.aten.view.default(addmm_4, [8, 3137, 192]);  addmm_4 = None
    view_25: "f32[8, 3137, 3, 8, 8]" = torch.ops.aten.view.default(view_24, [8, 3137, 3, 8, 8]);  view_24 = None
    permute_15: "f32[3, 8, 8, 3137, 8]" = torch.ops.aten.permute.default(view_25, [2, 0, 3, 1, 4]);  view_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:120, code: q, k, v = qkv.unbind(0)  # [B, h, N, Ch]
    unbind_1 = torch.ops.aten.unbind.int(permute_15);  permute_15 = None
    getitem_14: "f32[8, 8, 3137, 8]" = unbind_1[0]
    getitem_15: "f32[8, 8, 3137, 8]" = unbind_1[1]
    getitem_16: "f32[8, 8, 3137, 8]" = unbind_1[2];  unbind_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:123, code: k_softmax = k.softmax(dim=2)
    clone_8: "f32[8, 8, 3137, 8]" = torch.ops.aten.clone.default(getitem_15, memory_format = torch.contiguous_format);  getitem_15 = None
    amax_1: "f32[8, 8, 1, 8]" = torch.ops.aten.amax.default(clone_8, [2], True)
    sub_5: "f32[8, 8, 3137, 8]" = torch.ops.aten.sub.Tensor(clone_8, amax_1);  clone_8 = amax_1 = None
    exp_1: "f32[8, 8, 3137, 8]" = torch.ops.aten.exp.default(sub_5);  sub_5 = None
    sum_2: "f32[8, 8, 1, 8]" = torch.ops.aten.sum.dim_IntList(exp_1, [2], True)
    div_1: "f32[8, 8, 3137, 8]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    alias_1: "f32[8, 8, 3137, 8]" = torch.ops.aten.alias.default(div_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    permute_16: "f32[8, 8, 8, 3137]" = torch.ops.aten.permute.default(div_1, [0, 1, 3, 2]);  div_1 = None
    expand_5: "f32[8, 8, 8, 3137]" = torch.ops.aten.expand.default(permute_16, [8, 8, 8, 3137]);  permute_16 = None
    view_26: "f32[64, 8, 3137]" = torch.ops.aten.view.default(expand_5, [64, 8, 3137]);  expand_5 = None
    expand_6: "f32[8, 8, 3137, 8]" = torch.ops.aten.expand.default(getitem_16, [8, 8, 3137, 8])
    clone_9: "f32[8, 8, 3137, 8]" = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format);  expand_6 = None
    view_27: "f32[64, 3137, 8]" = torch.ops.aten.view.default(clone_9, [64, 3137, 8]);  clone_9 = None
    bmm_2: "f32[64, 8, 8]" = torch.ops.aten.bmm.default(view_26, view_27)
    view_28: "f32[8, 8, 8, 8]" = torch.ops.aten.view.default(bmm_2, [8, 8, 8, 8]);  bmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    expand_7: "f32[8, 8, 3137, 8]" = torch.ops.aten.expand.default(getitem_14, [8, 8, 3137, 8])
    clone_10: "f32[8, 8, 3137, 8]" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
    view_29: "f32[64, 3137, 8]" = torch.ops.aten.view.default(clone_10, [64, 3137, 8]);  clone_10 = None
    expand_8: "f32[8, 8, 8, 8]" = torch.ops.aten.expand.default(view_28, [8, 8, 8, 8]);  view_28 = None
    view_30: "f32[64, 8, 8]" = torch.ops.aten.view.default(expand_8, [64, 8, 8]);  expand_8 = None
    bmm_3: "f32[64, 3137, 8]" = torch.ops.aten.bmm.default(view_29, view_30)
    view_31: "f32[8, 8, 3137, 8]" = torch.ops.aten.view.default(bmm_3, [8, 8, 3137, 8]);  bmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:75, code: q_img = q[:, :, 1:, :]  # [B, h, H*W, Ch]
    slice_17: "f32[8, 8, 3137, 8]" = torch.ops.aten.slice.Tensor(getitem_14, 0, 0, 9223372036854775807);  getitem_14 = None
    slice_18: "f32[8, 8, 3137, 8]" = torch.ops.aten.slice.Tensor(slice_17, 1, 0, 9223372036854775807);  slice_17 = None
    slice_19: "f32[8, 8, 3136, 8]" = torch.ops.aten.slice.Tensor(slice_18, 2, 1, 9223372036854775807);  slice_18 = None
    slice_20: "f32[8, 8, 3136, 8]" = torch.ops.aten.slice.Tensor(slice_19, 3, 0, 9223372036854775807);  slice_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:76, code: v_img = v[:, :, 1:, :]  # [B, h, H*W, Ch]
    slice_21: "f32[8, 8, 3137, 8]" = torch.ops.aten.slice.Tensor(getitem_16, 0, 0, 9223372036854775807);  getitem_16 = None
    slice_22: "f32[8, 8, 3137, 8]" = torch.ops.aten.slice.Tensor(slice_21, 1, 0, 9223372036854775807);  slice_21 = None
    slice_23: "f32[8, 8, 3136, 8]" = torch.ops.aten.slice.Tensor(slice_22, 2, 1, 9223372036854775807);  slice_22 = None
    slice_24: "f32[8, 8, 3136, 8]" = torch.ops.aten.slice.Tensor(slice_23, 3, 0, 9223372036854775807);  slice_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:78, code: v_img = v_img.transpose(-1, -2).reshape(B, num_heads * C, H, W)
    permute_17: "f32[8, 8, 8, 3136]" = torch.ops.aten.permute.default(slice_24, [0, 1, 3, 2]);  slice_24 = None
    view_32: "f32[8, 64, 56, 56]" = torch.ops.aten.view.default(permute_17, [8, 64, 56, 56]);  permute_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:79, code: v_img_list = torch.split(v_img, self.channel_splits, dim=1)  # Split according to channels
    split_with_sizes_1 = torch.ops.aten.split_with_sizes.default(view_32, [16, 24, 24], 1);  view_32 = None
    getitem_17: "f32[8, 16, 56, 56]" = split_with_sizes_1[0]
    getitem_18: "f32[8, 24, 56, 56]" = split_with_sizes_1[1]
    getitem_19: "f32[8, 24, 56, 56]" = split_with_sizes_1[2];  split_with_sizes_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    convolution_6: "f32[8, 16, 56, 56]" = torch.ops.aten.convolution.default(getitem_17, primals_47, primals_48, [1, 1], [1, 1], [1, 1], False, [0, 0], 16);  primals_48 = None
    convolution_7: "f32[8, 24, 56, 56]" = torch.ops.aten.convolution.default(getitem_18, primals_49, primals_50, [1, 1], [2, 2], [1, 1], False, [0, 0], 24);  primals_50 = None
    convolution_8: "f32[8, 24, 56, 56]" = torch.ops.aten.convolution.default(getitem_19, primals_51, primals_52, [1, 1], [3, 3], [1, 1], False, [0, 0], 24);  primals_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:83, code: conv_v_img = torch.cat(conv_v_img_list, dim=1)
    cat_4: "f32[8, 64, 56, 56]" = torch.ops.aten.cat.default([convolution_6, convolution_7, convolution_8], 1);  convolution_6 = convolution_7 = convolution_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:84, code: conv_v_img = conv_v_img.reshape(B, num_heads, C, H * W).transpose(-1, -2)
    view_33: "f32[8, 8, 8, 3136]" = torch.ops.aten.view.default(cat_4, [8, 8, 8, 3136]);  cat_4 = None
    permute_18: "f32[8, 8, 3136, 8]" = torch.ops.aten.permute.default(view_33, [0, 1, 3, 2]);  view_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:86, code: EV_hat = q_img * conv_v_img
    mul_13: "f32[8, 8, 3136, 8]" = torch.ops.aten.mul.Tensor(slice_20, permute_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:87, code: EV_hat = F.pad(EV_hat, (0, 0, 1, 0, 0, 0))  # [B, h, N, Ch].
    constant_pad_nd_1: "f32[8, 8, 3137, 8]" = torch.ops.aten.constant_pad_nd.default(mul_13, [0, 0, 1, 0, 0, 0], 0.0);  mul_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:131, code: x = self.scale * factor_att + crpe
    mul_14: "f32[8, 8, 3137, 8]" = torch.ops.aten.mul.Tensor(view_31, 0.3535533905932738);  view_31 = None
    add_14: "f32[8, 8, 3137, 8]" = torch.ops.aten.add.Tensor(mul_14, constant_pad_nd_1);  mul_14 = constant_pad_nd_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:132, code: x = x.transpose(1, 2).reshape(B, N, C)  # [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C]
    permute_19: "f32[8, 3137, 8, 8]" = torch.ops.aten.permute.default(add_14, [0, 2, 1, 3]);  add_14 = None
    clone_11: "f32[8, 3137, 8, 8]" = torch.ops.aten.clone.default(permute_19, memory_format = torch.contiguous_format);  permute_19 = None
    view_34: "f32[8, 3137, 64]" = torch.ops.aten.view.default(clone_11, [8, 3137, 64]);  clone_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:135, code: x = self.proj(x)
    view_35: "f32[25096, 64]" = torch.ops.aten.view.default(view_34, [25096, 64]);  view_34 = None
    permute_20: "f32[64, 64]" = torch.ops.aten.permute.default(primals_61, [1, 0]);  primals_61 = None
    addmm_5: "f32[25096, 64]" = torch.ops.aten.addmm.default(primals_62, view_35, permute_20);  primals_62 = None
    view_36: "f32[8, 3137, 64]" = torch.ops.aten.view.default(addmm_5, [8, 3137, 64]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:136, code: x = self.proj_drop(x)
    clone_12: "f32[8, 3137, 64]" = torch.ops.aten.clone.default(view_36);  view_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:216, code: x = x + self.drop_path(cur)
    add_15: "f32[8, 3137, 64]" = torch.ops.aten.add.Tensor(cat_3, clone_12);  clone_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_4 = torch.ops.aten.var_mean.correction(add_15, [2], correction = 0, keepdim = True)
    getitem_20: "f32[8, 3137, 1]" = var_mean_4[0]
    getitem_21: "f32[8, 3137, 1]" = var_mean_4[1];  var_mean_4 = None
    add_16: "f32[8, 3137, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-06);  getitem_20 = None
    rsqrt_4: "f32[8, 3137, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
    sub_6: "f32[8, 3137, 64]" = torch.ops.aten.sub.Tensor(add_15, getitem_21)
    mul_15: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = None
    mul_16: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(mul_15, primals_8);  mul_15 = None
    add_17: "f32[8, 3137, 64]" = torch.ops.aten.add.Tensor(mul_16, primals_9);  mul_16 = primals_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_37: "f32[25096, 64]" = torch.ops.aten.view.default(add_17, [25096, 64]);  add_17 = None
    permute_21: "f32[64, 512]" = torch.ops.aten.permute.default(primals_63, [1, 0]);  primals_63 = None
    addmm_6: "f32[25096, 512]" = torch.ops.aten.addmm.default(primals_64, view_37, permute_21);  primals_64 = None
    view_38: "f32[8, 3137, 512]" = torch.ops.aten.view.default(addmm_6, [8, 3137, 512]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_17: "f32[8, 3137, 512]" = torch.ops.aten.mul.Tensor(view_38, 0.5)
    mul_18: "f32[8, 3137, 512]" = torch.ops.aten.mul.Tensor(view_38, 0.7071067811865476)
    erf_1: "f32[8, 3137, 512]" = torch.ops.aten.erf.default(mul_18);  mul_18 = None
    add_18: "f32[8, 3137, 512]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_19: "f32[8, 3137, 512]" = torch.ops.aten.mul.Tensor(mul_17, add_18);  mul_17 = add_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_13: "f32[8, 3137, 512]" = torch.ops.aten.clone.default(mul_19);  mul_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_39: "f32[25096, 512]" = torch.ops.aten.view.default(clone_13, [25096, 512]);  clone_13 = None
    permute_22: "f32[512, 64]" = torch.ops.aten.permute.default(primals_65, [1, 0]);  primals_65 = None
    addmm_7: "f32[25096, 64]" = torch.ops.aten.addmm.default(primals_66, view_39, permute_22);  primals_66 = None
    view_40: "f32[8, 3137, 64]" = torch.ops.aten.view.default(addmm_7, [8, 3137, 64]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_14: "f32[8, 3137, 64]" = torch.ops.aten.clone.default(view_40);  view_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:221, code: x = x + self.drop_path(cur)
    add_19: "f32[8, 3137, 64]" = torch.ops.aten.add.Tensor(add_15, clone_14);  clone_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:684, code: return x[:, 1:, :]
    slice_25: "f32[8, 3137, 64]" = torch.ops.aten.slice.Tensor(add_19, 0, 0, 9223372036854775807);  add_19 = None
    slice_26: "f32[8, 3136, 64]" = torch.ops.aten.slice.Tensor(slice_25, 1, 1, 9223372036854775807);  slice_25 = None
    slice_27: "f32[8, 3136, 64]" = torch.ops.aten.slice.Tensor(slice_26, 2, 0, 9223372036854775807);  slice_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:579, code: x1_nocls = remove_cls(x1).reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
    view_41: "f32[8, 56, 56, 64]" = torch.ops.aten.view.default(slice_27, [8, 56, 56, 64]);  slice_27 = None
    permute_23: "f32[8, 64, 56, 56]" = torch.ops.aten.permute.default(view_41, [0, 3, 1, 2]);  view_41 = None
    clone_15: "f32[8, 64, 56, 56]" = torch.ops.aten.clone.default(permute_23, memory_format = torch.contiguous_format);  permute_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution_9: "f32[8, 128, 28, 28]" = torch.ops.aten.convolution.default(clone_15, primals_67, primals_68, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  primals_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:89, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    view_42: "f32[8, 128, 784]" = torch.ops.aten.view.default(convolution_9, [8, 128, 784]);  convolution_9 = None
    permute_24: "f32[8, 784, 128]" = torch.ops.aten.permute.default(view_42, [0, 2, 1]);  view_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:92, code: x = self.norm(x)
    clone_16: "f32[8, 784, 128]" = torch.ops.aten.clone.default(permute_24, memory_format = torch.contiguous_format)
    var_mean_5 = torch.ops.aten.var_mean.correction(clone_16, [2], correction = 0, keepdim = True)
    getitem_22: "f32[8, 784, 1]" = var_mean_5[0]
    getitem_23: "f32[8, 784, 1]" = var_mean_5[1];  var_mean_5 = None
    add_20: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
    rsqrt_5: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
    sub_7: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(clone_16, getitem_23);  clone_16 = None
    mul_20: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_5);  sub_7 = None
    mul_21: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_20, primals_69);  mul_20 = None
    add_21: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(mul_21, primals_70);  mul_21 = primals_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:677, code: cls_tokens = cls_token.expand(x.shape[0], -1, -1)
    expand_9: "f32[8, 1, 128]" = torch.ops.aten.expand.default(primals_10, [8, -1, -1]);  primals_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:678, code: x = torch.cat((cls_tokens, x), dim=1)
    cat_5: "f32[8, 785, 128]" = torch.ops.aten.cat.default([expand_9, add_21], 1);  expand_9 = add_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    slice_28: "f32[8, 785, 128]" = torch.ops.aten.slice.Tensor(cat_5, 0, 0, 9223372036854775807)
    slice_29: "f32[8, 1, 128]" = torch.ops.aten.slice.Tensor(slice_28, 1, 0, 1);  slice_28 = None
    slice_30: "f32[8, 785, 128]" = torch.ops.aten.slice.Tensor(cat_5, 0, 0, 9223372036854775807);  cat_5 = None
    slice_31: "f32[8, 784, 128]" = torch.ops.aten.slice.Tensor(slice_30, 1, 1, 9223372036854775807);  slice_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:158, code: feat = img_tokens.transpose(1, 2).view(B, C, H, W)
    permute_25: "f32[8, 128, 784]" = torch.ops.aten.permute.default(slice_31, [0, 2, 1]);  slice_31 = None
    view_43: "f32[8, 128, 28, 28]" = torch.ops.aten.view.default(permute_25, [8, 128, 28, 28]);  permute_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:159, code: x = self.proj(feat) + feat
    convolution_10: "f32[8, 128, 28, 28]" = torch.ops.aten.convolution.default(view_43, primals_71, primals_72, [1, 1], [1, 1], [1, 1], False, [0, 0], 128)
    add_22: "f32[8, 128, 28, 28]" = torch.ops.aten.add.Tensor(convolution_10, view_43);  convolution_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:160, code: x = x.flatten(2).transpose(1, 2)
    view_44: "f32[8, 128, 784]" = torch.ops.aten.view.default(add_22, [8, 128, 784]);  add_22 = None
    permute_26: "f32[8, 784, 128]" = torch.ops.aten.permute.default(view_44, [0, 2, 1]);  view_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:163, code: x = torch.cat((cls_token, x), dim=1)
    cat_6: "f32[8, 785, 128]" = torch.ops.aten.cat.default([slice_29, permute_26], 1);  slice_29 = permute_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_6 = torch.ops.aten.var_mean.correction(cat_6, [2], correction = 0, keepdim = True)
    getitem_24: "f32[8, 785, 1]" = var_mean_6[0]
    getitem_25: "f32[8, 785, 1]" = var_mean_6[1];  var_mean_6 = None
    add_23: "f32[8, 785, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-06);  getitem_24 = None
    rsqrt_6: "f32[8, 785, 1]" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
    sub_8: "f32[8, 785, 128]" = torch.ops.aten.sub.Tensor(cat_6, getitem_25)
    mul_22: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_6);  sub_8 = None
    mul_23: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(mul_22, primals_11);  mul_22 = None
    add_24: "f32[8, 785, 128]" = torch.ops.aten.add.Tensor(mul_23, primals_12);  mul_23 = primals_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:119, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_45: "f32[6280, 128]" = torch.ops.aten.view.default(add_24, [6280, 128]);  add_24 = None
    permute_27: "f32[128, 384]" = torch.ops.aten.permute.default(primals_73, [1, 0]);  primals_73 = None
    addmm_8: "f32[6280, 384]" = torch.ops.aten.addmm.default(primals_74, view_45, permute_27);  primals_74 = None
    view_46: "f32[8, 785, 384]" = torch.ops.aten.view.default(addmm_8, [8, 785, 384]);  addmm_8 = None
    view_47: "f32[8, 785, 3, 8, 16]" = torch.ops.aten.view.default(view_46, [8, 785, 3, 8, 16]);  view_46 = None
    permute_28: "f32[3, 8, 8, 785, 16]" = torch.ops.aten.permute.default(view_47, [2, 0, 3, 1, 4]);  view_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:120, code: q, k, v = qkv.unbind(0)  # [B, h, N, Ch]
    unbind_2 = torch.ops.aten.unbind.int(permute_28);  permute_28 = None
    getitem_26: "f32[8, 8, 785, 16]" = unbind_2[0]
    getitem_27: "f32[8, 8, 785, 16]" = unbind_2[1]
    getitem_28: "f32[8, 8, 785, 16]" = unbind_2[2];  unbind_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:123, code: k_softmax = k.softmax(dim=2)
    clone_17: "f32[8, 8, 785, 16]" = torch.ops.aten.clone.default(getitem_27, memory_format = torch.contiguous_format);  getitem_27 = None
    amax_2: "f32[8, 8, 1, 16]" = torch.ops.aten.amax.default(clone_17, [2], True)
    sub_9: "f32[8, 8, 785, 16]" = torch.ops.aten.sub.Tensor(clone_17, amax_2);  clone_17 = amax_2 = None
    exp_2: "f32[8, 8, 785, 16]" = torch.ops.aten.exp.default(sub_9);  sub_9 = None
    sum_3: "f32[8, 8, 1, 16]" = torch.ops.aten.sum.dim_IntList(exp_2, [2], True)
    div_2: "f32[8, 8, 785, 16]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    alias_2: "f32[8, 8, 785, 16]" = torch.ops.aten.alias.default(div_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    permute_29: "f32[8, 8, 16, 785]" = torch.ops.aten.permute.default(div_2, [0, 1, 3, 2]);  div_2 = None
    expand_10: "f32[8, 8, 16, 785]" = torch.ops.aten.expand.default(permute_29, [8, 8, 16, 785]);  permute_29 = None
    view_48: "f32[64, 16, 785]" = torch.ops.aten.view.default(expand_10, [64, 16, 785]);  expand_10 = None
    expand_11: "f32[8, 8, 785, 16]" = torch.ops.aten.expand.default(getitem_28, [8, 8, 785, 16])
    clone_18: "f32[8, 8, 785, 16]" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
    view_49: "f32[64, 785, 16]" = torch.ops.aten.view.default(clone_18, [64, 785, 16]);  clone_18 = None
    bmm_4: "f32[64, 16, 16]" = torch.ops.aten.bmm.default(view_48, view_49)
    view_50: "f32[8, 8, 16, 16]" = torch.ops.aten.view.default(bmm_4, [8, 8, 16, 16]);  bmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    expand_12: "f32[8, 8, 785, 16]" = torch.ops.aten.expand.default(getitem_26, [8, 8, 785, 16])
    clone_19: "f32[8, 8, 785, 16]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
    view_51: "f32[64, 785, 16]" = torch.ops.aten.view.default(clone_19, [64, 785, 16]);  clone_19 = None
    expand_13: "f32[8, 8, 16, 16]" = torch.ops.aten.expand.default(view_50, [8, 8, 16, 16]);  view_50 = None
    view_52: "f32[64, 16, 16]" = torch.ops.aten.view.default(expand_13, [64, 16, 16]);  expand_13 = None
    bmm_5: "f32[64, 785, 16]" = torch.ops.aten.bmm.default(view_51, view_52)
    view_53: "f32[8, 8, 785, 16]" = torch.ops.aten.view.default(bmm_5, [8, 8, 785, 16]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:75, code: q_img = q[:, :, 1:, :]  # [B, h, H*W, Ch]
    slice_32: "f32[8, 8, 785, 16]" = torch.ops.aten.slice.Tensor(getitem_26, 0, 0, 9223372036854775807);  getitem_26 = None
    slice_33: "f32[8, 8, 785, 16]" = torch.ops.aten.slice.Tensor(slice_32, 1, 0, 9223372036854775807);  slice_32 = None
    slice_34: "f32[8, 8, 784, 16]" = torch.ops.aten.slice.Tensor(slice_33, 2, 1, 9223372036854775807);  slice_33 = None
    slice_35: "f32[8, 8, 784, 16]" = torch.ops.aten.slice.Tensor(slice_34, 3, 0, 9223372036854775807);  slice_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:76, code: v_img = v[:, :, 1:, :]  # [B, h, H*W, Ch]
    slice_36: "f32[8, 8, 785, 16]" = torch.ops.aten.slice.Tensor(getitem_28, 0, 0, 9223372036854775807);  getitem_28 = None
    slice_37: "f32[8, 8, 785, 16]" = torch.ops.aten.slice.Tensor(slice_36, 1, 0, 9223372036854775807);  slice_36 = None
    slice_38: "f32[8, 8, 784, 16]" = torch.ops.aten.slice.Tensor(slice_37, 2, 1, 9223372036854775807);  slice_37 = None
    slice_39: "f32[8, 8, 784, 16]" = torch.ops.aten.slice.Tensor(slice_38, 3, 0, 9223372036854775807);  slice_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:78, code: v_img = v_img.transpose(-1, -2).reshape(B, num_heads * C, H, W)
    permute_30: "f32[8, 8, 16, 784]" = torch.ops.aten.permute.default(slice_39, [0, 1, 3, 2]);  slice_39 = None
    view_54: "f32[8, 128, 28, 28]" = torch.ops.aten.view.default(permute_30, [8, 128, 28, 28]);  permute_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:79, code: v_img_list = torch.split(v_img, self.channel_splits, dim=1)  # Split according to channels
    split_with_sizes_2 = torch.ops.aten.split_with_sizes.default(view_54, [32, 48, 48], 1);  view_54 = None
    getitem_29: "f32[8, 32, 28, 28]" = split_with_sizes_2[0]
    getitem_30: "f32[8, 48, 28, 28]" = split_with_sizes_2[1]
    getitem_31: "f32[8, 48, 28, 28]" = split_with_sizes_2[2];  split_with_sizes_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    convolution_11: "f32[8, 32, 28, 28]" = torch.ops.aten.convolution.default(getitem_29, primals_75, primals_76, [1, 1], [1, 1], [1, 1], False, [0, 0], 32)
    convolution_12: "f32[8, 48, 28, 28]" = torch.ops.aten.convolution.default(getitem_30, primals_77, primals_78, [1, 1], [2, 2], [1, 1], False, [0, 0], 48)
    convolution_13: "f32[8, 48, 28, 28]" = torch.ops.aten.convolution.default(getitem_31, primals_79, primals_80, [1, 1], [3, 3], [1, 1], False, [0, 0], 48)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:83, code: conv_v_img = torch.cat(conv_v_img_list, dim=1)
    cat_7: "f32[8, 128, 28, 28]" = torch.ops.aten.cat.default([convolution_11, convolution_12, convolution_13], 1);  convolution_11 = convolution_12 = convolution_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:84, code: conv_v_img = conv_v_img.reshape(B, num_heads, C, H * W).transpose(-1, -2)
    view_55: "f32[8, 8, 16, 784]" = torch.ops.aten.view.default(cat_7, [8, 8, 16, 784]);  cat_7 = None
    permute_31: "f32[8, 8, 784, 16]" = torch.ops.aten.permute.default(view_55, [0, 1, 3, 2]);  view_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:86, code: EV_hat = q_img * conv_v_img
    mul_24: "f32[8, 8, 784, 16]" = torch.ops.aten.mul.Tensor(slice_35, permute_31)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:87, code: EV_hat = F.pad(EV_hat, (0, 0, 1, 0, 0, 0))  # [B, h, N, Ch].
    constant_pad_nd_2: "f32[8, 8, 785, 16]" = torch.ops.aten.constant_pad_nd.default(mul_24, [0, 0, 1, 0, 0, 0], 0.0);  mul_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:131, code: x = self.scale * factor_att + crpe
    mul_25: "f32[8, 8, 785, 16]" = torch.ops.aten.mul.Tensor(view_53, 0.25);  view_53 = None
    add_25: "f32[8, 8, 785, 16]" = torch.ops.aten.add.Tensor(mul_25, constant_pad_nd_2);  mul_25 = constant_pad_nd_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:132, code: x = x.transpose(1, 2).reshape(B, N, C)  # [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C]
    permute_32: "f32[8, 785, 8, 16]" = torch.ops.aten.permute.default(add_25, [0, 2, 1, 3]);  add_25 = None
    clone_20: "f32[8, 785, 8, 16]" = torch.ops.aten.clone.default(permute_32, memory_format = torch.contiguous_format);  permute_32 = None
    view_56: "f32[8, 785, 128]" = torch.ops.aten.view.default(clone_20, [8, 785, 128]);  clone_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:135, code: x = self.proj(x)
    view_57: "f32[6280, 128]" = torch.ops.aten.view.default(view_56, [6280, 128]);  view_56 = None
    permute_33: "f32[128, 128]" = torch.ops.aten.permute.default(primals_81, [1, 0]);  primals_81 = None
    addmm_9: "f32[6280, 128]" = torch.ops.aten.addmm.default(primals_82, view_57, permute_33);  primals_82 = None
    view_58: "f32[8, 785, 128]" = torch.ops.aten.view.default(addmm_9, [8, 785, 128]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:136, code: x = self.proj_drop(x)
    clone_21: "f32[8, 785, 128]" = torch.ops.aten.clone.default(view_58);  view_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:216, code: x = x + self.drop_path(cur)
    add_26: "f32[8, 785, 128]" = torch.ops.aten.add.Tensor(cat_6, clone_21);  clone_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_7 = torch.ops.aten.var_mean.correction(add_26, [2], correction = 0, keepdim = True)
    getitem_32: "f32[8, 785, 1]" = var_mean_7[0]
    getitem_33: "f32[8, 785, 1]" = var_mean_7[1];  var_mean_7 = None
    add_27: "f32[8, 785, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-06);  getitem_32 = None
    rsqrt_7: "f32[8, 785, 1]" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
    sub_10: "f32[8, 785, 128]" = torch.ops.aten.sub.Tensor(add_26, getitem_33)
    mul_26: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_7);  sub_10 = None
    mul_27: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(mul_26, primals_13);  mul_26 = None
    add_28: "f32[8, 785, 128]" = torch.ops.aten.add.Tensor(mul_27, primals_14);  mul_27 = primals_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_59: "f32[6280, 128]" = torch.ops.aten.view.default(add_28, [6280, 128]);  add_28 = None
    permute_34: "f32[128, 1024]" = torch.ops.aten.permute.default(primals_83, [1, 0]);  primals_83 = None
    addmm_10: "f32[6280, 1024]" = torch.ops.aten.addmm.default(primals_84, view_59, permute_34);  primals_84 = None
    view_60: "f32[8, 785, 1024]" = torch.ops.aten.view.default(addmm_10, [8, 785, 1024]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_28: "f32[8, 785, 1024]" = torch.ops.aten.mul.Tensor(view_60, 0.5)
    mul_29: "f32[8, 785, 1024]" = torch.ops.aten.mul.Tensor(view_60, 0.7071067811865476)
    erf_2: "f32[8, 785, 1024]" = torch.ops.aten.erf.default(mul_29);  mul_29 = None
    add_29: "f32[8, 785, 1024]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_30: "f32[8, 785, 1024]" = torch.ops.aten.mul.Tensor(mul_28, add_29);  mul_28 = add_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_22: "f32[8, 785, 1024]" = torch.ops.aten.clone.default(mul_30);  mul_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_61: "f32[6280, 1024]" = torch.ops.aten.view.default(clone_22, [6280, 1024]);  clone_22 = None
    permute_35: "f32[1024, 128]" = torch.ops.aten.permute.default(primals_85, [1, 0]);  primals_85 = None
    addmm_11: "f32[6280, 128]" = torch.ops.aten.addmm.default(primals_86, view_61, permute_35);  primals_86 = None
    view_62: "f32[8, 785, 128]" = torch.ops.aten.view.default(addmm_11, [8, 785, 128]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_23: "f32[8, 785, 128]" = torch.ops.aten.clone.default(view_62);  view_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:221, code: x = x + self.drop_path(cur)
    add_30: "f32[8, 785, 128]" = torch.ops.aten.add.Tensor(add_26, clone_23);  clone_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    slice_40: "f32[8, 785, 128]" = torch.ops.aten.slice.Tensor(add_30, 0, 0, 9223372036854775807)
    slice_41: "f32[8, 1, 128]" = torch.ops.aten.slice.Tensor(slice_40, 1, 0, 1);  slice_40 = None
    slice_42: "f32[8, 785, 128]" = torch.ops.aten.slice.Tensor(add_30, 0, 0, 9223372036854775807);  add_30 = None
    slice_43: "f32[8, 784, 128]" = torch.ops.aten.slice.Tensor(slice_42, 1, 1, 9223372036854775807);  slice_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:158, code: feat = img_tokens.transpose(1, 2).view(B, C, H, W)
    permute_36: "f32[8, 128, 784]" = torch.ops.aten.permute.default(slice_43, [0, 2, 1]);  slice_43 = None
    view_63: "f32[8, 128, 28, 28]" = torch.ops.aten.view.default(permute_36, [8, 128, 28, 28]);  permute_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:159, code: x = self.proj(feat) + feat
    convolution_14: "f32[8, 128, 28, 28]" = torch.ops.aten.convolution.default(view_63, primals_71, primals_72, [1, 1], [1, 1], [1, 1], False, [0, 0], 128);  primals_72 = None
    add_31: "f32[8, 128, 28, 28]" = torch.ops.aten.add.Tensor(convolution_14, view_63);  convolution_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:160, code: x = x.flatten(2).transpose(1, 2)
    view_64: "f32[8, 128, 784]" = torch.ops.aten.view.default(add_31, [8, 128, 784]);  add_31 = None
    permute_37: "f32[8, 784, 128]" = torch.ops.aten.permute.default(view_64, [0, 2, 1]);  view_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:163, code: x = torch.cat((cls_token, x), dim=1)
    cat_8: "f32[8, 785, 128]" = torch.ops.aten.cat.default([slice_41, permute_37], 1);  slice_41 = permute_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_8 = torch.ops.aten.var_mean.correction(cat_8, [2], correction = 0, keepdim = True)
    getitem_34: "f32[8, 785, 1]" = var_mean_8[0]
    getitem_35: "f32[8, 785, 1]" = var_mean_8[1];  var_mean_8 = None
    add_32: "f32[8, 785, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-06);  getitem_34 = None
    rsqrt_8: "f32[8, 785, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
    sub_11: "f32[8, 785, 128]" = torch.ops.aten.sub.Tensor(cat_8, getitem_35)
    mul_31: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_8);  sub_11 = None
    mul_32: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(mul_31, primals_15);  mul_31 = None
    add_33: "f32[8, 785, 128]" = torch.ops.aten.add.Tensor(mul_32, primals_16);  mul_32 = primals_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:119, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_65: "f32[6280, 128]" = torch.ops.aten.view.default(add_33, [6280, 128]);  add_33 = None
    permute_38: "f32[128, 384]" = torch.ops.aten.permute.default(primals_87, [1, 0]);  primals_87 = None
    addmm_12: "f32[6280, 384]" = torch.ops.aten.addmm.default(primals_88, view_65, permute_38);  primals_88 = None
    view_66: "f32[8, 785, 384]" = torch.ops.aten.view.default(addmm_12, [8, 785, 384]);  addmm_12 = None
    view_67: "f32[8, 785, 3, 8, 16]" = torch.ops.aten.view.default(view_66, [8, 785, 3, 8, 16]);  view_66 = None
    permute_39: "f32[3, 8, 8, 785, 16]" = torch.ops.aten.permute.default(view_67, [2, 0, 3, 1, 4]);  view_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:120, code: q, k, v = qkv.unbind(0)  # [B, h, N, Ch]
    unbind_3 = torch.ops.aten.unbind.int(permute_39);  permute_39 = None
    getitem_36: "f32[8, 8, 785, 16]" = unbind_3[0]
    getitem_37: "f32[8, 8, 785, 16]" = unbind_3[1]
    getitem_38: "f32[8, 8, 785, 16]" = unbind_3[2];  unbind_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:123, code: k_softmax = k.softmax(dim=2)
    clone_24: "f32[8, 8, 785, 16]" = torch.ops.aten.clone.default(getitem_37, memory_format = torch.contiguous_format);  getitem_37 = None
    amax_3: "f32[8, 8, 1, 16]" = torch.ops.aten.amax.default(clone_24, [2], True)
    sub_12: "f32[8, 8, 785, 16]" = torch.ops.aten.sub.Tensor(clone_24, amax_3);  clone_24 = amax_3 = None
    exp_3: "f32[8, 8, 785, 16]" = torch.ops.aten.exp.default(sub_12);  sub_12 = None
    sum_4: "f32[8, 8, 1, 16]" = torch.ops.aten.sum.dim_IntList(exp_3, [2], True)
    div_3: "f32[8, 8, 785, 16]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    alias_3: "f32[8, 8, 785, 16]" = torch.ops.aten.alias.default(div_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    permute_40: "f32[8, 8, 16, 785]" = torch.ops.aten.permute.default(div_3, [0, 1, 3, 2]);  div_3 = None
    expand_14: "f32[8, 8, 16, 785]" = torch.ops.aten.expand.default(permute_40, [8, 8, 16, 785]);  permute_40 = None
    view_68: "f32[64, 16, 785]" = torch.ops.aten.view.default(expand_14, [64, 16, 785]);  expand_14 = None
    expand_15: "f32[8, 8, 785, 16]" = torch.ops.aten.expand.default(getitem_38, [8, 8, 785, 16])
    clone_25: "f32[8, 8, 785, 16]" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
    view_69: "f32[64, 785, 16]" = torch.ops.aten.view.default(clone_25, [64, 785, 16]);  clone_25 = None
    bmm_6: "f32[64, 16, 16]" = torch.ops.aten.bmm.default(view_68, view_69)
    view_70: "f32[8, 8, 16, 16]" = torch.ops.aten.view.default(bmm_6, [8, 8, 16, 16]);  bmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    expand_16: "f32[8, 8, 785, 16]" = torch.ops.aten.expand.default(getitem_36, [8, 8, 785, 16])
    clone_26: "f32[8, 8, 785, 16]" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
    view_71: "f32[64, 785, 16]" = torch.ops.aten.view.default(clone_26, [64, 785, 16]);  clone_26 = None
    expand_17: "f32[8, 8, 16, 16]" = torch.ops.aten.expand.default(view_70, [8, 8, 16, 16]);  view_70 = None
    view_72: "f32[64, 16, 16]" = torch.ops.aten.view.default(expand_17, [64, 16, 16]);  expand_17 = None
    bmm_7: "f32[64, 785, 16]" = torch.ops.aten.bmm.default(view_71, view_72)
    view_73: "f32[8, 8, 785, 16]" = torch.ops.aten.view.default(bmm_7, [8, 8, 785, 16]);  bmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:75, code: q_img = q[:, :, 1:, :]  # [B, h, H*W, Ch]
    slice_44: "f32[8, 8, 785, 16]" = torch.ops.aten.slice.Tensor(getitem_36, 0, 0, 9223372036854775807);  getitem_36 = None
    slice_45: "f32[8, 8, 785, 16]" = torch.ops.aten.slice.Tensor(slice_44, 1, 0, 9223372036854775807);  slice_44 = None
    slice_46: "f32[8, 8, 784, 16]" = torch.ops.aten.slice.Tensor(slice_45, 2, 1, 9223372036854775807);  slice_45 = None
    slice_47: "f32[8, 8, 784, 16]" = torch.ops.aten.slice.Tensor(slice_46, 3, 0, 9223372036854775807);  slice_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:76, code: v_img = v[:, :, 1:, :]  # [B, h, H*W, Ch]
    slice_48: "f32[8, 8, 785, 16]" = torch.ops.aten.slice.Tensor(getitem_38, 0, 0, 9223372036854775807);  getitem_38 = None
    slice_49: "f32[8, 8, 785, 16]" = torch.ops.aten.slice.Tensor(slice_48, 1, 0, 9223372036854775807);  slice_48 = None
    slice_50: "f32[8, 8, 784, 16]" = torch.ops.aten.slice.Tensor(slice_49, 2, 1, 9223372036854775807);  slice_49 = None
    slice_51: "f32[8, 8, 784, 16]" = torch.ops.aten.slice.Tensor(slice_50, 3, 0, 9223372036854775807);  slice_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:78, code: v_img = v_img.transpose(-1, -2).reshape(B, num_heads * C, H, W)
    permute_41: "f32[8, 8, 16, 784]" = torch.ops.aten.permute.default(slice_51, [0, 1, 3, 2]);  slice_51 = None
    view_74: "f32[8, 128, 28, 28]" = torch.ops.aten.view.default(permute_41, [8, 128, 28, 28]);  permute_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:79, code: v_img_list = torch.split(v_img, self.channel_splits, dim=1)  # Split according to channels
    split_with_sizes_3 = torch.ops.aten.split_with_sizes.default(view_74, [32, 48, 48], 1);  view_74 = None
    getitem_39: "f32[8, 32, 28, 28]" = split_with_sizes_3[0]
    getitem_40: "f32[8, 48, 28, 28]" = split_with_sizes_3[1]
    getitem_41: "f32[8, 48, 28, 28]" = split_with_sizes_3[2];  split_with_sizes_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    convolution_15: "f32[8, 32, 28, 28]" = torch.ops.aten.convolution.default(getitem_39, primals_75, primals_76, [1, 1], [1, 1], [1, 1], False, [0, 0], 32);  primals_76 = None
    convolution_16: "f32[8, 48, 28, 28]" = torch.ops.aten.convolution.default(getitem_40, primals_77, primals_78, [1, 1], [2, 2], [1, 1], False, [0, 0], 48);  primals_78 = None
    convolution_17: "f32[8, 48, 28, 28]" = torch.ops.aten.convolution.default(getitem_41, primals_79, primals_80, [1, 1], [3, 3], [1, 1], False, [0, 0], 48);  primals_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:83, code: conv_v_img = torch.cat(conv_v_img_list, dim=1)
    cat_9: "f32[8, 128, 28, 28]" = torch.ops.aten.cat.default([convolution_15, convolution_16, convolution_17], 1);  convolution_15 = convolution_16 = convolution_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:84, code: conv_v_img = conv_v_img.reshape(B, num_heads, C, H * W).transpose(-1, -2)
    view_75: "f32[8, 8, 16, 784]" = torch.ops.aten.view.default(cat_9, [8, 8, 16, 784]);  cat_9 = None
    permute_42: "f32[8, 8, 784, 16]" = torch.ops.aten.permute.default(view_75, [0, 1, 3, 2]);  view_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:86, code: EV_hat = q_img * conv_v_img
    mul_33: "f32[8, 8, 784, 16]" = torch.ops.aten.mul.Tensor(slice_47, permute_42)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:87, code: EV_hat = F.pad(EV_hat, (0, 0, 1, 0, 0, 0))  # [B, h, N, Ch].
    constant_pad_nd_3: "f32[8, 8, 785, 16]" = torch.ops.aten.constant_pad_nd.default(mul_33, [0, 0, 1, 0, 0, 0], 0.0);  mul_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:131, code: x = self.scale * factor_att + crpe
    mul_34: "f32[8, 8, 785, 16]" = torch.ops.aten.mul.Tensor(view_73, 0.25);  view_73 = None
    add_34: "f32[8, 8, 785, 16]" = torch.ops.aten.add.Tensor(mul_34, constant_pad_nd_3);  mul_34 = constant_pad_nd_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:132, code: x = x.transpose(1, 2).reshape(B, N, C)  # [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C]
    permute_43: "f32[8, 785, 8, 16]" = torch.ops.aten.permute.default(add_34, [0, 2, 1, 3]);  add_34 = None
    clone_27: "f32[8, 785, 8, 16]" = torch.ops.aten.clone.default(permute_43, memory_format = torch.contiguous_format);  permute_43 = None
    view_76: "f32[8, 785, 128]" = torch.ops.aten.view.default(clone_27, [8, 785, 128]);  clone_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:135, code: x = self.proj(x)
    view_77: "f32[6280, 128]" = torch.ops.aten.view.default(view_76, [6280, 128]);  view_76 = None
    permute_44: "f32[128, 128]" = torch.ops.aten.permute.default(primals_89, [1, 0]);  primals_89 = None
    addmm_13: "f32[6280, 128]" = torch.ops.aten.addmm.default(primals_90, view_77, permute_44);  primals_90 = None
    view_78: "f32[8, 785, 128]" = torch.ops.aten.view.default(addmm_13, [8, 785, 128]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:136, code: x = self.proj_drop(x)
    clone_28: "f32[8, 785, 128]" = torch.ops.aten.clone.default(view_78);  view_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:216, code: x = x + self.drop_path(cur)
    add_35: "f32[8, 785, 128]" = torch.ops.aten.add.Tensor(cat_8, clone_28);  clone_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_9 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
    getitem_42: "f32[8, 785, 1]" = var_mean_9[0]
    getitem_43: "f32[8, 785, 1]" = var_mean_9[1];  var_mean_9 = None
    add_36: "f32[8, 785, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-06);  getitem_42 = None
    rsqrt_9: "f32[8, 785, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
    sub_13: "f32[8, 785, 128]" = torch.ops.aten.sub.Tensor(add_35, getitem_43)
    mul_35: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_9);  sub_13 = None
    mul_36: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(mul_35, primals_17);  mul_35 = None
    add_37: "f32[8, 785, 128]" = torch.ops.aten.add.Tensor(mul_36, primals_18);  mul_36 = primals_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_79: "f32[6280, 128]" = torch.ops.aten.view.default(add_37, [6280, 128]);  add_37 = None
    permute_45: "f32[128, 1024]" = torch.ops.aten.permute.default(primals_91, [1, 0]);  primals_91 = None
    addmm_14: "f32[6280, 1024]" = torch.ops.aten.addmm.default(primals_92, view_79, permute_45);  primals_92 = None
    view_80: "f32[8, 785, 1024]" = torch.ops.aten.view.default(addmm_14, [8, 785, 1024]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_37: "f32[8, 785, 1024]" = torch.ops.aten.mul.Tensor(view_80, 0.5)
    mul_38: "f32[8, 785, 1024]" = torch.ops.aten.mul.Tensor(view_80, 0.7071067811865476)
    erf_3: "f32[8, 785, 1024]" = torch.ops.aten.erf.default(mul_38);  mul_38 = None
    add_38: "f32[8, 785, 1024]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_39: "f32[8, 785, 1024]" = torch.ops.aten.mul.Tensor(mul_37, add_38);  mul_37 = add_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_29: "f32[8, 785, 1024]" = torch.ops.aten.clone.default(mul_39);  mul_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_81: "f32[6280, 1024]" = torch.ops.aten.view.default(clone_29, [6280, 1024]);  clone_29 = None
    permute_46: "f32[1024, 128]" = torch.ops.aten.permute.default(primals_93, [1, 0]);  primals_93 = None
    addmm_15: "f32[6280, 128]" = torch.ops.aten.addmm.default(primals_94, view_81, permute_46);  primals_94 = None
    view_82: "f32[8, 785, 128]" = torch.ops.aten.view.default(addmm_15, [8, 785, 128]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_30: "f32[8, 785, 128]" = torch.ops.aten.clone.default(view_82);  view_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:221, code: x = x + self.drop_path(cur)
    add_39: "f32[8, 785, 128]" = torch.ops.aten.add.Tensor(add_35, clone_30);  clone_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:684, code: return x[:, 1:, :]
    slice_52: "f32[8, 785, 128]" = torch.ops.aten.slice.Tensor(add_39, 0, 0, 9223372036854775807);  add_39 = None
    slice_53: "f32[8, 784, 128]" = torch.ops.aten.slice.Tensor(slice_52, 1, 1, 9223372036854775807);  slice_52 = None
    slice_54: "f32[8, 784, 128]" = torch.ops.aten.slice.Tensor(slice_53, 2, 0, 9223372036854775807);  slice_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:587, code: x2_nocls = remove_cls(x2).reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()
    view_83: "f32[8, 28, 28, 128]" = torch.ops.aten.view.default(slice_54, [8, 28, 28, 128]);  slice_54 = None
    permute_47: "f32[8, 128, 28, 28]" = torch.ops.aten.permute.default(view_83, [0, 3, 1, 2]);  view_83 = None
    clone_31: "f32[8, 128, 28, 28]" = torch.ops.aten.clone.default(permute_47, memory_format = torch.contiguous_format);  permute_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution_18: "f32[8, 320, 14, 14]" = torch.ops.aten.convolution.default(clone_31, primals_95, primals_96, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  primals_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:89, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    view_84: "f32[8, 320, 196]" = torch.ops.aten.view.default(convolution_18, [8, 320, 196]);  convolution_18 = None
    permute_48: "f32[8, 196, 320]" = torch.ops.aten.permute.default(view_84, [0, 2, 1]);  view_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:92, code: x = self.norm(x)
    clone_32: "f32[8, 196, 320]" = torch.ops.aten.clone.default(permute_48, memory_format = torch.contiguous_format)
    var_mean_10 = torch.ops.aten.var_mean.correction(clone_32, [2], correction = 0, keepdim = True)
    getitem_44: "f32[8, 196, 1]" = var_mean_10[0]
    getitem_45: "f32[8, 196, 1]" = var_mean_10[1];  var_mean_10 = None
    add_40: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
    rsqrt_10: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
    sub_14: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_32, getitem_45);  clone_32 = None
    mul_40: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_10);  sub_14 = None
    mul_41: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_40, primals_97);  mul_40 = None
    add_41: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_41, primals_98);  mul_41 = primals_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:677, code: cls_tokens = cls_token.expand(x.shape[0], -1, -1)
    expand_18: "f32[8, 1, 320]" = torch.ops.aten.expand.default(primals_19, [8, -1, -1]);  primals_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:678, code: x = torch.cat((cls_tokens, x), dim=1)
    cat_10: "f32[8, 197, 320]" = torch.ops.aten.cat.default([expand_18, add_41], 1);  expand_18 = add_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    slice_55: "f32[8, 197, 320]" = torch.ops.aten.slice.Tensor(cat_10, 0, 0, 9223372036854775807)
    slice_56: "f32[8, 1, 320]" = torch.ops.aten.slice.Tensor(slice_55, 1, 0, 1);  slice_55 = None
    slice_57: "f32[8, 197, 320]" = torch.ops.aten.slice.Tensor(cat_10, 0, 0, 9223372036854775807);  cat_10 = None
    slice_58: "f32[8, 196, 320]" = torch.ops.aten.slice.Tensor(slice_57, 1, 1, 9223372036854775807);  slice_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:158, code: feat = img_tokens.transpose(1, 2).view(B, C, H, W)
    permute_49: "f32[8, 320, 196]" = torch.ops.aten.permute.default(slice_58, [0, 2, 1]);  slice_58 = None
    view_85: "f32[8, 320, 14, 14]" = torch.ops.aten.view.default(permute_49, [8, 320, 14, 14]);  permute_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:159, code: x = self.proj(feat) + feat
    convolution_19: "f32[8, 320, 14, 14]" = torch.ops.aten.convolution.default(view_85, primals_99, primals_100, [1, 1], [1, 1], [1, 1], False, [0, 0], 320)
    add_42: "f32[8, 320, 14, 14]" = torch.ops.aten.add.Tensor(convolution_19, view_85);  convolution_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:160, code: x = x.flatten(2).transpose(1, 2)
    view_86: "f32[8, 320, 196]" = torch.ops.aten.view.default(add_42, [8, 320, 196]);  add_42 = None
    permute_50: "f32[8, 196, 320]" = torch.ops.aten.permute.default(view_86, [0, 2, 1]);  view_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:163, code: x = torch.cat((cls_token, x), dim=1)
    cat_11: "f32[8, 197, 320]" = torch.ops.aten.cat.default([slice_56, permute_50], 1);  slice_56 = permute_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_11 = torch.ops.aten.var_mean.correction(cat_11, [2], correction = 0, keepdim = True)
    getitem_46: "f32[8, 197, 1]" = var_mean_11[0]
    getitem_47: "f32[8, 197, 1]" = var_mean_11[1];  var_mean_11 = None
    add_43: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-06);  getitem_46 = None
    rsqrt_11: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
    sub_15: "f32[8, 197, 320]" = torch.ops.aten.sub.Tensor(cat_11, getitem_47)
    mul_42: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_11);  sub_15 = None
    mul_43: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(mul_42, primals_20);  mul_42 = None
    add_44: "f32[8, 197, 320]" = torch.ops.aten.add.Tensor(mul_43, primals_21);  mul_43 = primals_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:119, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_87: "f32[1576, 320]" = torch.ops.aten.view.default(add_44, [1576, 320]);  add_44 = None
    permute_51: "f32[320, 960]" = torch.ops.aten.permute.default(primals_101, [1, 0]);  primals_101 = None
    addmm_16: "f32[1576, 960]" = torch.ops.aten.addmm.default(primals_102, view_87, permute_51);  primals_102 = None
    view_88: "f32[8, 197, 960]" = torch.ops.aten.view.default(addmm_16, [8, 197, 960]);  addmm_16 = None
    view_89: "f32[8, 197, 3, 8, 40]" = torch.ops.aten.view.default(view_88, [8, 197, 3, 8, 40]);  view_88 = None
    permute_52: "f32[3, 8, 8, 197, 40]" = torch.ops.aten.permute.default(view_89, [2, 0, 3, 1, 4]);  view_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:120, code: q, k, v = qkv.unbind(0)  # [B, h, N, Ch]
    unbind_4 = torch.ops.aten.unbind.int(permute_52);  permute_52 = None
    getitem_48: "f32[8, 8, 197, 40]" = unbind_4[0]
    getitem_49: "f32[8, 8, 197, 40]" = unbind_4[1]
    getitem_50: "f32[8, 8, 197, 40]" = unbind_4[2];  unbind_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:123, code: k_softmax = k.softmax(dim=2)
    clone_33: "f32[8, 8, 197, 40]" = torch.ops.aten.clone.default(getitem_49, memory_format = torch.contiguous_format);  getitem_49 = None
    amax_4: "f32[8, 8, 1, 40]" = torch.ops.aten.amax.default(clone_33, [2], True)
    sub_16: "f32[8, 8, 197, 40]" = torch.ops.aten.sub.Tensor(clone_33, amax_4);  clone_33 = amax_4 = None
    exp_4: "f32[8, 8, 197, 40]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
    sum_5: "f32[8, 8, 1, 40]" = torch.ops.aten.sum.dim_IntList(exp_4, [2], True)
    div_4: "f32[8, 8, 197, 40]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    alias_4: "f32[8, 8, 197, 40]" = torch.ops.aten.alias.default(div_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    permute_53: "f32[8, 8, 40, 197]" = torch.ops.aten.permute.default(div_4, [0, 1, 3, 2]);  div_4 = None
    expand_19: "f32[8, 8, 40, 197]" = torch.ops.aten.expand.default(permute_53, [8, 8, 40, 197]);  permute_53 = None
    view_90: "f32[64, 40, 197]" = torch.ops.aten.view.default(expand_19, [64, 40, 197]);  expand_19 = None
    expand_20: "f32[8, 8, 197, 40]" = torch.ops.aten.expand.default(getitem_50, [8, 8, 197, 40])
    clone_34: "f32[8, 8, 197, 40]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
    view_91: "f32[64, 197, 40]" = torch.ops.aten.view.default(clone_34, [64, 197, 40]);  clone_34 = None
    bmm_8: "f32[64, 40, 40]" = torch.ops.aten.bmm.default(view_90, view_91)
    view_92: "f32[8, 8, 40, 40]" = torch.ops.aten.view.default(bmm_8, [8, 8, 40, 40]);  bmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    expand_21: "f32[8, 8, 197, 40]" = torch.ops.aten.expand.default(getitem_48, [8, 8, 197, 40])
    clone_35: "f32[8, 8, 197, 40]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
    view_93: "f32[64, 197, 40]" = torch.ops.aten.view.default(clone_35, [64, 197, 40]);  clone_35 = None
    expand_22: "f32[8, 8, 40, 40]" = torch.ops.aten.expand.default(view_92, [8, 8, 40, 40]);  view_92 = None
    view_94: "f32[64, 40, 40]" = torch.ops.aten.view.default(expand_22, [64, 40, 40]);  expand_22 = None
    bmm_9: "f32[64, 197, 40]" = torch.ops.aten.bmm.default(view_93, view_94)
    view_95: "f32[8, 8, 197, 40]" = torch.ops.aten.view.default(bmm_9, [8, 8, 197, 40]);  bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:75, code: q_img = q[:, :, 1:, :]  # [B, h, H*W, Ch]
    slice_59: "f32[8, 8, 197, 40]" = torch.ops.aten.slice.Tensor(getitem_48, 0, 0, 9223372036854775807);  getitem_48 = None
    slice_60: "f32[8, 8, 197, 40]" = torch.ops.aten.slice.Tensor(slice_59, 1, 0, 9223372036854775807);  slice_59 = None
    slice_61: "f32[8, 8, 196, 40]" = torch.ops.aten.slice.Tensor(slice_60, 2, 1, 9223372036854775807);  slice_60 = None
    slice_62: "f32[8, 8, 196, 40]" = torch.ops.aten.slice.Tensor(slice_61, 3, 0, 9223372036854775807);  slice_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:76, code: v_img = v[:, :, 1:, :]  # [B, h, H*W, Ch]
    slice_63: "f32[8, 8, 197, 40]" = torch.ops.aten.slice.Tensor(getitem_50, 0, 0, 9223372036854775807);  getitem_50 = None
    slice_64: "f32[8, 8, 197, 40]" = torch.ops.aten.slice.Tensor(slice_63, 1, 0, 9223372036854775807);  slice_63 = None
    slice_65: "f32[8, 8, 196, 40]" = torch.ops.aten.slice.Tensor(slice_64, 2, 1, 9223372036854775807);  slice_64 = None
    slice_66: "f32[8, 8, 196, 40]" = torch.ops.aten.slice.Tensor(slice_65, 3, 0, 9223372036854775807);  slice_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:78, code: v_img = v_img.transpose(-1, -2).reshape(B, num_heads * C, H, W)
    permute_54: "f32[8, 8, 40, 196]" = torch.ops.aten.permute.default(slice_66, [0, 1, 3, 2]);  slice_66 = None
    view_96: "f32[8, 320, 14, 14]" = torch.ops.aten.view.default(permute_54, [8, 320, 14, 14]);  permute_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:79, code: v_img_list = torch.split(v_img, self.channel_splits, dim=1)  # Split according to channels
    split_with_sizes_4 = torch.ops.aten.split_with_sizes.default(view_96, [80, 120, 120], 1);  view_96 = None
    getitem_51: "f32[8, 80, 14, 14]" = split_with_sizes_4[0]
    getitem_52: "f32[8, 120, 14, 14]" = split_with_sizes_4[1]
    getitem_53: "f32[8, 120, 14, 14]" = split_with_sizes_4[2];  split_with_sizes_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    convolution_20: "f32[8, 80, 14, 14]" = torch.ops.aten.convolution.default(getitem_51, primals_103, primals_104, [1, 1], [1, 1], [1, 1], False, [0, 0], 80)
    convolution_21: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(getitem_52, primals_105, primals_106, [1, 1], [2, 2], [1, 1], False, [0, 0], 120)
    convolution_22: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(getitem_53, primals_107, primals_108, [1, 1], [3, 3], [1, 1], False, [0, 0], 120)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:83, code: conv_v_img = torch.cat(conv_v_img_list, dim=1)
    cat_12: "f32[8, 320, 14, 14]" = torch.ops.aten.cat.default([convolution_20, convolution_21, convolution_22], 1);  convolution_20 = convolution_21 = convolution_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:84, code: conv_v_img = conv_v_img.reshape(B, num_heads, C, H * W).transpose(-1, -2)
    view_97: "f32[8, 8, 40, 196]" = torch.ops.aten.view.default(cat_12, [8, 8, 40, 196]);  cat_12 = None
    permute_55: "f32[8, 8, 196, 40]" = torch.ops.aten.permute.default(view_97, [0, 1, 3, 2]);  view_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:86, code: EV_hat = q_img * conv_v_img
    mul_44: "f32[8, 8, 196, 40]" = torch.ops.aten.mul.Tensor(slice_62, permute_55)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:87, code: EV_hat = F.pad(EV_hat, (0, 0, 1, 0, 0, 0))  # [B, h, N, Ch].
    constant_pad_nd_4: "f32[8, 8, 197, 40]" = torch.ops.aten.constant_pad_nd.default(mul_44, [0, 0, 1, 0, 0, 0], 0.0);  mul_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:131, code: x = self.scale * factor_att + crpe
    mul_45: "f32[8, 8, 197, 40]" = torch.ops.aten.mul.Tensor(view_95, 0.15811388300841897);  view_95 = None
    add_45: "f32[8, 8, 197, 40]" = torch.ops.aten.add.Tensor(mul_45, constant_pad_nd_4);  mul_45 = constant_pad_nd_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:132, code: x = x.transpose(1, 2).reshape(B, N, C)  # [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C]
    permute_56: "f32[8, 197, 8, 40]" = torch.ops.aten.permute.default(add_45, [0, 2, 1, 3]);  add_45 = None
    clone_36: "f32[8, 197, 8, 40]" = torch.ops.aten.clone.default(permute_56, memory_format = torch.contiguous_format);  permute_56 = None
    view_98: "f32[8, 197, 320]" = torch.ops.aten.view.default(clone_36, [8, 197, 320]);  clone_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:135, code: x = self.proj(x)
    view_99: "f32[1576, 320]" = torch.ops.aten.view.default(view_98, [1576, 320]);  view_98 = None
    permute_57: "f32[320, 320]" = torch.ops.aten.permute.default(primals_109, [1, 0]);  primals_109 = None
    addmm_17: "f32[1576, 320]" = torch.ops.aten.addmm.default(primals_110, view_99, permute_57);  primals_110 = None
    view_100: "f32[8, 197, 320]" = torch.ops.aten.view.default(addmm_17, [8, 197, 320]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:136, code: x = self.proj_drop(x)
    clone_37: "f32[8, 197, 320]" = torch.ops.aten.clone.default(view_100);  view_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:216, code: x = x + self.drop_path(cur)
    add_46: "f32[8, 197, 320]" = torch.ops.aten.add.Tensor(cat_11, clone_37);  clone_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_12 = torch.ops.aten.var_mean.correction(add_46, [2], correction = 0, keepdim = True)
    getitem_54: "f32[8, 197, 1]" = var_mean_12[0]
    getitem_55: "f32[8, 197, 1]" = var_mean_12[1];  var_mean_12 = None
    add_47: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-06);  getitem_54 = None
    rsqrt_12: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
    sub_17: "f32[8, 197, 320]" = torch.ops.aten.sub.Tensor(add_46, getitem_55)
    mul_46: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_12);  sub_17 = None
    mul_47: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(mul_46, primals_22);  mul_46 = None
    add_48: "f32[8, 197, 320]" = torch.ops.aten.add.Tensor(mul_47, primals_23);  mul_47 = primals_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_101: "f32[1576, 320]" = torch.ops.aten.view.default(add_48, [1576, 320]);  add_48 = None
    permute_58: "f32[320, 1280]" = torch.ops.aten.permute.default(primals_111, [1, 0]);  primals_111 = None
    addmm_18: "f32[1576, 1280]" = torch.ops.aten.addmm.default(primals_112, view_101, permute_58);  primals_112 = None
    view_102: "f32[8, 197, 1280]" = torch.ops.aten.view.default(addmm_18, [8, 197, 1280]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_48: "f32[8, 197, 1280]" = torch.ops.aten.mul.Tensor(view_102, 0.5)
    mul_49: "f32[8, 197, 1280]" = torch.ops.aten.mul.Tensor(view_102, 0.7071067811865476)
    erf_4: "f32[8, 197, 1280]" = torch.ops.aten.erf.default(mul_49);  mul_49 = None
    add_49: "f32[8, 197, 1280]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_50: "f32[8, 197, 1280]" = torch.ops.aten.mul.Tensor(mul_48, add_49);  mul_48 = add_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_38: "f32[8, 197, 1280]" = torch.ops.aten.clone.default(mul_50);  mul_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_103: "f32[1576, 1280]" = torch.ops.aten.view.default(clone_38, [1576, 1280]);  clone_38 = None
    permute_59: "f32[1280, 320]" = torch.ops.aten.permute.default(primals_113, [1, 0]);  primals_113 = None
    addmm_19: "f32[1576, 320]" = torch.ops.aten.addmm.default(primals_114, view_103, permute_59);  primals_114 = None
    view_104: "f32[8, 197, 320]" = torch.ops.aten.view.default(addmm_19, [8, 197, 320]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_39: "f32[8, 197, 320]" = torch.ops.aten.clone.default(view_104);  view_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:221, code: x = x + self.drop_path(cur)
    add_50: "f32[8, 197, 320]" = torch.ops.aten.add.Tensor(add_46, clone_39);  clone_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    slice_67: "f32[8, 197, 320]" = torch.ops.aten.slice.Tensor(add_50, 0, 0, 9223372036854775807)
    slice_68: "f32[8, 1, 320]" = torch.ops.aten.slice.Tensor(slice_67, 1, 0, 1);  slice_67 = None
    slice_69: "f32[8, 197, 320]" = torch.ops.aten.slice.Tensor(add_50, 0, 0, 9223372036854775807);  add_50 = None
    slice_70: "f32[8, 196, 320]" = torch.ops.aten.slice.Tensor(slice_69, 1, 1, 9223372036854775807);  slice_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:158, code: feat = img_tokens.transpose(1, 2).view(B, C, H, W)
    permute_60: "f32[8, 320, 196]" = torch.ops.aten.permute.default(slice_70, [0, 2, 1]);  slice_70 = None
    view_105: "f32[8, 320, 14, 14]" = torch.ops.aten.view.default(permute_60, [8, 320, 14, 14]);  permute_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:159, code: x = self.proj(feat) + feat
    convolution_23: "f32[8, 320, 14, 14]" = torch.ops.aten.convolution.default(view_105, primals_99, primals_100, [1, 1], [1, 1], [1, 1], False, [0, 0], 320);  primals_100 = None
    add_51: "f32[8, 320, 14, 14]" = torch.ops.aten.add.Tensor(convolution_23, view_105);  convolution_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:160, code: x = x.flatten(2).transpose(1, 2)
    view_106: "f32[8, 320, 196]" = torch.ops.aten.view.default(add_51, [8, 320, 196]);  add_51 = None
    permute_61: "f32[8, 196, 320]" = torch.ops.aten.permute.default(view_106, [0, 2, 1]);  view_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:163, code: x = torch.cat((cls_token, x), dim=1)
    cat_13: "f32[8, 197, 320]" = torch.ops.aten.cat.default([slice_68, permute_61], 1);  slice_68 = permute_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_13 = torch.ops.aten.var_mean.correction(cat_13, [2], correction = 0, keepdim = True)
    getitem_56: "f32[8, 197, 1]" = var_mean_13[0]
    getitem_57: "f32[8, 197, 1]" = var_mean_13[1];  var_mean_13 = None
    add_52: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-06);  getitem_56 = None
    rsqrt_13: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
    sub_18: "f32[8, 197, 320]" = torch.ops.aten.sub.Tensor(cat_13, getitem_57)
    mul_51: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_13);  sub_18 = None
    mul_52: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(mul_51, primals_24);  mul_51 = None
    add_53: "f32[8, 197, 320]" = torch.ops.aten.add.Tensor(mul_52, primals_25);  mul_52 = primals_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:119, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_107: "f32[1576, 320]" = torch.ops.aten.view.default(add_53, [1576, 320]);  add_53 = None
    permute_62: "f32[320, 960]" = torch.ops.aten.permute.default(primals_115, [1, 0]);  primals_115 = None
    addmm_20: "f32[1576, 960]" = torch.ops.aten.addmm.default(primals_116, view_107, permute_62);  primals_116 = None
    view_108: "f32[8, 197, 960]" = torch.ops.aten.view.default(addmm_20, [8, 197, 960]);  addmm_20 = None
    view_109: "f32[8, 197, 3, 8, 40]" = torch.ops.aten.view.default(view_108, [8, 197, 3, 8, 40]);  view_108 = None
    permute_63: "f32[3, 8, 8, 197, 40]" = torch.ops.aten.permute.default(view_109, [2, 0, 3, 1, 4]);  view_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:120, code: q, k, v = qkv.unbind(0)  # [B, h, N, Ch]
    unbind_5 = torch.ops.aten.unbind.int(permute_63);  permute_63 = None
    getitem_58: "f32[8, 8, 197, 40]" = unbind_5[0]
    getitem_59: "f32[8, 8, 197, 40]" = unbind_5[1]
    getitem_60: "f32[8, 8, 197, 40]" = unbind_5[2];  unbind_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:123, code: k_softmax = k.softmax(dim=2)
    clone_40: "f32[8, 8, 197, 40]" = torch.ops.aten.clone.default(getitem_59, memory_format = torch.contiguous_format);  getitem_59 = None
    amax_5: "f32[8, 8, 1, 40]" = torch.ops.aten.amax.default(clone_40, [2], True)
    sub_19: "f32[8, 8, 197, 40]" = torch.ops.aten.sub.Tensor(clone_40, amax_5);  clone_40 = amax_5 = None
    exp_5: "f32[8, 8, 197, 40]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
    sum_6: "f32[8, 8, 1, 40]" = torch.ops.aten.sum.dim_IntList(exp_5, [2], True)
    div_5: "f32[8, 8, 197, 40]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    alias_5: "f32[8, 8, 197, 40]" = torch.ops.aten.alias.default(div_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    permute_64: "f32[8, 8, 40, 197]" = torch.ops.aten.permute.default(div_5, [0, 1, 3, 2]);  div_5 = None
    expand_23: "f32[8, 8, 40, 197]" = torch.ops.aten.expand.default(permute_64, [8, 8, 40, 197]);  permute_64 = None
    view_110: "f32[64, 40, 197]" = torch.ops.aten.view.default(expand_23, [64, 40, 197]);  expand_23 = None
    expand_24: "f32[8, 8, 197, 40]" = torch.ops.aten.expand.default(getitem_60, [8, 8, 197, 40])
    clone_41: "f32[8, 8, 197, 40]" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
    view_111: "f32[64, 197, 40]" = torch.ops.aten.view.default(clone_41, [64, 197, 40]);  clone_41 = None
    bmm_10: "f32[64, 40, 40]" = torch.ops.aten.bmm.default(view_110, view_111)
    view_112: "f32[8, 8, 40, 40]" = torch.ops.aten.view.default(bmm_10, [8, 8, 40, 40]);  bmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    expand_25: "f32[8, 8, 197, 40]" = torch.ops.aten.expand.default(getitem_58, [8, 8, 197, 40])
    clone_42: "f32[8, 8, 197, 40]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
    view_113: "f32[64, 197, 40]" = torch.ops.aten.view.default(clone_42, [64, 197, 40]);  clone_42 = None
    expand_26: "f32[8, 8, 40, 40]" = torch.ops.aten.expand.default(view_112, [8, 8, 40, 40]);  view_112 = None
    view_114: "f32[64, 40, 40]" = torch.ops.aten.view.default(expand_26, [64, 40, 40]);  expand_26 = None
    bmm_11: "f32[64, 197, 40]" = torch.ops.aten.bmm.default(view_113, view_114)
    view_115: "f32[8, 8, 197, 40]" = torch.ops.aten.view.default(bmm_11, [8, 8, 197, 40]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:75, code: q_img = q[:, :, 1:, :]  # [B, h, H*W, Ch]
    slice_71: "f32[8, 8, 197, 40]" = torch.ops.aten.slice.Tensor(getitem_58, 0, 0, 9223372036854775807);  getitem_58 = None
    slice_72: "f32[8, 8, 197, 40]" = torch.ops.aten.slice.Tensor(slice_71, 1, 0, 9223372036854775807);  slice_71 = None
    slice_73: "f32[8, 8, 196, 40]" = torch.ops.aten.slice.Tensor(slice_72, 2, 1, 9223372036854775807);  slice_72 = None
    slice_74: "f32[8, 8, 196, 40]" = torch.ops.aten.slice.Tensor(slice_73, 3, 0, 9223372036854775807);  slice_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:76, code: v_img = v[:, :, 1:, :]  # [B, h, H*W, Ch]
    slice_75: "f32[8, 8, 197, 40]" = torch.ops.aten.slice.Tensor(getitem_60, 0, 0, 9223372036854775807);  getitem_60 = None
    slice_76: "f32[8, 8, 197, 40]" = torch.ops.aten.slice.Tensor(slice_75, 1, 0, 9223372036854775807);  slice_75 = None
    slice_77: "f32[8, 8, 196, 40]" = torch.ops.aten.slice.Tensor(slice_76, 2, 1, 9223372036854775807);  slice_76 = None
    slice_78: "f32[8, 8, 196, 40]" = torch.ops.aten.slice.Tensor(slice_77, 3, 0, 9223372036854775807);  slice_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:78, code: v_img = v_img.transpose(-1, -2).reshape(B, num_heads * C, H, W)
    permute_65: "f32[8, 8, 40, 196]" = torch.ops.aten.permute.default(slice_78, [0, 1, 3, 2]);  slice_78 = None
    view_116: "f32[8, 320, 14, 14]" = torch.ops.aten.view.default(permute_65, [8, 320, 14, 14]);  permute_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:79, code: v_img_list = torch.split(v_img, self.channel_splits, dim=1)  # Split according to channels
    split_with_sizes_5 = torch.ops.aten.split_with_sizes.default(view_116, [80, 120, 120], 1);  view_116 = None
    getitem_61: "f32[8, 80, 14, 14]" = split_with_sizes_5[0]
    getitem_62: "f32[8, 120, 14, 14]" = split_with_sizes_5[1]
    getitem_63: "f32[8, 120, 14, 14]" = split_with_sizes_5[2];  split_with_sizes_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    convolution_24: "f32[8, 80, 14, 14]" = torch.ops.aten.convolution.default(getitem_61, primals_103, primals_104, [1, 1], [1, 1], [1, 1], False, [0, 0], 80);  primals_104 = None
    convolution_25: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(getitem_62, primals_105, primals_106, [1, 1], [2, 2], [1, 1], False, [0, 0], 120);  primals_106 = None
    convolution_26: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(getitem_63, primals_107, primals_108, [1, 1], [3, 3], [1, 1], False, [0, 0], 120);  primals_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:83, code: conv_v_img = torch.cat(conv_v_img_list, dim=1)
    cat_14: "f32[8, 320, 14, 14]" = torch.ops.aten.cat.default([convolution_24, convolution_25, convolution_26], 1);  convolution_24 = convolution_25 = convolution_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:84, code: conv_v_img = conv_v_img.reshape(B, num_heads, C, H * W).transpose(-1, -2)
    view_117: "f32[8, 8, 40, 196]" = torch.ops.aten.view.default(cat_14, [8, 8, 40, 196]);  cat_14 = None
    permute_66: "f32[8, 8, 196, 40]" = torch.ops.aten.permute.default(view_117, [0, 1, 3, 2]);  view_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:86, code: EV_hat = q_img * conv_v_img
    mul_53: "f32[8, 8, 196, 40]" = torch.ops.aten.mul.Tensor(slice_74, permute_66)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:87, code: EV_hat = F.pad(EV_hat, (0, 0, 1, 0, 0, 0))  # [B, h, N, Ch].
    constant_pad_nd_5: "f32[8, 8, 197, 40]" = torch.ops.aten.constant_pad_nd.default(mul_53, [0, 0, 1, 0, 0, 0], 0.0);  mul_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:131, code: x = self.scale * factor_att + crpe
    mul_54: "f32[8, 8, 197, 40]" = torch.ops.aten.mul.Tensor(view_115, 0.15811388300841897);  view_115 = None
    add_54: "f32[8, 8, 197, 40]" = torch.ops.aten.add.Tensor(mul_54, constant_pad_nd_5);  mul_54 = constant_pad_nd_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:132, code: x = x.transpose(1, 2).reshape(B, N, C)  # [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C]
    permute_67: "f32[8, 197, 8, 40]" = torch.ops.aten.permute.default(add_54, [0, 2, 1, 3]);  add_54 = None
    clone_43: "f32[8, 197, 8, 40]" = torch.ops.aten.clone.default(permute_67, memory_format = torch.contiguous_format);  permute_67 = None
    view_118: "f32[8, 197, 320]" = torch.ops.aten.view.default(clone_43, [8, 197, 320]);  clone_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:135, code: x = self.proj(x)
    view_119: "f32[1576, 320]" = torch.ops.aten.view.default(view_118, [1576, 320]);  view_118 = None
    permute_68: "f32[320, 320]" = torch.ops.aten.permute.default(primals_117, [1, 0]);  primals_117 = None
    addmm_21: "f32[1576, 320]" = torch.ops.aten.addmm.default(primals_118, view_119, permute_68);  primals_118 = None
    view_120: "f32[8, 197, 320]" = torch.ops.aten.view.default(addmm_21, [8, 197, 320]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:136, code: x = self.proj_drop(x)
    clone_44: "f32[8, 197, 320]" = torch.ops.aten.clone.default(view_120);  view_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:216, code: x = x + self.drop_path(cur)
    add_55: "f32[8, 197, 320]" = torch.ops.aten.add.Tensor(cat_13, clone_44);  clone_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_14 = torch.ops.aten.var_mean.correction(add_55, [2], correction = 0, keepdim = True)
    getitem_64: "f32[8, 197, 1]" = var_mean_14[0]
    getitem_65: "f32[8, 197, 1]" = var_mean_14[1];  var_mean_14 = None
    add_56: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-06);  getitem_64 = None
    rsqrt_14: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
    sub_20: "f32[8, 197, 320]" = torch.ops.aten.sub.Tensor(add_55, getitem_65)
    mul_55: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_14);  sub_20 = None
    mul_56: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(mul_55, primals_26);  mul_55 = None
    add_57: "f32[8, 197, 320]" = torch.ops.aten.add.Tensor(mul_56, primals_27);  mul_56 = primals_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_121: "f32[1576, 320]" = torch.ops.aten.view.default(add_57, [1576, 320]);  add_57 = None
    permute_69: "f32[320, 1280]" = torch.ops.aten.permute.default(primals_119, [1, 0]);  primals_119 = None
    addmm_22: "f32[1576, 1280]" = torch.ops.aten.addmm.default(primals_120, view_121, permute_69);  primals_120 = None
    view_122: "f32[8, 197, 1280]" = torch.ops.aten.view.default(addmm_22, [8, 197, 1280]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_57: "f32[8, 197, 1280]" = torch.ops.aten.mul.Tensor(view_122, 0.5)
    mul_58: "f32[8, 197, 1280]" = torch.ops.aten.mul.Tensor(view_122, 0.7071067811865476)
    erf_5: "f32[8, 197, 1280]" = torch.ops.aten.erf.default(mul_58);  mul_58 = None
    add_58: "f32[8, 197, 1280]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_59: "f32[8, 197, 1280]" = torch.ops.aten.mul.Tensor(mul_57, add_58);  mul_57 = add_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_45: "f32[8, 197, 1280]" = torch.ops.aten.clone.default(mul_59);  mul_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_123: "f32[1576, 1280]" = torch.ops.aten.view.default(clone_45, [1576, 1280]);  clone_45 = None
    permute_70: "f32[1280, 320]" = torch.ops.aten.permute.default(primals_121, [1, 0]);  primals_121 = None
    addmm_23: "f32[1576, 320]" = torch.ops.aten.addmm.default(primals_122, view_123, permute_70);  primals_122 = None
    view_124: "f32[8, 197, 320]" = torch.ops.aten.view.default(addmm_23, [8, 197, 320]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_46: "f32[8, 197, 320]" = torch.ops.aten.clone.default(view_124);  view_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:221, code: x = x + self.drop_path(cur)
    add_59: "f32[8, 197, 320]" = torch.ops.aten.add.Tensor(add_55, clone_46);  clone_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:684, code: return x[:, 1:, :]
    slice_79: "f32[8, 197, 320]" = torch.ops.aten.slice.Tensor(add_59, 0, 0, 9223372036854775807);  add_59 = None
    slice_80: "f32[8, 196, 320]" = torch.ops.aten.slice.Tensor(slice_79, 1, 1, 9223372036854775807);  slice_79 = None
    slice_81: "f32[8, 196, 320]" = torch.ops.aten.slice.Tensor(slice_80, 2, 0, 9223372036854775807);  slice_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:595, code: x3_nocls = remove_cls(x3).reshape(B, H3, W3, -1).permute(0, 3, 1, 2).contiguous()
    view_125: "f32[8, 14, 14, 320]" = torch.ops.aten.view.default(slice_81, [8, 14, 14, 320]);  slice_81 = None
    permute_71: "f32[8, 320, 14, 14]" = torch.ops.aten.permute.default(view_125, [0, 3, 1, 2]);  view_125 = None
    clone_47: "f32[8, 320, 14, 14]" = torch.ops.aten.clone.default(permute_71, memory_format = torch.contiguous_format);  permute_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution_27: "f32[8, 512, 7, 7]" = torch.ops.aten.convolution.default(clone_47, primals_123, primals_124, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  primals_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:89, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    view_126: "f32[8, 512, 49]" = torch.ops.aten.view.default(convolution_27, [8, 512, 49]);  convolution_27 = None
    permute_72: "f32[8, 49, 512]" = torch.ops.aten.permute.default(view_126, [0, 2, 1]);  view_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:92, code: x = self.norm(x)
    clone_48: "f32[8, 49, 512]" = torch.ops.aten.clone.default(permute_72, memory_format = torch.contiguous_format)
    var_mean_15 = torch.ops.aten.var_mean.correction(clone_48, [2], correction = 0, keepdim = True)
    getitem_66: "f32[8, 49, 1]" = var_mean_15[0]
    getitem_67: "f32[8, 49, 1]" = var_mean_15[1];  var_mean_15 = None
    add_60: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
    rsqrt_15: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
    sub_21: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(clone_48, getitem_67);  clone_48 = None
    mul_60: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_15);  sub_21 = None
    mul_61: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_60, primals_125);  mul_60 = None
    add_61: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(mul_61, primals_126);  mul_61 = primals_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:677, code: cls_tokens = cls_token.expand(x.shape[0], -1, -1)
    expand_27: "f32[8, 1, 512]" = torch.ops.aten.expand.default(primals_28, [8, -1, -1]);  primals_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:678, code: x = torch.cat((cls_tokens, x), dim=1)
    cat_15: "f32[8, 50, 512]" = torch.ops.aten.cat.default([expand_27, add_61], 1);  expand_27 = add_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    slice_82: "f32[8, 50, 512]" = torch.ops.aten.slice.Tensor(cat_15, 0, 0, 9223372036854775807)
    slice_83: "f32[8, 1, 512]" = torch.ops.aten.slice.Tensor(slice_82, 1, 0, 1);  slice_82 = None
    slice_84: "f32[8, 50, 512]" = torch.ops.aten.slice.Tensor(cat_15, 0, 0, 9223372036854775807);  cat_15 = None
    slice_85: "f32[8, 49, 512]" = torch.ops.aten.slice.Tensor(slice_84, 1, 1, 9223372036854775807);  slice_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:158, code: feat = img_tokens.transpose(1, 2).view(B, C, H, W)
    permute_73: "f32[8, 512, 49]" = torch.ops.aten.permute.default(slice_85, [0, 2, 1]);  slice_85 = None
    view_127: "f32[8, 512, 7, 7]" = torch.ops.aten.view.default(permute_73, [8, 512, 7, 7]);  permute_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:159, code: x = self.proj(feat) + feat
    convolution_28: "f32[8, 512, 7, 7]" = torch.ops.aten.convolution.default(view_127, primals_127, primals_128, [1, 1], [1, 1], [1, 1], False, [0, 0], 512)
    add_62: "f32[8, 512, 7, 7]" = torch.ops.aten.add.Tensor(convolution_28, view_127);  convolution_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:160, code: x = x.flatten(2).transpose(1, 2)
    view_128: "f32[8, 512, 49]" = torch.ops.aten.view.default(add_62, [8, 512, 49]);  add_62 = None
    permute_74: "f32[8, 49, 512]" = torch.ops.aten.permute.default(view_128, [0, 2, 1]);  view_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:163, code: x = torch.cat((cls_token, x), dim=1)
    cat_16: "f32[8, 50, 512]" = torch.ops.aten.cat.default([slice_83, permute_74], 1);  slice_83 = permute_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_16 = torch.ops.aten.var_mean.correction(cat_16, [2], correction = 0, keepdim = True)
    getitem_68: "f32[8, 50, 1]" = var_mean_16[0]
    getitem_69: "f32[8, 50, 1]" = var_mean_16[1];  var_mean_16 = None
    add_63: "f32[8, 50, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-06);  getitem_68 = None
    rsqrt_16: "f32[8, 50, 1]" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
    sub_22: "f32[8, 50, 512]" = torch.ops.aten.sub.Tensor(cat_16, getitem_69)
    mul_62: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_16);  sub_22 = None
    mul_63: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(mul_62, primals_29);  mul_62 = None
    add_64: "f32[8, 50, 512]" = torch.ops.aten.add.Tensor(mul_63, primals_30);  mul_63 = primals_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:119, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_129: "f32[400, 512]" = torch.ops.aten.view.default(add_64, [400, 512]);  add_64 = None
    permute_75: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_129, [1, 0]);  primals_129 = None
    addmm_24: "f32[400, 1536]" = torch.ops.aten.addmm.default(primals_130, view_129, permute_75);  primals_130 = None
    view_130: "f32[8, 50, 1536]" = torch.ops.aten.view.default(addmm_24, [8, 50, 1536]);  addmm_24 = None
    view_131: "f32[8, 50, 3, 8, 64]" = torch.ops.aten.view.default(view_130, [8, 50, 3, 8, 64]);  view_130 = None
    permute_76: "f32[3, 8, 8, 50, 64]" = torch.ops.aten.permute.default(view_131, [2, 0, 3, 1, 4]);  view_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:120, code: q, k, v = qkv.unbind(0)  # [B, h, N, Ch]
    unbind_6 = torch.ops.aten.unbind.int(permute_76);  permute_76 = None
    getitem_70: "f32[8, 8, 50, 64]" = unbind_6[0]
    getitem_71: "f32[8, 8, 50, 64]" = unbind_6[1]
    getitem_72: "f32[8, 8, 50, 64]" = unbind_6[2];  unbind_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:123, code: k_softmax = k.softmax(dim=2)
    clone_49: "f32[8, 8, 50, 64]" = torch.ops.aten.clone.default(getitem_71, memory_format = torch.contiguous_format);  getitem_71 = None
    amax_6: "f32[8, 8, 1, 64]" = torch.ops.aten.amax.default(clone_49, [2], True)
    sub_23: "f32[8, 8, 50, 64]" = torch.ops.aten.sub.Tensor(clone_49, amax_6);  clone_49 = amax_6 = None
    exp_6: "f32[8, 8, 50, 64]" = torch.ops.aten.exp.default(sub_23);  sub_23 = None
    sum_7: "f32[8, 8, 1, 64]" = torch.ops.aten.sum.dim_IntList(exp_6, [2], True)
    div_6: "f32[8, 8, 50, 64]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    alias_6: "f32[8, 8, 50, 64]" = torch.ops.aten.alias.default(div_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    permute_77: "f32[8, 8, 64, 50]" = torch.ops.aten.permute.default(div_6, [0, 1, 3, 2]);  div_6 = None
    expand_28: "f32[8, 8, 64, 50]" = torch.ops.aten.expand.default(permute_77, [8, 8, 64, 50]);  permute_77 = None
    view_132: "f32[64, 64, 50]" = torch.ops.aten.view.default(expand_28, [64, 64, 50]);  expand_28 = None
    expand_29: "f32[8, 8, 50, 64]" = torch.ops.aten.expand.default(getitem_72, [8, 8, 50, 64])
    clone_50: "f32[8, 8, 50, 64]" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
    view_133: "f32[64, 50, 64]" = torch.ops.aten.view.default(clone_50, [64, 50, 64]);  clone_50 = None
    bmm_12: "f32[64, 64, 64]" = torch.ops.aten.bmm.default(view_132, view_133)
    view_134: "f32[8, 8, 64, 64]" = torch.ops.aten.view.default(bmm_12, [8, 8, 64, 64]);  bmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    expand_30: "f32[8, 8, 50, 64]" = torch.ops.aten.expand.default(getitem_70, [8, 8, 50, 64])
    clone_51: "f32[8, 8, 50, 64]" = torch.ops.aten.clone.default(expand_30, memory_format = torch.contiguous_format);  expand_30 = None
    view_135: "f32[64, 50, 64]" = torch.ops.aten.view.default(clone_51, [64, 50, 64]);  clone_51 = None
    expand_31: "f32[8, 8, 64, 64]" = torch.ops.aten.expand.default(view_134, [8, 8, 64, 64]);  view_134 = None
    view_136: "f32[64, 64, 64]" = torch.ops.aten.view.default(expand_31, [64, 64, 64]);  expand_31 = None
    bmm_13: "f32[64, 50, 64]" = torch.ops.aten.bmm.default(view_135, view_136)
    view_137: "f32[8, 8, 50, 64]" = torch.ops.aten.view.default(bmm_13, [8, 8, 50, 64]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:75, code: q_img = q[:, :, 1:, :]  # [B, h, H*W, Ch]
    slice_86: "f32[8, 8, 50, 64]" = torch.ops.aten.slice.Tensor(getitem_70, 0, 0, 9223372036854775807);  getitem_70 = None
    slice_87: "f32[8, 8, 50, 64]" = torch.ops.aten.slice.Tensor(slice_86, 1, 0, 9223372036854775807);  slice_86 = None
    slice_88: "f32[8, 8, 49, 64]" = torch.ops.aten.slice.Tensor(slice_87, 2, 1, 9223372036854775807);  slice_87 = None
    slice_89: "f32[8, 8, 49, 64]" = torch.ops.aten.slice.Tensor(slice_88, 3, 0, 9223372036854775807);  slice_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:76, code: v_img = v[:, :, 1:, :]  # [B, h, H*W, Ch]
    slice_90: "f32[8, 8, 50, 64]" = torch.ops.aten.slice.Tensor(getitem_72, 0, 0, 9223372036854775807);  getitem_72 = None
    slice_91: "f32[8, 8, 50, 64]" = torch.ops.aten.slice.Tensor(slice_90, 1, 0, 9223372036854775807);  slice_90 = None
    slice_92: "f32[8, 8, 49, 64]" = torch.ops.aten.slice.Tensor(slice_91, 2, 1, 9223372036854775807);  slice_91 = None
    slice_93: "f32[8, 8, 49, 64]" = torch.ops.aten.slice.Tensor(slice_92, 3, 0, 9223372036854775807);  slice_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:78, code: v_img = v_img.transpose(-1, -2).reshape(B, num_heads * C, H, W)
    permute_78: "f32[8, 8, 64, 49]" = torch.ops.aten.permute.default(slice_93, [0, 1, 3, 2]);  slice_93 = None
    view_138: "f32[8, 512, 7, 7]" = torch.ops.aten.view.default(permute_78, [8, 512, 7, 7]);  permute_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:79, code: v_img_list = torch.split(v_img, self.channel_splits, dim=1)  # Split according to channels
    split_with_sizes_6 = torch.ops.aten.split_with_sizes.default(view_138, [128, 192, 192], 1);  view_138 = None
    getitem_73: "f32[8, 128, 7, 7]" = split_with_sizes_6[0]
    getitem_74: "f32[8, 192, 7, 7]" = split_with_sizes_6[1]
    getitem_75: "f32[8, 192, 7, 7]" = split_with_sizes_6[2];  split_with_sizes_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    convolution_29: "f32[8, 128, 7, 7]" = torch.ops.aten.convolution.default(getitem_73, primals_131, primals_132, [1, 1], [1, 1], [1, 1], False, [0, 0], 128)
    convolution_30: "f32[8, 192, 7, 7]" = torch.ops.aten.convolution.default(getitem_74, primals_133, primals_134, [1, 1], [2, 2], [1, 1], False, [0, 0], 192)
    convolution_31: "f32[8, 192, 7, 7]" = torch.ops.aten.convolution.default(getitem_75, primals_135, primals_136, [1, 1], [3, 3], [1, 1], False, [0, 0], 192)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:83, code: conv_v_img = torch.cat(conv_v_img_list, dim=1)
    cat_17: "f32[8, 512, 7, 7]" = torch.ops.aten.cat.default([convolution_29, convolution_30, convolution_31], 1);  convolution_29 = convolution_30 = convolution_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:84, code: conv_v_img = conv_v_img.reshape(B, num_heads, C, H * W).transpose(-1, -2)
    view_139: "f32[8, 8, 64, 49]" = torch.ops.aten.view.default(cat_17, [8, 8, 64, 49]);  cat_17 = None
    permute_79: "f32[8, 8, 49, 64]" = torch.ops.aten.permute.default(view_139, [0, 1, 3, 2]);  view_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:86, code: EV_hat = q_img * conv_v_img
    mul_64: "f32[8, 8, 49, 64]" = torch.ops.aten.mul.Tensor(slice_89, permute_79)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:87, code: EV_hat = F.pad(EV_hat, (0, 0, 1, 0, 0, 0))  # [B, h, N, Ch].
    constant_pad_nd_6: "f32[8, 8, 50, 64]" = torch.ops.aten.constant_pad_nd.default(mul_64, [0, 0, 1, 0, 0, 0], 0.0);  mul_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:131, code: x = self.scale * factor_att + crpe
    mul_65: "f32[8, 8, 50, 64]" = torch.ops.aten.mul.Tensor(view_137, 0.125);  view_137 = None
    add_65: "f32[8, 8, 50, 64]" = torch.ops.aten.add.Tensor(mul_65, constant_pad_nd_6);  mul_65 = constant_pad_nd_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:132, code: x = x.transpose(1, 2).reshape(B, N, C)  # [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C]
    permute_80: "f32[8, 50, 8, 64]" = torch.ops.aten.permute.default(add_65, [0, 2, 1, 3]);  add_65 = None
    clone_52: "f32[8, 50, 8, 64]" = torch.ops.aten.clone.default(permute_80, memory_format = torch.contiguous_format);  permute_80 = None
    view_140: "f32[8, 50, 512]" = torch.ops.aten.view.default(clone_52, [8, 50, 512]);  clone_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:135, code: x = self.proj(x)
    view_141: "f32[400, 512]" = torch.ops.aten.view.default(view_140, [400, 512]);  view_140 = None
    permute_81: "f32[512, 512]" = torch.ops.aten.permute.default(primals_137, [1, 0]);  primals_137 = None
    addmm_25: "f32[400, 512]" = torch.ops.aten.addmm.default(primals_138, view_141, permute_81);  primals_138 = None
    view_142: "f32[8, 50, 512]" = torch.ops.aten.view.default(addmm_25, [8, 50, 512]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:136, code: x = self.proj_drop(x)
    clone_53: "f32[8, 50, 512]" = torch.ops.aten.clone.default(view_142);  view_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:216, code: x = x + self.drop_path(cur)
    add_66: "f32[8, 50, 512]" = torch.ops.aten.add.Tensor(cat_16, clone_53);  clone_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_17 = torch.ops.aten.var_mean.correction(add_66, [2], correction = 0, keepdim = True)
    getitem_76: "f32[8, 50, 1]" = var_mean_17[0]
    getitem_77: "f32[8, 50, 1]" = var_mean_17[1];  var_mean_17 = None
    add_67: "f32[8, 50, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-06);  getitem_76 = None
    rsqrt_17: "f32[8, 50, 1]" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
    sub_24: "f32[8, 50, 512]" = torch.ops.aten.sub.Tensor(add_66, getitem_77)
    mul_66: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_17);  sub_24 = None
    mul_67: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(mul_66, primals_31);  mul_66 = None
    add_68: "f32[8, 50, 512]" = torch.ops.aten.add.Tensor(mul_67, primals_32);  mul_67 = primals_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_143: "f32[400, 512]" = torch.ops.aten.view.default(add_68, [400, 512]);  add_68 = None
    permute_82: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_139, [1, 0]);  primals_139 = None
    addmm_26: "f32[400, 2048]" = torch.ops.aten.addmm.default(primals_140, view_143, permute_82);  primals_140 = None
    view_144: "f32[8, 50, 2048]" = torch.ops.aten.view.default(addmm_26, [8, 50, 2048]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_68: "f32[8, 50, 2048]" = torch.ops.aten.mul.Tensor(view_144, 0.5)
    mul_69: "f32[8, 50, 2048]" = torch.ops.aten.mul.Tensor(view_144, 0.7071067811865476)
    erf_6: "f32[8, 50, 2048]" = torch.ops.aten.erf.default(mul_69);  mul_69 = None
    add_69: "f32[8, 50, 2048]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_70: "f32[8, 50, 2048]" = torch.ops.aten.mul.Tensor(mul_68, add_69);  mul_68 = add_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_54: "f32[8, 50, 2048]" = torch.ops.aten.clone.default(mul_70);  mul_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_145: "f32[400, 2048]" = torch.ops.aten.view.default(clone_54, [400, 2048]);  clone_54 = None
    permute_83: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_141, [1, 0]);  primals_141 = None
    addmm_27: "f32[400, 512]" = torch.ops.aten.addmm.default(primals_142, view_145, permute_83);  primals_142 = None
    view_146: "f32[8, 50, 512]" = torch.ops.aten.view.default(addmm_27, [8, 50, 512]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_55: "f32[8, 50, 512]" = torch.ops.aten.clone.default(view_146);  view_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:221, code: x = x + self.drop_path(cur)
    add_70: "f32[8, 50, 512]" = torch.ops.aten.add.Tensor(add_66, clone_55);  clone_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    slice_94: "f32[8, 50, 512]" = torch.ops.aten.slice.Tensor(add_70, 0, 0, 9223372036854775807)
    slice_95: "f32[8, 1, 512]" = torch.ops.aten.slice.Tensor(slice_94, 1, 0, 1);  slice_94 = None
    slice_96: "f32[8, 50, 512]" = torch.ops.aten.slice.Tensor(add_70, 0, 0, 9223372036854775807);  add_70 = None
    slice_97: "f32[8, 49, 512]" = torch.ops.aten.slice.Tensor(slice_96, 1, 1, 9223372036854775807);  slice_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:158, code: feat = img_tokens.transpose(1, 2).view(B, C, H, W)
    permute_84: "f32[8, 512, 49]" = torch.ops.aten.permute.default(slice_97, [0, 2, 1]);  slice_97 = None
    view_147: "f32[8, 512, 7, 7]" = torch.ops.aten.view.default(permute_84, [8, 512, 7, 7]);  permute_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:159, code: x = self.proj(feat) + feat
    convolution_32: "f32[8, 512, 7, 7]" = torch.ops.aten.convolution.default(view_147, primals_127, primals_128, [1, 1], [1, 1], [1, 1], False, [0, 0], 512);  primals_128 = None
    add_71: "f32[8, 512, 7, 7]" = torch.ops.aten.add.Tensor(convolution_32, view_147);  convolution_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:160, code: x = x.flatten(2).transpose(1, 2)
    view_148: "f32[8, 512, 49]" = torch.ops.aten.view.default(add_71, [8, 512, 49]);  add_71 = None
    permute_85: "f32[8, 49, 512]" = torch.ops.aten.permute.default(view_148, [0, 2, 1]);  view_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:163, code: x = torch.cat((cls_token, x), dim=1)
    cat_18: "f32[8, 50, 512]" = torch.ops.aten.cat.default([slice_95, permute_85], 1);  slice_95 = permute_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_18 = torch.ops.aten.var_mean.correction(cat_18, [2], correction = 0, keepdim = True)
    getitem_78: "f32[8, 50, 1]" = var_mean_18[0]
    getitem_79: "f32[8, 50, 1]" = var_mean_18[1];  var_mean_18 = None
    add_72: "f32[8, 50, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-06);  getitem_78 = None
    rsqrt_18: "f32[8, 50, 1]" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
    sub_25: "f32[8, 50, 512]" = torch.ops.aten.sub.Tensor(cat_18, getitem_79)
    mul_71: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_18);  sub_25 = None
    mul_72: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(mul_71, primals_33);  mul_71 = None
    add_73: "f32[8, 50, 512]" = torch.ops.aten.add.Tensor(mul_72, primals_34);  mul_72 = primals_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:119, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_149: "f32[400, 512]" = torch.ops.aten.view.default(add_73, [400, 512]);  add_73 = None
    permute_86: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_143, [1, 0]);  primals_143 = None
    addmm_28: "f32[400, 1536]" = torch.ops.aten.addmm.default(primals_144, view_149, permute_86);  primals_144 = None
    view_150: "f32[8, 50, 1536]" = torch.ops.aten.view.default(addmm_28, [8, 50, 1536]);  addmm_28 = None
    view_151: "f32[8, 50, 3, 8, 64]" = torch.ops.aten.view.default(view_150, [8, 50, 3, 8, 64]);  view_150 = None
    permute_87: "f32[3, 8, 8, 50, 64]" = torch.ops.aten.permute.default(view_151, [2, 0, 3, 1, 4]);  view_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:120, code: q, k, v = qkv.unbind(0)  # [B, h, N, Ch]
    unbind_7 = torch.ops.aten.unbind.int(permute_87);  permute_87 = None
    getitem_80: "f32[8, 8, 50, 64]" = unbind_7[0]
    getitem_81: "f32[8, 8, 50, 64]" = unbind_7[1]
    getitem_82: "f32[8, 8, 50, 64]" = unbind_7[2];  unbind_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:123, code: k_softmax = k.softmax(dim=2)
    clone_56: "f32[8, 8, 50, 64]" = torch.ops.aten.clone.default(getitem_81, memory_format = torch.contiguous_format);  getitem_81 = None
    amax_7: "f32[8, 8, 1, 64]" = torch.ops.aten.amax.default(clone_56, [2], True)
    sub_26: "f32[8, 8, 50, 64]" = torch.ops.aten.sub.Tensor(clone_56, amax_7);  clone_56 = amax_7 = None
    exp_7: "f32[8, 8, 50, 64]" = torch.ops.aten.exp.default(sub_26);  sub_26 = None
    sum_8: "f32[8, 8, 1, 64]" = torch.ops.aten.sum.dim_IntList(exp_7, [2], True)
    div_7: "f32[8, 8, 50, 64]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    alias_7: "f32[8, 8, 50, 64]" = torch.ops.aten.alias.default(div_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    permute_88: "f32[8, 8, 64, 50]" = torch.ops.aten.permute.default(div_7, [0, 1, 3, 2]);  div_7 = None
    expand_32: "f32[8, 8, 64, 50]" = torch.ops.aten.expand.default(permute_88, [8, 8, 64, 50]);  permute_88 = None
    view_152: "f32[64, 64, 50]" = torch.ops.aten.view.default(expand_32, [64, 64, 50]);  expand_32 = None
    expand_33: "f32[8, 8, 50, 64]" = torch.ops.aten.expand.default(getitem_82, [8, 8, 50, 64])
    clone_57: "f32[8, 8, 50, 64]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
    view_153: "f32[64, 50, 64]" = torch.ops.aten.view.default(clone_57, [64, 50, 64]);  clone_57 = None
    bmm_14: "f32[64, 64, 64]" = torch.ops.aten.bmm.default(view_152, view_153)
    view_154: "f32[8, 8, 64, 64]" = torch.ops.aten.view.default(bmm_14, [8, 8, 64, 64]);  bmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    expand_34: "f32[8, 8, 50, 64]" = torch.ops.aten.expand.default(getitem_80, [8, 8, 50, 64])
    clone_58: "f32[8, 8, 50, 64]" = torch.ops.aten.clone.default(expand_34, memory_format = torch.contiguous_format);  expand_34 = None
    view_155: "f32[64, 50, 64]" = torch.ops.aten.view.default(clone_58, [64, 50, 64]);  clone_58 = None
    expand_35: "f32[8, 8, 64, 64]" = torch.ops.aten.expand.default(view_154, [8, 8, 64, 64]);  view_154 = None
    view_156: "f32[64, 64, 64]" = torch.ops.aten.view.default(expand_35, [64, 64, 64]);  expand_35 = None
    bmm_15: "f32[64, 50, 64]" = torch.ops.aten.bmm.default(view_155, view_156)
    view_157: "f32[8, 8, 50, 64]" = torch.ops.aten.view.default(bmm_15, [8, 8, 50, 64]);  bmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:75, code: q_img = q[:, :, 1:, :]  # [B, h, H*W, Ch]
    slice_98: "f32[8, 8, 50, 64]" = torch.ops.aten.slice.Tensor(getitem_80, 0, 0, 9223372036854775807);  getitem_80 = None
    slice_99: "f32[8, 8, 50, 64]" = torch.ops.aten.slice.Tensor(slice_98, 1, 0, 9223372036854775807);  slice_98 = None
    slice_100: "f32[8, 8, 49, 64]" = torch.ops.aten.slice.Tensor(slice_99, 2, 1, 9223372036854775807);  slice_99 = None
    slice_101: "f32[8, 8, 49, 64]" = torch.ops.aten.slice.Tensor(slice_100, 3, 0, 9223372036854775807);  slice_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:76, code: v_img = v[:, :, 1:, :]  # [B, h, H*W, Ch]
    slice_102: "f32[8, 8, 50, 64]" = torch.ops.aten.slice.Tensor(getitem_82, 0, 0, 9223372036854775807);  getitem_82 = None
    slice_103: "f32[8, 8, 50, 64]" = torch.ops.aten.slice.Tensor(slice_102, 1, 0, 9223372036854775807);  slice_102 = None
    slice_104: "f32[8, 8, 49, 64]" = torch.ops.aten.slice.Tensor(slice_103, 2, 1, 9223372036854775807);  slice_103 = None
    slice_105: "f32[8, 8, 49, 64]" = torch.ops.aten.slice.Tensor(slice_104, 3, 0, 9223372036854775807);  slice_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:78, code: v_img = v_img.transpose(-1, -2).reshape(B, num_heads * C, H, W)
    permute_89: "f32[8, 8, 64, 49]" = torch.ops.aten.permute.default(slice_105, [0, 1, 3, 2]);  slice_105 = None
    view_158: "f32[8, 512, 7, 7]" = torch.ops.aten.view.default(permute_89, [8, 512, 7, 7]);  permute_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:79, code: v_img_list = torch.split(v_img, self.channel_splits, dim=1)  # Split according to channels
    split_with_sizes_7 = torch.ops.aten.split_with_sizes.default(view_158, [128, 192, 192], 1);  view_158 = None
    getitem_83: "f32[8, 128, 7, 7]" = split_with_sizes_7[0]
    getitem_84: "f32[8, 192, 7, 7]" = split_with_sizes_7[1]
    getitem_85: "f32[8, 192, 7, 7]" = split_with_sizes_7[2];  split_with_sizes_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    convolution_33: "f32[8, 128, 7, 7]" = torch.ops.aten.convolution.default(getitem_83, primals_131, primals_132, [1, 1], [1, 1], [1, 1], False, [0, 0], 128);  primals_132 = None
    convolution_34: "f32[8, 192, 7, 7]" = torch.ops.aten.convolution.default(getitem_84, primals_133, primals_134, [1, 1], [2, 2], [1, 1], False, [0, 0], 192);  primals_134 = None
    convolution_35: "f32[8, 192, 7, 7]" = torch.ops.aten.convolution.default(getitem_85, primals_135, primals_136, [1, 1], [3, 3], [1, 1], False, [0, 0], 192);  primals_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:83, code: conv_v_img = torch.cat(conv_v_img_list, dim=1)
    cat_19: "f32[8, 512, 7, 7]" = torch.ops.aten.cat.default([convolution_33, convolution_34, convolution_35], 1);  convolution_33 = convolution_34 = convolution_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:84, code: conv_v_img = conv_v_img.reshape(B, num_heads, C, H * W).transpose(-1, -2)
    view_159: "f32[8, 8, 64, 49]" = torch.ops.aten.view.default(cat_19, [8, 8, 64, 49]);  cat_19 = None
    permute_90: "f32[8, 8, 49, 64]" = torch.ops.aten.permute.default(view_159, [0, 1, 3, 2]);  view_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:86, code: EV_hat = q_img * conv_v_img
    mul_73: "f32[8, 8, 49, 64]" = torch.ops.aten.mul.Tensor(slice_101, permute_90)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:87, code: EV_hat = F.pad(EV_hat, (0, 0, 1, 0, 0, 0))  # [B, h, N, Ch].
    constant_pad_nd_7: "f32[8, 8, 50, 64]" = torch.ops.aten.constant_pad_nd.default(mul_73, [0, 0, 1, 0, 0, 0], 0.0);  mul_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:131, code: x = self.scale * factor_att + crpe
    mul_74: "f32[8, 8, 50, 64]" = torch.ops.aten.mul.Tensor(view_157, 0.125);  view_157 = None
    add_74: "f32[8, 8, 50, 64]" = torch.ops.aten.add.Tensor(mul_74, constant_pad_nd_7);  mul_74 = constant_pad_nd_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:132, code: x = x.transpose(1, 2).reshape(B, N, C)  # [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C]
    permute_91: "f32[8, 50, 8, 64]" = torch.ops.aten.permute.default(add_74, [0, 2, 1, 3]);  add_74 = None
    clone_59: "f32[8, 50, 8, 64]" = torch.ops.aten.clone.default(permute_91, memory_format = torch.contiguous_format);  permute_91 = None
    view_160: "f32[8, 50, 512]" = torch.ops.aten.view.default(clone_59, [8, 50, 512]);  clone_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:135, code: x = self.proj(x)
    view_161: "f32[400, 512]" = torch.ops.aten.view.default(view_160, [400, 512]);  view_160 = None
    permute_92: "f32[512, 512]" = torch.ops.aten.permute.default(primals_145, [1, 0]);  primals_145 = None
    addmm_29: "f32[400, 512]" = torch.ops.aten.addmm.default(primals_146, view_161, permute_92);  primals_146 = None
    view_162: "f32[8, 50, 512]" = torch.ops.aten.view.default(addmm_29, [8, 50, 512]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:136, code: x = self.proj_drop(x)
    clone_60: "f32[8, 50, 512]" = torch.ops.aten.clone.default(view_162);  view_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:216, code: x = x + self.drop_path(cur)
    add_75: "f32[8, 50, 512]" = torch.ops.aten.add.Tensor(cat_18, clone_60);  clone_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_19 = torch.ops.aten.var_mean.correction(add_75, [2], correction = 0, keepdim = True)
    getitem_86: "f32[8, 50, 1]" = var_mean_19[0]
    getitem_87: "f32[8, 50, 1]" = var_mean_19[1];  var_mean_19 = None
    add_76: "f32[8, 50, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-06);  getitem_86 = None
    rsqrt_19: "f32[8, 50, 1]" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
    sub_27: "f32[8, 50, 512]" = torch.ops.aten.sub.Tensor(add_75, getitem_87)
    mul_75: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_19);  sub_27 = None
    mul_76: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(mul_75, primals_35);  mul_75 = None
    add_77: "f32[8, 50, 512]" = torch.ops.aten.add.Tensor(mul_76, primals_36);  mul_76 = primals_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_163: "f32[400, 512]" = torch.ops.aten.view.default(add_77, [400, 512]);  add_77 = None
    permute_93: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_147, [1, 0]);  primals_147 = None
    addmm_30: "f32[400, 2048]" = torch.ops.aten.addmm.default(primals_148, view_163, permute_93);  primals_148 = None
    view_164: "f32[8, 50, 2048]" = torch.ops.aten.view.default(addmm_30, [8, 50, 2048]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_77: "f32[8, 50, 2048]" = torch.ops.aten.mul.Tensor(view_164, 0.5)
    mul_78: "f32[8, 50, 2048]" = torch.ops.aten.mul.Tensor(view_164, 0.7071067811865476)
    erf_7: "f32[8, 50, 2048]" = torch.ops.aten.erf.default(mul_78);  mul_78 = None
    add_78: "f32[8, 50, 2048]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_79: "f32[8, 50, 2048]" = torch.ops.aten.mul.Tensor(mul_77, add_78);  mul_77 = add_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_61: "f32[8, 50, 2048]" = torch.ops.aten.clone.default(mul_79);  mul_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_165: "f32[400, 2048]" = torch.ops.aten.view.default(clone_61, [400, 2048]);  clone_61 = None
    permute_94: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_149, [1, 0]);  primals_149 = None
    addmm_31: "f32[400, 512]" = torch.ops.aten.addmm.default(primals_150, view_165, permute_94);  primals_150 = None
    view_166: "f32[8, 50, 512]" = torch.ops.aten.view.default(addmm_31, [8, 50, 512]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_62: "f32[8, 50, 512]" = torch.ops.aten.clone.default(view_166);  view_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:221, code: x = x + self.drop_path(cur)
    add_79: "f32[8, 50, 512]" = torch.ops.aten.add.Tensor(add_75, clone_62);  clone_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_20 = torch.ops.aten.var_mean.correction(add_79, [2], correction = 0, keepdim = True)
    getitem_88: "f32[8, 50, 1]" = var_mean_20[0]
    getitem_89: "f32[8, 50, 1]" = var_mean_20[1];  var_mean_20 = None
    add_80: "f32[8, 50, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-06);  getitem_88 = None
    rsqrt_20: "f32[8, 50, 1]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
    sub_28: "f32[8, 50, 512]" = torch.ops.aten.sub.Tensor(add_79, getitem_89)
    mul_80: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_20);  sub_28 = None
    mul_81: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(mul_80, primals_37);  mul_80 = None
    add_81: "f32[8, 50, 512]" = torch.ops.aten.add.Tensor(mul_81, primals_38);  mul_81 = primals_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:660, code: x = x_feat[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x_feat[:, 0]
    slice_109: "f32[8, 50, 512]" = torch.ops.aten.slice.Tensor(add_81, 0, 0, 9223372036854775807);  add_81 = None
    select: "f32[8, 512]" = torch.ops.aten.select.int(slice_109, 1, 0);  slice_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:661, code: x = self.head_drop(x)
    clone_64: "f32[8, 512]" = torch.ops.aten.clone.default(select);  select = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:662, code: return x if pre_logits else self.head(x)
    permute_96: "f32[512, 1000]" = torch.ops.aten.permute.default(primals_151, [1, 0]);  primals_151 = None
    addmm_32: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_152, clone_64, permute_96);  primals_152 = None
    permute_97: "f32[1000, 512]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
    mm: "f32[8, 512]" = torch.ops.aten.mm.default(tangents_1, permute_97);  permute_97 = None
    permute_98: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 512]" = torch.ops.aten.mm.default(permute_98, clone_64);  permute_98 = clone_64 = None
    permute_99: "f32[512, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_9: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_168: "f32[1000]" = torch.ops.aten.view.default(sum_9, [1000]);  sum_9 = None
    permute_100: "f32[1000, 512]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:660, code: x = x_feat[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x_feat[:, 0]
    full: "f32[8, 50, 512]" = torch.ops.aten.full.default([8, 50, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    select_scatter: "f32[8, 50, 512]" = torch.ops.aten.select_scatter.default(full, mm, 1, 0);  full = mm = None
    full_1: "f32[8, 50, 512]" = torch.ops.aten.full.default([8, 50, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter: "f32[8, 50, 512]" = torch.ops.aten.slice_scatter.default(full_1, select_scatter, 0, 0, 9223372036854775807);  full_1 = select_scatter = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_29: "f32[8, 50, 512]" = torch.ops.aten.sub.Tensor(add_79, getitem_89);  add_79 = getitem_89 = None
    mul_82: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_20);  sub_29 = None
    mul_83: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(slice_scatter, primals_37);  primals_37 = None
    mul_84: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(mul_83, 512)
    sum_10: "f32[8, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_83, [2], True)
    mul_85: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(mul_83, mul_82);  mul_83 = None
    sum_11: "f32[8, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_85, [2], True);  mul_85 = None
    mul_86: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(mul_82, sum_11);  sum_11 = None
    sub_30: "f32[8, 50, 512]" = torch.ops.aten.sub.Tensor(mul_84, sum_10);  mul_84 = sum_10 = None
    sub_31: "f32[8, 50, 512]" = torch.ops.aten.sub.Tensor(sub_30, mul_86);  sub_30 = mul_86 = None
    div_8: "f32[8, 50, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 512);  rsqrt_20 = None
    mul_87: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(div_8, sub_31);  div_8 = sub_31 = None
    mul_88: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(slice_scatter, mul_82);  mul_82 = None
    sum_12: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_88, [0, 1]);  mul_88 = None
    sum_13: "f32[512]" = torch.ops.aten.sum.dim_IntList(slice_scatter, [0, 1]);  slice_scatter = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_169: "f32[400, 512]" = torch.ops.aten.view.default(mul_87, [400, 512])
    permute_101: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_94, [1, 0]);  permute_94 = None
    mm_2: "f32[400, 2048]" = torch.ops.aten.mm.default(view_169, permute_101);  permute_101 = None
    permute_102: "f32[512, 400]" = torch.ops.aten.permute.default(view_169, [1, 0])
    mm_3: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_102, view_165);  permute_102 = view_165 = None
    permute_103: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
    sum_14: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_169, [0], True);  view_169 = None
    view_170: "f32[512]" = torch.ops.aten.view.default(sum_14, [512]);  sum_14 = None
    permute_104: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_103, [1, 0]);  permute_103 = None
    view_171: "f32[8, 50, 2048]" = torch.ops.aten.view.default(mm_2, [8, 50, 2048]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_89: "f32[8, 50, 2048]" = torch.ops.aten.mul.Tensor(view_164, 0.7071067811865476)
    erf_8: "f32[8, 50, 2048]" = torch.ops.aten.erf.default(mul_89);  mul_89 = None
    add_82: "f32[8, 50, 2048]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_90: "f32[8, 50, 2048]" = torch.ops.aten.mul.Tensor(add_82, 0.5);  add_82 = None
    mul_91: "f32[8, 50, 2048]" = torch.ops.aten.mul.Tensor(view_164, view_164)
    mul_92: "f32[8, 50, 2048]" = torch.ops.aten.mul.Tensor(mul_91, -0.5);  mul_91 = None
    exp_8: "f32[8, 50, 2048]" = torch.ops.aten.exp.default(mul_92);  mul_92 = None
    mul_93: "f32[8, 50, 2048]" = torch.ops.aten.mul.Tensor(exp_8, 0.3989422804014327);  exp_8 = None
    mul_94: "f32[8, 50, 2048]" = torch.ops.aten.mul.Tensor(view_164, mul_93);  view_164 = mul_93 = None
    add_83: "f32[8, 50, 2048]" = torch.ops.aten.add.Tensor(mul_90, mul_94);  mul_90 = mul_94 = None
    mul_95: "f32[8, 50, 2048]" = torch.ops.aten.mul.Tensor(view_171, add_83);  view_171 = add_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_172: "f32[400, 2048]" = torch.ops.aten.view.default(mul_95, [400, 2048]);  mul_95 = None
    permute_105: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_93, [1, 0]);  permute_93 = None
    mm_4: "f32[400, 512]" = torch.ops.aten.mm.default(view_172, permute_105);  permute_105 = None
    permute_106: "f32[2048, 400]" = torch.ops.aten.permute.default(view_172, [1, 0])
    mm_5: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_106, view_163);  permute_106 = view_163 = None
    permute_107: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
    sum_15: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_172, [0], True);  view_172 = None
    view_173: "f32[2048]" = torch.ops.aten.view.default(sum_15, [2048]);  sum_15 = None
    permute_108: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
    view_174: "f32[8, 50, 512]" = torch.ops.aten.view.default(mm_4, [8, 50, 512]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_32: "f32[8, 50, 512]" = torch.ops.aten.sub.Tensor(add_75, getitem_87);  add_75 = getitem_87 = None
    mul_96: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_19);  sub_32 = None
    mul_97: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(view_174, primals_35);  primals_35 = None
    mul_98: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(mul_97, 512)
    sum_16: "f32[8, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_97, [2], True)
    mul_99: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(mul_97, mul_96);  mul_97 = None
    sum_17: "f32[8, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_99, [2], True);  mul_99 = None
    mul_100: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(mul_96, sum_17);  sum_17 = None
    sub_33: "f32[8, 50, 512]" = torch.ops.aten.sub.Tensor(mul_98, sum_16);  mul_98 = sum_16 = None
    sub_34: "f32[8, 50, 512]" = torch.ops.aten.sub.Tensor(sub_33, mul_100);  sub_33 = mul_100 = None
    div_9: "f32[8, 50, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 512);  rsqrt_19 = None
    mul_101: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(div_9, sub_34);  div_9 = sub_34 = None
    mul_102: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(view_174, mul_96);  mul_96 = None
    sum_18: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_102, [0, 1]);  mul_102 = None
    sum_19: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_174, [0, 1]);  view_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_84: "f32[8, 50, 512]" = torch.ops.aten.add.Tensor(mul_87, mul_101);  mul_87 = mul_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:135, code: x = self.proj(x)
    view_175: "f32[400, 512]" = torch.ops.aten.view.default(add_84, [400, 512])
    permute_109: "f32[512, 512]" = torch.ops.aten.permute.default(permute_92, [1, 0]);  permute_92 = None
    mm_6: "f32[400, 512]" = torch.ops.aten.mm.default(view_175, permute_109);  permute_109 = None
    permute_110: "f32[512, 400]" = torch.ops.aten.permute.default(view_175, [1, 0])
    mm_7: "f32[512, 512]" = torch.ops.aten.mm.default(permute_110, view_161);  permute_110 = view_161 = None
    permute_111: "f32[512, 512]" = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
    sum_20: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_175, [0], True);  view_175 = None
    view_176: "f32[512]" = torch.ops.aten.view.default(sum_20, [512]);  sum_20 = None
    permute_112: "f32[512, 512]" = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
    view_177: "f32[8, 50, 512]" = torch.ops.aten.view.default(mm_6, [8, 50, 512]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:132, code: x = x.transpose(1, 2).reshape(B, N, C)  # [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C]
    view_178: "f32[8, 50, 8, 64]" = torch.ops.aten.view.default(view_177, [8, 50, 8, 64]);  view_177 = None
    permute_113: "f32[8, 8, 50, 64]" = torch.ops.aten.permute.default(view_178, [0, 2, 1, 3]);  view_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:131, code: x = self.scale * factor_att + crpe
    mul_103: "f32[8, 8, 50, 64]" = torch.ops.aten.mul.Tensor(permute_113, 0.125)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:87, code: EV_hat = F.pad(EV_hat, (0, 0, 1, 0, 0, 0))  # [B, h, N, Ch].
    constant_pad_nd_8: "f32[8, 8, 49, 64]" = torch.ops.aten.constant_pad_nd.default(permute_113, [0, 0, -1, 0, 0, 0]);  permute_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:86, code: EV_hat = q_img * conv_v_img
    mul_104: "f32[8, 8, 49, 64]" = torch.ops.aten.mul.Tensor(constant_pad_nd_8, slice_101);  slice_101 = None
    mul_105: "f32[8, 8, 49, 64]" = torch.ops.aten.mul.Tensor(constant_pad_nd_8, permute_90);  constant_pad_nd_8 = permute_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:84, code: conv_v_img = conv_v_img.reshape(B, num_heads, C, H * W).transpose(-1, -2)
    permute_114: "f32[8, 8, 64, 49]" = torch.ops.aten.permute.default(mul_104, [0, 1, 3, 2]);  mul_104 = None
    view_179: "f32[8, 512, 7, 7]" = torch.ops.aten.view.default(permute_114, [8, 512, 7, 7]);  permute_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:83, code: conv_v_img = torch.cat(conv_v_img_list, dim=1)
    slice_110: "f32[8, 128, 7, 7]" = torch.ops.aten.slice.Tensor(view_179, 1, 0, 128)
    slice_111: "f32[8, 192, 7, 7]" = torch.ops.aten.slice.Tensor(view_179, 1, 128, 320)
    slice_112: "f32[8, 192, 7, 7]" = torch.ops.aten.slice.Tensor(view_179, 1, 320, 512);  view_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    convolution_backward = torch.ops.aten.convolution_backward.default(slice_112, getitem_85, primals_135, [192], [1, 1], [3, 3], [1, 1], False, [0, 0], 192, [True, True, True]);  slice_112 = getitem_85 = None
    getitem_90: "f32[8, 192, 7, 7]" = convolution_backward[0]
    getitem_91: "f32[192, 1, 7, 7]" = convolution_backward[1]
    getitem_92: "f32[192]" = convolution_backward[2];  convolution_backward = None
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(slice_111, getitem_84, primals_133, [192], [1, 1], [2, 2], [1, 1], False, [0, 0], 192, [True, True, True]);  slice_111 = getitem_84 = None
    getitem_93: "f32[8, 192, 7, 7]" = convolution_backward_1[0]
    getitem_94: "f32[192, 1, 5, 5]" = convolution_backward_1[1]
    getitem_95: "f32[192]" = convolution_backward_1[2];  convolution_backward_1 = None
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(slice_110, getitem_83, primals_131, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 128, [True, True, True]);  slice_110 = getitem_83 = None
    getitem_96: "f32[8, 128, 7, 7]" = convolution_backward_2[0]
    getitem_97: "f32[128, 1, 3, 3]" = convolution_backward_2[1]
    getitem_98: "f32[128]" = convolution_backward_2[2];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:79, code: v_img_list = torch.split(v_img, self.channel_splits, dim=1)  # Split according to channels
    cat_20: "f32[8, 512, 7, 7]" = torch.ops.aten.cat.default([getitem_96, getitem_93, getitem_90], 1);  getitem_96 = getitem_93 = getitem_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:78, code: v_img = v_img.transpose(-1, -2).reshape(B, num_heads * C, H, W)
    view_180: "f32[8, 8, 64, 49]" = torch.ops.aten.view.default(cat_20, [8, 8, 64, 49]);  cat_20 = None
    permute_115: "f32[8, 8, 49, 64]" = torch.ops.aten.permute.default(view_180, [0, 1, 3, 2]);  view_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:76, code: v_img = v[:, :, 1:, :]  # [B, h, H*W, Ch]
    full_2: "f32[8, 8, 49, 64]" = torch.ops.aten.full.default([8, 8, 49, 64], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_1: "f32[8, 8, 49, 64]" = torch.ops.aten.slice_scatter.default(full_2, permute_115, 3, 0, 9223372036854775807);  full_2 = permute_115 = None
    full_3: "f32[8, 8, 50, 64]" = torch.ops.aten.full.default([8, 8, 50, 64], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_2: "f32[8, 8, 50, 64]" = torch.ops.aten.slice_scatter.default(full_3, slice_scatter_1, 2, 1, 9223372036854775807);  full_3 = slice_scatter_1 = None
    full_4: "f32[8, 8, 50, 64]" = torch.ops.aten.full.default([8, 8, 50, 64], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_3: "f32[8, 8, 50, 64]" = torch.ops.aten.slice_scatter.default(full_4, slice_scatter_2, 1, 0, 9223372036854775807);  full_4 = slice_scatter_2 = None
    full_5: "f32[8, 8, 50, 64]" = torch.ops.aten.full.default([8, 8, 50, 64], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_4: "f32[8, 8, 50, 64]" = torch.ops.aten.slice_scatter.default(full_5, slice_scatter_3, 0, 0, 9223372036854775807);  full_5 = slice_scatter_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:75, code: q_img = q[:, :, 1:, :]  # [B, h, H*W, Ch]
    full_6: "f32[8, 8, 49, 64]" = torch.ops.aten.full.default([8, 8, 49, 64], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_5: "f32[8, 8, 49, 64]" = torch.ops.aten.slice_scatter.default(full_6, mul_105, 3, 0, 9223372036854775807);  full_6 = mul_105 = None
    full_7: "f32[8, 8, 50, 64]" = torch.ops.aten.full.default([8, 8, 50, 64], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_6: "f32[8, 8, 50, 64]" = torch.ops.aten.slice_scatter.default(full_7, slice_scatter_5, 2, 1, 9223372036854775807);  full_7 = slice_scatter_5 = None
    full_8: "f32[8, 8, 50, 64]" = torch.ops.aten.full.default([8, 8, 50, 64], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_7: "f32[8, 8, 50, 64]" = torch.ops.aten.slice_scatter.default(full_8, slice_scatter_6, 1, 0, 9223372036854775807);  full_8 = slice_scatter_6 = None
    full_9: "f32[8, 8, 50, 64]" = torch.ops.aten.full.default([8, 8, 50, 64], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_8: "f32[8, 8, 50, 64]" = torch.ops.aten.slice_scatter.default(full_9, slice_scatter_7, 0, 0, 9223372036854775807);  full_9 = slice_scatter_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    clone_65: "f32[8, 8, 50, 64]" = torch.ops.aten.clone.default(mul_103, memory_format = torch.contiguous_format);  mul_103 = None
    view_181: "f32[64, 50, 64]" = torch.ops.aten.view.default(clone_65, [64, 50, 64]);  clone_65 = None
    permute_116: "f32[64, 64, 50]" = torch.ops.aten.permute.default(view_155, [0, 2, 1]);  view_155 = None
    bmm_16: "f32[64, 64, 64]" = torch.ops.aten.bmm.default(permute_116, view_181);  permute_116 = None
    permute_117: "f32[64, 64, 64]" = torch.ops.aten.permute.default(view_156, [0, 2, 1]);  view_156 = None
    bmm_17: "f32[64, 50, 64]" = torch.ops.aten.bmm.default(view_181, permute_117);  view_181 = permute_117 = None
    view_182: "f32[8, 8, 64, 64]" = torch.ops.aten.view.default(bmm_16, [8, 8, 64, 64]);  bmm_16 = None
    view_183: "f32[8, 8, 50, 64]" = torch.ops.aten.view.default(bmm_17, [8, 8, 50, 64]);  bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    add_85: "f32[8, 8, 50, 64]" = torch.ops.aten.add.Tensor(slice_scatter_8, view_183);  slice_scatter_8 = view_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    view_184: "f32[64, 64, 64]" = torch.ops.aten.view.default(view_182, [64, 64, 64]);  view_182 = None
    permute_118: "f32[64, 50, 64]" = torch.ops.aten.permute.default(view_152, [0, 2, 1]);  view_152 = None
    bmm_18: "f32[64, 50, 64]" = torch.ops.aten.bmm.default(permute_118, view_184);  permute_118 = None
    permute_119: "f32[64, 64, 50]" = torch.ops.aten.permute.default(view_153, [0, 2, 1]);  view_153 = None
    bmm_19: "f32[64, 64, 50]" = torch.ops.aten.bmm.default(view_184, permute_119);  view_184 = permute_119 = None
    view_185: "f32[8, 8, 50, 64]" = torch.ops.aten.view.default(bmm_18, [8, 8, 50, 64]);  bmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    add_86: "f32[8, 8, 50, 64]" = torch.ops.aten.add.Tensor(slice_scatter_4, view_185);  slice_scatter_4 = view_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    view_186: "f32[8, 8, 64, 50]" = torch.ops.aten.view.default(bmm_19, [8, 8, 64, 50]);  bmm_19 = None
    permute_120: "f32[8, 8, 50, 64]" = torch.ops.aten.permute.default(view_186, [0, 1, 3, 2]);  view_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:123, code: k_softmax = k.softmax(dim=2)
    alias_8: "f32[8, 8, 50, 64]" = torch.ops.aten.alias.default(alias_7);  alias_7 = None
    mul_106: "f32[8, 8, 50, 64]" = torch.ops.aten.mul.Tensor(permute_120, alias_8);  permute_120 = None
    sum_21: "f32[8, 8, 1, 64]" = torch.ops.aten.sum.dim_IntList(mul_106, [2], True)
    mul_107: "f32[8, 8, 50, 64]" = torch.ops.aten.mul.Tensor(alias_8, sum_21);  alias_8 = sum_21 = None
    sub_35: "f32[8, 8, 50, 64]" = torch.ops.aten.sub.Tensor(mul_106, mul_107);  mul_106 = mul_107 = None
    clone_66: "f32[8, 8, 50, 64]" = torch.ops.aten.clone.default(sub_35, memory_format = torch.contiguous_format);  sub_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:120, code: q, k, v = qkv.unbind(0)  # [B, h, N, Ch]
    cat_21: "f32[24, 8, 50, 64]" = torch.ops.aten.cat.default([add_85, clone_66, add_86]);  add_85 = clone_66 = add_86 = None
    view_187: "f32[3, 8, 8, 50, 64]" = torch.ops.aten.view.default(cat_21, [3, 8, 8, 50, 64]);  cat_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:119, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_121: "f32[8, 50, 3, 8, 64]" = torch.ops.aten.permute.default(view_187, [1, 3, 0, 2, 4]);  view_187 = None
    clone_67: "f32[8, 50, 3, 8, 64]" = torch.ops.aten.clone.default(permute_121, memory_format = torch.contiguous_format);  permute_121 = None
    view_188: "f32[8, 50, 1536]" = torch.ops.aten.view.default(clone_67, [8, 50, 1536]);  clone_67 = None
    view_189: "f32[400, 1536]" = torch.ops.aten.view.default(view_188, [400, 1536]);  view_188 = None
    permute_122: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
    mm_8: "f32[400, 512]" = torch.ops.aten.mm.default(view_189, permute_122);  permute_122 = None
    permute_123: "f32[1536, 400]" = torch.ops.aten.permute.default(view_189, [1, 0])
    mm_9: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_123, view_149);  permute_123 = view_149 = None
    permute_124: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
    sum_22: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_189, [0], True);  view_189 = None
    view_190: "f32[1536]" = torch.ops.aten.view.default(sum_22, [1536]);  sum_22 = None
    permute_125: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_124, [1, 0]);  permute_124 = None
    view_191: "f32[8, 50, 512]" = torch.ops.aten.view.default(mm_8, [8, 50, 512]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_36: "f32[8, 50, 512]" = torch.ops.aten.sub.Tensor(cat_18, getitem_79);  cat_18 = getitem_79 = None
    mul_108: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_18);  sub_36 = None
    mul_109: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(view_191, primals_33);  primals_33 = None
    mul_110: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(mul_109, 512)
    sum_23: "f32[8, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_109, [2], True)
    mul_111: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(mul_109, mul_108);  mul_109 = None
    sum_24: "f32[8, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_111, [2], True);  mul_111 = None
    mul_112: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(mul_108, sum_24);  sum_24 = None
    sub_37: "f32[8, 50, 512]" = torch.ops.aten.sub.Tensor(mul_110, sum_23);  mul_110 = sum_23 = None
    sub_38: "f32[8, 50, 512]" = torch.ops.aten.sub.Tensor(sub_37, mul_112);  sub_37 = mul_112 = None
    div_10: "f32[8, 50, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 512);  rsqrt_18 = None
    mul_113: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(div_10, sub_38);  div_10 = sub_38 = None
    mul_114: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(view_191, mul_108);  mul_108 = None
    sum_25: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_114, [0, 1]);  mul_114 = None
    sum_26: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_191, [0, 1]);  view_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_87: "f32[8, 50, 512]" = torch.ops.aten.add.Tensor(add_84, mul_113);  add_84 = mul_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:163, code: x = torch.cat((cls_token, x), dim=1)
    slice_113: "f32[8, 1, 512]" = torch.ops.aten.slice.Tensor(add_87, 1, 0, 1)
    slice_114: "f32[8, 49, 512]" = torch.ops.aten.slice.Tensor(add_87, 1, 1, 50);  add_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:160, code: x = x.flatten(2).transpose(1, 2)
    permute_126: "f32[8, 512, 49]" = torch.ops.aten.permute.default(slice_114, [0, 2, 1]);  slice_114 = None
    view_192: "f32[8, 512, 7, 7]" = torch.ops.aten.view.default(permute_126, [8, 512, 7, 7]);  permute_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:159, code: x = self.proj(feat) + feat
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(view_192, view_147, primals_127, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 512, [True, True, True]);  view_147 = None
    getitem_99: "f32[8, 512, 7, 7]" = convolution_backward_3[0]
    getitem_100: "f32[512, 1, 3, 3]" = convolution_backward_3[1]
    getitem_101: "f32[512]" = convolution_backward_3[2];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:159, code: x = self.proj(feat) + feat
    add_88: "f32[8, 512, 7, 7]" = torch.ops.aten.add.Tensor(view_192, getitem_99);  view_192 = getitem_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:158, code: feat = img_tokens.transpose(1, 2).view(B, C, H, W)
    view_193: "f32[8, 512, 49]" = torch.ops.aten.view.default(add_88, [8, 512, 49]);  add_88 = None
    permute_127: "f32[8, 49, 512]" = torch.ops.aten.permute.default(view_193, [0, 2, 1]);  view_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    full_10: "f32[8, 50, 512]" = torch.ops.aten.full.default([8, 50, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_9: "f32[8, 50, 512]" = torch.ops.aten.slice_scatter.default(full_10, permute_127, 1, 1, 9223372036854775807);  full_10 = permute_127 = None
    full_11: "f32[8, 50, 512]" = torch.ops.aten.full.default([8, 50, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_10: "f32[8, 50, 512]" = torch.ops.aten.slice_scatter.default(full_11, slice_scatter_9, 0, 0, 9223372036854775807);  full_11 = slice_scatter_9 = None
    full_12: "f32[8, 50, 512]" = torch.ops.aten.full.default([8, 50, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_11: "f32[8, 50, 512]" = torch.ops.aten.slice_scatter.default(full_12, slice_113, 1, 0, 1);  full_12 = slice_113 = None
    full_13: "f32[8, 50, 512]" = torch.ops.aten.full.default([8, 50, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_12: "f32[8, 50, 512]" = torch.ops.aten.slice_scatter.default(full_13, slice_scatter_11, 0, 0, 9223372036854775807);  full_13 = slice_scatter_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    add_89: "f32[8, 50, 512]" = torch.ops.aten.add.Tensor(slice_scatter_10, slice_scatter_12);  slice_scatter_10 = slice_scatter_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_194: "f32[400, 512]" = torch.ops.aten.view.default(add_89, [400, 512])
    permute_128: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_83, [1, 0]);  permute_83 = None
    mm_10: "f32[400, 2048]" = torch.ops.aten.mm.default(view_194, permute_128);  permute_128 = None
    permute_129: "f32[512, 400]" = torch.ops.aten.permute.default(view_194, [1, 0])
    mm_11: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_129, view_145);  permute_129 = view_145 = None
    permute_130: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_11, [1, 0]);  mm_11 = None
    sum_27: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_194, [0], True);  view_194 = None
    view_195: "f32[512]" = torch.ops.aten.view.default(sum_27, [512]);  sum_27 = None
    permute_131: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
    view_196: "f32[8, 50, 2048]" = torch.ops.aten.view.default(mm_10, [8, 50, 2048]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_115: "f32[8, 50, 2048]" = torch.ops.aten.mul.Tensor(view_144, 0.7071067811865476)
    erf_9: "f32[8, 50, 2048]" = torch.ops.aten.erf.default(mul_115);  mul_115 = None
    add_90: "f32[8, 50, 2048]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_116: "f32[8, 50, 2048]" = torch.ops.aten.mul.Tensor(add_90, 0.5);  add_90 = None
    mul_117: "f32[8, 50, 2048]" = torch.ops.aten.mul.Tensor(view_144, view_144)
    mul_118: "f32[8, 50, 2048]" = torch.ops.aten.mul.Tensor(mul_117, -0.5);  mul_117 = None
    exp_9: "f32[8, 50, 2048]" = torch.ops.aten.exp.default(mul_118);  mul_118 = None
    mul_119: "f32[8, 50, 2048]" = torch.ops.aten.mul.Tensor(exp_9, 0.3989422804014327);  exp_9 = None
    mul_120: "f32[8, 50, 2048]" = torch.ops.aten.mul.Tensor(view_144, mul_119);  view_144 = mul_119 = None
    add_91: "f32[8, 50, 2048]" = torch.ops.aten.add.Tensor(mul_116, mul_120);  mul_116 = mul_120 = None
    mul_121: "f32[8, 50, 2048]" = torch.ops.aten.mul.Tensor(view_196, add_91);  view_196 = add_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_197: "f32[400, 2048]" = torch.ops.aten.view.default(mul_121, [400, 2048]);  mul_121 = None
    permute_132: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_82, [1, 0]);  permute_82 = None
    mm_12: "f32[400, 512]" = torch.ops.aten.mm.default(view_197, permute_132);  permute_132 = None
    permute_133: "f32[2048, 400]" = torch.ops.aten.permute.default(view_197, [1, 0])
    mm_13: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_133, view_143);  permute_133 = view_143 = None
    permute_134: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_13, [1, 0]);  mm_13 = None
    sum_28: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_197, [0], True);  view_197 = None
    view_198: "f32[2048]" = torch.ops.aten.view.default(sum_28, [2048]);  sum_28 = None
    permute_135: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_134, [1, 0]);  permute_134 = None
    view_199: "f32[8, 50, 512]" = torch.ops.aten.view.default(mm_12, [8, 50, 512]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_39: "f32[8, 50, 512]" = torch.ops.aten.sub.Tensor(add_66, getitem_77);  add_66 = getitem_77 = None
    mul_122: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_17);  sub_39 = None
    mul_123: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(view_199, primals_31);  primals_31 = None
    mul_124: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(mul_123, 512)
    sum_29: "f32[8, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_123, [2], True)
    mul_125: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(mul_123, mul_122);  mul_123 = None
    sum_30: "f32[8, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_125, [2], True);  mul_125 = None
    mul_126: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(mul_122, sum_30);  sum_30 = None
    sub_40: "f32[8, 50, 512]" = torch.ops.aten.sub.Tensor(mul_124, sum_29);  mul_124 = sum_29 = None
    sub_41: "f32[8, 50, 512]" = torch.ops.aten.sub.Tensor(sub_40, mul_126);  sub_40 = mul_126 = None
    div_11: "f32[8, 50, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 512);  rsqrt_17 = None
    mul_127: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(div_11, sub_41);  div_11 = sub_41 = None
    mul_128: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(view_199, mul_122);  mul_122 = None
    sum_31: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_128, [0, 1]);  mul_128 = None
    sum_32: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_199, [0, 1]);  view_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_92: "f32[8, 50, 512]" = torch.ops.aten.add.Tensor(add_89, mul_127);  add_89 = mul_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:135, code: x = self.proj(x)
    view_200: "f32[400, 512]" = torch.ops.aten.view.default(add_92, [400, 512])
    permute_136: "f32[512, 512]" = torch.ops.aten.permute.default(permute_81, [1, 0]);  permute_81 = None
    mm_14: "f32[400, 512]" = torch.ops.aten.mm.default(view_200, permute_136);  permute_136 = None
    permute_137: "f32[512, 400]" = torch.ops.aten.permute.default(view_200, [1, 0])
    mm_15: "f32[512, 512]" = torch.ops.aten.mm.default(permute_137, view_141);  permute_137 = view_141 = None
    permute_138: "f32[512, 512]" = torch.ops.aten.permute.default(mm_15, [1, 0]);  mm_15 = None
    sum_33: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_200, [0], True);  view_200 = None
    view_201: "f32[512]" = torch.ops.aten.view.default(sum_33, [512]);  sum_33 = None
    permute_139: "f32[512, 512]" = torch.ops.aten.permute.default(permute_138, [1, 0]);  permute_138 = None
    view_202: "f32[8, 50, 512]" = torch.ops.aten.view.default(mm_14, [8, 50, 512]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:132, code: x = x.transpose(1, 2).reshape(B, N, C)  # [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C]
    view_203: "f32[8, 50, 8, 64]" = torch.ops.aten.view.default(view_202, [8, 50, 8, 64]);  view_202 = None
    permute_140: "f32[8, 8, 50, 64]" = torch.ops.aten.permute.default(view_203, [0, 2, 1, 3]);  view_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:131, code: x = self.scale * factor_att + crpe
    mul_129: "f32[8, 8, 50, 64]" = torch.ops.aten.mul.Tensor(permute_140, 0.125)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:87, code: EV_hat = F.pad(EV_hat, (0, 0, 1, 0, 0, 0))  # [B, h, N, Ch].
    constant_pad_nd_9: "f32[8, 8, 49, 64]" = torch.ops.aten.constant_pad_nd.default(permute_140, [0, 0, -1, 0, 0, 0]);  permute_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:86, code: EV_hat = q_img * conv_v_img
    mul_130: "f32[8, 8, 49, 64]" = torch.ops.aten.mul.Tensor(constant_pad_nd_9, slice_89);  slice_89 = None
    mul_131: "f32[8, 8, 49, 64]" = torch.ops.aten.mul.Tensor(constant_pad_nd_9, permute_79);  constant_pad_nd_9 = permute_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:84, code: conv_v_img = conv_v_img.reshape(B, num_heads, C, H * W).transpose(-1, -2)
    permute_141: "f32[8, 8, 64, 49]" = torch.ops.aten.permute.default(mul_130, [0, 1, 3, 2]);  mul_130 = None
    view_204: "f32[8, 512, 7, 7]" = torch.ops.aten.view.default(permute_141, [8, 512, 7, 7]);  permute_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:83, code: conv_v_img = torch.cat(conv_v_img_list, dim=1)
    slice_115: "f32[8, 128, 7, 7]" = torch.ops.aten.slice.Tensor(view_204, 1, 0, 128)
    slice_116: "f32[8, 192, 7, 7]" = torch.ops.aten.slice.Tensor(view_204, 1, 128, 320)
    slice_117: "f32[8, 192, 7, 7]" = torch.ops.aten.slice.Tensor(view_204, 1, 320, 512);  view_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(slice_117, getitem_75, primals_135, [192], [1, 1], [3, 3], [1, 1], False, [0, 0], 192, [True, True, True]);  slice_117 = getitem_75 = primals_135 = None
    getitem_102: "f32[8, 192, 7, 7]" = convolution_backward_4[0]
    getitem_103: "f32[192, 1, 7, 7]" = convolution_backward_4[1]
    getitem_104: "f32[192]" = convolution_backward_4[2];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    add_93: "f32[192, 1, 7, 7]" = torch.ops.aten.add.Tensor(getitem_91, getitem_103);  getitem_91 = getitem_103 = None
    add_94: "f32[192]" = torch.ops.aten.add.Tensor(getitem_92, getitem_104);  getitem_92 = getitem_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(slice_116, getitem_74, primals_133, [192], [1, 1], [2, 2], [1, 1], False, [0, 0], 192, [True, True, True]);  slice_116 = getitem_74 = primals_133 = None
    getitem_105: "f32[8, 192, 7, 7]" = convolution_backward_5[0]
    getitem_106: "f32[192, 1, 5, 5]" = convolution_backward_5[1]
    getitem_107: "f32[192]" = convolution_backward_5[2];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    add_95: "f32[192, 1, 5, 5]" = torch.ops.aten.add.Tensor(getitem_94, getitem_106);  getitem_94 = getitem_106 = None
    add_96: "f32[192]" = torch.ops.aten.add.Tensor(getitem_95, getitem_107);  getitem_95 = getitem_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(slice_115, getitem_73, primals_131, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 128, [True, True, True]);  slice_115 = getitem_73 = primals_131 = None
    getitem_108: "f32[8, 128, 7, 7]" = convolution_backward_6[0]
    getitem_109: "f32[128, 1, 3, 3]" = convolution_backward_6[1]
    getitem_110: "f32[128]" = convolution_backward_6[2];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    add_97: "f32[128, 1, 3, 3]" = torch.ops.aten.add.Tensor(getitem_97, getitem_109);  getitem_97 = getitem_109 = None
    add_98: "f32[128]" = torch.ops.aten.add.Tensor(getitem_98, getitem_110);  getitem_98 = getitem_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:79, code: v_img_list = torch.split(v_img, self.channel_splits, dim=1)  # Split according to channels
    cat_22: "f32[8, 512, 7, 7]" = torch.ops.aten.cat.default([getitem_108, getitem_105, getitem_102], 1);  getitem_108 = getitem_105 = getitem_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:78, code: v_img = v_img.transpose(-1, -2).reshape(B, num_heads * C, H, W)
    view_205: "f32[8, 8, 64, 49]" = torch.ops.aten.view.default(cat_22, [8, 8, 64, 49]);  cat_22 = None
    permute_142: "f32[8, 8, 49, 64]" = torch.ops.aten.permute.default(view_205, [0, 1, 3, 2]);  view_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:76, code: v_img = v[:, :, 1:, :]  # [B, h, H*W, Ch]
    full_14: "f32[8, 8, 49, 64]" = torch.ops.aten.full.default([8, 8, 49, 64], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_13: "f32[8, 8, 49, 64]" = torch.ops.aten.slice_scatter.default(full_14, permute_142, 3, 0, 9223372036854775807);  full_14 = permute_142 = None
    full_15: "f32[8, 8, 50, 64]" = torch.ops.aten.full.default([8, 8, 50, 64], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_14: "f32[8, 8, 50, 64]" = torch.ops.aten.slice_scatter.default(full_15, slice_scatter_13, 2, 1, 9223372036854775807);  full_15 = slice_scatter_13 = None
    full_16: "f32[8, 8, 50, 64]" = torch.ops.aten.full.default([8, 8, 50, 64], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_15: "f32[8, 8, 50, 64]" = torch.ops.aten.slice_scatter.default(full_16, slice_scatter_14, 1, 0, 9223372036854775807);  full_16 = slice_scatter_14 = None
    full_17: "f32[8, 8, 50, 64]" = torch.ops.aten.full.default([8, 8, 50, 64], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_16: "f32[8, 8, 50, 64]" = torch.ops.aten.slice_scatter.default(full_17, slice_scatter_15, 0, 0, 9223372036854775807);  full_17 = slice_scatter_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:75, code: q_img = q[:, :, 1:, :]  # [B, h, H*W, Ch]
    full_18: "f32[8, 8, 49, 64]" = torch.ops.aten.full.default([8, 8, 49, 64], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_17: "f32[8, 8, 49, 64]" = torch.ops.aten.slice_scatter.default(full_18, mul_131, 3, 0, 9223372036854775807);  full_18 = mul_131 = None
    full_19: "f32[8, 8, 50, 64]" = torch.ops.aten.full.default([8, 8, 50, 64], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_18: "f32[8, 8, 50, 64]" = torch.ops.aten.slice_scatter.default(full_19, slice_scatter_17, 2, 1, 9223372036854775807);  full_19 = slice_scatter_17 = None
    full_20: "f32[8, 8, 50, 64]" = torch.ops.aten.full.default([8, 8, 50, 64], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_19: "f32[8, 8, 50, 64]" = torch.ops.aten.slice_scatter.default(full_20, slice_scatter_18, 1, 0, 9223372036854775807);  full_20 = slice_scatter_18 = None
    full_21: "f32[8, 8, 50, 64]" = torch.ops.aten.full.default([8, 8, 50, 64], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_20: "f32[8, 8, 50, 64]" = torch.ops.aten.slice_scatter.default(full_21, slice_scatter_19, 0, 0, 9223372036854775807);  full_21 = slice_scatter_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    clone_68: "f32[8, 8, 50, 64]" = torch.ops.aten.clone.default(mul_129, memory_format = torch.contiguous_format);  mul_129 = None
    view_206: "f32[64, 50, 64]" = torch.ops.aten.view.default(clone_68, [64, 50, 64]);  clone_68 = None
    permute_143: "f32[64, 64, 50]" = torch.ops.aten.permute.default(view_135, [0, 2, 1]);  view_135 = None
    bmm_20: "f32[64, 64, 64]" = torch.ops.aten.bmm.default(permute_143, view_206);  permute_143 = None
    permute_144: "f32[64, 64, 64]" = torch.ops.aten.permute.default(view_136, [0, 2, 1]);  view_136 = None
    bmm_21: "f32[64, 50, 64]" = torch.ops.aten.bmm.default(view_206, permute_144);  view_206 = permute_144 = None
    view_207: "f32[8, 8, 64, 64]" = torch.ops.aten.view.default(bmm_20, [8, 8, 64, 64]);  bmm_20 = None
    view_208: "f32[8, 8, 50, 64]" = torch.ops.aten.view.default(bmm_21, [8, 8, 50, 64]);  bmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    add_99: "f32[8, 8, 50, 64]" = torch.ops.aten.add.Tensor(slice_scatter_20, view_208);  slice_scatter_20 = view_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    view_209: "f32[64, 64, 64]" = torch.ops.aten.view.default(view_207, [64, 64, 64]);  view_207 = None
    permute_145: "f32[64, 50, 64]" = torch.ops.aten.permute.default(view_132, [0, 2, 1]);  view_132 = None
    bmm_22: "f32[64, 50, 64]" = torch.ops.aten.bmm.default(permute_145, view_209);  permute_145 = None
    permute_146: "f32[64, 64, 50]" = torch.ops.aten.permute.default(view_133, [0, 2, 1]);  view_133 = None
    bmm_23: "f32[64, 64, 50]" = torch.ops.aten.bmm.default(view_209, permute_146);  view_209 = permute_146 = None
    view_210: "f32[8, 8, 50, 64]" = torch.ops.aten.view.default(bmm_22, [8, 8, 50, 64]);  bmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    add_100: "f32[8, 8, 50, 64]" = torch.ops.aten.add.Tensor(slice_scatter_16, view_210);  slice_scatter_16 = view_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    view_211: "f32[8, 8, 64, 50]" = torch.ops.aten.view.default(bmm_23, [8, 8, 64, 50]);  bmm_23 = None
    permute_147: "f32[8, 8, 50, 64]" = torch.ops.aten.permute.default(view_211, [0, 1, 3, 2]);  view_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:123, code: k_softmax = k.softmax(dim=2)
    alias_9: "f32[8, 8, 50, 64]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    mul_132: "f32[8, 8, 50, 64]" = torch.ops.aten.mul.Tensor(permute_147, alias_9);  permute_147 = None
    sum_34: "f32[8, 8, 1, 64]" = torch.ops.aten.sum.dim_IntList(mul_132, [2], True)
    mul_133: "f32[8, 8, 50, 64]" = torch.ops.aten.mul.Tensor(alias_9, sum_34);  alias_9 = sum_34 = None
    sub_42: "f32[8, 8, 50, 64]" = torch.ops.aten.sub.Tensor(mul_132, mul_133);  mul_132 = mul_133 = None
    clone_69: "f32[8, 8, 50, 64]" = torch.ops.aten.clone.default(sub_42, memory_format = torch.contiguous_format);  sub_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:120, code: q, k, v = qkv.unbind(0)  # [B, h, N, Ch]
    cat_23: "f32[24, 8, 50, 64]" = torch.ops.aten.cat.default([add_99, clone_69, add_100]);  add_99 = clone_69 = add_100 = None
    view_212: "f32[3, 8, 8, 50, 64]" = torch.ops.aten.view.default(cat_23, [3, 8, 8, 50, 64]);  cat_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:119, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_148: "f32[8, 50, 3, 8, 64]" = torch.ops.aten.permute.default(view_212, [1, 3, 0, 2, 4]);  view_212 = None
    clone_70: "f32[8, 50, 3, 8, 64]" = torch.ops.aten.clone.default(permute_148, memory_format = torch.contiguous_format);  permute_148 = None
    view_213: "f32[8, 50, 1536]" = torch.ops.aten.view.default(clone_70, [8, 50, 1536]);  clone_70 = None
    view_214: "f32[400, 1536]" = torch.ops.aten.view.default(view_213, [400, 1536]);  view_213 = None
    permute_149: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
    mm_16: "f32[400, 512]" = torch.ops.aten.mm.default(view_214, permute_149);  permute_149 = None
    permute_150: "f32[1536, 400]" = torch.ops.aten.permute.default(view_214, [1, 0])
    mm_17: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_150, view_129);  permute_150 = view_129 = None
    permute_151: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_17, [1, 0]);  mm_17 = None
    sum_35: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_214, [0], True);  view_214 = None
    view_215: "f32[1536]" = torch.ops.aten.view.default(sum_35, [1536]);  sum_35 = None
    permute_152: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_151, [1, 0]);  permute_151 = None
    view_216: "f32[8, 50, 512]" = torch.ops.aten.view.default(mm_16, [8, 50, 512]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_43: "f32[8, 50, 512]" = torch.ops.aten.sub.Tensor(cat_16, getitem_69);  cat_16 = getitem_69 = None
    mul_134: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_16);  sub_43 = None
    mul_135: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(view_216, primals_29);  primals_29 = None
    mul_136: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(mul_135, 512)
    sum_36: "f32[8, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_135, [2], True)
    mul_137: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(mul_135, mul_134);  mul_135 = None
    sum_37: "f32[8, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_137, [2], True);  mul_137 = None
    mul_138: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(mul_134, sum_37);  sum_37 = None
    sub_44: "f32[8, 50, 512]" = torch.ops.aten.sub.Tensor(mul_136, sum_36);  mul_136 = sum_36 = None
    sub_45: "f32[8, 50, 512]" = torch.ops.aten.sub.Tensor(sub_44, mul_138);  sub_44 = mul_138 = None
    div_12: "f32[8, 50, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 512);  rsqrt_16 = None
    mul_139: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(div_12, sub_45);  div_12 = sub_45 = None
    mul_140: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(view_216, mul_134);  mul_134 = None
    sum_38: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_140, [0, 1]);  mul_140 = None
    sum_39: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_216, [0, 1]);  view_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_101: "f32[8, 50, 512]" = torch.ops.aten.add.Tensor(add_92, mul_139);  add_92 = mul_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:163, code: x = torch.cat((cls_token, x), dim=1)
    slice_118: "f32[8, 1, 512]" = torch.ops.aten.slice.Tensor(add_101, 1, 0, 1)
    slice_119: "f32[8, 49, 512]" = torch.ops.aten.slice.Tensor(add_101, 1, 1, 50);  add_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:160, code: x = x.flatten(2).transpose(1, 2)
    permute_153: "f32[8, 512, 49]" = torch.ops.aten.permute.default(slice_119, [0, 2, 1]);  slice_119 = None
    view_217: "f32[8, 512, 7, 7]" = torch.ops.aten.view.default(permute_153, [8, 512, 7, 7]);  permute_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:159, code: x = self.proj(feat) + feat
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(view_217, view_127, primals_127, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 512, [True, True, True]);  view_127 = primals_127 = None
    getitem_111: "f32[8, 512, 7, 7]" = convolution_backward_7[0]
    getitem_112: "f32[512, 1, 3, 3]" = convolution_backward_7[1]
    getitem_113: "f32[512]" = convolution_backward_7[2];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:159, code: x = self.proj(feat) + feat
    add_102: "f32[8, 512, 7, 7]" = torch.ops.aten.add.Tensor(view_217, getitem_111);  view_217 = getitem_111 = None
    add_103: "f32[512, 1, 3, 3]" = torch.ops.aten.add.Tensor(getitem_100, getitem_112);  getitem_100 = getitem_112 = None
    add_104: "f32[512]" = torch.ops.aten.add.Tensor(getitem_101, getitem_113);  getitem_101 = getitem_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:158, code: feat = img_tokens.transpose(1, 2).view(B, C, H, W)
    view_218: "f32[8, 512, 49]" = torch.ops.aten.view.default(add_102, [8, 512, 49]);  add_102 = None
    permute_154: "f32[8, 49, 512]" = torch.ops.aten.permute.default(view_218, [0, 2, 1]);  view_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    full_22: "f32[8, 50, 512]" = torch.ops.aten.full.default([8, 50, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_21: "f32[8, 50, 512]" = torch.ops.aten.slice_scatter.default(full_22, permute_154, 1, 1, 9223372036854775807);  full_22 = permute_154 = None
    full_23: "f32[8, 50, 512]" = torch.ops.aten.full.default([8, 50, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_22: "f32[8, 50, 512]" = torch.ops.aten.slice_scatter.default(full_23, slice_scatter_21, 0, 0, 9223372036854775807);  full_23 = slice_scatter_21 = None
    full_24: "f32[8, 50, 512]" = torch.ops.aten.full.default([8, 50, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_23: "f32[8, 50, 512]" = torch.ops.aten.slice_scatter.default(full_24, slice_118, 1, 0, 1);  full_24 = slice_118 = None
    full_25: "f32[8, 50, 512]" = torch.ops.aten.full.default([8, 50, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_24: "f32[8, 50, 512]" = torch.ops.aten.slice_scatter.default(full_25, slice_scatter_23, 0, 0, 9223372036854775807);  full_25 = slice_scatter_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    add_105: "f32[8, 50, 512]" = torch.ops.aten.add.Tensor(slice_scatter_22, slice_scatter_24);  slice_scatter_22 = slice_scatter_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:678, code: x = torch.cat((cls_tokens, x), dim=1)
    slice_120: "f32[8, 1, 512]" = torch.ops.aten.slice.Tensor(add_105, 1, 0, 1)
    slice_121: "f32[8, 49, 512]" = torch.ops.aten.slice.Tensor(add_105, 1, 1, 50);  add_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:677, code: cls_tokens = cls_token.expand(x.shape[0], -1, -1)
    sum_40: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(slice_120, [0], True);  slice_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:92, code: x = self.norm(x)
    clone_71: "f32[8, 49, 512]" = torch.ops.aten.clone.default(slice_121, memory_format = torch.contiguous_format);  slice_121 = None
    clone_72: "f32[8, 49, 512]" = torch.ops.aten.clone.default(permute_72, memory_format = torch.contiguous_format);  permute_72 = None
    sub_46: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(clone_72, getitem_67);  clone_72 = getitem_67 = None
    mul_141: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_15);  sub_46 = None
    mul_142: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(clone_71, primals_125);  primals_125 = None
    mul_143: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_142, 512)
    sum_41: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_142, [2], True)
    mul_144: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_142, mul_141);  mul_142 = None
    sum_42: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_144, [2], True);  mul_144 = None
    mul_145: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_141, sum_42);  sum_42 = None
    sub_47: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(mul_143, sum_41);  mul_143 = sum_41 = None
    sub_48: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(sub_47, mul_145);  sub_47 = mul_145 = None
    div_13: "f32[8, 49, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 512);  rsqrt_15 = None
    mul_146: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(div_13, sub_48);  div_13 = sub_48 = None
    mul_147: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(clone_71, mul_141);  mul_141 = None
    sum_43: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_147, [0, 1]);  mul_147 = None
    sum_44: "f32[512]" = torch.ops.aten.sum.dim_IntList(clone_71, [0, 1]);  clone_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:89, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    permute_155: "f32[8, 512, 49]" = torch.ops.aten.permute.default(mul_146, [0, 2, 1]);  mul_146 = None
    view_219: "f32[8, 512, 7, 7]" = torch.ops.aten.view.default(permute_155, [8, 512, 7, 7]);  permute_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(view_219, clone_47, primals_123, [512], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  view_219 = clone_47 = primals_123 = None
    getitem_114: "f32[8, 320, 14, 14]" = convolution_backward_8[0]
    getitem_115: "f32[512, 320, 2, 2]" = convolution_backward_8[1]
    getitem_116: "f32[512]" = convolution_backward_8[2];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:595, code: x3_nocls = remove_cls(x3).reshape(B, H3, W3, -1).permute(0, 3, 1, 2).contiguous()
    permute_156: "f32[8, 14, 14, 320]" = torch.ops.aten.permute.default(getitem_114, [0, 2, 3, 1]);  getitem_114 = None
    view_220: "f32[8, 196, 320]" = torch.ops.aten.view.default(permute_156, [8, 196, 320]);  permute_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:684, code: return x[:, 1:, :]
    full_26: "f32[8, 196, 320]" = torch.ops.aten.full.default([8, 196, 320], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_25: "f32[8, 196, 320]" = torch.ops.aten.slice_scatter.default(full_26, view_220, 2, 0, 9223372036854775807);  full_26 = view_220 = None
    full_27: "f32[8, 197, 320]" = torch.ops.aten.full.default([8, 197, 320], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_26: "f32[8, 197, 320]" = torch.ops.aten.slice_scatter.default(full_27, slice_scatter_25, 1, 1, 9223372036854775807);  full_27 = slice_scatter_25 = None
    full_28: "f32[8, 197, 320]" = torch.ops.aten.full.default([8, 197, 320], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_27: "f32[8, 197, 320]" = torch.ops.aten.slice_scatter.default(full_28, slice_scatter_26, 0, 0, 9223372036854775807);  full_28 = slice_scatter_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_221: "f32[1576, 320]" = torch.ops.aten.view.default(slice_scatter_27, [1576, 320])
    permute_157: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_70, [1, 0]);  permute_70 = None
    mm_18: "f32[1576, 1280]" = torch.ops.aten.mm.default(view_221, permute_157);  permute_157 = None
    permute_158: "f32[320, 1576]" = torch.ops.aten.permute.default(view_221, [1, 0])
    mm_19: "f32[320, 1280]" = torch.ops.aten.mm.default(permute_158, view_123);  permute_158 = view_123 = None
    permute_159: "f32[1280, 320]" = torch.ops.aten.permute.default(mm_19, [1, 0]);  mm_19 = None
    sum_45: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_221, [0], True);  view_221 = None
    view_222: "f32[320]" = torch.ops.aten.view.default(sum_45, [320]);  sum_45 = None
    permute_160: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_159, [1, 0]);  permute_159 = None
    view_223: "f32[8, 197, 1280]" = torch.ops.aten.view.default(mm_18, [8, 197, 1280]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_148: "f32[8, 197, 1280]" = torch.ops.aten.mul.Tensor(view_122, 0.7071067811865476)
    erf_10: "f32[8, 197, 1280]" = torch.ops.aten.erf.default(mul_148);  mul_148 = None
    add_106: "f32[8, 197, 1280]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_149: "f32[8, 197, 1280]" = torch.ops.aten.mul.Tensor(add_106, 0.5);  add_106 = None
    mul_150: "f32[8, 197, 1280]" = torch.ops.aten.mul.Tensor(view_122, view_122)
    mul_151: "f32[8, 197, 1280]" = torch.ops.aten.mul.Tensor(mul_150, -0.5);  mul_150 = None
    exp_10: "f32[8, 197, 1280]" = torch.ops.aten.exp.default(mul_151);  mul_151 = None
    mul_152: "f32[8, 197, 1280]" = torch.ops.aten.mul.Tensor(exp_10, 0.3989422804014327);  exp_10 = None
    mul_153: "f32[8, 197, 1280]" = torch.ops.aten.mul.Tensor(view_122, mul_152);  view_122 = mul_152 = None
    add_107: "f32[8, 197, 1280]" = torch.ops.aten.add.Tensor(mul_149, mul_153);  mul_149 = mul_153 = None
    mul_154: "f32[8, 197, 1280]" = torch.ops.aten.mul.Tensor(view_223, add_107);  view_223 = add_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_224: "f32[1576, 1280]" = torch.ops.aten.view.default(mul_154, [1576, 1280]);  mul_154 = None
    permute_161: "f32[1280, 320]" = torch.ops.aten.permute.default(permute_69, [1, 0]);  permute_69 = None
    mm_20: "f32[1576, 320]" = torch.ops.aten.mm.default(view_224, permute_161);  permute_161 = None
    permute_162: "f32[1280, 1576]" = torch.ops.aten.permute.default(view_224, [1, 0])
    mm_21: "f32[1280, 320]" = torch.ops.aten.mm.default(permute_162, view_121);  permute_162 = view_121 = None
    permute_163: "f32[320, 1280]" = torch.ops.aten.permute.default(mm_21, [1, 0]);  mm_21 = None
    sum_46: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_224, [0], True);  view_224 = None
    view_225: "f32[1280]" = torch.ops.aten.view.default(sum_46, [1280]);  sum_46 = None
    permute_164: "f32[1280, 320]" = torch.ops.aten.permute.default(permute_163, [1, 0]);  permute_163 = None
    view_226: "f32[8, 197, 320]" = torch.ops.aten.view.default(mm_20, [8, 197, 320]);  mm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_49: "f32[8, 197, 320]" = torch.ops.aten.sub.Tensor(add_55, getitem_65);  add_55 = getitem_65 = None
    mul_155: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_14);  sub_49 = None
    mul_156: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(view_226, primals_26);  primals_26 = None
    mul_157: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(mul_156, 320)
    sum_47: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_156, [2], True)
    mul_158: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(mul_156, mul_155);  mul_156 = None
    sum_48: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_158, [2], True);  mul_158 = None
    mul_159: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(mul_155, sum_48);  sum_48 = None
    sub_50: "f32[8, 197, 320]" = torch.ops.aten.sub.Tensor(mul_157, sum_47);  mul_157 = sum_47 = None
    sub_51: "f32[8, 197, 320]" = torch.ops.aten.sub.Tensor(sub_50, mul_159);  sub_50 = mul_159 = None
    div_14: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 320);  rsqrt_14 = None
    mul_160: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(div_14, sub_51);  div_14 = sub_51 = None
    mul_161: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(view_226, mul_155);  mul_155 = None
    sum_49: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_161, [0, 1]);  mul_161 = None
    sum_50: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_226, [0, 1]);  view_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_108: "f32[8, 197, 320]" = torch.ops.aten.add.Tensor(slice_scatter_27, mul_160);  slice_scatter_27 = mul_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:135, code: x = self.proj(x)
    view_227: "f32[1576, 320]" = torch.ops.aten.view.default(add_108, [1576, 320])
    permute_165: "f32[320, 320]" = torch.ops.aten.permute.default(permute_68, [1, 0]);  permute_68 = None
    mm_22: "f32[1576, 320]" = torch.ops.aten.mm.default(view_227, permute_165);  permute_165 = None
    permute_166: "f32[320, 1576]" = torch.ops.aten.permute.default(view_227, [1, 0])
    mm_23: "f32[320, 320]" = torch.ops.aten.mm.default(permute_166, view_119);  permute_166 = view_119 = None
    permute_167: "f32[320, 320]" = torch.ops.aten.permute.default(mm_23, [1, 0]);  mm_23 = None
    sum_51: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_227, [0], True);  view_227 = None
    view_228: "f32[320]" = torch.ops.aten.view.default(sum_51, [320]);  sum_51 = None
    permute_168: "f32[320, 320]" = torch.ops.aten.permute.default(permute_167, [1, 0]);  permute_167 = None
    view_229: "f32[8, 197, 320]" = torch.ops.aten.view.default(mm_22, [8, 197, 320]);  mm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:132, code: x = x.transpose(1, 2).reshape(B, N, C)  # [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C]
    view_230: "f32[8, 197, 8, 40]" = torch.ops.aten.view.default(view_229, [8, 197, 8, 40]);  view_229 = None
    permute_169: "f32[8, 8, 197, 40]" = torch.ops.aten.permute.default(view_230, [0, 2, 1, 3]);  view_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:131, code: x = self.scale * factor_att + crpe
    mul_162: "f32[8, 8, 197, 40]" = torch.ops.aten.mul.Tensor(permute_169, 0.15811388300841897)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:87, code: EV_hat = F.pad(EV_hat, (0, 0, 1, 0, 0, 0))  # [B, h, N, Ch].
    constant_pad_nd_10: "f32[8, 8, 196, 40]" = torch.ops.aten.constant_pad_nd.default(permute_169, [0, 0, -1, 0, 0, 0]);  permute_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:86, code: EV_hat = q_img * conv_v_img
    mul_163: "f32[8, 8, 196, 40]" = torch.ops.aten.mul.Tensor(constant_pad_nd_10, slice_74);  slice_74 = None
    mul_164: "f32[8, 8, 196, 40]" = torch.ops.aten.mul.Tensor(constant_pad_nd_10, permute_66);  constant_pad_nd_10 = permute_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:84, code: conv_v_img = conv_v_img.reshape(B, num_heads, C, H * W).transpose(-1, -2)
    permute_170: "f32[8, 8, 40, 196]" = torch.ops.aten.permute.default(mul_163, [0, 1, 3, 2]);  mul_163 = None
    view_231: "f32[8, 320, 14, 14]" = torch.ops.aten.view.default(permute_170, [8, 320, 14, 14]);  permute_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:83, code: conv_v_img = torch.cat(conv_v_img_list, dim=1)
    slice_122: "f32[8, 80, 14, 14]" = torch.ops.aten.slice.Tensor(view_231, 1, 0, 80)
    slice_123: "f32[8, 120, 14, 14]" = torch.ops.aten.slice.Tensor(view_231, 1, 80, 200)
    slice_124: "f32[8, 120, 14, 14]" = torch.ops.aten.slice.Tensor(view_231, 1, 200, 320);  view_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(slice_124, getitem_63, primals_107, [120], [1, 1], [3, 3], [1, 1], False, [0, 0], 120, [True, True, True]);  slice_124 = getitem_63 = None
    getitem_117: "f32[8, 120, 14, 14]" = convolution_backward_9[0]
    getitem_118: "f32[120, 1, 7, 7]" = convolution_backward_9[1]
    getitem_119: "f32[120]" = convolution_backward_9[2];  convolution_backward_9 = None
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(slice_123, getitem_62, primals_105, [120], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, True]);  slice_123 = getitem_62 = None
    getitem_120: "f32[8, 120, 14, 14]" = convolution_backward_10[0]
    getitem_121: "f32[120, 1, 5, 5]" = convolution_backward_10[1]
    getitem_122: "f32[120]" = convolution_backward_10[2];  convolution_backward_10 = None
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(slice_122, getitem_61, primals_103, [80], [1, 1], [1, 1], [1, 1], False, [0, 0], 80, [True, True, True]);  slice_122 = getitem_61 = None
    getitem_123: "f32[8, 80, 14, 14]" = convolution_backward_11[0]
    getitem_124: "f32[80, 1, 3, 3]" = convolution_backward_11[1]
    getitem_125: "f32[80]" = convolution_backward_11[2];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:79, code: v_img_list = torch.split(v_img, self.channel_splits, dim=1)  # Split according to channels
    cat_24: "f32[8, 320, 14, 14]" = torch.ops.aten.cat.default([getitem_123, getitem_120, getitem_117], 1);  getitem_123 = getitem_120 = getitem_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:78, code: v_img = v_img.transpose(-1, -2).reshape(B, num_heads * C, H, W)
    view_232: "f32[8, 8, 40, 196]" = torch.ops.aten.view.default(cat_24, [8, 8, 40, 196]);  cat_24 = None
    permute_171: "f32[8, 8, 196, 40]" = torch.ops.aten.permute.default(view_232, [0, 1, 3, 2]);  view_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:76, code: v_img = v[:, :, 1:, :]  # [B, h, H*W, Ch]
    full_29: "f32[8, 8, 196, 40]" = torch.ops.aten.full.default([8, 8, 196, 40], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_28: "f32[8, 8, 196, 40]" = torch.ops.aten.slice_scatter.default(full_29, permute_171, 3, 0, 9223372036854775807);  full_29 = permute_171 = None
    full_30: "f32[8, 8, 197, 40]" = torch.ops.aten.full.default([8, 8, 197, 40], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_29: "f32[8, 8, 197, 40]" = torch.ops.aten.slice_scatter.default(full_30, slice_scatter_28, 2, 1, 9223372036854775807);  full_30 = slice_scatter_28 = None
    full_31: "f32[8, 8, 197, 40]" = torch.ops.aten.full.default([8, 8, 197, 40], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_30: "f32[8, 8, 197, 40]" = torch.ops.aten.slice_scatter.default(full_31, slice_scatter_29, 1, 0, 9223372036854775807);  full_31 = slice_scatter_29 = None
    full_32: "f32[8, 8, 197, 40]" = torch.ops.aten.full.default([8, 8, 197, 40], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_31: "f32[8, 8, 197, 40]" = torch.ops.aten.slice_scatter.default(full_32, slice_scatter_30, 0, 0, 9223372036854775807);  full_32 = slice_scatter_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:75, code: q_img = q[:, :, 1:, :]  # [B, h, H*W, Ch]
    full_33: "f32[8, 8, 196, 40]" = torch.ops.aten.full.default([8, 8, 196, 40], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_32: "f32[8, 8, 196, 40]" = torch.ops.aten.slice_scatter.default(full_33, mul_164, 3, 0, 9223372036854775807);  full_33 = mul_164 = None
    full_34: "f32[8, 8, 197, 40]" = torch.ops.aten.full.default([8, 8, 197, 40], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_33: "f32[8, 8, 197, 40]" = torch.ops.aten.slice_scatter.default(full_34, slice_scatter_32, 2, 1, 9223372036854775807);  full_34 = slice_scatter_32 = None
    full_35: "f32[8, 8, 197, 40]" = torch.ops.aten.full.default([8, 8, 197, 40], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_34: "f32[8, 8, 197, 40]" = torch.ops.aten.slice_scatter.default(full_35, slice_scatter_33, 1, 0, 9223372036854775807);  full_35 = slice_scatter_33 = None
    full_36: "f32[8, 8, 197, 40]" = torch.ops.aten.full.default([8, 8, 197, 40], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_35: "f32[8, 8, 197, 40]" = torch.ops.aten.slice_scatter.default(full_36, slice_scatter_34, 0, 0, 9223372036854775807);  full_36 = slice_scatter_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    clone_73: "f32[8, 8, 197, 40]" = torch.ops.aten.clone.default(mul_162, memory_format = torch.contiguous_format);  mul_162 = None
    view_233: "f32[64, 197, 40]" = torch.ops.aten.view.default(clone_73, [64, 197, 40]);  clone_73 = None
    permute_172: "f32[64, 40, 197]" = torch.ops.aten.permute.default(view_113, [0, 2, 1]);  view_113 = None
    bmm_24: "f32[64, 40, 40]" = torch.ops.aten.bmm.default(permute_172, view_233);  permute_172 = None
    permute_173: "f32[64, 40, 40]" = torch.ops.aten.permute.default(view_114, [0, 2, 1]);  view_114 = None
    bmm_25: "f32[64, 197, 40]" = torch.ops.aten.bmm.default(view_233, permute_173);  view_233 = permute_173 = None
    view_234: "f32[8, 8, 40, 40]" = torch.ops.aten.view.default(bmm_24, [8, 8, 40, 40]);  bmm_24 = None
    view_235: "f32[8, 8, 197, 40]" = torch.ops.aten.view.default(bmm_25, [8, 8, 197, 40]);  bmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    add_109: "f32[8, 8, 197, 40]" = torch.ops.aten.add.Tensor(slice_scatter_35, view_235);  slice_scatter_35 = view_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    view_236: "f32[64, 40, 40]" = torch.ops.aten.view.default(view_234, [64, 40, 40]);  view_234 = None
    permute_174: "f32[64, 197, 40]" = torch.ops.aten.permute.default(view_110, [0, 2, 1]);  view_110 = None
    bmm_26: "f32[64, 197, 40]" = torch.ops.aten.bmm.default(permute_174, view_236);  permute_174 = None
    permute_175: "f32[64, 40, 197]" = torch.ops.aten.permute.default(view_111, [0, 2, 1]);  view_111 = None
    bmm_27: "f32[64, 40, 197]" = torch.ops.aten.bmm.default(view_236, permute_175);  view_236 = permute_175 = None
    view_237: "f32[8, 8, 197, 40]" = torch.ops.aten.view.default(bmm_26, [8, 8, 197, 40]);  bmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    add_110: "f32[8, 8, 197, 40]" = torch.ops.aten.add.Tensor(slice_scatter_31, view_237);  slice_scatter_31 = view_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    view_238: "f32[8, 8, 40, 197]" = torch.ops.aten.view.default(bmm_27, [8, 8, 40, 197]);  bmm_27 = None
    permute_176: "f32[8, 8, 197, 40]" = torch.ops.aten.permute.default(view_238, [0, 1, 3, 2]);  view_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:123, code: k_softmax = k.softmax(dim=2)
    alias_10: "f32[8, 8, 197, 40]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    mul_165: "f32[8, 8, 197, 40]" = torch.ops.aten.mul.Tensor(permute_176, alias_10);  permute_176 = None
    sum_52: "f32[8, 8, 1, 40]" = torch.ops.aten.sum.dim_IntList(mul_165, [2], True)
    mul_166: "f32[8, 8, 197, 40]" = torch.ops.aten.mul.Tensor(alias_10, sum_52);  alias_10 = sum_52 = None
    sub_52: "f32[8, 8, 197, 40]" = torch.ops.aten.sub.Tensor(mul_165, mul_166);  mul_165 = mul_166 = None
    clone_74: "f32[8, 8, 197, 40]" = torch.ops.aten.clone.default(sub_52, memory_format = torch.contiguous_format);  sub_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:120, code: q, k, v = qkv.unbind(0)  # [B, h, N, Ch]
    cat_25: "f32[24, 8, 197, 40]" = torch.ops.aten.cat.default([add_109, clone_74, add_110]);  add_109 = clone_74 = add_110 = None
    view_239: "f32[3, 8, 8, 197, 40]" = torch.ops.aten.view.default(cat_25, [3, 8, 8, 197, 40]);  cat_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:119, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_177: "f32[8, 197, 3, 8, 40]" = torch.ops.aten.permute.default(view_239, [1, 3, 0, 2, 4]);  view_239 = None
    clone_75: "f32[8, 197, 3, 8, 40]" = torch.ops.aten.clone.default(permute_177, memory_format = torch.contiguous_format);  permute_177 = None
    view_240: "f32[8, 197, 960]" = torch.ops.aten.view.default(clone_75, [8, 197, 960]);  clone_75 = None
    view_241: "f32[1576, 960]" = torch.ops.aten.view.default(view_240, [1576, 960]);  view_240 = None
    permute_178: "f32[960, 320]" = torch.ops.aten.permute.default(permute_62, [1, 0]);  permute_62 = None
    mm_24: "f32[1576, 320]" = torch.ops.aten.mm.default(view_241, permute_178);  permute_178 = None
    permute_179: "f32[960, 1576]" = torch.ops.aten.permute.default(view_241, [1, 0])
    mm_25: "f32[960, 320]" = torch.ops.aten.mm.default(permute_179, view_107);  permute_179 = view_107 = None
    permute_180: "f32[320, 960]" = torch.ops.aten.permute.default(mm_25, [1, 0]);  mm_25 = None
    sum_53: "f32[1, 960]" = torch.ops.aten.sum.dim_IntList(view_241, [0], True);  view_241 = None
    view_242: "f32[960]" = torch.ops.aten.view.default(sum_53, [960]);  sum_53 = None
    permute_181: "f32[960, 320]" = torch.ops.aten.permute.default(permute_180, [1, 0]);  permute_180 = None
    view_243: "f32[8, 197, 320]" = torch.ops.aten.view.default(mm_24, [8, 197, 320]);  mm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_53: "f32[8, 197, 320]" = torch.ops.aten.sub.Tensor(cat_13, getitem_57);  cat_13 = getitem_57 = None
    mul_167: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_13);  sub_53 = None
    mul_168: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(view_243, primals_24);  primals_24 = None
    mul_169: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(mul_168, 320)
    sum_54: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_168, [2], True)
    mul_170: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(mul_168, mul_167);  mul_168 = None
    sum_55: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_170, [2], True);  mul_170 = None
    mul_171: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(mul_167, sum_55);  sum_55 = None
    sub_54: "f32[8, 197, 320]" = torch.ops.aten.sub.Tensor(mul_169, sum_54);  mul_169 = sum_54 = None
    sub_55: "f32[8, 197, 320]" = torch.ops.aten.sub.Tensor(sub_54, mul_171);  sub_54 = mul_171 = None
    div_15: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 320);  rsqrt_13 = None
    mul_172: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(div_15, sub_55);  div_15 = sub_55 = None
    mul_173: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(view_243, mul_167);  mul_167 = None
    sum_56: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_173, [0, 1]);  mul_173 = None
    sum_57: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_243, [0, 1]);  view_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_111: "f32[8, 197, 320]" = torch.ops.aten.add.Tensor(add_108, mul_172);  add_108 = mul_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:163, code: x = torch.cat((cls_token, x), dim=1)
    slice_125: "f32[8, 1, 320]" = torch.ops.aten.slice.Tensor(add_111, 1, 0, 1)
    slice_126: "f32[8, 196, 320]" = torch.ops.aten.slice.Tensor(add_111, 1, 1, 197);  add_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:160, code: x = x.flatten(2).transpose(1, 2)
    permute_182: "f32[8, 320, 196]" = torch.ops.aten.permute.default(slice_126, [0, 2, 1]);  slice_126 = None
    view_244: "f32[8, 320, 14, 14]" = torch.ops.aten.view.default(permute_182, [8, 320, 14, 14]);  permute_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:159, code: x = self.proj(feat) + feat
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(view_244, view_105, primals_99, [320], [1, 1], [1, 1], [1, 1], False, [0, 0], 320, [True, True, True]);  view_105 = None
    getitem_126: "f32[8, 320, 14, 14]" = convolution_backward_12[0]
    getitem_127: "f32[320, 1, 3, 3]" = convolution_backward_12[1]
    getitem_128: "f32[320]" = convolution_backward_12[2];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:159, code: x = self.proj(feat) + feat
    add_112: "f32[8, 320, 14, 14]" = torch.ops.aten.add.Tensor(view_244, getitem_126);  view_244 = getitem_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:158, code: feat = img_tokens.transpose(1, 2).view(B, C, H, W)
    view_245: "f32[8, 320, 196]" = torch.ops.aten.view.default(add_112, [8, 320, 196]);  add_112 = None
    permute_183: "f32[8, 196, 320]" = torch.ops.aten.permute.default(view_245, [0, 2, 1]);  view_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    full_37: "f32[8, 197, 320]" = torch.ops.aten.full.default([8, 197, 320], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_36: "f32[8, 197, 320]" = torch.ops.aten.slice_scatter.default(full_37, permute_183, 1, 1, 9223372036854775807);  full_37 = permute_183 = None
    full_38: "f32[8, 197, 320]" = torch.ops.aten.full.default([8, 197, 320], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_37: "f32[8, 197, 320]" = torch.ops.aten.slice_scatter.default(full_38, slice_scatter_36, 0, 0, 9223372036854775807);  full_38 = slice_scatter_36 = None
    full_39: "f32[8, 197, 320]" = torch.ops.aten.full.default([8, 197, 320], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_38: "f32[8, 197, 320]" = torch.ops.aten.slice_scatter.default(full_39, slice_125, 1, 0, 1);  full_39 = slice_125 = None
    full_40: "f32[8, 197, 320]" = torch.ops.aten.full.default([8, 197, 320], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_39: "f32[8, 197, 320]" = torch.ops.aten.slice_scatter.default(full_40, slice_scatter_38, 0, 0, 9223372036854775807);  full_40 = slice_scatter_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    add_113: "f32[8, 197, 320]" = torch.ops.aten.add.Tensor(slice_scatter_37, slice_scatter_39);  slice_scatter_37 = slice_scatter_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_246: "f32[1576, 320]" = torch.ops.aten.view.default(add_113, [1576, 320])
    permute_184: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None
    mm_26: "f32[1576, 1280]" = torch.ops.aten.mm.default(view_246, permute_184);  permute_184 = None
    permute_185: "f32[320, 1576]" = torch.ops.aten.permute.default(view_246, [1, 0])
    mm_27: "f32[320, 1280]" = torch.ops.aten.mm.default(permute_185, view_103);  permute_185 = view_103 = None
    permute_186: "f32[1280, 320]" = torch.ops.aten.permute.default(mm_27, [1, 0]);  mm_27 = None
    sum_58: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_246, [0], True);  view_246 = None
    view_247: "f32[320]" = torch.ops.aten.view.default(sum_58, [320]);  sum_58 = None
    permute_187: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_186, [1, 0]);  permute_186 = None
    view_248: "f32[8, 197, 1280]" = torch.ops.aten.view.default(mm_26, [8, 197, 1280]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_174: "f32[8, 197, 1280]" = torch.ops.aten.mul.Tensor(view_102, 0.7071067811865476)
    erf_11: "f32[8, 197, 1280]" = torch.ops.aten.erf.default(mul_174);  mul_174 = None
    add_114: "f32[8, 197, 1280]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_175: "f32[8, 197, 1280]" = torch.ops.aten.mul.Tensor(add_114, 0.5);  add_114 = None
    mul_176: "f32[8, 197, 1280]" = torch.ops.aten.mul.Tensor(view_102, view_102)
    mul_177: "f32[8, 197, 1280]" = torch.ops.aten.mul.Tensor(mul_176, -0.5);  mul_176 = None
    exp_11: "f32[8, 197, 1280]" = torch.ops.aten.exp.default(mul_177);  mul_177 = None
    mul_178: "f32[8, 197, 1280]" = torch.ops.aten.mul.Tensor(exp_11, 0.3989422804014327);  exp_11 = None
    mul_179: "f32[8, 197, 1280]" = torch.ops.aten.mul.Tensor(view_102, mul_178);  view_102 = mul_178 = None
    add_115: "f32[8, 197, 1280]" = torch.ops.aten.add.Tensor(mul_175, mul_179);  mul_175 = mul_179 = None
    mul_180: "f32[8, 197, 1280]" = torch.ops.aten.mul.Tensor(view_248, add_115);  view_248 = add_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_249: "f32[1576, 1280]" = torch.ops.aten.view.default(mul_180, [1576, 1280]);  mul_180 = None
    permute_188: "f32[1280, 320]" = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
    mm_28: "f32[1576, 320]" = torch.ops.aten.mm.default(view_249, permute_188);  permute_188 = None
    permute_189: "f32[1280, 1576]" = torch.ops.aten.permute.default(view_249, [1, 0])
    mm_29: "f32[1280, 320]" = torch.ops.aten.mm.default(permute_189, view_101);  permute_189 = view_101 = None
    permute_190: "f32[320, 1280]" = torch.ops.aten.permute.default(mm_29, [1, 0]);  mm_29 = None
    sum_59: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_249, [0], True);  view_249 = None
    view_250: "f32[1280]" = torch.ops.aten.view.default(sum_59, [1280]);  sum_59 = None
    permute_191: "f32[1280, 320]" = torch.ops.aten.permute.default(permute_190, [1, 0]);  permute_190 = None
    view_251: "f32[8, 197, 320]" = torch.ops.aten.view.default(mm_28, [8, 197, 320]);  mm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_56: "f32[8, 197, 320]" = torch.ops.aten.sub.Tensor(add_46, getitem_55);  add_46 = getitem_55 = None
    mul_181: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_12);  sub_56 = None
    mul_182: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(view_251, primals_22);  primals_22 = None
    mul_183: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(mul_182, 320)
    sum_60: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_182, [2], True)
    mul_184: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(mul_182, mul_181);  mul_182 = None
    sum_61: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_184, [2], True);  mul_184 = None
    mul_185: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(mul_181, sum_61);  sum_61 = None
    sub_57: "f32[8, 197, 320]" = torch.ops.aten.sub.Tensor(mul_183, sum_60);  mul_183 = sum_60 = None
    sub_58: "f32[8, 197, 320]" = torch.ops.aten.sub.Tensor(sub_57, mul_185);  sub_57 = mul_185 = None
    div_16: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 320);  rsqrt_12 = None
    mul_186: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(div_16, sub_58);  div_16 = sub_58 = None
    mul_187: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(view_251, mul_181);  mul_181 = None
    sum_62: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_187, [0, 1]);  mul_187 = None
    sum_63: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_251, [0, 1]);  view_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_116: "f32[8, 197, 320]" = torch.ops.aten.add.Tensor(add_113, mul_186);  add_113 = mul_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:135, code: x = self.proj(x)
    view_252: "f32[1576, 320]" = torch.ops.aten.view.default(add_116, [1576, 320])
    permute_192: "f32[320, 320]" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
    mm_30: "f32[1576, 320]" = torch.ops.aten.mm.default(view_252, permute_192);  permute_192 = None
    permute_193: "f32[320, 1576]" = torch.ops.aten.permute.default(view_252, [1, 0])
    mm_31: "f32[320, 320]" = torch.ops.aten.mm.default(permute_193, view_99);  permute_193 = view_99 = None
    permute_194: "f32[320, 320]" = torch.ops.aten.permute.default(mm_31, [1, 0]);  mm_31 = None
    sum_64: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_252, [0], True);  view_252 = None
    view_253: "f32[320]" = torch.ops.aten.view.default(sum_64, [320]);  sum_64 = None
    permute_195: "f32[320, 320]" = torch.ops.aten.permute.default(permute_194, [1, 0]);  permute_194 = None
    view_254: "f32[8, 197, 320]" = torch.ops.aten.view.default(mm_30, [8, 197, 320]);  mm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:132, code: x = x.transpose(1, 2).reshape(B, N, C)  # [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C]
    view_255: "f32[8, 197, 8, 40]" = torch.ops.aten.view.default(view_254, [8, 197, 8, 40]);  view_254 = None
    permute_196: "f32[8, 8, 197, 40]" = torch.ops.aten.permute.default(view_255, [0, 2, 1, 3]);  view_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:131, code: x = self.scale * factor_att + crpe
    mul_188: "f32[8, 8, 197, 40]" = torch.ops.aten.mul.Tensor(permute_196, 0.15811388300841897)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:87, code: EV_hat = F.pad(EV_hat, (0, 0, 1, 0, 0, 0))  # [B, h, N, Ch].
    constant_pad_nd_11: "f32[8, 8, 196, 40]" = torch.ops.aten.constant_pad_nd.default(permute_196, [0, 0, -1, 0, 0, 0]);  permute_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:86, code: EV_hat = q_img * conv_v_img
    mul_189: "f32[8, 8, 196, 40]" = torch.ops.aten.mul.Tensor(constant_pad_nd_11, slice_62);  slice_62 = None
    mul_190: "f32[8, 8, 196, 40]" = torch.ops.aten.mul.Tensor(constant_pad_nd_11, permute_55);  constant_pad_nd_11 = permute_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:84, code: conv_v_img = conv_v_img.reshape(B, num_heads, C, H * W).transpose(-1, -2)
    permute_197: "f32[8, 8, 40, 196]" = torch.ops.aten.permute.default(mul_189, [0, 1, 3, 2]);  mul_189 = None
    view_256: "f32[8, 320, 14, 14]" = torch.ops.aten.view.default(permute_197, [8, 320, 14, 14]);  permute_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:83, code: conv_v_img = torch.cat(conv_v_img_list, dim=1)
    slice_127: "f32[8, 80, 14, 14]" = torch.ops.aten.slice.Tensor(view_256, 1, 0, 80)
    slice_128: "f32[8, 120, 14, 14]" = torch.ops.aten.slice.Tensor(view_256, 1, 80, 200)
    slice_129: "f32[8, 120, 14, 14]" = torch.ops.aten.slice.Tensor(view_256, 1, 200, 320);  view_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(slice_129, getitem_53, primals_107, [120], [1, 1], [3, 3], [1, 1], False, [0, 0], 120, [True, True, True]);  slice_129 = getitem_53 = primals_107 = None
    getitem_129: "f32[8, 120, 14, 14]" = convolution_backward_13[0]
    getitem_130: "f32[120, 1, 7, 7]" = convolution_backward_13[1]
    getitem_131: "f32[120]" = convolution_backward_13[2];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    add_117: "f32[120, 1, 7, 7]" = torch.ops.aten.add.Tensor(getitem_118, getitem_130);  getitem_118 = getitem_130 = None
    add_118: "f32[120]" = torch.ops.aten.add.Tensor(getitem_119, getitem_131);  getitem_119 = getitem_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(slice_128, getitem_52, primals_105, [120], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, True]);  slice_128 = getitem_52 = primals_105 = None
    getitem_132: "f32[8, 120, 14, 14]" = convolution_backward_14[0]
    getitem_133: "f32[120, 1, 5, 5]" = convolution_backward_14[1]
    getitem_134: "f32[120]" = convolution_backward_14[2];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    add_119: "f32[120, 1, 5, 5]" = torch.ops.aten.add.Tensor(getitem_121, getitem_133);  getitem_121 = getitem_133 = None
    add_120: "f32[120]" = torch.ops.aten.add.Tensor(getitem_122, getitem_134);  getitem_122 = getitem_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(slice_127, getitem_51, primals_103, [80], [1, 1], [1, 1], [1, 1], False, [0, 0], 80, [True, True, True]);  slice_127 = getitem_51 = primals_103 = None
    getitem_135: "f32[8, 80, 14, 14]" = convolution_backward_15[0]
    getitem_136: "f32[80, 1, 3, 3]" = convolution_backward_15[1]
    getitem_137: "f32[80]" = convolution_backward_15[2];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    add_121: "f32[80, 1, 3, 3]" = torch.ops.aten.add.Tensor(getitem_124, getitem_136);  getitem_124 = getitem_136 = None
    add_122: "f32[80]" = torch.ops.aten.add.Tensor(getitem_125, getitem_137);  getitem_125 = getitem_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:79, code: v_img_list = torch.split(v_img, self.channel_splits, dim=1)  # Split according to channels
    cat_26: "f32[8, 320, 14, 14]" = torch.ops.aten.cat.default([getitem_135, getitem_132, getitem_129], 1);  getitem_135 = getitem_132 = getitem_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:78, code: v_img = v_img.transpose(-1, -2).reshape(B, num_heads * C, H, W)
    view_257: "f32[8, 8, 40, 196]" = torch.ops.aten.view.default(cat_26, [8, 8, 40, 196]);  cat_26 = None
    permute_198: "f32[8, 8, 196, 40]" = torch.ops.aten.permute.default(view_257, [0, 1, 3, 2]);  view_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:76, code: v_img = v[:, :, 1:, :]  # [B, h, H*W, Ch]
    full_41: "f32[8, 8, 196, 40]" = torch.ops.aten.full.default([8, 8, 196, 40], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_40: "f32[8, 8, 196, 40]" = torch.ops.aten.slice_scatter.default(full_41, permute_198, 3, 0, 9223372036854775807);  full_41 = permute_198 = None
    full_42: "f32[8, 8, 197, 40]" = torch.ops.aten.full.default([8, 8, 197, 40], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_41: "f32[8, 8, 197, 40]" = torch.ops.aten.slice_scatter.default(full_42, slice_scatter_40, 2, 1, 9223372036854775807);  full_42 = slice_scatter_40 = None
    full_43: "f32[8, 8, 197, 40]" = torch.ops.aten.full.default([8, 8, 197, 40], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_42: "f32[8, 8, 197, 40]" = torch.ops.aten.slice_scatter.default(full_43, slice_scatter_41, 1, 0, 9223372036854775807);  full_43 = slice_scatter_41 = None
    full_44: "f32[8, 8, 197, 40]" = torch.ops.aten.full.default([8, 8, 197, 40], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_43: "f32[8, 8, 197, 40]" = torch.ops.aten.slice_scatter.default(full_44, slice_scatter_42, 0, 0, 9223372036854775807);  full_44 = slice_scatter_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:75, code: q_img = q[:, :, 1:, :]  # [B, h, H*W, Ch]
    full_45: "f32[8, 8, 196, 40]" = torch.ops.aten.full.default([8, 8, 196, 40], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_44: "f32[8, 8, 196, 40]" = torch.ops.aten.slice_scatter.default(full_45, mul_190, 3, 0, 9223372036854775807);  full_45 = mul_190 = None
    full_46: "f32[8, 8, 197, 40]" = torch.ops.aten.full.default([8, 8, 197, 40], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_45: "f32[8, 8, 197, 40]" = torch.ops.aten.slice_scatter.default(full_46, slice_scatter_44, 2, 1, 9223372036854775807);  full_46 = slice_scatter_44 = None
    full_47: "f32[8, 8, 197, 40]" = torch.ops.aten.full.default([8, 8, 197, 40], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_46: "f32[8, 8, 197, 40]" = torch.ops.aten.slice_scatter.default(full_47, slice_scatter_45, 1, 0, 9223372036854775807);  full_47 = slice_scatter_45 = None
    full_48: "f32[8, 8, 197, 40]" = torch.ops.aten.full.default([8, 8, 197, 40], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_47: "f32[8, 8, 197, 40]" = torch.ops.aten.slice_scatter.default(full_48, slice_scatter_46, 0, 0, 9223372036854775807);  full_48 = slice_scatter_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    clone_76: "f32[8, 8, 197, 40]" = torch.ops.aten.clone.default(mul_188, memory_format = torch.contiguous_format);  mul_188 = None
    view_258: "f32[64, 197, 40]" = torch.ops.aten.view.default(clone_76, [64, 197, 40]);  clone_76 = None
    permute_199: "f32[64, 40, 197]" = torch.ops.aten.permute.default(view_93, [0, 2, 1]);  view_93 = None
    bmm_28: "f32[64, 40, 40]" = torch.ops.aten.bmm.default(permute_199, view_258);  permute_199 = None
    permute_200: "f32[64, 40, 40]" = torch.ops.aten.permute.default(view_94, [0, 2, 1]);  view_94 = None
    bmm_29: "f32[64, 197, 40]" = torch.ops.aten.bmm.default(view_258, permute_200);  view_258 = permute_200 = None
    view_259: "f32[8, 8, 40, 40]" = torch.ops.aten.view.default(bmm_28, [8, 8, 40, 40]);  bmm_28 = None
    view_260: "f32[8, 8, 197, 40]" = torch.ops.aten.view.default(bmm_29, [8, 8, 197, 40]);  bmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    add_123: "f32[8, 8, 197, 40]" = torch.ops.aten.add.Tensor(slice_scatter_47, view_260);  slice_scatter_47 = view_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    view_261: "f32[64, 40, 40]" = torch.ops.aten.view.default(view_259, [64, 40, 40]);  view_259 = None
    permute_201: "f32[64, 197, 40]" = torch.ops.aten.permute.default(view_90, [0, 2, 1]);  view_90 = None
    bmm_30: "f32[64, 197, 40]" = torch.ops.aten.bmm.default(permute_201, view_261);  permute_201 = None
    permute_202: "f32[64, 40, 197]" = torch.ops.aten.permute.default(view_91, [0, 2, 1]);  view_91 = None
    bmm_31: "f32[64, 40, 197]" = torch.ops.aten.bmm.default(view_261, permute_202);  view_261 = permute_202 = None
    view_262: "f32[8, 8, 197, 40]" = torch.ops.aten.view.default(bmm_30, [8, 8, 197, 40]);  bmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    add_124: "f32[8, 8, 197, 40]" = torch.ops.aten.add.Tensor(slice_scatter_43, view_262);  slice_scatter_43 = view_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    view_263: "f32[8, 8, 40, 197]" = torch.ops.aten.view.default(bmm_31, [8, 8, 40, 197]);  bmm_31 = None
    permute_203: "f32[8, 8, 197, 40]" = torch.ops.aten.permute.default(view_263, [0, 1, 3, 2]);  view_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:123, code: k_softmax = k.softmax(dim=2)
    alias_11: "f32[8, 8, 197, 40]" = torch.ops.aten.alias.default(alias_4);  alias_4 = None
    mul_191: "f32[8, 8, 197, 40]" = torch.ops.aten.mul.Tensor(permute_203, alias_11);  permute_203 = None
    sum_65: "f32[8, 8, 1, 40]" = torch.ops.aten.sum.dim_IntList(mul_191, [2], True)
    mul_192: "f32[8, 8, 197, 40]" = torch.ops.aten.mul.Tensor(alias_11, sum_65);  alias_11 = sum_65 = None
    sub_59: "f32[8, 8, 197, 40]" = torch.ops.aten.sub.Tensor(mul_191, mul_192);  mul_191 = mul_192 = None
    clone_77: "f32[8, 8, 197, 40]" = torch.ops.aten.clone.default(sub_59, memory_format = torch.contiguous_format);  sub_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:120, code: q, k, v = qkv.unbind(0)  # [B, h, N, Ch]
    cat_27: "f32[24, 8, 197, 40]" = torch.ops.aten.cat.default([add_123, clone_77, add_124]);  add_123 = clone_77 = add_124 = None
    view_264: "f32[3, 8, 8, 197, 40]" = torch.ops.aten.view.default(cat_27, [3, 8, 8, 197, 40]);  cat_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:119, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_204: "f32[8, 197, 3, 8, 40]" = torch.ops.aten.permute.default(view_264, [1, 3, 0, 2, 4]);  view_264 = None
    clone_78: "f32[8, 197, 3, 8, 40]" = torch.ops.aten.clone.default(permute_204, memory_format = torch.contiguous_format);  permute_204 = None
    view_265: "f32[8, 197, 960]" = torch.ops.aten.view.default(clone_78, [8, 197, 960]);  clone_78 = None
    view_266: "f32[1576, 960]" = torch.ops.aten.view.default(view_265, [1576, 960]);  view_265 = None
    permute_205: "f32[960, 320]" = torch.ops.aten.permute.default(permute_51, [1, 0]);  permute_51 = None
    mm_32: "f32[1576, 320]" = torch.ops.aten.mm.default(view_266, permute_205);  permute_205 = None
    permute_206: "f32[960, 1576]" = torch.ops.aten.permute.default(view_266, [1, 0])
    mm_33: "f32[960, 320]" = torch.ops.aten.mm.default(permute_206, view_87);  permute_206 = view_87 = None
    permute_207: "f32[320, 960]" = torch.ops.aten.permute.default(mm_33, [1, 0]);  mm_33 = None
    sum_66: "f32[1, 960]" = torch.ops.aten.sum.dim_IntList(view_266, [0], True);  view_266 = None
    view_267: "f32[960]" = torch.ops.aten.view.default(sum_66, [960]);  sum_66 = None
    permute_208: "f32[960, 320]" = torch.ops.aten.permute.default(permute_207, [1, 0]);  permute_207 = None
    view_268: "f32[8, 197, 320]" = torch.ops.aten.view.default(mm_32, [8, 197, 320]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_60: "f32[8, 197, 320]" = torch.ops.aten.sub.Tensor(cat_11, getitem_47);  cat_11 = getitem_47 = None
    mul_193: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_11);  sub_60 = None
    mul_194: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(view_268, primals_20);  primals_20 = None
    mul_195: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(mul_194, 320)
    sum_67: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_194, [2], True)
    mul_196: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(mul_194, mul_193);  mul_194 = None
    sum_68: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_196, [2], True);  mul_196 = None
    mul_197: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(mul_193, sum_68);  sum_68 = None
    sub_61: "f32[8, 197, 320]" = torch.ops.aten.sub.Tensor(mul_195, sum_67);  mul_195 = sum_67 = None
    sub_62: "f32[8, 197, 320]" = torch.ops.aten.sub.Tensor(sub_61, mul_197);  sub_61 = mul_197 = None
    div_17: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 320);  rsqrt_11 = None
    mul_198: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(div_17, sub_62);  div_17 = sub_62 = None
    mul_199: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(view_268, mul_193);  mul_193 = None
    sum_69: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_199, [0, 1]);  mul_199 = None
    sum_70: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_268, [0, 1]);  view_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_125: "f32[8, 197, 320]" = torch.ops.aten.add.Tensor(add_116, mul_198);  add_116 = mul_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:163, code: x = torch.cat((cls_token, x), dim=1)
    slice_130: "f32[8, 1, 320]" = torch.ops.aten.slice.Tensor(add_125, 1, 0, 1)
    slice_131: "f32[8, 196, 320]" = torch.ops.aten.slice.Tensor(add_125, 1, 1, 197);  add_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:160, code: x = x.flatten(2).transpose(1, 2)
    permute_209: "f32[8, 320, 196]" = torch.ops.aten.permute.default(slice_131, [0, 2, 1]);  slice_131 = None
    view_269: "f32[8, 320, 14, 14]" = torch.ops.aten.view.default(permute_209, [8, 320, 14, 14]);  permute_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:159, code: x = self.proj(feat) + feat
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(view_269, view_85, primals_99, [320], [1, 1], [1, 1], [1, 1], False, [0, 0], 320, [True, True, True]);  view_85 = primals_99 = None
    getitem_138: "f32[8, 320, 14, 14]" = convolution_backward_16[0]
    getitem_139: "f32[320, 1, 3, 3]" = convolution_backward_16[1]
    getitem_140: "f32[320]" = convolution_backward_16[2];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:159, code: x = self.proj(feat) + feat
    add_126: "f32[8, 320, 14, 14]" = torch.ops.aten.add.Tensor(view_269, getitem_138);  view_269 = getitem_138 = None
    add_127: "f32[320, 1, 3, 3]" = torch.ops.aten.add.Tensor(getitem_127, getitem_139);  getitem_127 = getitem_139 = None
    add_128: "f32[320]" = torch.ops.aten.add.Tensor(getitem_128, getitem_140);  getitem_128 = getitem_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:158, code: feat = img_tokens.transpose(1, 2).view(B, C, H, W)
    view_270: "f32[8, 320, 196]" = torch.ops.aten.view.default(add_126, [8, 320, 196]);  add_126 = None
    permute_210: "f32[8, 196, 320]" = torch.ops.aten.permute.default(view_270, [0, 2, 1]);  view_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    full_49: "f32[8, 197, 320]" = torch.ops.aten.full.default([8, 197, 320], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_48: "f32[8, 197, 320]" = torch.ops.aten.slice_scatter.default(full_49, permute_210, 1, 1, 9223372036854775807);  full_49 = permute_210 = None
    full_50: "f32[8, 197, 320]" = torch.ops.aten.full.default([8, 197, 320], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_49: "f32[8, 197, 320]" = torch.ops.aten.slice_scatter.default(full_50, slice_scatter_48, 0, 0, 9223372036854775807);  full_50 = slice_scatter_48 = None
    full_51: "f32[8, 197, 320]" = torch.ops.aten.full.default([8, 197, 320], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_50: "f32[8, 197, 320]" = torch.ops.aten.slice_scatter.default(full_51, slice_130, 1, 0, 1);  full_51 = slice_130 = None
    full_52: "f32[8, 197, 320]" = torch.ops.aten.full.default([8, 197, 320], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_51: "f32[8, 197, 320]" = torch.ops.aten.slice_scatter.default(full_52, slice_scatter_50, 0, 0, 9223372036854775807);  full_52 = slice_scatter_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    add_129: "f32[8, 197, 320]" = torch.ops.aten.add.Tensor(slice_scatter_49, slice_scatter_51);  slice_scatter_49 = slice_scatter_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:678, code: x = torch.cat((cls_tokens, x), dim=1)
    slice_132: "f32[8, 1, 320]" = torch.ops.aten.slice.Tensor(add_129, 1, 0, 1)
    slice_133: "f32[8, 196, 320]" = torch.ops.aten.slice.Tensor(add_129, 1, 1, 197);  add_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:677, code: cls_tokens = cls_token.expand(x.shape[0], -1, -1)
    sum_71: "f32[1, 1, 320]" = torch.ops.aten.sum.dim_IntList(slice_132, [0], True);  slice_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:92, code: x = self.norm(x)
    clone_79: "f32[8, 196, 320]" = torch.ops.aten.clone.default(slice_133, memory_format = torch.contiguous_format);  slice_133 = None
    clone_80: "f32[8, 196, 320]" = torch.ops.aten.clone.default(permute_48, memory_format = torch.contiguous_format);  permute_48 = None
    sub_63: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_80, getitem_45);  clone_80 = getitem_45 = None
    mul_200: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_10);  sub_63 = None
    mul_201: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_79, primals_97);  primals_97 = None
    mul_202: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_201, 320)
    sum_72: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_201, [2], True)
    mul_203: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_201, mul_200);  mul_201 = None
    sum_73: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_203, [2], True);  mul_203 = None
    mul_204: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_200, sum_73);  sum_73 = None
    sub_64: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_202, sum_72);  mul_202 = sum_72 = None
    sub_65: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_64, mul_204);  sub_64 = mul_204 = None
    div_18: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 320);  rsqrt_10 = None
    mul_205: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_18, sub_65);  div_18 = sub_65 = None
    mul_206: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_79, mul_200);  mul_200 = None
    sum_74: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_206, [0, 1]);  mul_206 = None
    sum_75: "f32[320]" = torch.ops.aten.sum.dim_IntList(clone_79, [0, 1]);  clone_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:89, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    permute_211: "f32[8, 320, 196]" = torch.ops.aten.permute.default(mul_205, [0, 2, 1]);  mul_205 = None
    view_271: "f32[8, 320, 14, 14]" = torch.ops.aten.view.default(permute_211, [8, 320, 14, 14]);  permute_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(view_271, clone_31, primals_95, [320], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  view_271 = clone_31 = primals_95 = None
    getitem_141: "f32[8, 128, 28, 28]" = convolution_backward_17[0]
    getitem_142: "f32[320, 128, 2, 2]" = convolution_backward_17[1]
    getitem_143: "f32[320]" = convolution_backward_17[2];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:587, code: x2_nocls = remove_cls(x2).reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()
    permute_212: "f32[8, 28, 28, 128]" = torch.ops.aten.permute.default(getitem_141, [0, 2, 3, 1]);  getitem_141 = None
    view_272: "f32[8, 784, 128]" = torch.ops.aten.view.default(permute_212, [8, 784, 128]);  permute_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:684, code: return x[:, 1:, :]
    full_53: "f32[8, 784, 128]" = torch.ops.aten.full.default([8, 784, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_52: "f32[8, 784, 128]" = torch.ops.aten.slice_scatter.default(full_53, view_272, 2, 0, 9223372036854775807);  full_53 = view_272 = None
    full_54: "f32[8, 785, 128]" = torch.ops.aten.full.default([8, 785, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_53: "f32[8, 785, 128]" = torch.ops.aten.slice_scatter.default(full_54, slice_scatter_52, 1, 1, 9223372036854775807);  full_54 = slice_scatter_52 = None
    full_55: "f32[8, 785, 128]" = torch.ops.aten.full.default([8, 785, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_54: "f32[8, 785, 128]" = torch.ops.aten.slice_scatter.default(full_55, slice_scatter_53, 0, 0, 9223372036854775807);  full_55 = slice_scatter_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_273: "f32[6280, 128]" = torch.ops.aten.view.default(slice_scatter_54, [6280, 128])
    permute_213: "f32[128, 1024]" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
    mm_34: "f32[6280, 1024]" = torch.ops.aten.mm.default(view_273, permute_213);  permute_213 = None
    permute_214: "f32[128, 6280]" = torch.ops.aten.permute.default(view_273, [1, 0])
    mm_35: "f32[128, 1024]" = torch.ops.aten.mm.default(permute_214, view_81);  permute_214 = view_81 = None
    permute_215: "f32[1024, 128]" = torch.ops.aten.permute.default(mm_35, [1, 0]);  mm_35 = None
    sum_76: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_273, [0], True);  view_273 = None
    view_274: "f32[128]" = torch.ops.aten.view.default(sum_76, [128]);  sum_76 = None
    permute_216: "f32[128, 1024]" = torch.ops.aten.permute.default(permute_215, [1, 0]);  permute_215 = None
    view_275: "f32[8, 785, 1024]" = torch.ops.aten.view.default(mm_34, [8, 785, 1024]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_207: "f32[8, 785, 1024]" = torch.ops.aten.mul.Tensor(view_80, 0.7071067811865476)
    erf_12: "f32[8, 785, 1024]" = torch.ops.aten.erf.default(mul_207);  mul_207 = None
    add_130: "f32[8, 785, 1024]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_208: "f32[8, 785, 1024]" = torch.ops.aten.mul.Tensor(add_130, 0.5);  add_130 = None
    mul_209: "f32[8, 785, 1024]" = torch.ops.aten.mul.Tensor(view_80, view_80)
    mul_210: "f32[8, 785, 1024]" = torch.ops.aten.mul.Tensor(mul_209, -0.5);  mul_209 = None
    exp_12: "f32[8, 785, 1024]" = torch.ops.aten.exp.default(mul_210);  mul_210 = None
    mul_211: "f32[8, 785, 1024]" = torch.ops.aten.mul.Tensor(exp_12, 0.3989422804014327);  exp_12 = None
    mul_212: "f32[8, 785, 1024]" = torch.ops.aten.mul.Tensor(view_80, mul_211);  view_80 = mul_211 = None
    add_131: "f32[8, 785, 1024]" = torch.ops.aten.add.Tensor(mul_208, mul_212);  mul_208 = mul_212 = None
    mul_213: "f32[8, 785, 1024]" = torch.ops.aten.mul.Tensor(view_275, add_131);  view_275 = add_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_276: "f32[6280, 1024]" = torch.ops.aten.view.default(mul_213, [6280, 1024]);  mul_213 = None
    permute_217: "f32[1024, 128]" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
    mm_36: "f32[6280, 128]" = torch.ops.aten.mm.default(view_276, permute_217);  permute_217 = None
    permute_218: "f32[1024, 6280]" = torch.ops.aten.permute.default(view_276, [1, 0])
    mm_37: "f32[1024, 128]" = torch.ops.aten.mm.default(permute_218, view_79);  permute_218 = view_79 = None
    permute_219: "f32[128, 1024]" = torch.ops.aten.permute.default(mm_37, [1, 0]);  mm_37 = None
    sum_77: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_276, [0], True);  view_276 = None
    view_277: "f32[1024]" = torch.ops.aten.view.default(sum_77, [1024]);  sum_77 = None
    permute_220: "f32[1024, 128]" = torch.ops.aten.permute.default(permute_219, [1, 0]);  permute_219 = None
    view_278: "f32[8, 785, 128]" = torch.ops.aten.view.default(mm_36, [8, 785, 128]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_66: "f32[8, 785, 128]" = torch.ops.aten.sub.Tensor(add_35, getitem_43);  add_35 = getitem_43 = None
    mul_214: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_9);  sub_66 = None
    mul_215: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(view_278, primals_17);  primals_17 = None
    mul_216: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(mul_215, 128)
    sum_78: "f32[8, 785, 1]" = torch.ops.aten.sum.dim_IntList(mul_215, [2], True)
    mul_217: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(mul_215, mul_214);  mul_215 = None
    sum_79: "f32[8, 785, 1]" = torch.ops.aten.sum.dim_IntList(mul_217, [2], True);  mul_217 = None
    mul_218: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(mul_214, sum_79);  sum_79 = None
    sub_67: "f32[8, 785, 128]" = torch.ops.aten.sub.Tensor(mul_216, sum_78);  mul_216 = sum_78 = None
    sub_68: "f32[8, 785, 128]" = torch.ops.aten.sub.Tensor(sub_67, mul_218);  sub_67 = mul_218 = None
    div_19: "f32[8, 785, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 128);  rsqrt_9 = None
    mul_219: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(div_19, sub_68);  div_19 = sub_68 = None
    mul_220: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(view_278, mul_214);  mul_214 = None
    sum_80: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_220, [0, 1]);  mul_220 = None
    sum_81: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_278, [0, 1]);  view_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_132: "f32[8, 785, 128]" = torch.ops.aten.add.Tensor(slice_scatter_54, mul_219);  slice_scatter_54 = mul_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:135, code: x = self.proj(x)
    view_279: "f32[6280, 128]" = torch.ops.aten.view.default(add_132, [6280, 128])
    permute_221: "f32[128, 128]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
    mm_38: "f32[6280, 128]" = torch.ops.aten.mm.default(view_279, permute_221);  permute_221 = None
    permute_222: "f32[128, 6280]" = torch.ops.aten.permute.default(view_279, [1, 0])
    mm_39: "f32[128, 128]" = torch.ops.aten.mm.default(permute_222, view_77);  permute_222 = view_77 = None
    permute_223: "f32[128, 128]" = torch.ops.aten.permute.default(mm_39, [1, 0]);  mm_39 = None
    sum_82: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_279, [0], True);  view_279 = None
    view_280: "f32[128]" = torch.ops.aten.view.default(sum_82, [128]);  sum_82 = None
    permute_224: "f32[128, 128]" = torch.ops.aten.permute.default(permute_223, [1, 0]);  permute_223 = None
    view_281: "f32[8, 785, 128]" = torch.ops.aten.view.default(mm_38, [8, 785, 128]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:132, code: x = x.transpose(1, 2).reshape(B, N, C)  # [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C]
    view_282: "f32[8, 785, 8, 16]" = torch.ops.aten.view.default(view_281, [8, 785, 8, 16]);  view_281 = None
    permute_225: "f32[8, 8, 785, 16]" = torch.ops.aten.permute.default(view_282, [0, 2, 1, 3]);  view_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:131, code: x = self.scale * factor_att + crpe
    mul_221: "f32[8, 8, 785, 16]" = torch.ops.aten.mul.Tensor(permute_225, 0.25)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:87, code: EV_hat = F.pad(EV_hat, (0, 0, 1, 0, 0, 0))  # [B, h, N, Ch].
    constant_pad_nd_12: "f32[8, 8, 784, 16]" = torch.ops.aten.constant_pad_nd.default(permute_225, [0, 0, -1, 0, 0, 0]);  permute_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:86, code: EV_hat = q_img * conv_v_img
    mul_222: "f32[8, 8, 784, 16]" = torch.ops.aten.mul.Tensor(constant_pad_nd_12, slice_47);  slice_47 = None
    mul_223: "f32[8, 8, 784, 16]" = torch.ops.aten.mul.Tensor(constant_pad_nd_12, permute_42);  constant_pad_nd_12 = permute_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:84, code: conv_v_img = conv_v_img.reshape(B, num_heads, C, H * W).transpose(-1, -2)
    permute_226: "f32[8, 8, 16, 784]" = torch.ops.aten.permute.default(mul_222, [0, 1, 3, 2]);  mul_222 = None
    view_283: "f32[8, 128, 28, 28]" = torch.ops.aten.view.default(permute_226, [8, 128, 28, 28]);  permute_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:83, code: conv_v_img = torch.cat(conv_v_img_list, dim=1)
    slice_134: "f32[8, 32, 28, 28]" = torch.ops.aten.slice.Tensor(view_283, 1, 0, 32)
    slice_135: "f32[8, 48, 28, 28]" = torch.ops.aten.slice.Tensor(view_283, 1, 32, 80)
    slice_136: "f32[8, 48, 28, 28]" = torch.ops.aten.slice.Tensor(view_283, 1, 80, 128);  view_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(slice_136, getitem_41, primals_79, [48], [1, 1], [3, 3], [1, 1], False, [0, 0], 48, [True, True, True]);  slice_136 = getitem_41 = None
    getitem_144: "f32[8, 48, 28, 28]" = convolution_backward_18[0]
    getitem_145: "f32[48, 1, 7, 7]" = convolution_backward_18[1]
    getitem_146: "f32[48]" = convolution_backward_18[2];  convolution_backward_18 = None
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(slice_135, getitem_40, primals_77, [48], [1, 1], [2, 2], [1, 1], False, [0, 0], 48, [True, True, True]);  slice_135 = getitem_40 = None
    getitem_147: "f32[8, 48, 28, 28]" = convolution_backward_19[0]
    getitem_148: "f32[48, 1, 5, 5]" = convolution_backward_19[1]
    getitem_149: "f32[48]" = convolution_backward_19[2];  convolution_backward_19 = None
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(slice_134, getitem_39, primals_75, [32], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, True]);  slice_134 = getitem_39 = None
    getitem_150: "f32[8, 32, 28, 28]" = convolution_backward_20[0]
    getitem_151: "f32[32, 1, 3, 3]" = convolution_backward_20[1]
    getitem_152: "f32[32]" = convolution_backward_20[2];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:79, code: v_img_list = torch.split(v_img, self.channel_splits, dim=1)  # Split according to channels
    cat_28: "f32[8, 128, 28, 28]" = torch.ops.aten.cat.default([getitem_150, getitem_147, getitem_144], 1);  getitem_150 = getitem_147 = getitem_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:78, code: v_img = v_img.transpose(-1, -2).reshape(B, num_heads * C, H, W)
    view_284: "f32[8, 8, 16, 784]" = torch.ops.aten.view.default(cat_28, [8, 8, 16, 784]);  cat_28 = None
    permute_227: "f32[8, 8, 784, 16]" = torch.ops.aten.permute.default(view_284, [0, 1, 3, 2]);  view_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:76, code: v_img = v[:, :, 1:, :]  # [B, h, H*W, Ch]
    full_56: "f32[8, 8, 784, 16]" = torch.ops.aten.full.default([8, 8, 784, 16], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_55: "f32[8, 8, 784, 16]" = torch.ops.aten.slice_scatter.default(full_56, permute_227, 3, 0, 9223372036854775807);  full_56 = permute_227 = None
    full_57: "f32[8, 8, 785, 16]" = torch.ops.aten.full.default([8, 8, 785, 16], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_56: "f32[8, 8, 785, 16]" = torch.ops.aten.slice_scatter.default(full_57, slice_scatter_55, 2, 1, 9223372036854775807);  full_57 = slice_scatter_55 = None
    full_58: "f32[8, 8, 785, 16]" = torch.ops.aten.full.default([8, 8, 785, 16], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_57: "f32[8, 8, 785, 16]" = torch.ops.aten.slice_scatter.default(full_58, slice_scatter_56, 1, 0, 9223372036854775807);  full_58 = slice_scatter_56 = None
    full_59: "f32[8, 8, 785, 16]" = torch.ops.aten.full.default([8, 8, 785, 16], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_58: "f32[8, 8, 785, 16]" = torch.ops.aten.slice_scatter.default(full_59, slice_scatter_57, 0, 0, 9223372036854775807);  full_59 = slice_scatter_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:75, code: q_img = q[:, :, 1:, :]  # [B, h, H*W, Ch]
    full_60: "f32[8, 8, 784, 16]" = torch.ops.aten.full.default([8, 8, 784, 16], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_59: "f32[8, 8, 784, 16]" = torch.ops.aten.slice_scatter.default(full_60, mul_223, 3, 0, 9223372036854775807);  full_60 = mul_223 = None
    full_61: "f32[8, 8, 785, 16]" = torch.ops.aten.full.default([8, 8, 785, 16], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_60: "f32[8, 8, 785, 16]" = torch.ops.aten.slice_scatter.default(full_61, slice_scatter_59, 2, 1, 9223372036854775807);  full_61 = slice_scatter_59 = None
    full_62: "f32[8, 8, 785, 16]" = torch.ops.aten.full.default([8, 8, 785, 16], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_61: "f32[8, 8, 785, 16]" = torch.ops.aten.slice_scatter.default(full_62, slice_scatter_60, 1, 0, 9223372036854775807);  full_62 = slice_scatter_60 = None
    full_63: "f32[8, 8, 785, 16]" = torch.ops.aten.full.default([8, 8, 785, 16], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_62: "f32[8, 8, 785, 16]" = torch.ops.aten.slice_scatter.default(full_63, slice_scatter_61, 0, 0, 9223372036854775807);  full_63 = slice_scatter_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    clone_81: "f32[8, 8, 785, 16]" = torch.ops.aten.clone.default(mul_221, memory_format = torch.contiguous_format);  mul_221 = None
    view_285: "f32[64, 785, 16]" = torch.ops.aten.view.default(clone_81, [64, 785, 16]);  clone_81 = None
    permute_228: "f32[64, 16, 785]" = torch.ops.aten.permute.default(view_71, [0, 2, 1]);  view_71 = None
    bmm_32: "f32[64, 16, 16]" = torch.ops.aten.bmm.default(permute_228, view_285);  permute_228 = None
    permute_229: "f32[64, 16, 16]" = torch.ops.aten.permute.default(view_72, [0, 2, 1]);  view_72 = None
    bmm_33: "f32[64, 785, 16]" = torch.ops.aten.bmm.default(view_285, permute_229);  view_285 = permute_229 = None
    view_286: "f32[8, 8, 16, 16]" = torch.ops.aten.view.default(bmm_32, [8, 8, 16, 16]);  bmm_32 = None
    view_287: "f32[8, 8, 785, 16]" = torch.ops.aten.view.default(bmm_33, [8, 8, 785, 16]);  bmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    add_133: "f32[8, 8, 785, 16]" = torch.ops.aten.add.Tensor(slice_scatter_62, view_287);  slice_scatter_62 = view_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    view_288: "f32[64, 16, 16]" = torch.ops.aten.view.default(view_286, [64, 16, 16]);  view_286 = None
    permute_230: "f32[64, 785, 16]" = torch.ops.aten.permute.default(view_68, [0, 2, 1]);  view_68 = None
    bmm_34: "f32[64, 785, 16]" = torch.ops.aten.bmm.default(permute_230, view_288);  permute_230 = None
    permute_231: "f32[64, 16, 785]" = torch.ops.aten.permute.default(view_69, [0, 2, 1]);  view_69 = None
    bmm_35: "f32[64, 16, 785]" = torch.ops.aten.bmm.default(view_288, permute_231);  view_288 = permute_231 = None
    view_289: "f32[8, 8, 785, 16]" = torch.ops.aten.view.default(bmm_34, [8, 8, 785, 16]);  bmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    add_134: "f32[8, 8, 785, 16]" = torch.ops.aten.add.Tensor(slice_scatter_58, view_289);  slice_scatter_58 = view_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    view_290: "f32[8, 8, 16, 785]" = torch.ops.aten.view.default(bmm_35, [8, 8, 16, 785]);  bmm_35 = None
    permute_232: "f32[8, 8, 785, 16]" = torch.ops.aten.permute.default(view_290, [0, 1, 3, 2]);  view_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:123, code: k_softmax = k.softmax(dim=2)
    alias_12: "f32[8, 8, 785, 16]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    mul_224: "f32[8, 8, 785, 16]" = torch.ops.aten.mul.Tensor(permute_232, alias_12);  permute_232 = None
    sum_83: "f32[8, 8, 1, 16]" = torch.ops.aten.sum.dim_IntList(mul_224, [2], True)
    mul_225: "f32[8, 8, 785, 16]" = torch.ops.aten.mul.Tensor(alias_12, sum_83);  alias_12 = sum_83 = None
    sub_69: "f32[8, 8, 785, 16]" = torch.ops.aten.sub.Tensor(mul_224, mul_225);  mul_224 = mul_225 = None
    clone_82: "f32[8, 8, 785, 16]" = torch.ops.aten.clone.default(sub_69, memory_format = torch.contiguous_format);  sub_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:120, code: q, k, v = qkv.unbind(0)  # [B, h, N, Ch]
    cat_29: "f32[24, 8, 785, 16]" = torch.ops.aten.cat.default([add_133, clone_82, add_134]);  add_133 = clone_82 = add_134 = None
    view_291: "f32[3, 8, 8, 785, 16]" = torch.ops.aten.view.default(cat_29, [3, 8, 8, 785, 16]);  cat_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:119, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_233: "f32[8, 785, 3, 8, 16]" = torch.ops.aten.permute.default(view_291, [1, 3, 0, 2, 4]);  view_291 = None
    clone_83: "f32[8, 785, 3, 8, 16]" = torch.ops.aten.clone.default(permute_233, memory_format = torch.contiguous_format);  permute_233 = None
    view_292: "f32[8, 785, 384]" = torch.ops.aten.view.default(clone_83, [8, 785, 384]);  clone_83 = None
    view_293: "f32[6280, 384]" = torch.ops.aten.view.default(view_292, [6280, 384]);  view_292 = None
    permute_234: "f32[384, 128]" = torch.ops.aten.permute.default(permute_38, [1, 0]);  permute_38 = None
    mm_40: "f32[6280, 128]" = torch.ops.aten.mm.default(view_293, permute_234);  permute_234 = None
    permute_235: "f32[384, 6280]" = torch.ops.aten.permute.default(view_293, [1, 0])
    mm_41: "f32[384, 128]" = torch.ops.aten.mm.default(permute_235, view_65);  permute_235 = view_65 = None
    permute_236: "f32[128, 384]" = torch.ops.aten.permute.default(mm_41, [1, 0]);  mm_41 = None
    sum_84: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_293, [0], True);  view_293 = None
    view_294: "f32[384]" = torch.ops.aten.view.default(sum_84, [384]);  sum_84 = None
    permute_237: "f32[384, 128]" = torch.ops.aten.permute.default(permute_236, [1, 0]);  permute_236 = None
    view_295: "f32[8, 785, 128]" = torch.ops.aten.view.default(mm_40, [8, 785, 128]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_70: "f32[8, 785, 128]" = torch.ops.aten.sub.Tensor(cat_8, getitem_35);  cat_8 = getitem_35 = None
    mul_226: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_8);  sub_70 = None
    mul_227: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(view_295, primals_15);  primals_15 = None
    mul_228: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(mul_227, 128)
    sum_85: "f32[8, 785, 1]" = torch.ops.aten.sum.dim_IntList(mul_227, [2], True)
    mul_229: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(mul_227, mul_226);  mul_227 = None
    sum_86: "f32[8, 785, 1]" = torch.ops.aten.sum.dim_IntList(mul_229, [2], True);  mul_229 = None
    mul_230: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(mul_226, sum_86);  sum_86 = None
    sub_71: "f32[8, 785, 128]" = torch.ops.aten.sub.Tensor(mul_228, sum_85);  mul_228 = sum_85 = None
    sub_72: "f32[8, 785, 128]" = torch.ops.aten.sub.Tensor(sub_71, mul_230);  sub_71 = mul_230 = None
    div_20: "f32[8, 785, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 128);  rsqrt_8 = None
    mul_231: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(div_20, sub_72);  div_20 = sub_72 = None
    mul_232: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(view_295, mul_226);  mul_226 = None
    sum_87: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_232, [0, 1]);  mul_232 = None
    sum_88: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_295, [0, 1]);  view_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_135: "f32[8, 785, 128]" = torch.ops.aten.add.Tensor(add_132, mul_231);  add_132 = mul_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:163, code: x = torch.cat((cls_token, x), dim=1)
    slice_137: "f32[8, 1, 128]" = torch.ops.aten.slice.Tensor(add_135, 1, 0, 1)
    slice_138: "f32[8, 784, 128]" = torch.ops.aten.slice.Tensor(add_135, 1, 1, 785);  add_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:160, code: x = x.flatten(2).transpose(1, 2)
    permute_238: "f32[8, 128, 784]" = torch.ops.aten.permute.default(slice_138, [0, 2, 1]);  slice_138 = None
    view_296: "f32[8, 128, 28, 28]" = torch.ops.aten.view.default(permute_238, [8, 128, 28, 28]);  permute_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:159, code: x = self.proj(feat) + feat
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(view_296, view_63, primals_71, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 128, [True, True, True]);  view_63 = None
    getitem_153: "f32[8, 128, 28, 28]" = convolution_backward_21[0]
    getitem_154: "f32[128, 1, 3, 3]" = convolution_backward_21[1]
    getitem_155: "f32[128]" = convolution_backward_21[2];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:159, code: x = self.proj(feat) + feat
    add_136: "f32[8, 128, 28, 28]" = torch.ops.aten.add.Tensor(view_296, getitem_153);  view_296 = getitem_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:158, code: feat = img_tokens.transpose(1, 2).view(B, C, H, W)
    view_297: "f32[8, 128, 784]" = torch.ops.aten.view.default(add_136, [8, 128, 784]);  add_136 = None
    permute_239: "f32[8, 784, 128]" = torch.ops.aten.permute.default(view_297, [0, 2, 1]);  view_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    full_64: "f32[8, 785, 128]" = torch.ops.aten.full.default([8, 785, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_63: "f32[8, 785, 128]" = torch.ops.aten.slice_scatter.default(full_64, permute_239, 1, 1, 9223372036854775807);  full_64 = permute_239 = None
    full_65: "f32[8, 785, 128]" = torch.ops.aten.full.default([8, 785, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_64: "f32[8, 785, 128]" = torch.ops.aten.slice_scatter.default(full_65, slice_scatter_63, 0, 0, 9223372036854775807);  full_65 = slice_scatter_63 = None
    full_66: "f32[8, 785, 128]" = torch.ops.aten.full.default([8, 785, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_65: "f32[8, 785, 128]" = torch.ops.aten.slice_scatter.default(full_66, slice_137, 1, 0, 1);  full_66 = slice_137 = None
    full_67: "f32[8, 785, 128]" = torch.ops.aten.full.default([8, 785, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_66: "f32[8, 785, 128]" = torch.ops.aten.slice_scatter.default(full_67, slice_scatter_65, 0, 0, 9223372036854775807);  full_67 = slice_scatter_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    add_137: "f32[8, 785, 128]" = torch.ops.aten.add.Tensor(slice_scatter_64, slice_scatter_66);  slice_scatter_64 = slice_scatter_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_298: "f32[6280, 128]" = torch.ops.aten.view.default(add_137, [6280, 128])
    permute_240: "f32[128, 1024]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
    mm_42: "f32[6280, 1024]" = torch.ops.aten.mm.default(view_298, permute_240);  permute_240 = None
    permute_241: "f32[128, 6280]" = torch.ops.aten.permute.default(view_298, [1, 0])
    mm_43: "f32[128, 1024]" = torch.ops.aten.mm.default(permute_241, view_61);  permute_241 = view_61 = None
    permute_242: "f32[1024, 128]" = torch.ops.aten.permute.default(mm_43, [1, 0]);  mm_43 = None
    sum_89: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_298, [0], True);  view_298 = None
    view_299: "f32[128]" = torch.ops.aten.view.default(sum_89, [128]);  sum_89 = None
    permute_243: "f32[128, 1024]" = torch.ops.aten.permute.default(permute_242, [1, 0]);  permute_242 = None
    view_300: "f32[8, 785, 1024]" = torch.ops.aten.view.default(mm_42, [8, 785, 1024]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_233: "f32[8, 785, 1024]" = torch.ops.aten.mul.Tensor(view_60, 0.7071067811865476)
    erf_13: "f32[8, 785, 1024]" = torch.ops.aten.erf.default(mul_233);  mul_233 = None
    add_138: "f32[8, 785, 1024]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    mul_234: "f32[8, 785, 1024]" = torch.ops.aten.mul.Tensor(add_138, 0.5);  add_138 = None
    mul_235: "f32[8, 785, 1024]" = torch.ops.aten.mul.Tensor(view_60, view_60)
    mul_236: "f32[8, 785, 1024]" = torch.ops.aten.mul.Tensor(mul_235, -0.5);  mul_235 = None
    exp_13: "f32[8, 785, 1024]" = torch.ops.aten.exp.default(mul_236);  mul_236 = None
    mul_237: "f32[8, 785, 1024]" = torch.ops.aten.mul.Tensor(exp_13, 0.3989422804014327);  exp_13 = None
    mul_238: "f32[8, 785, 1024]" = torch.ops.aten.mul.Tensor(view_60, mul_237);  view_60 = mul_237 = None
    add_139: "f32[8, 785, 1024]" = torch.ops.aten.add.Tensor(mul_234, mul_238);  mul_234 = mul_238 = None
    mul_239: "f32[8, 785, 1024]" = torch.ops.aten.mul.Tensor(view_300, add_139);  view_300 = add_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_301: "f32[6280, 1024]" = torch.ops.aten.view.default(mul_239, [6280, 1024]);  mul_239 = None
    permute_244: "f32[1024, 128]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
    mm_44: "f32[6280, 128]" = torch.ops.aten.mm.default(view_301, permute_244);  permute_244 = None
    permute_245: "f32[1024, 6280]" = torch.ops.aten.permute.default(view_301, [1, 0])
    mm_45: "f32[1024, 128]" = torch.ops.aten.mm.default(permute_245, view_59);  permute_245 = view_59 = None
    permute_246: "f32[128, 1024]" = torch.ops.aten.permute.default(mm_45, [1, 0]);  mm_45 = None
    sum_90: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_301, [0], True);  view_301 = None
    view_302: "f32[1024]" = torch.ops.aten.view.default(sum_90, [1024]);  sum_90 = None
    permute_247: "f32[1024, 128]" = torch.ops.aten.permute.default(permute_246, [1, 0]);  permute_246 = None
    view_303: "f32[8, 785, 128]" = torch.ops.aten.view.default(mm_44, [8, 785, 128]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_73: "f32[8, 785, 128]" = torch.ops.aten.sub.Tensor(add_26, getitem_33);  add_26 = getitem_33 = None
    mul_240: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_7);  sub_73 = None
    mul_241: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(view_303, primals_13);  primals_13 = None
    mul_242: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(mul_241, 128)
    sum_91: "f32[8, 785, 1]" = torch.ops.aten.sum.dim_IntList(mul_241, [2], True)
    mul_243: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(mul_241, mul_240);  mul_241 = None
    sum_92: "f32[8, 785, 1]" = torch.ops.aten.sum.dim_IntList(mul_243, [2], True);  mul_243 = None
    mul_244: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(mul_240, sum_92);  sum_92 = None
    sub_74: "f32[8, 785, 128]" = torch.ops.aten.sub.Tensor(mul_242, sum_91);  mul_242 = sum_91 = None
    sub_75: "f32[8, 785, 128]" = torch.ops.aten.sub.Tensor(sub_74, mul_244);  sub_74 = mul_244 = None
    div_21: "f32[8, 785, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 128);  rsqrt_7 = None
    mul_245: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(div_21, sub_75);  div_21 = sub_75 = None
    mul_246: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(view_303, mul_240);  mul_240 = None
    sum_93: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_246, [0, 1]);  mul_246 = None
    sum_94: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_303, [0, 1]);  view_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_140: "f32[8, 785, 128]" = torch.ops.aten.add.Tensor(add_137, mul_245);  add_137 = mul_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:135, code: x = self.proj(x)
    view_304: "f32[6280, 128]" = torch.ops.aten.view.default(add_140, [6280, 128])
    permute_248: "f32[128, 128]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
    mm_46: "f32[6280, 128]" = torch.ops.aten.mm.default(view_304, permute_248);  permute_248 = None
    permute_249: "f32[128, 6280]" = torch.ops.aten.permute.default(view_304, [1, 0])
    mm_47: "f32[128, 128]" = torch.ops.aten.mm.default(permute_249, view_57);  permute_249 = view_57 = None
    permute_250: "f32[128, 128]" = torch.ops.aten.permute.default(mm_47, [1, 0]);  mm_47 = None
    sum_95: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_304, [0], True);  view_304 = None
    view_305: "f32[128]" = torch.ops.aten.view.default(sum_95, [128]);  sum_95 = None
    permute_251: "f32[128, 128]" = torch.ops.aten.permute.default(permute_250, [1, 0]);  permute_250 = None
    view_306: "f32[8, 785, 128]" = torch.ops.aten.view.default(mm_46, [8, 785, 128]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:132, code: x = x.transpose(1, 2).reshape(B, N, C)  # [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C]
    view_307: "f32[8, 785, 8, 16]" = torch.ops.aten.view.default(view_306, [8, 785, 8, 16]);  view_306 = None
    permute_252: "f32[8, 8, 785, 16]" = torch.ops.aten.permute.default(view_307, [0, 2, 1, 3]);  view_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:131, code: x = self.scale * factor_att + crpe
    mul_247: "f32[8, 8, 785, 16]" = torch.ops.aten.mul.Tensor(permute_252, 0.25)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:87, code: EV_hat = F.pad(EV_hat, (0, 0, 1, 0, 0, 0))  # [B, h, N, Ch].
    constant_pad_nd_13: "f32[8, 8, 784, 16]" = torch.ops.aten.constant_pad_nd.default(permute_252, [0, 0, -1, 0, 0, 0]);  permute_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:86, code: EV_hat = q_img * conv_v_img
    mul_248: "f32[8, 8, 784, 16]" = torch.ops.aten.mul.Tensor(constant_pad_nd_13, slice_35);  slice_35 = None
    mul_249: "f32[8, 8, 784, 16]" = torch.ops.aten.mul.Tensor(constant_pad_nd_13, permute_31);  constant_pad_nd_13 = permute_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:84, code: conv_v_img = conv_v_img.reshape(B, num_heads, C, H * W).transpose(-1, -2)
    permute_253: "f32[8, 8, 16, 784]" = torch.ops.aten.permute.default(mul_248, [0, 1, 3, 2]);  mul_248 = None
    view_308: "f32[8, 128, 28, 28]" = torch.ops.aten.view.default(permute_253, [8, 128, 28, 28]);  permute_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:83, code: conv_v_img = torch.cat(conv_v_img_list, dim=1)
    slice_139: "f32[8, 32, 28, 28]" = torch.ops.aten.slice.Tensor(view_308, 1, 0, 32)
    slice_140: "f32[8, 48, 28, 28]" = torch.ops.aten.slice.Tensor(view_308, 1, 32, 80)
    slice_141: "f32[8, 48, 28, 28]" = torch.ops.aten.slice.Tensor(view_308, 1, 80, 128);  view_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(slice_141, getitem_31, primals_79, [48], [1, 1], [3, 3], [1, 1], False, [0, 0], 48, [True, True, True]);  slice_141 = getitem_31 = primals_79 = None
    getitem_156: "f32[8, 48, 28, 28]" = convolution_backward_22[0]
    getitem_157: "f32[48, 1, 7, 7]" = convolution_backward_22[1]
    getitem_158: "f32[48]" = convolution_backward_22[2];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    add_141: "f32[48, 1, 7, 7]" = torch.ops.aten.add.Tensor(getitem_145, getitem_157);  getitem_145 = getitem_157 = None
    add_142: "f32[48]" = torch.ops.aten.add.Tensor(getitem_146, getitem_158);  getitem_146 = getitem_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(slice_140, getitem_30, primals_77, [48], [1, 1], [2, 2], [1, 1], False, [0, 0], 48, [True, True, True]);  slice_140 = getitem_30 = primals_77 = None
    getitem_159: "f32[8, 48, 28, 28]" = convolution_backward_23[0]
    getitem_160: "f32[48, 1, 5, 5]" = convolution_backward_23[1]
    getitem_161: "f32[48]" = convolution_backward_23[2];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    add_143: "f32[48, 1, 5, 5]" = torch.ops.aten.add.Tensor(getitem_148, getitem_160);  getitem_148 = getitem_160 = None
    add_144: "f32[48]" = torch.ops.aten.add.Tensor(getitem_149, getitem_161);  getitem_149 = getitem_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(slice_139, getitem_29, primals_75, [32], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, True]);  slice_139 = getitem_29 = primals_75 = None
    getitem_162: "f32[8, 32, 28, 28]" = convolution_backward_24[0]
    getitem_163: "f32[32, 1, 3, 3]" = convolution_backward_24[1]
    getitem_164: "f32[32]" = convolution_backward_24[2];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    add_145: "f32[32, 1, 3, 3]" = torch.ops.aten.add.Tensor(getitem_151, getitem_163);  getitem_151 = getitem_163 = None
    add_146: "f32[32]" = torch.ops.aten.add.Tensor(getitem_152, getitem_164);  getitem_152 = getitem_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:79, code: v_img_list = torch.split(v_img, self.channel_splits, dim=1)  # Split according to channels
    cat_30: "f32[8, 128, 28, 28]" = torch.ops.aten.cat.default([getitem_162, getitem_159, getitem_156], 1);  getitem_162 = getitem_159 = getitem_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:78, code: v_img = v_img.transpose(-1, -2).reshape(B, num_heads * C, H, W)
    view_309: "f32[8, 8, 16, 784]" = torch.ops.aten.view.default(cat_30, [8, 8, 16, 784]);  cat_30 = None
    permute_254: "f32[8, 8, 784, 16]" = torch.ops.aten.permute.default(view_309, [0, 1, 3, 2]);  view_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:76, code: v_img = v[:, :, 1:, :]  # [B, h, H*W, Ch]
    full_68: "f32[8, 8, 784, 16]" = torch.ops.aten.full.default([8, 8, 784, 16], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_67: "f32[8, 8, 784, 16]" = torch.ops.aten.slice_scatter.default(full_68, permute_254, 3, 0, 9223372036854775807);  full_68 = permute_254 = None
    full_69: "f32[8, 8, 785, 16]" = torch.ops.aten.full.default([8, 8, 785, 16], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_68: "f32[8, 8, 785, 16]" = torch.ops.aten.slice_scatter.default(full_69, slice_scatter_67, 2, 1, 9223372036854775807);  full_69 = slice_scatter_67 = None
    full_70: "f32[8, 8, 785, 16]" = torch.ops.aten.full.default([8, 8, 785, 16], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_69: "f32[8, 8, 785, 16]" = torch.ops.aten.slice_scatter.default(full_70, slice_scatter_68, 1, 0, 9223372036854775807);  full_70 = slice_scatter_68 = None
    full_71: "f32[8, 8, 785, 16]" = torch.ops.aten.full.default([8, 8, 785, 16], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_70: "f32[8, 8, 785, 16]" = torch.ops.aten.slice_scatter.default(full_71, slice_scatter_69, 0, 0, 9223372036854775807);  full_71 = slice_scatter_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:75, code: q_img = q[:, :, 1:, :]  # [B, h, H*W, Ch]
    full_72: "f32[8, 8, 784, 16]" = torch.ops.aten.full.default([8, 8, 784, 16], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_71: "f32[8, 8, 784, 16]" = torch.ops.aten.slice_scatter.default(full_72, mul_249, 3, 0, 9223372036854775807);  full_72 = mul_249 = None
    full_73: "f32[8, 8, 785, 16]" = torch.ops.aten.full.default([8, 8, 785, 16], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_72: "f32[8, 8, 785, 16]" = torch.ops.aten.slice_scatter.default(full_73, slice_scatter_71, 2, 1, 9223372036854775807);  full_73 = slice_scatter_71 = None
    full_74: "f32[8, 8, 785, 16]" = torch.ops.aten.full.default([8, 8, 785, 16], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_73: "f32[8, 8, 785, 16]" = torch.ops.aten.slice_scatter.default(full_74, slice_scatter_72, 1, 0, 9223372036854775807);  full_74 = slice_scatter_72 = None
    full_75: "f32[8, 8, 785, 16]" = torch.ops.aten.full.default([8, 8, 785, 16], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_74: "f32[8, 8, 785, 16]" = torch.ops.aten.slice_scatter.default(full_75, slice_scatter_73, 0, 0, 9223372036854775807);  full_75 = slice_scatter_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    clone_84: "f32[8, 8, 785, 16]" = torch.ops.aten.clone.default(mul_247, memory_format = torch.contiguous_format);  mul_247 = None
    view_310: "f32[64, 785, 16]" = torch.ops.aten.view.default(clone_84, [64, 785, 16]);  clone_84 = None
    permute_255: "f32[64, 16, 785]" = torch.ops.aten.permute.default(view_51, [0, 2, 1]);  view_51 = None
    bmm_36: "f32[64, 16, 16]" = torch.ops.aten.bmm.default(permute_255, view_310);  permute_255 = None
    permute_256: "f32[64, 16, 16]" = torch.ops.aten.permute.default(view_52, [0, 2, 1]);  view_52 = None
    bmm_37: "f32[64, 785, 16]" = torch.ops.aten.bmm.default(view_310, permute_256);  view_310 = permute_256 = None
    view_311: "f32[8, 8, 16, 16]" = torch.ops.aten.view.default(bmm_36, [8, 8, 16, 16]);  bmm_36 = None
    view_312: "f32[8, 8, 785, 16]" = torch.ops.aten.view.default(bmm_37, [8, 8, 785, 16]);  bmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    add_147: "f32[8, 8, 785, 16]" = torch.ops.aten.add.Tensor(slice_scatter_74, view_312);  slice_scatter_74 = view_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    view_313: "f32[64, 16, 16]" = torch.ops.aten.view.default(view_311, [64, 16, 16]);  view_311 = None
    permute_257: "f32[64, 785, 16]" = torch.ops.aten.permute.default(view_48, [0, 2, 1]);  view_48 = None
    bmm_38: "f32[64, 785, 16]" = torch.ops.aten.bmm.default(permute_257, view_313);  permute_257 = None
    permute_258: "f32[64, 16, 785]" = torch.ops.aten.permute.default(view_49, [0, 2, 1]);  view_49 = None
    bmm_39: "f32[64, 16, 785]" = torch.ops.aten.bmm.default(view_313, permute_258);  view_313 = permute_258 = None
    view_314: "f32[8, 8, 785, 16]" = torch.ops.aten.view.default(bmm_38, [8, 8, 785, 16]);  bmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    add_148: "f32[8, 8, 785, 16]" = torch.ops.aten.add.Tensor(slice_scatter_70, view_314);  slice_scatter_70 = view_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    view_315: "f32[8, 8, 16, 785]" = torch.ops.aten.view.default(bmm_39, [8, 8, 16, 785]);  bmm_39 = None
    permute_259: "f32[8, 8, 785, 16]" = torch.ops.aten.permute.default(view_315, [0, 1, 3, 2]);  view_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:123, code: k_softmax = k.softmax(dim=2)
    alias_13: "f32[8, 8, 785, 16]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    mul_250: "f32[8, 8, 785, 16]" = torch.ops.aten.mul.Tensor(permute_259, alias_13);  permute_259 = None
    sum_96: "f32[8, 8, 1, 16]" = torch.ops.aten.sum.dim_IntList(mul_250, [2], True)
    mul_251: "f32[8, 8, 785, 16]" = torch.ops.aten.mul.Tensor(alias_13, sum_96);  alias_13 = sum_96 = None
    sub_76: "f32[8, 8, 785, 16]" = torch.ops.aten.sub.Tensor(mul_250, mul_251);  mul_250 = mul_251 = None
    clone_85: "f32[8, 8, 785, 16]" = torch.ops.aten.clone.default(sub_76, memory_format = torch.contiguous_format);  sub_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:120, code: q, k, v = qkv.unbind(0)  # [B, h, N, Ch]
    cat_31: "f32[24, 8, 785, 16]" = torch.ops.aten.cat.default([add_147, clone_85, add_148]);  add_147 = clone_85 = add_148 = None
    view_316: "f32[3, 8, 8, 785, 16]" = torch.ops.aten.view.default(cat_31, [3, 8, 8, 785, 16]);  cat_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:119, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_260: "f32[8, 785, 3, 8, 16]" = torch.ops.aten.permute.default(view_316, [1, 3, 0, 2, 4]);  view_316 = None
    clone_86: "f32[8, 785, 3, 8, 16]" = torch.ops.aten.clone.default(permute_260, memory_format = torch.contiguous_format);  permute_260 = None
    view_317: "f32[8, 785, 384]" = torch.ops.aten.view.default(clone_86, [8, 785, 384]);  clone_86 = None
    view_318: "f32[6280, 384]" = torch.ops.aten.view.default(view_317, [6280, 384]);  view_317 = None
    permute_261: "f32[384, 128]" = torch.ops.aten.permute.default(permute_27, [1, 0]);  permute_27 = None
    mm_48: "f32[6280, 128]" = torch.ops.aten.mm.default(view_318, permute_261);  permute_261 = None
    permute_262: "f32[384, 6280]" = torch.ops.aten.permute.default(view_318, [1, 0])
    mm_49: "f32[384, 128]" = torch.ops.aten.mm.default(permute_262, view_45);  permute_262 = view_45 = None
    permute_263: "f32[128, 384]" = torch.ops.aten.permute.default(mm_49, [1, 0]);  mm_49 = None
    sum_97: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_318, [0], True);  view_318 = None
    view_319: "f32[384]" = torch.ops.aten.view.default(sum_97, [384]);  sum_97 = None
    permute_264: "f32[384, 128]" = torch.ops.aten.permute.default(permute_263, [1, 0]);  permute_263 = None
    view_320: "f32[8, 785, 128]" = torch.ops.aten.view.default(mm_48, [8, 785, 128]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_77: "f32[8, 785, 128]" = torch.ops.aten.sub.Tensor(cat_6, getitem_25);  cat_6 = getitem_25 = None
    mul_252: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(sub_77, rsqrt_6);  sub_77 = None
    mul_253: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(view_320, primals_11);  primals_11 = None
    mul_254: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(mul_253, 128)
    sum_98: "f32[8, 785, 1]" = torch.ops.aten.sum.dim_IntList(mul_253, [2], True)
    mul_255: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(mul_253, mul_252);  mul_253 = None
    sum_99: "f32[8, 785, 1]" = torch.ops.aten.sum.dim_IntList(mul_255, [2], True);  mul_255 = None
    mul_256: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(mul_252, sum_99);  sum_99 = None
    sub_78: "f32[8, 785, 128]" = torch.ops.aten.sub.Tensor(mul_254, sum_98);  mul_254 = sum_98 = None
    sub_79: "f32[8, 785, 128]" = torch.ops.aten.sub.Tensor(sub_78, mul_256);  sub_78 = mul_256 = None
    div_22: "f32[8, 785, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 128);  rsqrt_6 = None
    mul_257: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(div_22, sub_79);  div_22 = sub_79 = None
    mul_258: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(view_320, mul_252);  mul_252 = None
    sum_100: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_258, [0, 1]);  mul_258 = None
    sum_101: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_320, [0, 1]);  view_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_149: "f32[8, 785, 128]" = torch.ops.aten.add.Tensor(add_140, mul_257);  add_140 = mul_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:163, code: x = torch.cat((cls_token, x), dim=1)
    slice_142: "f32[8, 1, 128]" = torch.ops.aten.slice.Tensor(add_149, 1, 0, 1)
    slice_143: "f32[8, 784, 128]" = torch.ops.aten.slice.Tensor(add_149, 1, 1, 785);  add_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:160, code: x = x.flatten(2).transpose(1, 2)
    permute_265: "f32[8, 128, 784]" = torch.ops.aten.permute.default(slice_143, [0, 2, 1]);  slice_143 = None
    view_321: "f32[8, 128, 28, 28]" = torch.ops.aten.view.default(permute_265, [8, 128, 28, 28]);  permute_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:159, code: x = self.proj(feat) + feat
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(view_321, view_43, primals_71, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 128, [True, True, True]);  view_43 = primals_71 = None
    getitem_165: "f32[8, 128, 28, 28]" = convolution_backward_25[0]
    getitem_166: "f32[128, 1, 3, 3]" = convolution_backward_25[1]
    getitem_167: "f32[128]" = convolution_backward_25[2];  convolution_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:159, code: x = self.proj(feat) + feat
    add_150: "f32[8, 128, 28, 28]" = torch.ops.aten.add.Tensor(view_321, getitem_165);  view_321 = getitem_165 = None
    add_151: "f32[128, 1, 3, 3]" = torch.ops.aten.add.Tensor(getitem_154, getitem_166);  getitem_154 = getitem_166 = None
    add_152: "f32[128]" = torch.ops.aten.add.Tensor(getitem_155, getitem_167);  getitem_155 = getitem_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:158, code: feat = img_tokens.transpose(1, 2).view(B, C, H, W)
    view_322: "f32[8, 128, 784]" = torch.ops.aten.view.default(add_150, [8, 128, 784]);  add_150 = None
    permute_266: "f32[8, 784, 128]" = torch.ops.aten.permute.default(view_322, [0, 2, 1]);  view_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    full_76: "f32[8, 785, 128]" = torch.ops.aten.full.default([8, 785, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_75: "f32[8, 785, 128]" = torch.ops.aten.slice_scatter.default(full_76, permute_266, 1, 1, 9223372036854775807);  full_76 = permute_266 = None
    full_77: "f32[8, 785, 128]" = torch.ops.aten.full.default([8, 785, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_76: "f32[8, 785, 128]" = torch.ops.aten.slice_scatter.default(full_77, slice_scatter_75, 0, 0, 9223372036854775807);  full_77 = slice_scatter_75 = None
    full_78: "f32[8, 785, 128]" = torch.ops.aten.full.default([8, 785, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_77: "f32[8, 785, 128]" = torch.ops.aten.slice_scatter.default(full_78, slice_142, 1, 0, 1);  full_78 = slice_142 = None
    full_79: "f32[8, 785, 128]" = torch.ops.aten.full.default([8, 785, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_78: "f32[8, 785, 128]" = torch.ops.aten.slice_scatter.default(full_79, slice_scatter_77, 0, 0, 9223372036854775807);  full_79 = slice_scatter_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    add_153: "f32[8, 785, 128]" = torch.ops.aten.add.Tensor(slice_scatter_76, slice_scatter_78);  slice_scatter_76 = slice_scatter_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:678, code: x = torch.cat((cls_tokens, x), dim=1)
    slice_144: "f32[8, 1, 128]" = torch.ops.aten.slice.Tensor(add_153, 1, 0, 1)
    slice_145: "f32[8, 784, 128]" = torch.ops.aten.slice.Tensor(add_153, 1, 1, 785);  add_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:677, code: cls_tokens = cls_token.expand(x.shape[0], -1, -1)
    sum_102: "f32[1, 1, 128]" = torch.ops.aten.sum.dim_IntList(slice_144, [0], True);  slice_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:92, code: x = self.norm(x)
    clone_87: "f32[8, 784, 128]" = torch.ops.aten.clone.default(slice_145, memory_format = torch.contiguous_format);  slice_145 = None
    clone_88: "f32[8, 784, 128]" = torch.ops.aten.clone.default(permute_24, memory_format = torch.contiguous_format);  permute_24 = None
    sub_80: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(clone_88, getitem_23);  clone_88 = getitem_23 = None
    mul_259: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(sub_80, rsqrt_5);  sub_80 = None
    mul_260: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(clone_87, primals_69);  primals_69 = None
    mul_261: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_260, 128)
    sum_103: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_260, [2], True)
    mul_262: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_260, mul_259);  mul_260 = None
    sum_104: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_262, [2], True);  mul_262 = None
    mul_263: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_259, sum_104);  sum_104 = None
    sub_81: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(mul_261, sum_103);  mul_261 = sum_103 = None
    sub_82: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(sub_81, mul_263);  sub_81 = mul_263 = None
    div_23: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 128);  rsqrt_5 = None
    mul_264: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(div_23, sub_82);  div_23 = sub_82 = None
    mul_265: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(clone_87, mul_259);  mul_259 = None
    sum_105: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_265, [0, 1]);  mul_265 = None
    sum_106: "f32[128]" = torch.ops.aten.sum.dim_IntList(clone_87, [0, 1]);  clone_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:89, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    permute_267: "f32[8, 128, 784]" = torch.ops.aten.permute.default(mul_264, [0, 2, 1]);  mul_264 = None
    view_323: "f32[8, 128, 28, 28]" = torch.ops.aten.view.default(permute_267, [8, 128, 28, 28]);  permute_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(view_323, clone_15, primals_67, [128], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  view_323 = clone_15 = primals_67 = None
    getitem_168: "f32[8, 64, 56, 56]" = convolution_backward_26[0]
    getitem_169: "f32[128, 64, 2, 2]" = convolution_backward_26[1]
    getitem_170: "f32[128]" = convolution_backward_26[2];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:579, code: x1_nocls = remove_cls(x1).reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
    permute_268: "f32[8, 56, 56, 64]" = torch.ops.aten.permute.default(getitem_168, [0, 2, 3, 1]);  getitem_168 = None
    view_324: "f32[8, 3136, 64]" = torch.ops.aten.view.default(permute_268, [8, 3136, 64]);  permute_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:684, code: return x[:, 1:, :]
    full_80: "f32[8, 3136, 64]" = torch.ops.aten.full.default([8, 3136, 64], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_79: "f32[8, 3136, 64]" = torch.ops.aten.slice_scatter.default(full_80, view_324, 2, 0, 9223372036854775807);  full_80 = view_324 = None
    full_81: "f32[8, 3137, 64]" = torch.ops.aten.full.default([8, 3137, 64], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_80: "f32[8, 3137, 64]" = torch.ops.aten.slice_scatter.default(full_81, slice_scatter_79, 1, 1, 9223372036854775807);  full_81 = slice_scatter_79 = None
    full_82: "f32[8, 3137, 64]" = torch.ops.aten.full.default([8, 3137, 64], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_81: "f32[8, 3137, 64]" = torch.ops.aten.slice_scatter.default(full_82, slice_scatter_80, 0, 0, 9223372036854775807);  full_82 = slice_scatter_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_325: "f32[25096, 64]" = torch.ops.aten.view.default(slice_scatter_81, [25096, 64])
    permute_269: "f32[64, 512]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
    mm_50: "f32[25096, 512]" = torch.ops.aten.mm.default(view_325, permute_269);  permute_269 = None
    permute_270: "f32[64, 25096]" = torch.ops.aten.permute.default(view_325, [1, 0])
    mm_51: "f32[64, 512]" = torch.ops.aten.mm.default(permute_270, view_39);  permute_270 = view_39 = None
    permute_271: "f32[512, 64]" = torch.ops.aten.permute.default(mm_51, [1, 0]);  mm_51 = None
    sum_107: "f32[1, 64]" = torch.ops.aten.sum.dim_IntList(view_325, [0], True);  view_325 = None
    view_326: "f32[64]" = torch.ops.aten.view.default(sum_107, [64]);  sum_107 = None
    permute_272: "f32[64, 512]" = torch.ops.aten.permute.default(permute_271, [1, 0]);  permute_271 = None
    view_327: "f32[8, 3137, 512]" = torch.ops.aten.view.default(mm_50, [8, 3137, 512]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_266: "f32[8, 3137, 512]" = torch.ops.aten.mul.Tensor(view_38, 0.7071067811865476)
    erf_14: "f32[8, 3137, 512]" = torch.ops.aten.erf.default(mul_266);  mul_266 = None
    add_154: "f32[8, 3137, 512]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    mul_267: "f32[8, 3137, 512]" = torch.ops.aten.mul.Tensor(add_154, 0.5);  add_154 = None
    mul_268: "f32[8, 3137, 512]" = torch.ops.aten.mul.Tensor(view_38, view_38)
    mul_269: "f32[8, 3137, 512]" = torch.ops.aten.mul.Tensor(mul_268, -0.5);  mul_268 = None
    exp_14: "f32[8, 3137, 512]" = torch.ops.aten.exp.default(mul_269);  mul_269 = None
    mul_270: "f32[8, 3137, 512]" = torch.ops.aten.mul.Tensor(exp_14, 0.3989422804014327);  exp_14 = None
    mul_271: "f32[8, 3137, 512]" = torch.ops.aten.mul.Tensor(view_38, mul_270);  view_38 = mul_270 = None
    add_155: "f32[8, 3137, 512]" = torch.ops.aten.add.Tensor(mul_267, mul_271);  mul_267 = mul_271 = None
    mul_272: "f32[8, 3137, 512]" = torch.ops.aten.mul.Tensor(view_327, add_155);  view_327 = add_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_328: "f32[25096, 512]" = torch.ops.aten.view.default(mul_272, [25096, 512]);  mul_272 = None
    permute_273: "f32[512, 64]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
    mm_52: "f32[25096, 64]" = torch.ops.aten.mm.default(view_328, permute_273);  permute_273 = None
    permute_274: "f32[512, 25096]" = torch.ops.aten.permute.default(view_328, [1, 0])
    mm_53: "f32[512, 64]" = torch.ops.aten.mm.default(permute_274, view_37);  permute_274 = view_37 = None
    permute_275: "f32[64, 512]" = torch.ops.aten.permute.default(mm_53, [1, 0]);  mm_53 = None
    sum_108: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_328, [0], True);  view_328 = None
    view_329: "f32[512]" = torch.ops.aten.view.default(sum_108, [512]);  sum_108 = None
    permute_276: "f32[512, 64]" = torch.ops.aten.permute.default(permute_275, [1, 0]);  permute_275 = None
    view_330: "f32[8, 3137, 64]" = torch.ops.aten.view.default(mm_52, [8, 3137, 64]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_83: "f32[8, 3137, 64]" = torch.ops.aten.sub.Tensor(add_15, getitem_21);  add_15 = getitem_21 = None
    mul_273: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(sub_83, rsqrt_4);  sub_83 = None
    mul_274: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(view_330, primals_8);  primals_8 = None
    mul_275: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(mul_274, 64)
    sum_109: "f32[8, 3137, 1]" = torch.ops.aten.sum.dim_IntList(mul_274, [2], True)
    mul_276: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(mul_274, mul_273);  mul_274 = None
    sum_110: "f32[8, 3137, 1]" = torch.ops.aten.sum.dim_IntList(mul_276, [2], True);  mul_276 = None
    mul_277: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(mul_273, sum_110);  sum_110 = None
    sub_84: "f32[8, 3137, 64]" = torch.ops.aten.sub.Tensor(mul_275, sum_109);  mul_275 = sum_109 = None
    sub_85: "f32[8, 3137, 64]" = torch.ops.aten.sub.Tensor(sub_84, mul_277);  sub_84 = mul_277 = None
    div_24: "f32[8, 3137, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 64);  rsqrt_4 = None
    mul_278: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(div_24, sub_85);  div_24 = sub_85 = None
    mul_279: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(view_330, mul_273);  mul_273 = None
    sum_111: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_279, [0, 1]);  mul_279 = None
    sum_112: "f32[64]" = torch.ops.aten.sum.dim_IntList(view_330, [0, 1]);  view_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_156: "f32[8, 3137, 64]" = torch.ops.aten.add.Tensor(slice_scatter_81, mul_278);  slice_scatter_81 = mul_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:135, code: x = self.proj(x)
    view_331: "f32[25096, 64]" = torch.ops.aten.view.default(add_156, [25096, 64])
    permute_277: "f32[64, 64]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
    mm_54: "f32[25096, 64]" = torch.ops.aten.mm.default(view_331, permute_277);  permute_277 = None
    permute_278: "f32[64, 25096]" = torch.ops.aten.permute.default(view_331, [1, 0])
    mm_55: "f32[64, 64]" = torch.ops.aten.mm.default(permute_278, view_35);  permute_278 = view_35 = None
    permute_279: "f32[64, 64]" = torch.ops.aten.permute.default(mm_55, [1, 0]);  mm_55 = None
    sum_113: "f32[1, 64]" = torch.ops.aten.sum.dim_IntList(view_331, [0], True);  view_331 = None
    view_332: "f32[64]" = torch.ops.aten.view.default(sum_113, [64]);  sum_113 = None
    permute_280: "f32[64, 64]" = torch.ops.aten.permute.default(permute_279, [1, 0]);  permute_279 = None
    view_333: "f32[8, 3137, 64]" = torch.ops.aten.view.default(mm_54, [8, 3137, 64]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:132, code: x = x.transpose(1, 2).reshape(B, N, C)  # [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C]
    view_334: "f32[8, 3137, 8, 8]" = torch.ops.aten.view.default(view_333, [8, 3137, 8, 8]);  view_333 = None
    permute_281: "f32[8, 8, 3137, 8]" = torch.ops.aten.permute.default(view_334, [0, 2, 1, 3]);  view_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:131, code: x = self.scale * factor_att + crpe
    mul_280: "f32[8, 8, 3137, 8]" = torch.ops.aten.mul.Tensor(permute_281, 0.3535533905932738)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:87, code: EV_hat = F.pad(EV_hat, (0, 0, 1, 0, 0, 0))  # [B, h, N, Ch].
    constant_pad_nd_14: "f32[8, 8, 3136, 8]" = torch.ops.aten.constant_pad_nd.default(permute_281, [0, 0, -1, 0, 0, 0]);  permute_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:86, code: EV_hat = q_img * conv_v_img
    mul_281: "f32[8, 8, 3136, 8]" = torch.ops.aten.mul.Tensor(constant_pad_nd_14, slice_20);  slice_20 = None
    mul_282: "f32[8, 8, 3136, 8]" = torch.ops.aten.mul.Tensor(constant_pad_nd_14, permute_18);  constant_pad_nd_14 = permute_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:84, code: conv_v_img = conv_v_img.reshape(B, num_heads, C, H * W).transpose(-1, -2)
    permute_282: "f32[8, 8, 8, 3136]" = torch.ops.aten.permute.default(mul_281, [0, 1, 3, 2]);  mul_281 = None
    view_335: "f32[8, 64, 56, 56]" = torch.ops.aten.view.default(permute_282, [8, 64, 56, 56]);  permute_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:83, code: conv_v_img = torch.cat(conv_v_img_list, dim=1)
    slice_146: "f32[8, 16, 56, 56]" = torch.ops.aten.slice.Tensor(view_335, 1, 0, 16)
    slice_147: "f32[8, 24, 56, 56]" = torch.ops.aten.slice.Tensor(view_335, 1, 16, 40)
    slice_148: "f32[8, 24, 56, 56]" = torch.ops.aten.slice.Tensor(view_335, 1, 40, 64);  view_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(slice_148, getitem_19, primals_51, [24], [1, 1], [3, 3], [1, 1], False, [0, 0], 24, [True, True, True]);  slice_148 = getitem_19 = None
    getitem_171: "f32[8, 24, 56, 56]" = convolution_backward_27[0]
    getitem_172: "f32[24, 1, 7, 7]" = convolution_backward_27[1]
    getitem_173: "f32[24]" = convolution_backward_27[2];  convolution_backward_27 = None
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(slice_147, getitem_18, primals_49, [24], [1, 1], [2, 2], [1, 1], False, [0, 0], 24, [True, True, True]);  slice_147 = getitem_18 = None
    getitem_174: "f32[8, 24, 56, 56]" = convolution_backward_28[0]
    getitem_175: "f32[24, 1, 5, 5]" = convolution_backward_28[1]
    getitem_176: "f32[24]" = convolution_backward_28[2];  convolution_backward_28 = None
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(slice_146, getitem_17, primals_47, [16], [1, 1], [1, 1], [1, 1], False, [0, 0], 16, [True, True, True]);  slice_146 = getitem_17 = None
    getitem_177: "f32[8, 16, 56, 56]" = convolution_backward_29[0]
    getitem_178: "f32[16, 1, 3, 3]" = convolution_backward_29[1]
    getitem_179: "f32[16]" = convolution_backward_29[2];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:79, code: v_img_list = torch.split(v_img, self.channel_splits, dim=1)  # Split according to channels
    cat_32: "f32[8, 64, 56, 56]" = torch.ops.aten.cat.default([getitem_177, getitem_174, getitem_171], 1);  getitem_177 = getitem_174 = getitem_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:78, code: v_img = v_img.transpose(-1, -2).reshape(B, num_heads * C, H, W)
    view_336: "f32[8, 8, 8, 3136]" = torch.ops.aten.view.default(cat_32, [8, 8, 8, 3136]);  cat_32 = None
    permute_283: "f32[8, 8, 3136, 8]" = torch.ops.aten.permute.default(view_336, [0, 1, 3, 2]);  view_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:76, code: v_img = v[:, :, 1:, :]  # [B, h, H*W, Ch]
    full_83: "f32[8, 8, 3136, 8]" = torch.ops.aten.full.default([8, 8, 3136, 8], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_82: "f32[8, 8, 3136, 8]" = torch.ops.aten.slice_scatter.default(full_83, permute_283, 3, 0, 9223372036854775807);  full_83 = permute_283 = None
    full_84: "f32[8, 8, 3137, 8]" = torch.ops.aten.full.default([8, 8, 3137, 8], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_83: "f32[8, 8, 3137, 8]" = torch.ops.aten.slice_scatter.default(full_84, slice_scatter_82, 2, 1, 9223372036854775807);  full_84 = slice_scatter_82 = None
    full_85: "f32[8, 8, 3137, 8]" = torch.ops.aten.full.default([8, 8, 3137, 8], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_84: "f32[8, 8, 3137, 8]" = torch.ops.aten.slice_scatter.default(full_85, slice_scatter_83, 1, 0, 9223372036854775807);  full_85 = slice_scatter_83 = None
    full_86: "f32[8, 8, 3137, 8]" = torch.ops.aten.full.default([8, 8, 3137, 8], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_85: "f32[8, 8, 3137, 8]" = torch.ops.aten.slice_scatter.default(full_86, slice_scatter_84, 0, 0, 9223372036854775807);  full_86 = slice_scatter_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:75, code: q_img = q[:, :, 1:, :]  # [B, h, H*W, Ch]
    full_87: "f32[8, 8, 3136, 8]" = torch.ops.aten.full.default([8, 8, 3136, 8], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_86: "f32[8, 8, 3136, 8]" = torch.ops.aten.slice_scatter.default(full_87, mul_282, 3, 0, 9223372036854775807);  full_87 = mul_282 = None
    full_88: "f32[8, 8, 3137, 8]" = torch.ops.aten.full.default([8, 8, 3137, 8], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_87: "f32[8, 8, 3137, 8]" = torch.ops.aten.slice_scatter.default(full_88, slice_scatter_86, 2, 1, 9223372036854775807);  full_88 = slice_scatter_86 = None
    full_89: "f32[8, 8, 3137, 8]" = torch.ops.aten.full.default([8, 8, 3137, 8], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_88: "f32[8, 8, 3137, 8]" = torch.ops.aten.slice_scatter.default(full_89, slice_scatter_87, 1, 0, 9223372036854775807);  full_89 = slice_scatter_87 = None
    full_90: "f32[8, 8, 3137, 8]" = torch.ops.aten.full.default([8, 8, 3137, 8], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_89: "f32[8, 8, 3137, 8]" = torch.ops.aten.slice_scatter.default(full_90, slice_scatter_88, 0, 0, 9223372036854775807);  full_90 = slice_scatter_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    clone_89: "f32[8, 8, 3137, 8]" = torch.ops.aten.clone.default(mul_280, memory_format = torch.contiguous_format);  mul_280 = None
    view_337: "f32[64, 3137, 8]" = torch.ops.aten.view.default(clone_89, [64, 3137, 8]);  clone_89 = None
    permute_284: "f32[64, 8, 3137]" = torch.ops.aten.permute.default(view_29, [0, 2, 1]);  view_29 = None
    bmm_40: "f32[64, 8, 8]" = torch.ops.aten.bmm.default(permute_284, view_337);  permute_284 = None
    permute_285: "f32[64, 8, 8]" = torch.ops.aten.permute.default(view_30, [0, 2, 1]);  view_30 = None
    bmm_41: "f32[64, 3137, 8]" = torch.ops.aten.bmm.default(view_337, permute_285);  view_337 = permute_285 = None
    view_338: "f32[8, 8, 8, 8]" = torch.ops.aten.view.default(bmm_40, [8, 8, 8, 8]);  bmm_40 = None
    view_339: "f32[8, 8, 3137, 8]" = torch.ops.aten.view.default(bmm_41, [8, 8, 3137, 8]);  bmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    add_157: "f32[8, 8, 3137, 8]" = torch.ops.aten.add.Tensor(slice_scatter_89, view_339);  slice_scatter_89 = view_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    view_340: "f32[64, 8, 8]" = torch.ops.aten.view.default(view_338, [64, 8, 8]);  view_338 = None
    permute_286: "f32[64, 3137, 8]" = torch.ops.aten.permute.default(view_26, [0, 2, 1]);  view_26 = None
    bmm_42: "f32[64, 3137, 8]" = torch.ops.aten.bmm.default(permute_286, view_340);  permute_286 = None
    permute_287: "f32[64, 8, 3137]" = torch.ops.aten.permute.default(view_27, [0, 2, 1]);  view_27 = None
    bmm_43: "f32[64, 8, 3137]" = torch.ops.aten.bmm.default(view_340, permute_287);  view_340 = permute_287 = None
    view_341: "f32[8, 8, 3137, 8]" = torch.ops.aten.view.default(bmm_42, [8, 8, 3137, 8]);  bmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    add_158: "f32[8, 8, 3137, 8]" = torch.ops.aten.add.Tensor(slice_scatter_85, view_341);  slice_scatter_85 = view_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    view_342: "f32[8, 8, 8, 3137]" = torch.ops.aten.view.default(bmm_43, [8, 8, 8, 3137]);  bmm_43 = None
    permute_288: "f32[8, 8, 3137, 8]" = torch.ops.aten.permute.default(view_342, [0, 1, 3, 2]);  view_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:123, code: k_softmax = k.softmax(dim=2)
    alias_14: "f32[8, 8, 3137, 8]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    mul_283: "f32[8, 8, 3137, 8]" = torch.ops.aten.mul.Tensor(permute_288, alias_14);  permute_288 = None
    sum_114: "f32[8, 8, 1, 8]" = torch.ops.aten.sum.dim_IntList(mul_283, [2], True)
    mul_284: "f32[8, 8, 3137, 8]" = torch.ops.aten.mul.Tensor(alias_14, sum_114);  alias_14 = sum_114 = None
    sub_86: "f32[8, 8, 3137, 8]" = torch.ops.aten.sub.Tensor(mul_283, mul_284);  mul_283 = mul_284 = None
    clone_90: "f32[8, 8, 3137, 8]" = torch.ops.aten.clone.default(sub_86, memory_format = torch.contiguous_format);  sub_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:120, code: q, k, v = qkv.unbind(0)  # [B, h, N, Ch]
    cat_33: "f32[24, 8, 3137, 8]" = torch.ops.aten.cat.default([add_157, clone_90, add_158]);  add_157 = clone_90 = add_158 = None
    view_343: "f32[3, 8, 8, 3137, 8]" = torch.ops.aten.view.default(cat_33, [3, 8, 8, 3137, 8]);  cat_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:119, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_289: "f32[8, 3137, 3, 8, 8]" = torch.ops.aten.permute.default(view_343, [1, 3, 0, 2, 4]);  view_343 = None
    clone_91: "f32[8, 3137, 3, 8, 8]" = torch.ops.aten.clone.default(permute_289, memory_format = torch.contiguous_format);  permute_289 = None
    view_344: "f32[8, 3137, 192]" = torch.ops.aten.view.default(clone_91, [8, 3137, 192]);  clone_91 = None
    view_345: "f32[25096, 192]" = torch.ops.aten.view.default(view_344, [25096, 192]);  view_344 = None
    permute_290: "f32[192, 64]" = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None
    mm_56: "f32[25096, 64]" = torch.ops.aten.mm.default(view_345, permute_290);  permute_290 = None
    permute_291: "f32[192, 25096]" = torch.ops.aten.permute.default(view_345, [1, 0])
    mm_57: "f32[192, 64]" = torch.ops.aten.mm.default(permute_291, view_23);  permute_291 = view_23 = None
    permute_292: "f32[64, 192]" = torch.ops.aten.permute.default(mm_57, [1, 0]);  mm_57 = None
    sum_115: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_345, [0], True);  view_345 = None
    view_346: "f32[192]" = torch.ops.aten.view.default(sum_115, [192]);  sum_115 = None
    permute_293: "f32[192, 64]" = torch.ops.aten.permute.default(permute_292, [1, 0]);  permute_292 = None
    view_347: "f32[8, 3137, 64]" = torch.ops.aten.view.default(mm_56, [8, 3137, 64]);  mm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_87: "f32[8, 3137, 64]" = torch.ops.aten.sub.Tensor(cat_3, getitem_13);  cat_3 = getitem_13 = None
    mul_285: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(sub_87, rsqrt_3);  sub_87 = None
    mul_286: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(view_347, primals_6);  primals_6 = None
    mul_287: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(mul_286, 64)
    sum_116: "f32[8, 3137, 1]" = torch.ops.aten.sum.dim_IntList(mul_286, [2], True)
    mul_288: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(mul_286, mul_285);  mul_286 = None
    sum_117: "f32[8, 3137, 1]" = torch.ops.aten.sum.dim_IntList(mul_288, [2], True);  mul_288 = None
    mul_289: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(mul_285, sum_117);  sum_117 = None
    sub_88: "f32[8, 3137, 64]" = torch.ops.aten.sub.Tensor(mul_287, sum_116);  mul_287 = sum_116 = None
    sub_89: "f32[8, 3137, 64]" = torch.ops.aten.sub.Tensor(sub_88, mul_289);  sub_88 = mul_289 = None
    div_25: "f32[8, 3137, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 64);  rsqrt_3 = None
    mul_290: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(div_25, sub_89);  div_25 = sub_89 = None
    mul_291: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(view_347, mul_285);  mul_285 = None
    sum_118: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_291, [0, 1]);  mul_291 = None
    sum_119: "f32[64]" = torch.ops.aten.sum.dim_IntList(view_347, [0, 1]);  view_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_159: "f32[8, 3137, 64]" = torch.ops.aten.add.Tensor(add_156, mul_290);  add_156 = mul_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:163, code: x = torch.cat((cls_token, x), dim=1)
    slice_149: "f32[8, 1, 64]" = torch.ops.aten.slice.Tensor(add_159, 1, 0, 1)
    slice_150: "f32[8, 3136, 64]" = torch.ops.aten.slice.Tensor(add_159, 1, 1, 3137);  add_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:160, code: x = x.flatten(2).transpose(1, 2)
    permute_294: "f32[8, 64, 3136]" = torch.ops.aten.permute.default(slice_150, [0, 2, 1]);  slice_150 = None
    view_348: "f32[8, 64, 56, 56]" = torch.ops.aten.view.default(permute_294, [8, 64, 56, 56]);  permute_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:159, code: x = self.proj(feat) + feat
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(view_348, view_21, primals_43, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 64, [True, True, True]);  view_21 = None
    getitem_180: "f32[8, 64, 56, 56]" = convolution_backward_30[0]
    getitem_181: "f32[64, 1, 3, 3]" = convolution_backward_30[1]
    getitem_182: "f32[64]" = convolution_backward_30[2];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:159, code: x = self.proj(feat) + feat
    add_160: "f32[8, 64, 56, 56]" = torch.ops.aten.add.Tensor(view_348, getitem_180);  view_348 = getitem_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:158, code: feat = img_tokens.transpose(1, 2).view(B, C, H, W)
    view_349: "f32[8, 64, 3136]" = torch.ops.aten.view.default(add_160, [8, 64, 3136]);  add_160 = None
    permute_295: "f32[8, 3136, 64]" = torch.ops.aten.permute.default(view_349, [0, 2, 1]);  view_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    full_91: "f32[8, 3137, 64]" = torch.ops.aten.full.default([8, 3137, 64], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_90: "f32[8, 3137, 64]" = torch.ops.aten.slice_scatter.default(full_91, permute_295, 1, 1, 9223372036854775807);  full_91 = permute_295 = None
    full_92: "f32[8, 3137, 64]" = torch.ops.aten.full.default([8, 3137, 64], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_91: "f32[8, 3137, 64]" = torch.ops.aten.slice_scatter.default(full_92, slice_scatter_90, 0, 0, 9223372036854775807);  full_92 = slice_scatter_90 = None
    full_93: "f32[8, 3137, 64]" = torch.ops.aten.full.default([8, 3137, 64], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_92: "f32[8, 3137, 64]" = torch.ops.aten.slice_scatter.default(full_93, slice_149, 1, 0, 1);  full_93 = slice_149 = None
    full_94: "f32[8, 3137, 64]" = torch.ops.aten.full.default([8, 3137, 64], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_93: "f32[8, 3137, 64]" = torch.ops.aten.slice_scatter.default(full_94, slice_scatter_92, 0, 0, 9223372036854775807);  full_94 = slice_scatter_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    add_161: "f32[8, 3137, 64]" = torch.ops.aten.add.Tensor(slice_scatter_91, slice_scatter_93);  slice_scatter_91 = slice_scatter_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_350: "f32[25096, 64]" = torch.ops.aten.view.default(add_161, [25096, 64])
    permute_296: "f32[64, 512]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
    mm_58: "f32[25096, 512]" = torch.ops.aten.mm.default(view_350, permute_296);  permute_296 = None
    permute_297: "f32[64, 25096]" = torch.ops.aten.permute.default(view_350, [1, 0])
    mm_59: "f32[64, 512]" = torch.ops.aten.mm.default(permute_297, view_19);  permute_297 = view_19 = None
    permute_298: "f32[512, 64]" = torch.ops.aten.permute.default(mm_59, [1, 0]);  mm_59 = None
    sum_120: "f32[1, 64]" = torch.ops.aten.sum.dim_IntList(view_350, [0], True);  view_350 = None
    view_351: "f32[64]" = torch.ops.aten.view.default(sum_120, [64]);  sum_120 = None
    permute_299: "f32[64, 512]" = torch.ops.aten.permute.default(permute_298, [1, 0]);  permute_298 = None
    view_352: "f32[8, 3137, 512]" = torch.ops.aten.view.default(mm_58, [8, 3137, 512]);  mm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_292: "f32[8, 3137, 512]" = torch.ops.aten.mul.Tensor(view_18, 0.7071067811865476)
    erf_15: "f32[8, 3137, 512]" = torch.ops.aten.erf.default(mul_292);  mul_292 = None
    add_162: "f32[8, 3137, 512]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    mul_293: "f32[8, 3137, 512]" = torch.ops.aten.mul.Tensor(add_162, 0.5);  add_162 = None
    mul_294: "f32[8, 3137, 512]" = torch.ops.aten.mul.Tensor(view_18, view_18)
    mul_295: "f32[8, 3137, 512]" = torch.ops.aten.mul.Tensor(mul_294, -0.5);  mul_294 = None
    exp_15: "f32[8, 3137, 512]" = torch.ops.aten.exp.default(mul_295);  mul_295 = None
    mul_296: "f32[8, 3137, 512]" = torch.ops.aten.mul.Tensor(exp_15, 0.3989422804014327);  exp_15 = None
    mul_297: "f32[8, 3137, 512]" = torch.ops.aten.mul.Tensor(view_18, mul_296);  view_18 = mul_296 = None
    add_163: "f32[8, 3137, 512]" = torch.ops.aten.add.Tensor(mul_293, mul_297);  mul_293 = mul_297 = None
    mul_298: "f32[8, 3137, 512]" = torch.ops.aten.mul.Tensor(view_352, add_163);  view_352 = add_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_353: "f32[25096, 512]" = torch.ops.aten.view.default(mul_298, [25096, 512]);  mul_298 = None
    permute_300: "f32[512, 64]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    mm_60: "f32[25096, 64]" = torch.ops.aten.mm.default(view_353, permute_300);  permute_300 = None
    permute_301: "f32[512, 25096]" = torch.ops.aten.permute.default(view_353, [1, 0])
    mm_61: "f32[512, 64]" = torch.ops.aten.mm.default(permute_301, view_17);  permute_301 = view_17 = None
    permute_302: "f32[64, 512]" = torch.ops.aten.permute.default(mm_61, [1, 0]);  mm_61 = None
    sum_121: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_353, [0], True);  view_353 = None
    view_354: "f32[512]" = torch.ops.aten.view.default(sum_121, [512]);  sum_121 = None
    permute_303: "f32[512, 64]" = torch.ops.aten.permute.default(permute_302, [1, 0]);  permute_302 = None
    view_355: "f32[8, 3137, 64]" = torch.ops.aten.view.default(mm_60, [8, 3137, 64]);  mm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_90: "f32[8, 3137, 64]" = torch.ops.aten.sub.Tensor(add_6, getitem_11);  add_6 = getitem_11 = None
    mul_299: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(sub_90, rsqrt_2);  sub_90 = None
    mul_300: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(view_355, primals_4);  primals_4 = None
    mul_301: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(mul_300, 64)
    sum_122: "f32[8, 3137, 1]" = torch.ops.aten.sum.dim_IntList(mul_300, [2], True)
    mul_302: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(mul_300, mul_299);  mul_300 = None
    sum_123: "f32[8, 3137, 1]" = torch.ops.aten.sum.dim_IntList(mul_302, [2], True);  mul_302 = None
    mul_303: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(mul_299, sum_123);  sum_123 = None
    sub_91: "f32[8, 3137, 64]" = torch.ops.aten.sub.Tensor(mul_301, sum_122);  mul_301 = sum_122 = None
    sub_92: "f32[8, 3137, 64]" = torch.ops.aten.sub.Tensor(sub_91, mul_303);  sub_91 = mul_303 = None
    div_26: "f32[8, 3137, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 64);  rsqrt_2 = None
    mul_304: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(div_26, sub_92);  div_26 = sub_92 = None
    mul_305: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(view_355, mul_299);  mul_299 = None
    sum_124: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_305, [0, 1]);  mul_305 = None
    sum_125: "f32[64]" = torch.ops.aten.sum.dim_IntList(view_355, [0, 1]);  view_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_164: "f32[8, 3137, 64]" = torch.ops.aten.add.Tensor(add_161, mul_304);  add_161 = mul_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:135, code: x = self.proj(x)
    view_356: "f32[25096, 64]" = torch.ops.aten.view.default(add_164, [25096, 64])
    permute_304: "f32[64, 64]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
    mm_62: "f32[25096, 64]" = torch.ops.aten.mm.default(view_356, permute_304);  permute_304 = None
    permute_305: "f32[64, 25096]" = torch.ops.aten.permute.default(view_356, [1, 0])
    mm_63: "f32[64, 64]" = torch.ops.aten.mm.default(permute_305, view_15);  permute_305 = view_15 = None
    permute_306: "f32[64, 64]" = torch.ops.aten.permute.default(mm_63, [1, 0]);  mm_63 = None
    sum_126: "f32[1, 64]" = torch.ops.aten.sum.dim_IntList(view_356, [0], True);  view_356 = None
    view_357: "f32[64]" = torch.ops.aten.view.default(sum_126, [64]);  sum_126 = None
    permute_307: "f32[64, 64]" = torch.ops.aten.permute.default(permute_306, [1, 0]);  permute_306 = None
    view_358: "f32[8, 3137, 64]" = torch.ops.aten.view.default(mm_62, [8, 3137, 64]);  mm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:132, code: x = x.transpose(1, 2).reshape(B, N, C)  # [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C]
    view_359: "f32[8, 3137, 8, 8]" = torch.ops.aten.view.default(view_358, [8, 3137, 8, 8]);  view_358 = None
    permute_308: "f32[8, 8, 3137, 8]" = torch.ops.aten.permute.default(view_359, [0, 2, 1, 3]);  view_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:131, code: x = self.scale * factor_att + crpe
    mul_306: "f32[8, 8, 3137, 8]" = torch.ops.aten.mul.Tensor(permute_308, 0.3535533905932738)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:87, code: EV_hat = F.pad(EV_hat, (0, 0, 1, 0, 0, 0))  # [B, h, N, Ch].
    constant_pad_nd_15: "f32[8, 8, 3136, 8]" = torch.ops.aten.constant_pad_nd.default(permute_308, [0, 0, -1, 0, 0, 0]);  permute_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:86, code: EV_hat = q_img * conv_v_img
    mul_307: "f32[8, 8, 3136, 8]" = torch.ops.aten.mul.Tensor(constant_pad_nd_15, slice_8);  slice_8 = None
    mul_308: "f32[8, 8, 3136, 8]" = torch.ops.aten.mul.Tensor(constant_pad_nd_15, permute_7);  constant_pad_nd_15 = permute_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:84, code: conv_v_img = conv_v_img.reshape(B, num_heads, C, H * W).transpose(-1, -2)
    permute_309: "f32[8, 8, 8, 3136]" = torch.ops.aten.permute.default(mul_307, [0, 1, 3, 2]);  mul_307 = None
    view_360: "f32[8, 64, 56, 56]" = torch.ops.aten.view.default(permute_309, [8, 64, 56, 56]);  permute_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:83, code: conv_v_img = torch.cat(conv_v_img_list, dim=1)
    slice_151: "f32[8, 16, 56, 56]" = torch.ops.aten.slice.Tensor(view_360, 1, 0, 16)
    slice_152: "f32[8, 24, 56, 56]" = torch.ops.aten.slice.Tensor(view_360, 1, 16, 40)
    slice_153: "f32[8, 24, 56, 56]" = torch.ops.aten.slice.Tensor(view_360, 1, 40, 64);  view_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(slice_153, getitem_9, primals_51, [24], [1, 1], [3, 3], [1, 1], False, [0, 0], 24, [True, True, True]);  slice_153 = getitem_9 = primals_51 = None
    getitem_183: "f32[8, 24, 56, 56]" = convolution_backward_31[0]
    getitem_184: "f32[24, 1, 7, 7]" = convolution_backward_31[1]
    getitem_185: "f32[24]" = convolution_backward_31[2];  convolution_backward_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    add_165: "f32[24, 1, 7, 7]" = torch.ops.aten.add.Tensor(getitem_172, getitem_184);  getitem_172 = getitem_184 = None
    add_166: "f32[24]" = torch.ops.aten.add.Tensor(getitem_173, getitem_185);  getitem_173 = getitem_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(slice_152, getitem_8, primals_49, [24], [1, 1], [2, 2], [1, 1], False, [0, 0], 24, [True, True, True]);  slice_152 = getitem_8 = primals_49 = None
    getitem_186: "f32[8, 24, 56, 56]" = convolution_backward_32[0]
    getitem_187: "f32[24, 1, 5, 5]" = convolution_backward_32[1]
    getitem_188: "f32[24]" = convolution_backward_32[2];  convolution_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    add_167: "f32[24, 1, 5, 5]" = torch.ops.aten.add.Tensor(getitem_175, getitem_187);  getitem_175 = getitem_187 = None
    add_168: "f32[24]" = torch.ops.aten.add.Tensor(getitem_176, getitem_188);  getitem_176 = getitem_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(slice_151, getitem_7, primals_47, [16], [1, 1], [1, 1], [1, 1], False, [0, 0], 16, [True, True, True]);  slice_151 = getitem_7 = primals_47 = None
    getitem_189: "f32[8, 16, 56, 56]" = convolution_backward_33[0]
    getitem_190: "f32[16, 1, 3, 3]" = convolution_backward_33[1]
    getitem_191: "f32[16]" = convolution_backward_33[2];  convolution_backward_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    add_169: "f32[16, 1, 3, 3]" = torch.ops.aten.add.Tensor(getitem_178, getitem_190);  getitem_178 = getitem_190 = None
    add_170: "f32[16]" = torch.ops.aten.add.Tensor(getitem_179, getitem_191);  getitem_179 = getitem_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:79, code: v_img_list = torch.split(v_img, self.channel_splits, dim=1)  # Split according to channels
    cat_34: "f32[8, 64, 56, 56]" = torch.ops.aten.cat.default([getitem_189, getitem_186, getitem_183], 1);  getitem_189 = getitem_186 = getitem_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:78, code: v_img = v_img.transpose(-1, -2).reshape(B, num_heads * C, H, W)
    view_361: "f32[8, 8, 8, 3136]" = torch.ops.aten.view.default(cat_34, [8, 8, 8, 3136]);  cat_34 = None
    permute_310: "f32[8, 8, 3136, 8]" = torch.ops.aten.permute.default(view_361, [0, 1, 3, 2]);  view_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:76, code: v_img = v[:, :, 1:, :]  # [B, h, H*W, Ch]
    full_95: "f32[8, 8, 3136, 8]" = torch.ops.aten.full.default([8, 8, 3136, 8], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_94: "f32[8, 8, 3136, 8]" = torch.ops.aten.slice_scatter.default(full_95, permute_310, 3, 0, 9223372036854775807);  full_95 = permute_310 = None
    full_96: "f32[8, 8, 3137, 8]" = torch.ops.aten.full.default([8, 8, 3137, 8], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_95: "f32[8, 8, 3137, 8]" = torch.ops.aten.slice_scatter.default(full_96, slice_scatter_94, 2, 1, 9223372036854775807);  full_96 = slice_scatter_94 = None
    full_97: "f32[8, 8, 3137, 8]" = torch.ops.aten.full.default([8, 8, 3137, 8], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_96: "f32[8, 8, 3137, 8]" = torch.ops.aten.slice_scatter.default(full_97, slice_scatter_95, 1, 0, 9223372036854775807);  full_97 = slice_scatter_95 = None
    full_98: "f32[8, 8, 3137, 8]" = torch.ops.aten.full.default([8, 8, 3137, 8], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_97: "f32[8, 8, 3137, 8]" = torch.ops.aten.slice_scatter.default(full_98, slice_scatter_96, 0, 0, 9223372036854775807);  full_98 = slice_scatter_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:75, code: q_img = q[:, :, 1:, :]  # [B, h, H*W, Ch]
    full_99: "f32[8, 8, 3136, 8]" = torch.ops.aten.full.default([8, 8, 3136, 8], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_98: "f32[8, 8, 3136, 8]" = torch.ops.aten.slice_scatter.default(full_99, mul_308, 3, 0, 9223372036854775807);  full_99 = mul_308 = None
    full_100: "f32[8, 8, 3137, 8]" = torch.ops.aten.full.default([8, 8, 3137, 8], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_99: "f32[8, 8, 3137, 8]" = torch.ops.aten.slice_scatter.default(full_100, slice_scatter_98, 2, 1, 9223372036854775807);  full_100 = slice_scatter_98 = None
    full_101: "f32[8, 8, 3137, 8]" = torch.ops.aten.full.default([8, 8, 3137, 8], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_100: "f32[8, 8, 3137, 8]" = torch.ops.aten.slice_scatter.default(full_101, slice_scatter_99, 1, 0, 9223372036854775807);  full_101 = slice_scatter_99 = None
    full_102: "f32[8, 8, 3137, 8]" = torch.ops.aten.full.default([8, 8, 3137, 8], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_101: "f32[8, 8, 3137, 8]" = torch.ops.aten.slice_scatter.default(full_102, slice_scatter_100, 0, 0, 9223372036854775807);  full_102 = slice_scatter_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    clone_92: "f32[8, 8, 3137, 8]" = torch.ops.aten.clone.default(mul_306, memory_format = torch.contiguous_format);  mul_306 = None
    view_362: "f32[64, 3137, 8]" = torch.ops.aten.view.default(clone_92, [64, 3137, 8]);  clone_92 = None
    permute_311: "f32[64, 8, 3137]" = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
    bmm_44: "f32[64, 8, 8]" = torch.ops.aten.bmm.default(permute_311, view_362);  permute_311 = None
    permute_312: "f32[64, 8, 8]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
    bmm_45: "f32[64, 3137, 8]" = torch.ops.aten.bmm.default(view_362, permute_312);  view_362 = permute_312 = None
    view_363: "f32[8, 8, 8, 8]" = torch.ops.aten.view.default(bmm_44, [8, 8, 8, 8]);  bmm_44 = None
    view_364: "f32[8, 8, 3137, 8]" = torch.ops.aten.view.default(bmm_45, [8, 8, 3137, 8]);  bmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    add_171: "f32[8, 8, 3137, 8]" = torch.ops.aten.add.Tensor(slice_scatter_101, view_364);  slice_scatter_101 = view_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    view_365: "f32[64, 8, 8]" = torch.ops.aten.view.default(view_363, [64, 8, 8]);  view_363 = None
    permute_313: "f32[64, 3137, 8]" = torch.ops.aten.permute.default(view_6, [0, 2, 1]);  view_6 = None
    bmm_46: "f32[64, 3137, 8]" = torch.ops.aten.bmm.default(permute_313, view_365);  permute_313 = None
    permute_314: "f32[64, 8, 3137]" = torch.ops.aten.permute.default(view_7, [0, 2, 1]);  view_7 = None
    bmm_47: "f32[64, 8, 3137]" = torch.ops.aten.bmm.default(view_365, permute_314);  view_365 = permute_314 = None
    view_366: "f32[8, 8, 3137, 8]" = torch.ops.aten.view.default(bmm_46, [8, 8, 3137, 8]);  bmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    add_172: "f32[8, 8, 3137, 8]" = torch.ops.aten.add.Tensor(slice_scatter_97, view_366);  slice_scatter_97 = view_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    view_367: "f32[8, 8, 8, 3137]" = torch.ops.aten.view.default(bmm_47, [8, 8, 8, 3137]);  bmm_47 = None
    permute_315: "f32[8, 8, 3137, 8]" = torch.ops.aten.permute.default(view_367, [0, 1, 3, 2]);  view_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:123, code: k_softmax = k.softmax(dim=2)
    alias_15: "f32[8, 8, 3137, 8]" = torch.ops.aten.alias.default(alias);  alias = None
    mul_309: "f32[8, 8, 3137, 8]" = torch.ops.aten.mul.Tensor(permute_315, alias_15);  permute_315 = None
    sum_127: "f32[8, 8, 1, 8]" = torch.ops.aten.sum.dim_IntList(mul_309, [2], True)
    mul_310: "f32[8, 8, 3137, 8]" = torch.ops.aten.mul.Tensor(alias_15, sum_127);  alias_15 = sum_127 = None
    sub_93: "f32[8, 8, 3137, 8]" = torch.ops.aten.sub.Tensor(mul_309, mul_310);  mul_309 = mul_310 = None
    clone_93: "f32[8, 8, 3137, 8]" = torch.ops.aten.clone.default(sub_93, memory_format = torch.contiguous_format);  sub_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:120, code: q, k, v = qkv.unbind(0)  # [B, h, N, Ch]
    cat_35: "f32[24, 8, 3137, 8]" = torch.ops.aten.cat.default([add_171, clone_93, add_172]);  add_171 = clone_93 = add_172 = None
    view_368: "f32[3, 8, 8, 3137, 8]" = torch.ops.aten.view.default(cat_35, [3, 8, 8, 3137, 8]);  cat_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:119, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_316: "f32[8, 3137, 3, 8, 8]" = torch.ops.aten.permute.default(view_368, [1, 3, 0, 2, 4]);  view_368 = None
    clone_94: "f32[8, 3137, 3, 8, 8]" = torch.ops.aten.clone.default(permute_316, memory_format = torch.contiguous_format);  permute_316 = None
    view_369: "f32[8, 3137, 192]" = torch.ops.aten.view.default(clone_94, [8, 3137, 192]);  clone_94 = None
    view_370: "f32[25096, 192]" = torch.ops.aten.view.default(view_369, [25096, 192]);  view_369 = None
    permute_317: "f32[192, 64]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    mm_64: "f32[25096, 64]" = torch.ops.aten.mm.default(view_370, permute_317);  permute_317 = None
    permute_318: "f32[192, 25096]" = torch.ops.aten.permute.default(view_370, [1, 0])
    mm_65: "f32[192, 64]" = torch.ops.aten.mm.default(permute_318, view_3);  permute_318 = view_3 = None
    permute_319: "f32[64, 192]" = torch.ops.aten.permute.default(mm_65, [1, 0]);  mm_65 = None
    sum_128: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_370, [0], True);  view_370 = None
    view_371: "f32[192]" = torch.ops.aten.view.default(sum_128, [192]);  sum_128 = None
    permute_320: "f32[192, 64]" = torch.ops.aten.permute.default(permute_319, [1, 0]);  permute_319 = None
    view_372: "f32[8, 3137, 64]" = torch.ops.aten.view.default(mm_64, [8, 3137, 64]);  mm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_94: "f32[8, 3137, 64]" = torch.ops.aten.sub.Tensor(cat_1, getitem_3);  cat_1 = getitem_3 = None
    mul_311: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(sub_94, rsqrt_1);  sub_94 = None
    mul_312: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(view_372, primals_2);  primals_2 = None
    mul_313: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(mul_312, 64)
    sum_129: "f32[8, 3137, 1]" = torch.ops.aten.sum.dim_IntList(mul_312, [2], True)
    mul_314: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(mul_312, mul_311);  mul_312 = None
    sum_130: "f32[8, 3137, 1]" = torch.ops.aten.sum.dim_IntList(mul_314, [2], True);  mul_314 = None
    mul_315: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(mul_311, sum_130);  sum_130 = None
    sub_95: "f32[8, 3137, 64]" = torch.ops.aten.sub.Tensor(mul_313, sum_129);  mul_313 = sum_129 = None
    sub_96: "f32[8, 3137, 64]" = torch.ops.aten.sub.Tensor(sub_95, mul_315);  sub_95 = mul_315 = None
    div_27: "f32[8, 3137, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 64);  rsqrt_1 = None
    mul_316: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(div_27, sub_96);  div_27 = sub_96 = None
    mul_317: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(view_372, mul_311);  mul_311 = None
    sum_131: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_317, [0, 1]);  mul_317 = None
    sum_132: "f32[64]" = torch.ops.aten.sum.dim_IntList(view_372, [0, 1]);  view_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_173: "f32[8, 3137, 64]" = torch.ops.aten.add.Tensor(add_164, mul_316);  add_164 = mul_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:163, code: x = torch.cat((cls_token, x), dim=1)
    slice_154: "f32[8, 1, 64]" = torch.ops.aten.slice.Tensor(add_173, 1, 0, 1)
    slice_155: "f32[8, 3136, 64]" = torch.ops.aten.slice.Tensor(add_173, 1, 1, 3137);  add_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:160, code: x = x.flatten(2).transpose(1, 2)
    permute_321: "f32[8, 64, 3136]" = torch.ops.aten.permute.default(slice_155, [0, 2, 1]);  slice_155 = None
    view_373: "f32[8, 64, 56, 56]" = torch.ops.aten.view.default(permute_321, [8, 64, 56, 56]);  permute_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:159, code: x = self.proj(feat) + feat
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(view_373, view_1, primals_43, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 64, [True, True, True]);  view_1 = primals_43 = None
    getitem_192: "f32[8, 64, 56, 56]" = convolution_backward_34[0]
    getitem_193: "f32[64, 1, 3, 3]" = convolution_backward_34[1]
    getitem_194: "f32[64]" = convolution_backward_34[2];  convolution_backward_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:159, code: x = self.proj(feat) + feat
    add_174: "f32[8, 64, 56, 56]" = torch.ops.aten.add.Tensor(view_373, getitem_192);  view_373 = getitem_192 = None
    add_175: "f32[64, 1, 3, 3]" = torch.ops.aten.add.Tensor(getitem_181, getitem_193);  getitem_181 = getitem_193 = None
    add_176: "f32[64]" = torch.ops.aten.add.Tensor(getitem_182, getitem_194);  getitem_182 = getitem_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:158, code: feat = img_tokens.transpose(1, 2).view(B, C, H, W)
    view_374: "f32[8, 64, 3136]" = torch.ops.aten.view.default(add_174, [8, 64, 3136]);  add_174 = None
    permute_322: "f32[8, 3136, 64]" = torch.ops.aten.permute.default(view_374, [0, 2, 1]);  view_374 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    full_103: "f32[8, 3137, 64]" = torch.ops.aten.full.default([8, 3137, 64], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_102: "f32[8, 3137, 64]" = torch.ops.aten.slice_scatter.default(full_103, permute_322, 1, 1, 9223372036854775807);  full_103 = permute_322 = None
    full_104: "f32[8, 3137, 64]" = torch.ops.aten.full.default([8, 3137, 64], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_103: "f32[8, 3137, 64]" = torch.ops.aten.slice_scatter.default(full_104, slice_scatter_102, 0, 0, 9223372036854775807);  full_104 = slice_scatter_102 = None
    full_105: "f32[8, 3137, 64]" = torch.ops.aten.full.default([8, 3137, 64], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_104: "f32[8, 3137, 64]" = torch.ops.aten.slice_scatter.default(full_105, slice_154, 1, 0, 1);  full_105 = slice_154 = None
    full_106: "f32[8, 3137, 64]" = torch.ops.aten.full.default([8, 3137, 64], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_105: "f32[8, 3137, 64]" = torch.ops.aten.slice_scatter.default(full_106, slice_scatter_104, 0, 0, 9223372036854775807);  full_106 = slice_scatter_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    add_177: "f32[8, 3137, 64]" = torch.ops.aten.add.Tensor(slice_scatter_103, slice_scatter_105);  slice_scatter_103 = slice_scatter_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:678, code: x = torch.cat((cls_tokens, x), dim=1)
    slice_156: "f32[8, 1, 64]" = torch.ops.aten.slice.Tensor(add_177, 1, 0, 1)
    slice_157: "f32[8, 3136, 64]" = torch.ops.aten.slice.Tensor(add_177, 1, 1, 3137);  add_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:677, code: cls_tokens = cls_token.expand(x.shape[0], -1, -1)
    sum_133: "f32[1, 1, 64]" = torch.ops.aten.sum.dim_IntList(slice_156, [0], True);  slice_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:92, code: x = self.norm(x)
    clone_95: "f32[8, 3136, 64]" = torch.ops.aten.clone.default(slice_157, memory_format = torch.contiguous_format);  slice_157 = None
    clone_96: "f32[8, 3136, 64]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
    sub_97: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(clone_96, getitem_1);  clone_96 = getitem_1 = None
    mul_318: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(sub_97, rsqrt);  sub_97 = None
    mul_319: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(clone_95, primals_41);  primals_41 = None
    mul_320: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_319, 64)
    sum_134: "f32[8, 3136, 1]" = torch.ops.aten.sum.dim_IntList(mul_319, [2], True)
    mul_321: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_319, mul_318);  mul_319 = None
    sum_135: "f32[8, 3136, 1]" = torch.ops.aten.sum.dim_IntList(mul_321, [2], True);  mul_321 = None
    mul_322: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_318, sum_135);  sum_135 = None
    sub_98: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(mul_320, sum_134);  mul_320 = sum_134 = None
    sub_99: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(sub_98, mul_322);  sub_98 = mul_322 = None
    div_28: "f32[8, 3136, 1]" = torch.ops.aten.div.Tensor(rsqrt, 64);  rsqrt = None
    mul_323: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(div_28, sub_99);  div_28 = sub_99 = None
    mul_324: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(clone_95, mul_318);  mul_318 = None
    sum_136: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_324, [0, 1]);  mul_324 = None
    sum_137: "f32[64]" = torch.ops.aten.sum.dim_IntList(clone_95, [0, 1]);  clone_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:89, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    permute_323: "f32[8, 64, 3136]" = torch.ops.aten.permute.default(mul_323, [0, 2, 1]);  mul_323 = None
    view_375: "f32[8, 64, 56, 56]" = torch.ops.aten.view.default(permute_323, [8, 64, 56, 56]);  permute_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(view_375, primals_153, primals_39, [64], [4, 4], [0, 0], [1, 1], False, [0, 0], 1, [False, True, True]);  view_375 = primals_153 = primals_39 = None
    getitem_196: "f32[64, 3, 4, 4]" = convolution_backward_35[1]
    getitem_197: "f32[64]" = convolution_backward_35[2];  convolution_backward_35 = None
    return pytree.tree_unflatten([addmm_32, sum_133, sum_131, sum_132, sum_124, sum_125, sum_118, sum_119, sum_111, sum_112, sum_102, sum_100, sum_101, sum_93, sum_94, sum_87, sum_88, sum_80, sum_81, sum_71, sum_69, sum_70, sum_62, sum_63, sum_56, sum_57, sum_49, sum_50, sum_40, sum_38, sum_39, sum_31, sum_32, sum_25, sum_26, sum_18, sum_19, sum_12, sum_13, getitem_196, getitem_197, sum_136, sum_137, add_175, add_176, permute_320, view_371, add_169, add_170, add_167, add_168, add_165, add_166, permute_307, view_357, permute_303, view_354, permute_299, view_351, permute_293, view_346, permute_280, view_332, permute_276, view_329, permute_272, view_326, getitem_169, getitem_170, sum_105, sum_106, add_151, add_152, permute_264, view_319, add_145, add_146, add_143, add_144, add_141, add_142, permute_251, view_305, permute_247, view_302, permute_243, view_299, permute_237, view_294, permute_224, view_280, permute_220, view_277, permute_216, view_274, getitem_142, getitem_143, sum_74, sum_75, add_127, add_128, permute_208, view_267, add_121, add_122, add_119, add_120, add_117, add_118, permute_195, view_253, permute_191, view_250, permute_187, view_247, permute_181, view_242, permute_168, view_228, permute_164, view_225, permute_160, view_222, getitem_115, getitem_116, sum_43, sum_44, add_103, add_104, permute_152, view_215, add_97, add_98, add_95, add_96, add_93, add_94, permute_139, view_201, permute_135, view_198, permute_131, view_195, permute_125, view_190, permute_112, view_176, permute_108, view_173, permute_104, view_170, permute_100, view_168, None], self._out_spec)
    