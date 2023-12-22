from __future__ import annotations



def forward(self, primals_1: "f32[1, 1, 64]", primals_2: "f32[64]", primals_3: "f32[64]", primals_4: "f32[64]", primals_5: "f32[64]", primals_6: "f32[64]", primals_7: "f32[64]", primals_8: "f32[64]", primals_9: "f32[64]", primals_10: "f32[1, 1, 128]", primals_11: "f32[128]", primals_12: "f32[128]", primals_13: "f32[128]", primals_14: "f32[128]", primals_15: "f32[128]", primals_16: "f32[128]", primals_17: "f32[128]", primals_18: "f32[128]", primals_19: "f32[1, 1, 320]", primals_20: "f32[320]", primals_21: "f32[320]", primals_22: "f32[320]", primals_23: "f32[320]", primals_24: "f32[320]", primals_25: "f32[320]", primals_26: "f32[320]", primals_27: "f32[320]", primals_28: "f32[1, 1, 512]", primals_29: "f32[512]", primals_30: "f32[512]", primals_31: "f32[512]", primals_32: "f32[512]", primals_33: "f32[512]", primals_34: "f32[512]", primals_35: "f32[512]", primals_36: "f32[512]", primals_37: "f32[512]", primals_38: "f32[512]", primals_39: "f32[64, 3, 4, 4]", primals_40: "f32[64]", primals_41: "f32[64]", primals_42: "f32[64]", primals_43: "f32[64, 1, 3, 3]", primals_44: "f32[64]", primals_45: "f32[192, 64]", primals_46: "f32[192]", primals_47: "f32[16, 1, 3, 3]", primals_48: "f32[16]", primals_49: "f32[24, 1, 5, 5]", primals_50: "f32[24]", primals_51: "f32[24, 1, 7, 7]", primals_52: "f32[24]", primals_53: "f32[64, 64]", primals_54: "f32[64]", primals_55: "f32[512, 64]", primals_56: "f32[512]", primals_57: "f32[64, 512]", primals_58: "f32[64]", primals_59: "f32[192, 64]", primals_60: "f32[192]", primals_61: "f32[64, 64]", primals_62: "f32[64]", primals_63: "f32[512, 64]", primals_64: "f32[512]", primals_65: "f32[64, 512]", primals_66: "f32[64]", primals_67: "f32[128, 64, 2, 2]", primals_68: "f32[128]", primals_69: "f32[128]", primals_70: "f32[128]", primals_71: "f32[128, 1, 3, 3]", primals_72: "f32[128]", primals_73: "f32[384, 128]", primals_74: "f32[384]", primals_75: "f32[32, 1, 3, 3]", primals_76: "f32[32]", primals_77: "f32[48, 1, 5, 5]", primals_78: "f32[48]", primals_79: "f32[48, 1, 7, 7]", primals_80: "f32[48]", primals_81: "f32[128, 128]", primals_82: "f32[128]", primals_83: "f32[1024, 128]", primals_84: "f32[1024]", primals_85: "f32[128, 1024]", primals_86: "f32[128]", primals_87: "f32[384, 128]", primals_88: "f32[384]", primals_89: "f32[128, 128]", primals_90: "f32[128]", primals_91: "f32[1024, 128]", primals_92: "f32[1024]", primals_93: "f32[128, 1024]", primals_94: "f32[128]", primals_95: "f32[320, 128, 2, 2]", primals_96: "f32[320]", primals_97: "f32[320]", primals_98: "f32[320]", primals_99: "f32[320, 1, 3, 3]", primals_100: "f32[320]", primals_101: "f32[960, 320]", primals_102: "f32[960]", primals_103: "f32[80, 1, 3, 3]", primals_104: "f32[80]", primals_105: "f32[120, 1, 5, 5]", primals_106: "f32[120]", primals_107: "f32[120, 1, 7, 7]", primals_108: "f32[120]", primals_109: "f32[320, 320]", primals_110: "f32[320]", primals_111: "f32[1280, 320]", primals_112: "f32[1280]", primals_113: "f32[320, 1280]", primals_114: "f32[320]", primals_115: "f32[960, 320]", primals_116: "f32[960]", primals_117: "f32[320, 320]", primals_118: "f32[320]", primals_119: "f32[1280, 320]", primals_120: "f32[1280]", primals_121: "f32[320, 1280]", primals_122: "f32[320]", primals_123: "f32[512, 320, 2, 2]", primals_124: "f32[512]", primals_125: "f32[512]", primals_126: "f32[512]", primals_127: "f32[512, 1, 3, 3]", primals_128: "f32[512]", primals_129: "f32[1536, 512]", primals_130: "f32[1536]", primals_131: "f32[128, 1, 3, 3]", primals_132: "f32[128]", primals_133: "f32[192, 1, 5, 5]", primals_134: "f32[192]", primals_135: "f32[192, 1, 7, 7]", primals_136: "f32[192]", primals_137: "f32[512, 512]", primals_138: "f32[512]", primals_139: "f32[2048, 512]", primals_140: "f32[2048]", primals_141: "f32[512, 2048]", primals_142: "f32[512]", primals_143: "f32[1536, 512]", primals_144: "f32[1536]", primals_145: "f32[512, 512]", primals_146: "f32[512]", primals_147: "f32[2048, 512]", primals_148: "f32[2048]", primals_149: "f32[512, 2048]", primals_150: "f32[512]", primals_151: "f32[1000, 512]", primals_152: "f32[1000]", primals_153: "f32[8, 3, 224, 224]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution: "f32[8, 64, 56, 56]" = torch.ops.aten.convolution.default(primals_153, primals_39, primals_40, [4, 4], [0, 0], [1, 1], False, [0, 0], 1);  primals_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:89, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    view: "f32[8, 64, 3136]" = torch.ops.aten.view.default(convolution, [8, 64, 3136]);  convolution = None
    permute: "f32[8, 3136, 64]" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:92, code: x = self.norm(x)
    clone: "f32[8, 3136, 64]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
    var_mean = torch.ops.aten.var_mean.correction(clone, [2], correction = 0, keepdim = True)
    getitem: "f32[8, 3136, 1]" = var_mean[0]
    getitem_1: "f32[8, 3136, 1]" = var_mean[1];  var_mean = None
    add: "f32[8, 3136, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
    rsqrt: "f32[8, 3136, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
    sub: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(clone, getitem_1);  clone = getitem_1 = None
    mul: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    mul_1: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul, primals_41)
    add_1: "f32[8, 3136, 64]" = torch.ops.aten.add.Tensor(mul_1, primals_42);  mul_1 = primals_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:677, code: cls_tokens = cls_token.expand(x.shape[0], -1, -1)
    expand: "f32[8, 1, 64]" = torch.ops.aten.expand.default(primals_1, [8, -1, -1]);  primals_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:678, code: x = torch.cat((cls_tokens, x), dim=1)
    cat: "f32[8, 3137, 64]" = torch.ops.aten.cat.default([expand, add_1], 1);  expand = add_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    slice_1: "f32[8, 3137, 64]" = torch.ops.aten.slice.Tensor(cat, 0, 0, 9223372036854775807);  cat = None
    slice_2: "f32[8, 1, 64]" = torch.ops.aten.slice.Tensor(slice_1, 1, 0, 1)
    slice_4: "f32[8, 3136, 64]" = torch.ops.aten.slice.Tensor(slice_1, 1, 1, 9223372036854775807);  slice_1 = None
    
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
    view_13: "f32[8, 8, 8, 3136]" = torch.ops.aten.view.default(cat_2, [8, 8, 8, 3136])
    permute_7: "f32[8, 8, 3136, 8]" = torch.ops.aten.permute.default(view_13, [0, 1, 3, 2]);  view_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:86, code: EV_hat = q_img * conv_v_img
    mul_4: "f32[8, 8, 3136, 8]" = torch.ops.aten.mul.Tensor(slice_8, permute_7);  permute_7 = None
    
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
    sub_3: "f32[8, 3137, 64]" = torch.ops.aten.sub.Tensor(add_6, getitem_11);  getitem_11 = None
    mul_6: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = None
    mul_7: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(mul_6, primals_4)
    add_8: "f32[8, 3137, 64]" = torch.ops.aten.add.Tensor(mul_7, primals_5);  mul_7 = primals_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_17: "f32[25096, 64]" = torch.ops.aten.view.default(add_8, [25096, 64]);  add_8 = None
    permute_10: "f32[64, 512]" = torch.ops.aten.permute.default(primals_55, [1, 0]);  primals_55 = None
    addmm_2: "f32[25096, 512]" = torch.ops.aten.addmm.default(primals_56, view_17, permute_10);  primals_56 = None
    view_18: "f32[8, 3137, 512]" = torch.ops.aten.view.default(addmm_2, [8, 3137, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_8: "f32[8, 3137, 512]" = torch.ops.aten.mul.Tensor(view_18, 0.5)
    mul_9: "f32[8, 3137, 512]" = torch.ops.aten.mul.Tensor(view_18, 0.7071067811865476);  view_18 = None
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
    add_10: "f32[8, 3137, 64]" = torch.ops.aten.add.Tensor(add_6, clone_7);  add_6 = clone_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    slice_13: "f32[8, 3137, 64]" = torch.ops.aten.slice.Tensor(add_10, 0, 0, 9223372036854775807);  add_10 = None
    slice_14: "f32[8, 1, 64]" = torch.ops.aten.slice.Tensor(slice_13, 1, 0, 1)
    slice_16: "f32[8, 3136, 64]" = torch.ops.aten.slice.Tensor(slice_13, 1, 1, 9223372036854775807);  slice_13 = None
    
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
    view_33: "f32[8, 8, 8, 3136]" = torch.ops.aten.view.default(cat_4, [8, 8, 8, 3136])
    permute_18: "f32[8, 8, 3136, 8]" = torch.ops.aten.permute.default(view_33, [0, 1, 3, 2]);  view_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:86, code: EV_hat = q_img * conv_v_img
    mul_13: "f32[8, 8, 3136, 8]" = torch.ops.aten.mul.Tensor(slice_20, permute_18);  permute_18 = None
    
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
    sub_6: "f32[8, 3137, 64]" = torch.ops.aten.sub.Tensor(add_15, getitem_21);  getitem_21 = None
    mul_15: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = None
    mul_16: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(mul_15, primals_8)
    add_17: "f32[8, 3137, 64]" = torch.ops.aten.add.Tensor(mul_16, primals_9);  mul_16 = primals_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_37: "f32[25096, 64]" = torch.ops.aten.view.default(add_17, [25096, 64]);  add_17 = None
    permute_21: "f32[64, 512]" = torch.ops.aten.permute.default(primals_63, [1, 0]);  primals_63 = None
    addmm_6: "f32[25096, 512]" = torch.ops.aten.addmm.default(primals_64, view_37, permute_21);  primals_64 = None
    view_38: "f32[8, 3137, 512]" = torch.ops.aten.view.default(addmm_6, [8, 3137, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_17: "f32[8, 3137, 512]" = torch.ops.aten.mul.Tensor(view_38, 0.5)
    mul_18: "f32[8, 3137, 512]" = torch.ops.aten.mul.Tensor(view_38, 0.7071067811865476);  view_38 = None
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
    add_19: "f32[8, 3137, 64]" = torch.ops.aten.add.Tensor(add_15, clone_14);  add_15 = clone_14 = None
    
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
    clone_16: "f32[8, 784, 128]" = torch.ops.aten.clone.default(permute_24, memory_format = torch.contiguous_format);  permute_24 = None
    var_mean_5 = torch.ops.aten.var_mean.correction(clone_16, [2], correction = 0, keepdim = True)
    getitem_22: "f32[8, 784, 1]" = var_mean_5[0]
    getitem_23: "f32[8, 784, 1]" = var_mean_5[1];  var_mean_5 = None
    add_20: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
    rsqrt_5: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
    sub_7: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(clone_16, getitem_23);  clone_16 = getitem_23 = None
    mul_20: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_5);  sub_7 = None
    mul_21: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_20, primals_69)
    add_21: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(mul_21, primals_70);  mul_21 = primals_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:677, code: cls_tokens = cls_token.expand(x.shape[0], -1, -1)
    expand_9: "f32[8, 1, 128]" = torch.ops.aten.expand.default(primals_10, [8, -1, -1]);  primals_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:678, code: x = torch.cat((cls_tokens, x), dim=1)
    cat_5: "f32[8, 785, 128]" = torch.ops.aten.cat.default([expand_9, add_21], 1);  expand_9 = add_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    slice_28: "f32[8, 785, 128]" = torch.ops.aten.slice.Tensor(cat_5, 0, 0, 9223372036854775807);  cat_5 = None
    slice_29: "f32[8, 1, 128]" = torch.ops.aten.slice.Tensor(slice_28, 1, 0, 1)
    slice_31: "f32[8, 784, 128]" = torch.ops.aten.slice.Tensor(slice_28, 1, 1, 9223372036854775807);  slice_28 = None
    
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
    view_55: "f32[8, 8, 16, 784]" = torch.ops.aten.view.default(cat_7, [8, 8, 16, 784])
    permute_31: "f32[8, 8, 784, 16]" = torch.ops.aten.permute.default(view_55, [0, 1, 3, 2]);  view_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:86, code: EV_hat = q_img * conv_v_img
    mul_24: "f32[8, 8, 784, 16]" = torch.ops.aten.mul.Tensor(slice_35, permute_31);  permute_31 = None
    
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
    sub_10: "f32[8, 785, 128]" = torch.ops.aten.sub.Tensor(add_26, getitem_33);  getitem_33 = None
    mul_26: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_7);  sub_10 = None
    mul_27: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(mul_26, primals_13)
    add_28: "f32[8, 785, 128]" = torch.ops.aten.add.Tensor(mul_27, primals_14);  mul_27 = primals_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_59: "f32[6280, 128]" = torch.ops.aten.view.default(add_28, [6280, 128]);  add_28 = None
    permute_34: "f32[128, 1024]" = torch.ops.aten.permute.default(primals_83, [1, 0]);  primals_83 = None
    addmm_10: "f32[6280, 1024]" = torch.ops.aten.addmm.default(primals_84, view_59, permute_34);  primals_84 = None
    view_60: "f32[8, 785, 1024]" = torch.ops.aten.view.default(addmm_10, [8, 785, 1024])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_28: "f32[8, 785, 1024]" = torch.ops.aten.mul.Tensor(view_60, 0.5)
    mul_29: "f32[8, 785, 1024]" = torch.ops.aten.mul.Tensor(view_60, 0.7071067811865476);  view_60 = None
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
    add_30: "f32[8, 785, 128]" = torch.ops.aten.add.Tensor(add_26, clone_23);  add_26 = clone_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    slice_40: "f32[8, 785, 128]" = torch.ops.aten.slice.Tensor(add_30, 0, 0, 9223372036854775807);  add_30 = None
    slice_41: "f32[8, 1, 128]" = torch.ops.aten.slice.Tensor(slice_40, 1, 0, 1)
    slice_43: "f32[8, 784, 128]" = torch.ops.aten.slice.Tensor(slice_40, 1, 1, 9223372036854775807);  slice_40 = None
    
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
    view_75: "f32[8, 8, 16, 784]" = torch.ops.aten.view.default(cat_9, [8, 8, 16, 784])
    permute_42: "f32[8, 8, 784, 16]" = torch.ops.aten.permute.default(view_75, [0, 1, 3, 2]);  view_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:86, code: EV_hat = q_img * conv_v_img
    mul_33: "f32[8, 8, 784, 16]" = torch.ops.aten.mul.Tensor(slice_47, permute_42);  permute_42 = None
    
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
    sub_13: "f32[8, 785, 128]" = torch.ops.aten.sub.Tensor(add_35, getitem_43);  getitem_43 = None
    mul_35: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_9);  sub_13 = None
    mul_36: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(mul_35, primals_17)
    add_37: "f32[8, 785, 128]" = torch.ops.aten.add.Tensor(mul_36, primals_18);  mul_36 = primals_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_79: "f32[6280, 128]" = torch.ops.aten.view.default(add_37, [6280, 128]);  add_37 = None
    permute_45: "f32[128, 1024]" = torch.ops.aten.permute.default(primals_91, [1, 0]);  primals_91 = None
    addmm_14: "f32[6280, 1024]" = torch.ops.aten.addmm.default(primals_92, view_79, permute_45);  primals_92 = None
    view_80: "f32[8, 785, 1024]" = torch.ops.aten.view.default(addmm_14, [8, 785, 1024])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_37: "f32[8, 785, 1024]" = torch.ops.aten.mul.Tensor(view_80, 0.5)
    mul_38: "f32[8, 785, 1024]" = torch.ops.aten.mul.Tensor(view_80, 0.7071067811865476);  view_80 = None
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
    add_39: "f32[8, 785, 128]" = torch.ops.aten.add.Tensor(add_35, clone_30);  add_35 = clone_30 = None
    
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
    clone_32: "f32[8, 196, 320]" = torch.ops.aten.clone.default(permute_48, memory_format = torch.contiguous_format);  permute_48 = None
    var_mean_10 = torch.ops.aten.var_mean.correction(clone_32, [2], correction = 0, keepdim = True)
    getitem_44: "f32[8, 196, 1]" = var_mean_10[0]
    getitem_45: "f32[8, 196, 1]" = var_mean_10[1];  var_mean_10 = None
    add_40: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
    rsqrt_10: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
    sub_14: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_32, getitem_45);  clone_32 = getitem_45 = None
    mul_40: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_10);  sub_14 = None
    mul_41: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_40, primals_97)
    add_41: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_41, primals_98);  mul_41 = primals_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:677, code: cls_tokens = cls_token.expand(x.shape[0], -1, -1)
    expand_18: "f32[8, 1, 320]" = torch.ops.aten.expand.default(primals_19, [8, -1, -1]);  primals_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:678, code: x = torch.cat((cls_tokens, x), dim=1)
    cat_10: "f32[8, 197, 320]" = torch.ops.aten.cat.default([expand_18, add_41], 1);  expand_18 = add_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    slice_55: "f32[8, 197, 320]" = torch.ops.aten.slice.Tensor(cat_10, 0, 0, 9223372036854775807);  cat_10 = None
    slice_56: "f32[8, 1, 320]" = torch.ops.aten.slice.Tensor(slice_55, 1, 0, 1)
    slice_58: "f32[8, 196, 320]" = torch.ops.aten.slice.Tensor(slice_55, 1, 1, 9223372036854775807);  slice_55 = None
    
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
    view_97: "f32[8, 8, 40, 196]" = torch.ops.aten.view.default(cat_12, [8, 8, 40, 196])
    permute_55: "f32[8, 8, 196, 40]" = torch.ops.aten.permute.default(view_97, [0, 1, 3, 2]);  view_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:86, code: EV_hat = q_img * conv_v_img
    mul_44: "f32[8, 8, 196, 40]" = torch.ops.aten.mul.Tensor(slice_62, permute_55);  permute_55 = None
    
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
    sub_17: "f32[8, 197, 320]" = torch.ops.aten.sub.Tensor(add_46, getitem_55);  getitem_55 = None
    mul_46: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_12);  sub_17 = None
    mul_47: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(mul_46, primals_22)
    add_48: "f32[8, 197, 320]" = torch.ops.aten.add.Tensor(mul_47, primals_23);  mul_47 = primals_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_101: "f32[1576, 320]" = torch.ops.aten.view.default(add_48, [1576, 320]);  add_48 = None
    permute_58: "f32[320, 1280]" = torch.ops.aten.permute.default(primals_111, [1, 0]);  primals_111 = None
    addmm_18: "f32[1576, 1280]" = torch.ops.aten.addmm.default(primals_112, view_101, permute_58);  primals_112 = None
    view_102: "f32[8, 197, 1280]" = torch.ops.aten.view.default(addmm_18, [8, 197, 1280])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_48: "f32[8, 197, 1280]" = torch.ops.aten.mul.Tensor(view_102, 0.5)
    mul_49: "f32[8, 197, 1280]" = torch.ops.aten.mul.Tensor(view_102, 0.7071067811865476);  view_102 = None
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
    add_50: "f32[8, 197, 320]" = torch.ops.aten.add.Tensor(add_46, clone_39);  add_46 = clone_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    slice_67: "f32[8, 197, 320]" = torch.ops.aten.slice.Tensor(add_50, 0, 0, 9223372036854775807);  add_50 = None
    slice_68: "f32[8, 1, 320]" = torch.ops.aten.slice.Tensor(slice_67, 1, 0, 1)
    slice_70: "f32[8, 196, 320]" = torch.ops.aten.slice.Tensor(slice_67, 1, 1, 9223372036854775807);  slice_67 = None
    
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
    view_117: "f32[8, 8, 40, 196]" = torch.ops.aten.view.default(cat_14, [8, 8, 40, 196])
    permute_66: "f32[8, 8, 196, 40]" = torch.ops.aten.permute.default(view_117, [0, 1, 3, 2]);  view_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:86, code: EV_hat = q_img * conv_v_img
    mul_53: "f32[8, 8, 196, 40]" = torch.ops.aten.mul.Tensor(slice_74, permute_66);  permute_66 = None
    
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
    sub_20: "f32[8, 197, 320]" = torch.ops.aten.sub.Tensor(add_55, getitem_65);  getitem_65 = None
    mul_55: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_14);  sub_20 = None
    mul_56: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(mul_55, primals_26)
    add_57: "f32[8, 197, 320]" = torch.ops.aten.add.Tensor(mul_56, primals_27);  mul_56 = primals_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_121: "f32[1576, 320]" = torch.ops.aten.view.default(add_57, [1576, 320]);  add_57 = None
    permute_69: "f32[320, 1280]" = torch.ops.aten.permute.default(primals_119, [1, 0]);  primals_119 = None
    addmm_22: "f32[1576, 1280]" = torch.ops.aten.addmm.default(primals_120, view_121, permute_69);  primals_120 = None
    view_122: "f32[8, 197, 1280]" = torch.ops.aten.view.default(addmm_22, [8, 197, 1280])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_57: "f32[8, 197, 1280]" = torch.ops.aten.mul.Tensor(view_122, 0.5)
    mul_58: "f32[8, 197, 1280]" = torch.ops.aten.mul.Tensor(view_122, 0.7071067811865476);  view_122 = None
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
    add_59: "f32[8, 197, 320]" = torch.ops.aten.add.Tensor(add_55, clone_46);  add_55 = clone_46 = None
    
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
    clone_48: "f32[8, 49, 512]" = torch.ops.aten.clone.default(permute_72, memory_format = torch.contiguous_format);  permute_72 = None
    var_mean_15 = torch.ops.aten.var_mean.correction(clone_48, [2], correction = 0, keepdim = True)
    getitem_66: "f32[8, 49, 1]" = var_mean_15[0]
    getitem_67: "f32[8, 49, 1]" = var_mean_15[1];  var_mean_15 = None
    add_60: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
    rsqrt_15: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
    sub_21: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(clone_48, getitem_67);  clone_48 = getitem_67 = None
    mul_60: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_15);  sub_21 = None
    mul_61: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_60, primals_125)
    add_61: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(mul_61, primals_126);  mul_61 = primals_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:677, code: cls_tokens = cls_token.expand(x.shape[0], -1, -1)
    expand_27: "f32[8, 1, 512]" = torch.ops.aten.expand.default(primals_28, [8, -1, -1]);  primals_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:678, code: x = torch.cat((cls_tokens, x), dim=1)
    cat_15: "f32[8, 50, 512]" = torch.ops.aten.cat.default([expand_27, add_61], 1);  expand_27 = add_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    slice_82: "f32[8, 50, 512]" = torch.ops.aten.slice.Tensor(cat_15, 0, 0, 9223372036854775807);  cat_15 = None
    slice_83: "f32[8, 1, 512]" = torch.ops.aten.slice.Tensor(slice_82, 1, 0, 1)
    slice_85: "f32[8, 49, 512]" = torch.ops.aten.slice.Tensor(slice_82, 1, 1, 9223372036854775807);  slice_82 = None
    
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
    view_139: "f32[8, 8, 64, 49]" = torch.ops.aten.view.default(cat_17, [8, 8, 64, 49])
    permute_79: "f32[8, 8, 49, 64]" = torch.ops.aten.permute.default(view_139, [0, 1, 3, 2]);  view_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:86, code: EV_hat = q_img * conv_v_img
    mul_64: "f32[8, 8, 49, 64]" = torch.ops.aten.mul.Tensor(slice_89, permute_79);  permute_79 = None
    
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
    sub_24: "f32[8, 50, 512]" = torch.ops.aten.sub.Tensor(add_66, getitem_77);  getitem_77 = None
    mul_66: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_17);  sub_24 = None
    mul_67: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(mul_66, primals_31)
    add_68: "f32[8, 50, 512]" = torch.ops.aten.add.Tensor(mul_67, primals_32);  mul_67 = primals_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_143: "f32[400, 512]" = torch.ops.aten.view.default(add_68, [400, 512]);  add_68 = None
    permute_82: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_139, [1, 0]);  primals_139 = None
    addmm_26: "f32[400, 2048]" = torch.ops.aten.addmm.default(primals_140, view_143, permute_82);  primals_140 = None
    view_144: "f32[8, 50, 2048]" = torch.ops.aten.view.default(addmm_26, [8, 50, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_68: "f32[8, 50, 2048]" = torch.ops.aten.mul.Tensor(view_144, 0.5)
    mul_69: "f32[8, 50, 2048]" = torch.ops.aten.mul.Tensor(view_144, 0.7071067811865476);  view_144 = None
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
    add_70: "f32[8, 50, 512]" = torch.ops.aten.add.Tensor(add_66, clone_55);  add_66 = clone_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    slice_94: "f32[8, 50, 512]" = torch.ops.aten.slice.Tensor(add_70, 0, 0, 9223372036854775807);  add_70 = None
    slice_95: "f32[8, 1, 512]" = torch.ops.aten.slice.Tensor(slice_94, 1, 0, 1)
    slice_97: "f32[8, 49, 512]" = torch.ops.aten.slice.Tensor(slice_94, 1, 1, 9223372036854775807);  slice_94 = None
    
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
    view_159: "f32[8, 8, 64, 49]" = torch.ops.aten.view.default(cat_19, [8, 8, 64, 49])
    permute_90: "f32[8, 8, 49, 64]" = torch.ops.aten.permute.default(view_159, [0, 1, 3, 2]);  view_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:86, code: EV_hat = q_img * conv_v_img
    mul_73: "f32[8, 8, 49, 64]" = torch.ops.aten.mul.Tensor(slice_101, permute_90);  permute_90 = None
    
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
    sub_27: "f32[8, 50, 512]" = torch.ops.aten.sub.Tensor(add_75, getitem_87);  getitem_87 = None
    mul_75: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_19);  sub_27 = None
    mul_76: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(mul_75, primals_35)
    add_77: "f32[8, 50, 512]" = torch.ops.aten.add.Tensor(mul_76, primals_36);  mul_76 = primals_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_163: "f32[400, 512]" = torch.ops.aten.view.default(add_77, [400, 512]);  add_77 = None
    permute_93: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_147, [1, 0]);  primals_147 = None
    addmm_30: "f32[400, 2048]" = torch.ops.aten.addmm.default(primals_148, view_163, permute_93);  primals_148 = None
    view_164: "f32[8, 50, 2048]" = torch.ops.aten.view.default(addmm_30, [8, 50, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_77: "f32[8, 50, 2048]" = torch.ops.aten.mul.Tensor(view_164, 0.5)
    mul_78: "f32[8, 50, 2048]" = torch.ops.aten.mul.Tensor(view_164, 0.7071067811865476);  view_164 = None
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
    add_79: "f32[8, 50, 512]" = torch.ops.aten.add.Tensor(add_75, clone_62);  add_75 = clone_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_20 = torch.ops.aten.var_mean.correction(add_79, [2], correction = 0, keepdim = True)
    getitem_88: "f32[8, 50, 1]" = var_mean_20[0]
    getitem_89: "f32[8, 50, 1]" = var_mean_20[1];  var_mean_20 = None
    add_80: "f32[8, 50, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-06);  getitem_88 = None
    rsqrt_20: "f32[8, 50, 1]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
    sub_28: "f32[8, 50, 512]" = torch.ops.aten.sub.Tensor(add_79, getitem_89);  add_79 = getitem_89 = None
    mul_80: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_20);  sub_28 = None
    mul_81: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(mul_80, primals_37)
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_8: "f32[8, 50, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 512);  rsqrt_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_101: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_94, [1, 0]);  permute_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_105: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_93, [1, 0]);  permute_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_9: "f32[8, 50, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 512);  rsqrt_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:135, code: x = self.proj(x)
    permute_109: "f32[512, 512]" = torch.ops.aten.permute.default(permute_92, [1, 0]);  permute_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    permute_116: "f32[64, 64, 50]" = torch.ops.aten.permute.default(view_155, [0, 2, 1]);  view_155 = None
    permute_117: "f32[64, 64, 64]" = torch.ops.aten.permute.default(view_156, [0, 2, 1]);  view_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    permute_118: "f32[64, 50, 64]" = torch.ops.aten.permute.default(view_152, [0, 2, 1]);  view_152 = None
    permute_119: "f32[64, 64, 50]" = torch.ops.aten.permute.default(view_153, [0, 2, 1]);  view_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:123, code: k_softmax = k.softmax(dim=2)
    alias_8: "f32[8, 8, 50, 64]" = torch.ops.aten.alias.default(alias_7);  alias_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:119, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_122: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_128: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_83, [1, 0]);  permute_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_132: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_82, [1, 0]);  permute_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_11: "f32[8, 50, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 512);  rsqrt_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:135, code: x = self.proj(x)
    permute_136: "f32[512, 512]" = torch.ops.aten.permute.default(permute_81, [1, 0]);  permute_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    permute_143: "f32[64, 64, 50]" = torch.ops.aten.permute.default(view_135, [0, 2, 1]);  view_135 = None
    permute_144: "f32[64, 64, 64]" = torch.ops.aten.permute.default(view_136, [0, 2, 1]);  view_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    permute_145: "f32[64, 50, 64]" = torch.ops.aten.permute.default(view_132, [0, 2, 1]);  view_132 = None
    permute_146: "f32[64, 64, 50]" = torch.ops.aten.permute.default(view_133, [0, 2, 1]);  view_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:123, code: k_softmax = k.softmax(dim=2)
    alias_9: "f32[8, 8, 50, 64]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:119, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_149: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:92, code: x = self.norm(x)
    div_13: "f32[8, 49, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 512);  rsqrt_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_157: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_70, [1, 0]);  permute_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_161: "f32[1280, 320]" = torch.ops.aten.permute.default(permute_69, [1, 0]);  permute_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_14: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 320);  rsqrt_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:135, code: x = self.proj(x)
    permute_165: "f32[320, 320]" = torch.ops.aten.permute.default(permute_68, [1, 0]);  permute_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    permute_172: "f32[64, 40, 197]" = torch.ops.aten.permute.default(view_113, [0, 2, 1]);  view_113 = None
    permute_173: "f32[64, 40, 40]" = torch.ops.aten.permute.default(view_114, [0, 2, 1]);  view_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    permute_174: "f32[64, 197, 40]" = torch.ops.aten.permute.default(view_110, [0, 2, 1]);  view_110 = None
    permute_175: "f32[64, 40, 197]" = torch.ops.aten.permute.default(view_111, [0, 2, 1]);  view_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:123, code: k_softmax = k.softmax(dim=2)
    alias_10: "f32[8, 8, 197, 40]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:119, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_178: "f32[960, 320]" = torch.ops.aten.permute.default(permute_62, [1, 0]);  permute_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_184: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_188: "f32[1280, 320]" = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_16: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 320);  rsqrt_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:135, code: x = self.proj(x)
    permute_192: "f32[320, 320]" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    permute_199: "f32[64, 40, 197]" = torch.ops.aten.permute.default(view_93, [0, 2, 1]);  view_93 = None
    permute_200: "f32[64, 40, 40]" = torch.ops.aten.permute.default(view_94, [0, 2, 1]);  view_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    permute_201: "f32[64, 197, 40]" = torch.ops.aten.permute.default(view_90, [0, 2, 1]);  view_90 = None
    permute_202: "f32[64, 40, 197]" = torch.ops.aten.permute.default(view_91, [0, 2, 1]);  view_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:123, code: k_softmax = k.softmax(dim=2)
    alias_11: "f32[8, 8, 197, 40]" = torch.ops.aten.alias.default(alias_4);  alias_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:119, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_205: "f32[960, 320]" = torch.ops.aten.permute.default(permute_51, [1, 0]);  permute_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:92, code: x = self.norm(x)
    div_18: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 320);  rsqrt_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_213: "f32[128, 1024]" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_217: "f32[1024, 128]" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_19: "f32[8, 785, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 128);  rsqrt_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:135, code: x = self.proj(x)
    permute_221: "f32[128, 128]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    permute_228: "f32[64, 16, 785]" = torch.ops.aten.permute.default(view_71, [0, 2, 1]);  view_71 = None
    permute_229: "f32[64, 16, 16]" = torch.ops.aten.permute.default(view_72, [0, 2, 1]);  view_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    permute_230: "f32[64, 785, 16]" = torch.ops.aten.permute.default(view_68, [0, 2, 1]);  view_68 = None
    permute_231: "f32[64, 16, 785]" = torch.ops.aten.permute.default(view_69, [0, 2, 1]);  view_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:123, code: k_softmax = k.softmax(dim=2)
    alias_12: "f32[8, 8, 785, 16]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:119, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_234: "f32[384, 128]" = torch.ops.aten.permute.default(permute_38, [1, 0]);  permute_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_240: "f32[128, 1024]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_244: "f32[1024, 128]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_21: "f32[8, 785, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 128);  rsqrt_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:135, code: x = self.proj(x)
    permute_248: "f32[128, 128]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    permute_255: "f32[64, 16, 785]" = torch.ops.aten.permute.default(view_51, [0, 2, 1]);  view_51 = None
    permute_256: "f32[64, 16, 16]" = torch.ops.aten.permute.default(view_52, [0, 2, 1]);  view_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    permute_257: "f32[64, 785, 16]" = torch.ops.aten.permute.default(view_48, [0, 2, 1]);  view_48 = None
    permute_258: "f32[64, 16, 785]" = torch.ops.aten.permute.default(view_49, [0, 2, 1]);  view_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:123, code: k_softmax = k.softmax(dim=2)
    alias_13: "f32[8, 8, 785, 16]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:119, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_261: "f32[384, 128]" = torch.ops.aten.permute.default(permute_27, [1, 0]);  permute_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:92, code: x = self.norm(x)
    div_23: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 128);  rsqrt_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_269: "f32[64, 512]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_273: "f32[512, 64]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_24: "f32[8, 3137, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 64);  rsqrt_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:135, code: x = self.proj(x)
    permute_277: "f32[64, 64]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    permute_284: "f32[64, 8, 3137]" = torch.ops.aten.permute.default(view_29, [0, 2, 1]);  view_29 = None
    permute_285: "f32[64, 8, 8]" = torch.ops.aten.permute.default(view_30, [0, 2, 1]);  view_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    permute_286: "f32[64, 3137, 8]" = torch.ops.aten.permute.default(view_26, [0, 2, 1]);  view_26 = None
    permute_287: "f32[64, 8, 3137]" = torch.ops.aten.permute.default(view_27, [0, 2, 1]);  view_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:123, code: k_softmax = k.softmax(dim=2)
    alias_14: "f32[8, 8, 3137, 8]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:119, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_290: "f32[192, 64]" = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_296: "f32[64, 512]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_300: "f32[512, 64]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_26: "f32[8, 3137, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 64);  rsqrt_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:135, code: x = self.proj(x)
    permute_304: "f32[64, 64]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    permute_311: "f32[64, 8, 3137]" = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
    permute_312: "f32[64, 8, 8]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    permute_313: "f32[64, 3137, 8]" = torch.ops.aten.permute.default(view_6, [0, 2, 1]);  view_6 = None
    permute_314: "f32[64, 8, 3137]" = torch.ops.aten.permute.default(view_7, [0, 2, 1]);  view_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:123, code: k_softmax = k.softmax(dim=2)
    alias_15: "f32[8, 8, 3137, 8]" = torch.ops.aten.alias.default(alias);  alias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:119, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_317: "f32[192, 64]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:92, code: x = self.norm(x)
    div_28: "f32[8, 3136, 1]" = torch.ops.aten.div.Tensor(rsqrt, 64);  rsqrt = None
    return [addmm_32, primals_2, primals_4, primals_6, primals_8, primals_11, primals_13, primals_15, primals_17, primals_20, primals_22, primals_24, primals_26, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_47, primals_49, primals_51, primals_67, primals_69, primals_71, primals_75, primals_77, primals_79, primals_95, primals_97, primals_99, primals_103, primals_105, primals_107, primals_123, primals_125, primals_127, primals_131, primals_133, primals_135, primals_153, mul, view_1, cat_1, getitem_3, rsqrt_1, view_3, slice_8, getitem_7, getitem_8, getitem_9, cat_2, view_15, mul_6, view_17, addmm_2, view_19, view_21, cat_3, getitem_13, rsqrt_3, view_23, slice_20, getitem_17, getitem_18, getitem_19, cat_4, view_35, mul_15, view_37, addmm_6, view_39, clone_15, mul_20, view_43, cat_6, getitem_25, rsqrt_6, view_45, slice_35, getitem_29, getitem_30, getitem_31, cat_7, view_57, mul_26, view_59, addmm_10, view_61, view_63, cat_8, getitem_35, rsqrt_8, view_65, slice_47, getitem_39, getitem_40, getitem_41, cat_9, view_77, mul_35, view_79, addmm_14, view_81, clone_31, mul_40, view_85, cat_11, getitem_47, rsqrt_11, view_87, slice_62, getitem_51, getitem_52, getitem_53, cat_12, view_99, mul_46, view_101, addmm_18, view_103, view_105, cat_13, getitem_57, rsqrt_13, view_107, slice_74, getitem_61, getitem_62, getitem_63, cat_14, view_119, mul_55, view_121, addmm_22, view_123, clone_47, mul_60, view_127, cat_16, getitem_69, rsqrt_16, view_129, slice_89, getitem_73, getitem_74, getitem_75, cat_17, view_141, mul_66, view_143, addmm_26, view_145, view_147, cat_18, getitem_79, rsqrt_18, view_149, slice_101, getitem_83, getitem_84, getitem_85, cat_19, view_161, mul_75, view_163, addmm_30, view_165, mul_80, clone_64, permute_97, div_8, permute_101, permute_105, div_9, permute_109, permute_116, permute_117, permute_118, permute_119, alias_8, permute_122, permute_128, permute_132, div_11, permute_136, permute_143, permute_144, permute_145, permute_146, alias_9, permute_149, div_13, permute_157, permute_161, div_14, permute_165, permute_172, permute_173, permute_174, permute_175, alias_10, permute_178, permute_184, permute_188, div_16, permute_192, permute_199, permute_200, permute_201, permute_202, alias_11, permute_205, div_18, permute_213, permute_217, div_19, permute_221, permute_228, permute_229, permute_230, permute_231, alias_12, permute_234, permute_240, permute_244, div_21, permute_248, permute_255, permute_256, permute_257, permute_258, alias_13, permute_261, div_23, permute_269, permute_273, div_24, permute_277, permute_284, permute_285, permute_286, permute_287, alias_14, permute_290, permute_296, permute_300, div_26, permute_304, permute_311, permute_312, permute_313, permute_314, alias_15, permute_317, div_28]
    