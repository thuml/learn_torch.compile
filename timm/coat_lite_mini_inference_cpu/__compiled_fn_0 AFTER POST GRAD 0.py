from __future__ import annotations



def forward(self, arg0_1: "f32[1, 1, 64]", arg1_1: "f32[64]", arg2_1: "f32[64]", arg3_1: "f32[64]", arg4_1: "f32[64]", arg5_1: "f32[64]", arg6_1: "f32[64]", arg7_1: "f32[64]", arg8_1: "f32[64]", arg9_1: "f32[1, 1, 128]", arg10_1: "f32[128]", arg11_1: "f32[128]", arg12_1: "f32[128]", arg13_1: "f32[128]", arg14_1: "f32[128]", arg15_1: "f32[128]", arg16_1: "f32[128]", arg17_1: "f32[128]", arg18_1: "f32[1, 1, 320]", arg19_1: "f32[320]", arg20_1: "f32[320]", arg21_1: "f32[320]", arg22_1: "f32[320]", arg23_1: "f32[320]", arg24_1: "f32[320]", arg25_1: "f32[320]", arg26_1: "f32[320]", arg27_1: "f32[1, 1, 512]", arg28_1: "f32[512]", arg29_1: "f32[512]", arg30_1: "f32[512]", arg31_1: "f32[512]", arg32_1: "f32[512]", arg33_1: "f32[512]", arg34_1: "f32[512]", arg35_1: "f32[512]", arg36_1: "f32[512]", arg37_1: "f32[512]", arg38_1: "f32[64, 3, 4, 4]", arg39_1: "f32[64]", arg40_1: "f32[64]", arg41_1: "f32[64]", arg42_1: "f32[64, 1, 3, 3]", arg43_1: "f32[64]", arg44_1: "f32[192, 64]", arg45_1: "f32[192]", arg46_1: "f32[16, 1, 3, 3]", arg47_1: "f32[16]", arg48_1: "f32[24, 1, 5, 5]", arg49_1: "f32[24]", arg50_1: "f32[24, 1, 7, 7]", arg51_1: "f32[24]", arg52_1: "f32[64, 64]", arg53_1: "f32[64]", arg54_1: "f32[512, 64]", arg55_1: "f32[512]", arg56_1: "f32[64, 512]", arg57_1: "f32[64]", arg58_1: "f32[192, 64]", arg59_1: "f32[192]", arg60_1: "f32[64, 64]", arg61_1: "f32[64]", arg62_1: "f32[512, 64]", arg63_1: "f32[512]", arg64_1: "f32[64, 512]", arg65_1: "f32[64]", arg66_1: "f32[128, 64, 2, 2]", arg67_1: "f32[128]", arg68_1: "f32[128]", arg69_1: "f32[128]", arg70_1: "f32[128, 1, 3, 3]", arg71_1: "f32[128]", arg72_1: "f32[384, 128]", arg73_1: "f32[384]", arg74_1: "f32[32, 1, 3, 3]", arg75_1: "f32[32]", arg76_1: "f32[48, 1, 5, 5]", arg77_1: "f32[48]", arg78_1: "f32[48, 1, 7, 7]", arg79_1: "f32[48]", arg80_1: "f32[128, 128]", arg81_1: "f32[128]", arg82_1: "f32[1024, 128]", arg83_1: "f32[1024]", arg84_1: "f32[128, 1024]", arg85_1: "f32[128]", arg86_1: "f32[384, 128]", arg87_1: "f32[384]", arg88_1: "f32[128, 128]", arg89_1: "f32[128]", arg90_1: "f32[1024, 128]", arg91_1: "f32[1024]", arg92_1: "f32[128, 1024]", arg93_1: "f32[128]", arg94_1: "f32[320, 128, 2, 2]", arg95_1: "f32[320]", arg96_1: "f32[320]", arg97_1: "f32[320]", arg98_1: "f32[320, 1, 3, 3]", arg99_1: "f32[320]", arg100_1: "f32[960, 320]", arg101_1: "f32[960]", arg102_1: "f32[80, 1, 3, 3]", arg103_1: "f32[80]", arg104_1: "f32[120, 1, 5, 5]", arg105_1: "f32[120]", arg106_1: "f32[120, 1, 7, 7]", arg107_1: "f32[120]", arg108_1: "f32[320, 320]", arg109_1: "f32[320]", arg110_1: "f32[1280, 320]", arg111_1: "f32[1280]", arg112_1: "f32[320, 1280]", arg113_1: "f32[320]", arg114_1: "f32[960, 320]", arg115_1: "f32[960]", arg116_1: "f32[320, 320]", arg117_1: "f32[320]", arg118_1: "f32[1280, 320]", arg119_1: "f32[1280]", arg120_1: "f32[320, 1280]", arg121_1: "f32[320]", arg122_1: "f32[512, 320, 2, 2]", arg123_1: "f32[512]", arg124_1: "f32[512]", arg125_1: "f32[512]", arg126_1: "f32[512, 1, 3, 3]", arg127_1: "f32[512]", arg128_1: "f32[1536, 512]", arg129_1: "f32[1536]", arg130_1: "f32[128, 1, 3, 3]", arg131_1: "f32[128]", arg132_1: "f32[192, 1, 5, 5]", arg133_1: "f32[192]", arg134_1: "f32[192, 1, 7, 7]", arg135_1: "f32[192]", arg136_1: "f32[512, 512]", arg137_1: "f32[512]", arg138_1: "f32[2048, 512]", arg139_1: "f32[2048]", arg140_1: "f32[512, 2048]", arg141_1: "f32[512]", arg142_1: "f32[1536, 512]", arg143_1: "f32[1536]", arg144_1: "f32[512, 512]", arg145_1: "f32[512]", arg146_1: "f32[2048, 512]", arg147_1: "f32[2048]", arg148_1: "f32[512, 2048]", arg149_1: "f32[512]", arg150_1: "f32[1000, 512]", arg151_1: "f32[1000]", arg152_1: "f32[8, 3, 224, 224]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution: "f32[8, 64, 56, 56]" = torch.ops.aten.convolution.default(arg152_1, arg38_1, arg39_1, [4, 4], [0, 0], [1, 1], False, [0, 0], 1);  arg152_1 = arg38_1 = arg39_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:89, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    view: "f32[8, 64, 3136]" = torch.ops.aten.reshape.default(convolution, [8, 64, 3136]);  convolution = None
    permute: "f32[8, 3136, 64]" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:92, code: x = self.norm(x)
    clone: "f32[8, 3136, 64]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
    var_mean = torch.ops.aten.var_mean.correction(clone, [2], correction = 0, keepdim = True)
    getitem: "f32[8, 3136, 1]" = var_mean[0]
    getitem_1: "f32[8, 3136, 1]" = var_mean[1];  var_mean = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:677, code: cls_tokens = cls_token.expand(x.shape[0], -1, -1)
    expand: "f32[8, 1, 64]" = torch.ops.aten.expand.default(arg0_1, [8, -1, -1]);  arg0_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:92, code: x = self.norm(x)
    sub: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(clone, getitem_1);  clone = getitem_1 = None
    add: "f32[8, 3136, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
    rsqrt: "f32[8, 3136, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
    mul: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
    mul_1: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul, arg40_1);  mul = arg40_1 = None
    add_1: "f32[8, 3136, 64]" = torch.ops.aten.add.Tensor(mul_1, arg41_1);  mul_1 = arg41_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:678, code: x = torch.cat((cls_tokens, x), dim=1)
    cat: "f32[8, 3137, 64]" = torch.ops.aten.cat.default([expand, add_1], 1);  expand = add_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    slice_2: "f32[8, 1, 64]" = torch.ops.aten.slice.Tensor(cat, 1, 0, 1)
    slice_4: "f32[8, 3136, 64]" = torch.ops.aten.slice.Tensor(cat, 1, 1, 9223372036854775807);  cat = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:158, code: feat = img_tokens.transpose(1, 2).view(B, C, H, W)
    permute_1: "f32[8, 64, 3136]" = torch.ops.aten.permute.default(slice_4, [0, 2, 1]);  slice_4 = None
    view_1: "f32[8, 64, 56, 56]" = torch.ops.aten.reshape.default(permute_1, [8, 64, 56, 56]);  permute_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:159, code: x = self.proj(feat) + feat
    convolution_1: "f32[8, 64, 56, 56]" = torch.ops.aten.convolution.default(view_1, arg42_1, arg43_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 64)
    add_2: "f32[8, 64, 56, 56]" = torch.ops.aten.add.Tensor(convolution_1, view_1);  convolution_1 = view_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:160, code: x = x.flatten(2).transpose(1, 2)
    view_2: "f32[8, 64, 3136]" = torch.ops.aten.reshape.default(add_2, [8, 64, 3136]);  add_2 = None
    permute_2: "f32[8, 3136, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1]);  view_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:163, code: x = torch.cat((cls_token, x), dim=1)
    cat_1: "f32[8, 3137, 64]" = torch.ops.aten.cat.default([slice_2, permute_2], 1);  slice_2 = permute_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_1 = torch.ops.aten.var_mean.correction(cat_1, [2], correction = 0, keepdim = True)
    getitem_2: "f32[8, 3137, 1]" = var_mean_1[0]
    getitem_3: "f32[8, 3137, 1]" = var_mean_1[1];  var_mean_1 = None
    sub_1: "f32[8, 3137, 64]" = torch.ops.aten.sub.Tensor(cat_1, getitem_3);  getitem_3 = None
    add_3: "f32[8, 3137, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-06);  getitem_2 = None
    rsqrt_1: "f32[8, 3137, 1]" = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
    mul_2: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = rsqrt_1 = None
    mul_3: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(mul_2, arg1_1);  mul_2 = arg1_1 = None
    add_4: "f32[8, 3137, 64]" = torch.ops.aten.add.Tensor(mul_3, arg2_1);  mul_3 = arg2_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:119, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_3: "f32[25096, 64]" = torch.ops.aten.reshape.default(add_4, [25096, 64]);  add_4 = None
    permute_3: "f32[64, 192]" = torch.ops.aten.permute.default(arg44_1, [1, 0]);  arg44_1 = None
    addmm: "f32[25096, 192]" = torch.ops.aten.addmm.default(arg45_1, view_3, permute_3);  arg45_1 = view_3 = permute_3 = None
    view_4: "f32[8, 3137, 192]" = torch.ops.aten.reshape.default(addmm, [8, 3137, 192]);  addmm = None
    view_5: "f32[8, 3137, 3, 8, 8]" = torch.ops.aten.reshape.default(view_4, [8, 3137, 3, 8, 8]);  view_4 = None
    permute_4: "f32[3, 8, 8, 3137, 8]" = torch.ops.aten.permute.default(view_5, [2, 0, 3, 1, 4]);  view_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:120, code: q, k, v = qkv.unbind(0)  # [B, h, N, Ch]
    unbind = torch.ops.aten.unbind.int(permute_4);  permute_4 = None
    getitem_4: "f32[8, 8, 3137, 8]" = unbind[0]
    getitem_5: "f32[8, 8, 3137, 8]" = unbind[1]
    getitem_6: "f32[8, 8, 3137, 8]" = unbind[2];  unbind = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:76, code: v_img = v[:, :, 1:, :]  # [B, h, H*W, Ch]
    slice_11: "f32[8, 8, 3136, 8]" = torch.ops.aten.slice.Tensor(getitem_6, 2, 1, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:78, code: v_img = v_img.transpose(-1, -2).reshape(B, num_heads * C, H, W)
    permute_6: "f32[8, 8, 8, 3136]" = torch.ops.aten.permute.default(slice_11, [0, 1, 3, 2]);  slice_11 = None
    view_12: "f32[8, 64, 56, 56]" = torch.ops.aten.reshape.default(permute_6, [8, 64, 56, 56]);  permute_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:79, code: v_img_list = torch.split(v_img, self.channel_splits, dim=1)  # Split according to channels
    split_with_sizes = torch.ops.aten.split_with_sizes.default(view_12, [16, 24, 24], 1);  view_12 = None
    getitem_7: "f32[8, 16, 56, 56]" = split_with_sizes[0]
    getitem_8: "f32[8, 24, 56, 56]" = split_with_sizes[1]
    getitem_9: "f32[8, 24, 56, 56]" = split_with_sizes[2];  split_with_sizes = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    expand_3: "f32[8, 8, 3137, 8]" = torch.ops.aten.expand.default(getitem_4, [8, 8, 3137, 8])
    clone_3: "f32[8, 8, 3137, 8]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
    view_9: "f32[64, 3137, 8]" = torch.ops.aten.reshape.default(clone_3, [64, 3137, 8]);  clone_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:123, code: k_softmax = k.softmax(dim=2)
    clone_1: "f32[8, 8, 3137, 8]" = torch.ops.aten.clone.default(getitem_5, memory_format = torch.contiguous_format);  getitem_5 = None
    amax: "f32[8, 8, 1, 8]" = torch.ops.aten.amax.default(clone_1, [2], True)
    sub_2: "f32[8, 8, 3137, 8]" = torch.ops.aten.sub.Tensor(clone_1, amax);  clone_1 = amax = None
    exp: "f32[8, 8, 3137, 8]" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
    sum_1: "f32[8, 8, 1, 8]" = torch.ops.aten.sum.dim_IntList(exp, [2], True)
    div: "f32[8, 8, 3137, 8]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    permute_5: "f32[8, 8, 8, 3137]" = torch.ops.aten.permute.default(div, [0, 1, 3, 2]);  div = None
    expand_1: "f32[8, 8, 8, 3137]" = torch.ops.aten.expand.default(permute_5, [8, 8, 8, 3137]);  permute_5 = None
    view_6: "f32[64, 8, 3137]" = torch.ops.aten.reshape.default(expand_1, [64, 8, 3137]);  expand_1 = None
    expand_2: "f32[8, 8, 3137, 8]" = torch.ops.aten.expand.default(getitem_6, [8, 8, 3137, 8]);  getitem_6 = None
    clone_2: "f32[8, 8, 3137, 8]" = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
    view_7: "f32[64, 3137, 8]" = torch.ops.aten.reshape.default(clone_2, [64, 3137, 8]);  clone_2 = None
    bmm: "f32[64, 8, 8]" = torch.ops.aten.bmm.default(view_6, view_7);  view_6 = view_7 = None
    view_8: "f32[8, 8, 8, 8]" = torch.ops.aten.reshape.default(bmm, [8, 8, 8, 8]);  bmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    expand_4: "f32[8, 8, 8, 8]" = torch.ops.aten.expand.default(view_8, [8, 8, 8, 8]);  view_8 = None
    view_10: "f32[64, 8, 8]" = torch.ops.aten.reshape.default(expand_4, [64, 8, 8]);  expand_4 = None
    bmm_1: "f32[64, 3137, 8]" = torch.ops.aten.bmm.default(view_9, view_10);  view_9 = view_10 = None
    view_11: "f32[8, 8, 3137, 8]" = torch.ops.aten.reshape.default(bmm_1, [8, 8, 3137, 8]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:131, code: x = self.scale * factor_att + crpe
    mul_5: "f32[8, 8, 3137, 8]" = torch.ops.aten.mul.Tensor(view_11, 0.3535533905932738);  view_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:75, code: q_img = q[:, :, 1:, :]  # [B, h, H*W, Ch]
    slice_7: "f32[8, 8, 3136, 8]" = torch.ops.aten.slice.Tensor(getitem_4, 2, 1, 9223372036854775807);  getitem_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    convolution_2: "f32[8, 16, 56, 56]" = torch.ops.aten.convolution.default(getitem_7, arg46_1, arg47_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 16);  getitem_7 = None
    convolution_3: "f32[8, 24, 56, 56]" = torch.ops.aten.convolution.default(getitem_8, arg48_1, arg49_1, [1, 1], [2, 2], [1, 1], False, [0, 0], 24);  getitem_8 = None
    convolution_4: "f32[8, 24, 56, 56]" = torch.ops.aten.convolution.default(getitem_9, arg50_1, arg51_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 24);  getitem_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:83, code: conv_v_img = torch.cat(conv_v_img_list, dim=1)
    cat_2: "f32[8, 64, 56, 56]" = torch.ops.aten.cat.default([convolution_2, convolution_3, convolution_4], 1);  convolution_2 = convolution_3 = convolution_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:84, code: conv_v_img = conv_v_img.reshape(B, num_heads, C, H * W).transpose(-1, -2)
    view_13: "f32[8, 8, 8, 3136]" = torch.ops.aten.reshape.default(cat_2, [8, 8, 8, 3136]);  cat_2 = None
    permute_7: "f32[8, 8, 3136, 8]" = torch.ops.aten.permute.default(view_13, [0, 1, 3, 2]);  view_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:86, code: EV_hat = q_img * conv_v_img
    mul_4: "f32[8, 8, 3136, 8]" = torch.ops.aten.mul.Tensor(slice_7, permute_7);  slice_7 = permute_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:87, code: EV_hat = F.pad(EV_hat, (0, 0, 1, 0, 0, 0))  # [B, h, N, Ch].
    constant_pad_nd: "f32[8, 8, 3137, 8]" = torch.ops.aten.constant_pad_nd.default(mul_4, [0, 0, 1, 0, 0, 0], 0.0);  mul_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:131, code: x = self.scale * factor_att + crpe
    add_5: "f32[8, 8, 3137, 8]" = torch.ops.aten.add.Tensor(mul_5, constant_pad_nd);  mul_5 = constant_pad_nd = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:132, code: x = x.transpose(1, 2).reshape(B, N, C)  # [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C]
    permute_8: "f32[8, 3137, 8, 8]" = torch.ops.aten.permute.default(add_5, [0, 2, 1, 3]);  add_5 = None
    clone_4: "f32[8, 3137, 8, 8]" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None
    view_14: "f32[8, 3137, 64]" = torch.ops.aten.reshape.default(clone_4, [8, 3137, 64]);  clone_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:135, code: x = self.proj(x)
    view_15: "f32[25096, 64]" = torch.ops.aten.reshape.default(view_14, [25096, 64]);  view_14 = None
    permute_9: "f32[64, 64]" = torch.ops.aten.permute.default(arg52_1, [1, 0]);  arg52_1 = None
    addmm_1: "f32[25096, 64]" = torch.ops.aten.addmm.default(arg53_1, view_15, permute_9);  arg53_1 = view_15 = permute_9 = None
    view_16: "f32[8, 3137, 64]" = torch.ops.aten.reshape.default(addmm_1, [8, 3137, 64]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:216, code: x = x + self.drop_path(cur)
    add_6: "f32[8, 3137, 64]" = torch.ops.aten.add.Tensor(cat_1, view_16);  cat_1 = view_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_2 = torch.ops.aten.var_mean.correction(add_6, [2], correction = 0, keepdim = True)
    getitem_10: "f32[8, 3137, 1]" = var_mean_2[0]
    getitem_11: "f32[8, 3137, 1]" = var_mean_2[1];  var_mean_2 = None
    sub_3: "f32[8, 3137, 64]" = torch.ops.aten.sub.Tensor(add_6, getitem_11);  getitem_11 = None
    add_7: "f32[8, 3137, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-06);  getitem_10 = None
    rsqrt_2: "f32[8, 3137, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
    mul_6: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = rsqrt_2 = None
    mul_7: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(mul_6, arg3_1);  mul_6 = arg3_1 = None
    add_8: "f32[8, 3137, 64]" = torch.ops.aten.add.Tensor(mul_7, arg4_1);  mul_7 = arg4_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_17: "f32[25096, 64]" = torch.ops.aten.reshape.default(add_8, [25096, 64]);  add_8 = None
    permute_10: "f32[64, 512]" = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
    addmm_2: "f32[25096, 512]" = torch.ops.aten.addmm.default(arg55_1, view_17, permute_10);  arg55_1 = view_17 = permute_10 = None
    view_18: "f32[8, 3137, 512]" = torch.ops.aten.reshape.default(addmm_2, [8, 3137, 512]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_8: "f32[8, 3137, 512]" = torch.ops.aten.mul.Tensor(view_18, 0.5)
    mul_9: "f32[8, 3137, 512]" = torch.ops.aten.mul.Tensor(view_18, 0.7071067811865476);  view_18 = None
    erf: "f32[8, 3137, 512]" = torch.ops.aten.erf.default(mul_9);  mul_9 = None
    add_9: "f32[8, 3137, 512]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_10: "f32[8, 3137, 512]" = torch.ops.aten.mul.Tensor(mul_8, add_9);  mul_8 = add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_19: "f32[25096, 512]" = torch.ops.aten.reshape.default(mul_10, [25096, 512]);  mul_10 = None
    permute_11: "f32[512, 64]" = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
    addmm_3: "f32[25096, 64]" = torch.ops.aten.addmm.default(arg57_1, view_19, permute_11);  arg57_1 = view_19 = permute_11 = None
    view_20: "f32[8, 3137, 64]" = torch.ops.aten.reshape.default(addmm_3, [8, 3137, 64]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:221, code: x = x + self.drop_path(cur)
    add_10: "f32[8, 3137, 64]" = torch.ops.aten.add.Tensor(add_6, view_20);  add_6 = view_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    slice_14: "f32[8, 1, 64]" = torch.ops.aten.slice.Tensor(add_10, 1, 0, 1)
    slice_16: "f32[8, 3136, 64]" = torch.ops.aten.slice.Tensor(add_10, 1, 1, 9223372036854775807);  add_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:158, code: feat = img_tokens.transpose(1, 2).view(B, C, H, W)
    permute_12: "f32[8, 64, 3136]" = torch.ops.aten.permute.default(slice_16, [0, 2, 1]);  slice_16 = None
    view_21: "f32[8, 64, 56, 56]" = torch.ops.aten.reshape.default(permute_12, [8, 64, 56, 56]);  permute_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:159, code: x = self.proj(feat) + feat
    convolution_5: "f32[8, 64, 56, 56]" = torch.ops.aten.convolution.default(view_21, arg42_1, arg43_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 64);  arg42_1 = arg43_1 = None
    add_11: "f32[8, 64, 56, 56]" = torch.ops.aten.add.Tensor(convolution_5, view_21);  convolution_5 = view_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:160, code: x = x.flatten(2).transpose(1, 2)
    view_22: "f32[8, 64, 3136]" = torch.ops.aten.reshape.default(add_11, [8, 64, 3136]);  add_11 = None
    permute_13: "f32[8, 3136, 64]" = torch.ops.aten.permute.default(view_22, [0, 2, 1]);  view_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:163, code: x = torch.cat((cls_token, x), dim=1)
    cat_3: "f32[8, 3137, 64]" = torch.ops.aten.cat.default([slice_14, permute_13], 1);  slice_14 = permute_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_3 = torch.ops.aten.var_mean.correction(cat_3, [2], correction = 0, keepdim = True)
    getitem_12: "f32[8, 3137, 1]" = var_mean_3[0]
    getitem_13: "f32[8, 3137, 1]" = var_mean_3[1];  var_mean_3 = None
    sub_4: "f32[8, 3137, 64]" = torch.ops.aten.sub.Tensor(cat_3, getitem_13);  getitem_13 = None
    add_12: "f32[8, 3137, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-06);  getitem_12 = None
    rsqrt_3: "f32[8, 3137, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
    mul_11: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_3);  sub_4 = rsqrt_3 = None
    mul_12: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(mul_11, arg5_1);  mul_11 = arg5_1 = None
    add_13: "f32[8, 3137, 64]" = torch.ops.aten.add.Tensor(mul_12, arg6_1);  mul_12 = arg6_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:119, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_23: "f32[25096, 64]" = torch.ops.aten.reshape.default(add_13, [25096, 64]);  add_13 = None
    permute_14: "f32[64, 192]" = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
    addmm_4: "f32[25096, 192]" = torch.ops.aten.addmm.default(arg59_1, view_23, permute_14);  arg59_1 = view_23 = permute_14 = None
    view_24: "f32[8, 3137, 192]" = torch.ops.aten.reshape.default(addmm_4, [8, 3137, 192]);  addmm_4 = None
    view_25: "f32[8, 3137, 3, 8, 8]" = torch.ops.aten.reshape.default(view_24, [8, 3137, 3, 8, 8]);  view_24 = None
    permute_15: "f32[3, 8, 8, 3137, 8]" = torch.ops.aten.permute.default(view_25, [2, 0, 3, 1, 4]);  view_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:120, code: q, k, v = qkv.unbind(0)  # [B, h, N, Ch]
    unbind_1 = torch.ops.aten.unbind.int(permute_15);  permute_15 = None
    getitem_14: "f32[8, 8, 3137, 8]" = unbind_1[0]
    getitem_15: "f32[8, 8, 3137, 8]" = unbind_1[1]
    getitem_16: "f32[8, 8, 3137, 8]" = unbind_1[2];  unbind_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:76, code: v_img = v[:, :, 1:, :]  # [B, h, H*W, Ch]
    slice_23: "f32[8, 8, 3136, 8]" = torch.ops.aten.slice.Tensor(getitem_16, 2, 1, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:78, code: v_img = v_img.transpose(-1, -2).reshape(B, num_heads * C, H, W)
    permute_17: "f32[8, 8, 8, 3136]" = torch.ops.aten.permute.default(slice_23, [0, 1, 3, 2]);  slice_23 = None
    view_32: "f32[8, 64, 56, 56]" = torch.ops.aten.reshape.default(permute_17, [8, 64, 56, 56]);  permute_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:79, code: v_img_list = torch.split(v_img, self.channel_splits, dim=1)  # Split according to channels
    split_with_sizes_1 = torch.ops.aten.split_with_sizes.default(view_32, [16, 24, 24], 1);  view_32 = None
    getitem_17: "f32[8, 16, 56, 56]" = split_with_sizes_1[0]
    getitem_18: "f32[8, 24, 56, 56]" = split_with_sizes_1[1]
    getitem_19: "f32[8, 24, 56, 56]" = split_with_sizes_1[2];  split_with_sizes_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    expand_7: "f32[8, 8, 3137, 8]" = torch.ops.aten.expand.default(getitem_14, [8, 8, 3137, 8])
    clone_10: "f32[8, 8, 3137, 8]" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
    view_29: "f32[64, 3137, 8]" = torch.ops.aten.reshape.default(clone_10, [64, 3137, 8]);  clone_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:123, code: k_softmax = k.softmax(dim=2)
    clone_8: "f32[8, 8, 3137, 8]" = torch.ops.aten.clone.default(getitem_15, memory_format = torch.contiguous_format);  getitem_15 = None
    amax_1: "f32[8, 8, 1, 8]" = torch.ops.aten.amax.default(clone_8, [2], True)
    sub_5: "f32[8, 8, 3137, 8]" = torch.ops.aten.sub.Tensor(clone_8, amax_1);  clone_8 = amax_1 = None
    exp_1: "f32[8, 8, 3137, 8]" = torch.ops.aten.exp.default(sub_5);  sub_5 = None
    sum_2: "f32[8, 8, 1, 8]" = torch.ops.aten.sum.dim_IntList(exp_1, [2], True)
    div_1: "f32[8, 8, 3137, 8]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    permute_16: "f32[8, 8, 8, 3137]" = torch.ops.aten.permute.default(div_1, [0, 1, 3, 2]);  div_1 = None
    expand_5: "f32[8, 8, 8, 3137]" = torch.ops.aten.expand.default(permute_16, [8, 8, 8, 3137]);  permute_16 = None
    view_26: "f32[64, 8, 3137]" = torch.ops.aten.reshape.default(expand_5, [64, 8, 3137]);  expand_5 = None
    expand_6: "f32[8, 8, 3137, 8]" = torch.ops.aten.expand.default(getitem_16, [8, 8, 3137, 8]);  getitem_16 = None
    clone_9: "f32[8, 8, 3137, 8]" = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format);  expand_6 = None
    view_27: "f32[64, 3137, 8]" = torch.ops.aten.reshape.default(clone_9, [64, 3137, 8]);  clone_9 = None
    bmm_2: "f32[64, 8, 8]" = torch.ops.aten.bmm.default(view_26, view_27);  view_26 = view_27 = None
    view_28: "f32[8, 8, 8, 8]" = torch.ops.aten.reshape.default(bmm_2, [8, 8, 8, 8]);  bmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    expand_8: "f32[8, 8, 8, 8]" = torch.ops.aten.expand.default(view_28, [8, 8, 8, 8]);  view_28 = None
    view_30: "f32[64, 8, 8]" = torch.ops.aten.reshape.default(expand_8, [64, 8, 8]);  expand_8 = None
    bmm_3: "f32[64, 3137, 8]" = torch.ops.aten.bmm.default(view_29, view_30);  view_29 = view_30 = None
    view_31: "f32[8, 8, 3137, 8]" = torch.ops.aten.reshape.default(bmm_3, [8, 8, 3137, 8]);  bmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:131, code: x = self.scale * factor_att + crpe
    mul_14: "f32[8, 8, 3137, 8]" = torch.ops.aten.mul.Tensor(view_31, 0.3535533905932738);  view_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:75, code: q_img = q[:, :, 1:, :]  # [B, h, H*W, Ch]
    slice_19: "f32[8, 8, 3136, 8]" = torch.ops.aten.slice.Tensor(getitem_14, 2, 1, 9223372036854775807);  getitem_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    convolution_6: "f32[8, 16, 56, 56]" = torch.ops.aten.convolution.default(getitem_17, arg46_1, arg47_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 16);  getitem_17 = arg46_1 = arg47_1 = None
    convolution_7: "f32[8, 24, 56, 56]" = torch.ops.aten.convolution.default(getitem_18, arg48_1, arg49_1, [1, 1], [2, 2], [1, 1], False, [0, 0], 24);  getitem_18 = arg48_1 = arg49_1 = None
    convolution_8: "f32[8, 24, 56, 56]" = torch.ops.aten.convolution.default(getitem_19, arg50_1, arg51_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 24);  getitem_19 = arg50_1 = arg51_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:83, code: conv_v_img = torch.cat(conv_v_img_list, dim=1)
    cat_4: "f32[8, 64, 56, 56]" = torch.ops.aten.cat.default([convolution_6, convolution_7, convolution_8], 1);  convolution_6 = convolution_7 = convolution_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:84, code: conv_v_img = conv_v_img.reshape(B, num_heads, C, H * W).transpose(-1, -2)
    view_33: "f32[8, 8, 8, 3136]" = torch.ops.aten.reshape.default(cat_4, [8, 8, 8, 3136]);  cat_4 = None
    permute_18: "f32[8, 8, 3136, 8]" = torch.ops.aten.permute.default(view_33, [0, 1, 3, 2]);  view_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:86, code: EV_hat = q_img * conv_v_img
    mul_13: "f32[8, 8, 3136, 8]" = torch.ops.aten.mul.Tensor(slice_19, permute_18);  slice_19 = permute_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:87, code: EV_hat = F.pad(EV_hat, (0, 0, 1, 0, 0, 0))  # [B, h, N, Ch].
    constant_pad_nd_1: "f32[8, 8, 3137, 8]" = torch.ops.aten.constant_pad_nd.default(mul_13, [0, 0, 1, 0, 0, 0], 0.0);  mul_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:131, code: x = self.scale * factor_att + crpe
    add_14: "f32[8, 8, 3137, 8]" = torch.ops.aten.add.Tensor(mul_14, constant_pad_nd_1);  mul_14 = constant_pad_nd_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:132, code: x = x.transpose(1, 2).reshape(B, N, C)  # [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C]
    permute_19: "f32[8, 3137, 8, 8]" = torch.ops.aten.permute.default(add_14, [0, 2, 1, 3]);  add_14 = None
    clone_11: "f32[8, 3137, 8, 8]" = torch.ops.aten.clone.default(permute_19, memory_format = torch.contiguous_format);  permute_19 = None
    view_34: "f32[8, 3137, 64]" = torch.ops.aten.reshape.default(clone_11, [8, 3137, 64]);  clone_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:135, code: x = self.proj(x)
    view_35: "f32[25096, 64]" = torch.ops.aten.reshape.default(view_34, [25096, 64]);  view_34 = None
    permute_20: "f32[64, 64]" = torch.ops.aten.permute.default(arg60_1, [1, 0]);  arg60_1 = None
    addmm_5: "f32[25096, 64]" = torch.ops.aten.addmm.default(arg61_1, view_35, permute_20);  arg61_1 = view_35 = permute_20 = None
    view_36: "f32[8, 3137, 64]" = torch.ops.aten.reshape.default(addmm_5, [8, 3137, 64]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:216, code: x = x + self.drop_path(cur)
    add_15: "f32[8, 3137, 64]" = torch.ops.aten.add.Tensor(cat_3, view_36);  cat_3 = view_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_4 = torch.ops.aten.var_mean.correction(add_15, [2], correction = 0, keepdim = True)
    getitem_20: "f32[8, 3137, 1]" = var_mean_4[0]
    getitem_21: "f32[8, 3137, 1]" = var_mean_4[1];  var_mean_4 = None
    sub_6: "f32[8, 3137, 64]" = torch.ops.aten.sub.Tensor(add_15, getitem_21);  getitem_21 = None
    add_16: "f32[8, 3137, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-06);  getitem_20 = None
    rsqrt_4: "f32[8, 3137, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
    mul_15: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = rsqrt_4 = None
    mul_16: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(mul_15, arg7_1);  mul_15 = arg7_1 = None
    add_17: "f32[8, 3137, 64]" = torch.ops.aten.add.Tensor(mul_16, arg8_1);  mul_16 = arg8_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_37: "f32[25096, 64]" = torch.ops.aten.reshape.default(add_17, [25096, 64]);  add_17 = None
    permute_21: "f32[64, 512]" = torch.ops.aten.permute.default(arg62_1, [1, 0]);  arg62_1 = None
    addmm_6: "f32[25096, 512]" = torch.ops.aten.addmm.default(arg63_1, view_37, permute_21);  arg63_1 = view_37 = permute_21 = None
    view_38: "f32[8, 3137, 512]" = torch.ops.aten.reshape.default(addmm_6, [8, 3137, 512]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_17: "f32[8, 3137, 512]" = torch.ops.aten.mul.Tensor(view_38, 0.5)
    mul_18: "f32[8, 3137, 512]" = torch.ops.aten.mul.Tensor(view_38, 0.7071067811865476);  view_38 = None
    erf_1: "f32[8, 3137, 512]" = torch.ops.aten.erf.default(mul_18);  mul_18 = None
    add_18: "f32[8, 3137, 512]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_19: "f32[8, 3137, 512]" = torch.ops.aten.mul.Tensor(mul_17, add_18);  mul_17 = add_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_39: "f32[25096, 512]" = torch.ops.aten.reshape.default(mul_19, [25096, 512]);  mul_19 = None
    permute_22: "f32[512, 64]" = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
    addmm_7: "f32[25096, 64]" = torch.ops.aten.addmm.default(arg65_1, view_39, permute_22);  arg65_1 = view_39 = permute_22 = None
    view_40: "f32[8, 3137, 64]" = torch.ops.aten.reshape.default(addmm_7, [8, 3137, 64]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:221, code: x = x + self.drop_path(cur)
    add_19: "f32[8, 3137, 64]" = torch.ops.aten.add.Tensor(add_15, view_40);  add_15 = view_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:684, code: return x[:, 1:, :]
    slice_26: "f32[8, 3136, 64]" = torch.ops.aten.slice.Tensor(add_19, 1, 1, 9223372036854775807);  add_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:579, code: x1_nocls = remove_cls(x1).reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
    view_41: "f32[8, 56, 56, 64]" = torch.ops.aten.reshape.default(slice_26, [8, 56, 56, 64]);  slice_26 = None
    permute_23: "f32[8, 64, 56, 56]" = torch.ops.aten.permute.default(view_41, [0, 3, 1, 2]);  view_41 = None
    clone_15: "f32[8, 64, 56, 56]" = torch.ops.aten.clone.default(permute_23, memory_format = torch.contiguous_format);  permute_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution_9: "f32[8, 128, 28, 28]" = torch.ops.aten.convolution.default(clone_15, arg66_1, arg67_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  clone_15 = arg66_1 = arg67_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:89, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    view_42: "f32[8, 128, 784]" = torch.ops.aten.reshape.default(convolution_9, [8, 128, 784]);  convolution_9 = None
    permute_24: "f32[8, 784, 128]" = torch.ops.aten.permute.default(view_42, [0, 2, 1]);  view_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:92, code: x = self.norm(x)
    clone_16: "f32[8, 784, 128]" = torch.ops.aten.clone.default(permute_24, memory_format = torch.contiguous_format);  permute_24 = None
    var_mean_5 = torch.ops.aten.var_mean.correction(clone_16, [2], correction = 0, keepdim = True)
    getitem_22: "f32[8, 784, 1]" = var_mean_5[0]
    getitem_23: "f32[8, 784, 1]" = var_mean_5[1];  var_mean_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:677, code: cls_tokens = cls_token.expand(x.shape[0], -1, -1)
    expand_9: "f32[8, 1, 128]" = torch.ops.aten.expand.default(arg9_1, [8, -1, -1]);  arg9_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:92, code: x = self.norm(x)
    sub_7: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(clone_16, getitem_23);  clone_16 = getitem_23 = None
    add_20: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
    rsqrt_5: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
    mul_20: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_5);  sub_7 = rsqrt_5 = None
    mul_21: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_20, arg68_1);  mul_20 = arg68_1 = None
    add_21: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(mul_21, arg69_1);  mul_21 = arg69_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:678, code: x = torch.cat((cls_tokens, x), dim=1)
    cat_5: "f32[8, 785, 128]" = torch.ops.aten.cat.default([expand_9, add_21], 1);  expand_9 = add_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    slice_29: "f32[8, 1, 128]" = torch.ops.aten.slice.Tensor(cat_5, 1, 0, 1)
    slice_31: "f32[8, 784, 128]" = torch.ops.aten.slice.Tensor(cat_5, 1, 1, 9223372036854775807);  cat_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:158, code: feat = img_tokens.transpose(1, 2).view(B, C, H, W)
    permute_25: "f32[8, 128, 784]" = torch.ops.aten.permute.default(slice_31, [0, 2, 1]);  slice_31 = None
    view_43: "f32[8, 128, 28, 28]" = torch.ops.aten.reshape.default(permute_25, [8, 128, 28, 28]);  permute_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:159, code: x = self.proj(feat) + feat
    convolution_10: "f32[8, 128, 28, 28]" = torch.ops.aten.convolution.default(view_43, arg70_1, arg71_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 128)
    add_22: "f32[8, 128, 28, 28]" = torch.ops.aten.add.Tensor(convolution_10, view_43);  convolution_10 = view_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:160, code: x = x.flatten(2).transpose(1, 2)
    view_44: "f32[8, 128, 784]" = torch.ops.aten.reshape.default(add_22, [8, 128, 784]);  add_22 = None
    permute_26: "f32[8, 784, 128]" = torch.ops.aten.permute.default(view_44, [0, 2, 1]);  view_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:163, code: x = torch.cat((cls_token, x), dim=1)
    cat_6: "f32[8, 785, 128]" = torch.ops.aten.cat.default([slice_29, permute_26], 1);  slice_29 = permute_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_6 = torch.ops.aten.var_mean.correction(cat_6, [2], correction = 0, keepdim = True)
    getitem_24: "f32[8, 785, 1]" = var_mean_6[0]
    getitem_25: "f32[8, 785, 1]" = var_mean_6[1];  var_mean_6 = None
    sub_8: "f32[8, 785, 128]" = torch.ops.aten.sub.Tensor(cat_6, getitem_25);  getitem_25 = None
    add_23: "f32[8, 785, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-06);  getitem_24 = None
    rsqrt_6: "f32[8, 785, 1]" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
    mul_22: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_6);  sub_8 = rsqrt_6 = None
    mul_23: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(mul_22, arg10_1);  mul_22 = arg10_1 = None
    add_24: "f32[8, 785, 128]" = torch.ops.aten.add.Tensor(mul_23, arg11_1);  mul_23 = arg11_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:119, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_45: "f32[6280, 128]" = torch.ops.aten.reshape.default(add_24, [6280, 128]);  add_24 = None
    permute_27: "f32[128, 384]" = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
    addmm_8: "f32[6280, 384]" = torch.ops.aten.addmm.default(arg73_1, view_45, permute_27);  arg73_1 = view_45 = permute_27 = None
    view_46: "f32[8, 785, 384]" = torch.ops.aten.reshape.default(addmm_8, [8, 785, 384]);  addmm_8 = None
    view_47: "f32[8, 785, 3, 8, 16]" = torch.ops.aten.reshape.default(view_46, [8, 785, 3, 8, 16]);  view_46 = None
    permute_28: "f32[3, 8, 8, 785, 16]" = torch.ops.aten.permute.default(view_47, [2, 0, 3, 1, 4]);  view_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:120, code: q, k, v = qkv.unbind(0)  # [B, h, N, Ch]
    unbind_2 = torch.ops.aten.unbind.int(permute_28);  permute_28 = None
    getitem_26: "f32[8, 8, 785, 16]" = unbind_2[0]
    getitem_27: "f32[8, 8, 785, 16]" = unbind_2[1]
    getitem_28: "f32[8, 8, 785, 16]" = unbind_2[2];  unbind_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:76, code: v_img = v[:, :, 1:, :]  # [B, h, H*W, Ch]
    slice_38: "f32[8, 8, 784, 16]" = torch.ops.aten.slice.Tensor(getitem_28, 2, 1, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:78, code: v_img = v_img.transpose(-1, -2).reshape(B, num_heads * C, H, W)
    permute_30: "f32[8, 8, 16, 784]" = torch.ops.aten.permute.default(slice_38, [0, 1, 3, 2]);  slice_38 = None
    view_54: "f32[8, 128, 28, 28]" = torch.ops.aten.reshape.default(permute_30, [8, 128, 28, 28]);  permute_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:79, code: v_img_list = torch.split(v_img, self.channel_splits, dim=1)  # Split according to channels
    split_with_sizes_2 = torch.ops.aten.split_with_sizes.default(view_54, [32, 48, 48], 1);  view_54 = None
    getitem_29: "f32[8, 32, 28, 28]" = split_with_sizes_2[0]
    getitem_30: "f32[8, 48, 28, 28]" = split_with_sizes_2[1]
    getitem_31: "f32[8, 48, 28, 28]" = split_with_sizes_2[2];  split_with_sizes_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    expand_12: "f32[8, 8, 785, 16]" = torch.ops.aten.expand.default(getitem_26, [8, 8, 785, 16])
    clone_19: "f32[8, 8, 785, 16]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
    view_51: "f32[64, 785, 16]" = torch.ops.aten.reshape.default(clone_19, [64, 785, 16]);  clone_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:123, code: k_softmax = k.softmax(dim=2)
    clone_17: "f32[8, 8, 785, 16]" = torch.ops.aten.clone.default(getitem_27, memory_format = torch.contiguous_format);  getitem_27 = None
    amax_2: "f32[8, 8, 1, 16]" = torch.ops.aten.amax.default(clone_17, [2], True)
    sub_9: "f32[8, 8, 785, 16]" = torch.ops.aten.sub.Tensor(clone_17, amax_2);  clone_17 = amax_2 = None
    exp_2: "f32[8, 8, 785, 16]" = torch.ops.aten.exp.default(sub_9);  sub_9 = None
    sum_3: "f32[8, 8, 1, 16]" = torch.ops.aten.sum.dim_IntList(exp_2, [2], True)
    div_2: "f32[8, 8, 785, 16]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    permute_29: "f32[8, 8, 16, 785]" = torch.ops.aten.permute.default(div_2, [0, 1, 3, 2]);  div_2 = None
    expand_10: "f32[8, 8, 16, 785]" = torch.ops.aten.expand.default(permute_29, [8, 8, 16, 785]);  permute_29 = None
    view_48: "f32[64, 16, 785]" = torch.ops.aten.reshape.default(expand_10, [64, 16, 785]);  expand_10 = None
    expand_11: "f32[8, 8, 785, 16]" = torch.ops.aten.expand.default(getitem_28, [8, 8, 785, 16]);  getitem_28 = None
    clone_18: "f32[8, 8, 785, 16]" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
    view_49: "f32[64, 785, 16]" = torch.ops.aten.reshape.default(clone_18, [64, 785, 16]);  clone_18 = None
    bmm_4: "f32[64, 16, 16]" = torch.ops.aten.bmm.default(view_48, view_49);  view_48 = view_49 = None
    view_50: "f32[8, 8, 16, 16]" = torch.ops.aten.reshape.default(bmm_4, [8, 8, 16, 16]);  bmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    expand_13: "f32[8, 8, 16, 16]" = torch.ops.aten.expand.default(view_50, [8, 8, 16, 16]);  view_50 = None
    view_52: "f32[64, 16, 16]" = torch.ops.aten.reshape.default(expand_13, [64, 16, 16]);  expand_13 = None
    bmm_5: "f32[64, 785, 16]" = torch.ops.aten.bmm.default(view_51, view_52);  view_51 = view_52 = None
    view_53: "f32[8, 8, 785, 16]" = torch.ops.aten.reshape.default(bmm_5, [8, 8, 785, 16]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:131, code: x = self.scale * factor_att + crpe
    mul_25: "f32[8, 8, 785, 16]" = torch.ops.aten.mul.Tensor(view_53, 0.25);  view_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:75, code: q_img = q[:, :, 1:, :]  # [B, h, H*W, Ch]
    slice_34: "f32[8, 8, 784, 16]" = torch.ops.aten.slice.Tensor(getitem_26, 2, 1, 9223372036854775807);  getitem_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    convolution_11: "f32[8, 32, 28, 28]" = torch.ops.aten.convolution.default(getitem_29, arg74_1, arg75_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 32);  getitem_29 = None
    convolution_12: "f32[8, 48, 28, 28]" = torch.ops.aten.convolution.default(getitem_30, arg76_1, arg77_1, [1, 1], [2, 2], [1, 1], False, [0, 0], 48);  getitem_30 = None
    convolution_13: "f32[8, 48, 28, 28]" = torch.ops.aten.convolution.default(getitem_31, arg78_1, arg79_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 48);  getitem_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:83, code: conv_v_img = torch.cat(conv_v_img_list, dim=1)
    cat_7: "f32[8, 128, 28, 28]" = torch.ops.aten.cat.default([convolution_11, convolution_12, convolution_13], 1);  convolution_11 = convolution_12 = convolution_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:84, code: conv_v_img = conv_v_img.reshape(B, num_heads, C, H * W).transpose(-1, -2)
    view_55: "f32[8, 8, 16, 784]" = torch.ops.aten.reshape.default(cat_7, [8, 8, 16, 784]);  cat_7 = None
    permute_31: "f32[8, 8, 784, 16]" = torch.ops.aten.permute.default(view_55, [0, 1, 3, 2]);  view_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:86, code: EV_hat = q_img * conv_v_img
    mul_24: "f32[8, 8, 784, 16]" = torch.ops.aten.mul.Tensor(slice_34, permute_31);  slice_34 = permute_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:87, code: EV_hat = F.pad(EV_hat, (0, 0, 1, 0, 0, 0))  # [B, h, N, Ch].
    constant_pad_nd_2: "f32[8, 8, 785, 16]" = torch.ops.aten.constant_pad_nd.default(mul_24, [0, 0, 1, 0, 0, 0], 0.0);  mul_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:131, code: x = self.scale * factor_att + crpe
    add_25: "f32[8, 8, 785, 16]" = torch.ops.aten.add.Tensor(mul_25, constant_pad_nd_2);  mul_25 = constant_pad_nd_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:132, code: x = x.transpose(1, 2).reshape(B, N, C)  # [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C]
    permute_32: "f32[8, 785, 8, 16]" = torch.ops.aten.permute.default(add_25, [0, 2, 1, 3]);  add_25 = None
    clone_20: "f32[8, 785, 8, 16]" = torch.ops.aten.clone.default(permute_32, memory_format = torch.contiguous_format);  permute_32 = None
    view_56: "f32[8, 785, 128]" = torch.ops.aten.reshape.default(clone_20, [8, 785, 128]);  clone_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:135, code: x = self.proj(x)
    view_57: "f32[6280, 128]" = torch.ops.aten.reshape.default(view_56, [6280, 128]);  view_56 = None
    permute_33: "f32[128, 128]" = torch.ops.aten.permute.default(arg80_1, [1, 0]);  arg80_1 = None
    addmm_9: "f32[6280, 128]" = torch.ops.aten.addmm.default(arg81_1, view_57, permute_33);  arg81_1 = view_57 = permute_33 = None
    view_58: "f32[8, 785, 128]" = torch.ops.aten.reshape.default(addmm_9, [8, 785, 128]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:216, code: x = x + self.drop_path(cur)
    add_26: "f32[8, 785, 128]" = torch.ops.aten.add.Tensor(cat_6, view_58);  cat_6 = view_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_7 = torch.ops.aten.var_mean.correction(add_26, [2], correction = 0, keepdim = True)
    getitem_32: "f32[8, 785, 1]" = var_mean_7[0]
    getitem_33: "f32[8, 785, 1]" = var_mean_7[1];  var_mean_7 = None
    sub_10: "f32[8, 785, 128]" = torch.ops.aten.sub.Tensor(add_26, getitem_33);  getitem_33 = None
    add_27: "f32[8, 785, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-06);  getitem_32 = None
    rsqrt_7: "f32[8, 785, 1]" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
    mul_26: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_7);  sub_10 = rsqrt_7 = None
    mul_27: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(mul_26, arg12_1);  mul_26 = arg12_1 = None
    add_28: "f32[8, 785, 128]" = torch.ops.aten.add.Tensor(mul_27, arg13_1);  mul_27 = arg13_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_59: "f32[6280, 128]" = torch.ops.aten.reshape.default(add_28, [6280, 128]);  add_28 = None
    permute_34: "f32[128, 1024]" = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
    addmm_10: "f32[6280, 1024]" = torch.ops.aten.addmm.default(arg83_1, view_59, permute_34);  arg83_1 = view_59 = permute_34 = None
    view_60: "f32[8, 785, 1024]" = torch.ops.aten.reshape.default(addmm_10, [8, 785, 1024]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_28: "f32[8, 785, 1024]" = torch.ops.aten.mul.Tensor(view_60, 0.5)
    mul_29: "f32[8, 785, 1024]" = torch.ops.aten.mul.Tensor(view_60, 0.7071067811865476);  view_60 = None
    erf_2: "f32[8, 785, 1024]" = torch.ops.aten.erf.default(mul_29);  mul_29 = None
    add_29: "f32[8, 785, 1024]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_30: "f32[8, 785, 1024]" = torch.ops.aten.mul.Tensor(mul_28, add_29);  mul_28 = add_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_61: "f32[6280, 1024]" = torch.ops.aten.reshape.default(mul_30, [6280, 1024]);  mul_30 = None
    permute_35: "f32[1024, 128]" = torch.ops.aten.permute.default(arg84_1, [1, 0]);  arg84_1 = None
    addmm_11: "f32[6280, 128]" = torch.ops.aten.addmm.default(arg85_1, view_61, permute_35);  arg85_1 = view_61 = permute_35 = None
    view_62: "f32[8, 785, 128]" = torch.ops.aten.reshape.default(addmm_11, [8, 785, 128]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:221, code: x = x + self.drop_path(cur)
    add_30: "f32[8, 785, 128]" = torch.ops.aten.add.Tensor(add_26, view_62);  add_26 = view_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    slice_41: "f32[8, 1, 128]" = torch.ops.aten.slice.Tensor(add_30, 1, 0, 1)
    slice_43: "f32[8, 784, 128]" = torch.ops.aten.slice.Tensor(add_30, 1, 1, 9223372036854775807);  add_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:158, code: feat = img_tokens.transpose(1, 2).view(B, C, H, W)
    permute_36: "f32[8, 128, 784]" = torch.ops.aten.permute.default(slice_43, [0, 2, 1]);  slice_43 = None
    view_63: "f32[8, 128, 28, 28]" = torch.ops.aten.reshape.default(permute_36, [8, 128, 28, 28]);  permute_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:159, code: x = self.proj(feat) + feat
    convolution_14: "f32[8, 128, 28, 28]" = torch.ops.aten.convolution.default(view_63, arg70_1, arg71_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 128);  arg70_1 = arg71_1 = None
    add_31: "f32[8, 128, 28, 28]" = torch.ops.aten.add.Tensor(convolution_14, view_63);  convolution_14 = view_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:160, code: x = x.flatten(2).transpose(1, 2)
    view_64: "f32[8, 128, 784]" = torch.ops.aten.reshape.default(add_31, [8, 128, 784]);  add_31 = None
    permute_37: "f32[8, 784, 128]" = torch.ops.aten.permute.default(view_64, [0, 2, 1]);  view_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:163, code: x = torch.cat((cls_token, x), dim=1)
    cat_8: "f32[8, 785, 128]" = torch.ops.aten.cat.default([slice_41, permute_37], 1);  slice_41 = permute_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_8 = torch.ops.aten.var_mean.correction(cat_8, [2], correction = 0, keepdim = True)
    getitem_34: "f32[8, 785, 1]" = var_mean_8[0]
    getitem_35: "f32[8, 785, 1]" = var_mean_8[1];  var_mean_8 = None
    sub_11: "f32[8, 785, 128]" = torch.ops.aten.sub.Tensor(cat_8, getitem_35);  getitem_35 = None
    add_32: "f32[8, 785, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-06);  getitem_34 = None
    rsqrt_8: "f32[8, 785, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
    mul_31: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_8);  sub_11 = rsqrt_8 = None
    mul_32: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(mul_31, arg14_1);  mul_31 = arg14_1 = None
    add_33: "f32[8, 785, 128]" = torch.ops.aten.add.Tensor(mul_32, arg15_1);  mul_32 = arg15_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:119, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_65: "f32[6280, 128]" = torch.ops.aten.reshape.default(add_33, [6280, 128]);  add_33 = None
    permute_38: "f32[128, 384]" = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
    addmm_12: "f32[6280, 384]" = torch.ops.aten.addmm.default(arg87_1, view_65, permute_38);  arg87_1 = view_65 = permute_38 = None
    view_66: "f32[8, 785, 384]" = torch.ops.aten.reshape.default(addmm_12, [8, 785, 384]);  addmm_12 = None
    view_67: "f32[8, 785, 3, 8, 16]" = torch.ops.aten.reshape.default(view_66, [8, 785, 3, 8, 16]);  view_66 = None
    permute_39: "f32[3, 8, 8, 785, 16]" = torch.ops.aten.permute.default(view_67, [2, 0, 3, 1, 4]);  view_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:120, code: q, k, v = qkv.unbind(0)  # [B, h, N, Ch]
    unbind_3 = torch.ops.aten.unbind.int(permute_39);  permute_39 = None
    getitem_36: "f32[8, 8, 785, 16]" = unbind_3[0]
    getitem_37: "f32[8, 8, 785, 16]" = unbind_3[1]
    getitem_38: "f32[8, 8, 785, 16]" = unbind_3[2];  unbind_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:76, code: v_img = v[:, :, 1:, :]  # [B, h, H*W, Ch]
    slice_50: "f32[8, 8, 784, 16]" = torch.ops.aten.slice.Tensor(getitem_38, 2, 1, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:78, code: v_img = v_img.transpose(-1, -2).reshape(B, num_heads * C, H, W)
    permute_41: "f32[8, 8, 16, 784]" = torch.ops.aten.permute.default(slice_50, [0, 1, 3, 2]);  slice_50 = None
    view_74: "f32[8, 128, 28, 28]" = torch.ops.aten.reshape.default(permute_41, [8, 128, 28, 28]);  permute_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:79, code: v_img_list = torch.split(v_img, self.channel_splits, dim=1)  # Split according to channels
    split_with_sizes_3 = torch.ops.aten.split_with_sizes.default(view_74, [32, 48, 48], 1);  view_74 = None
    getitem_39: "f32[8, 32, 28, 28]" = split_with_sizes_3[0]
    getitem_40: "f32[8, 48, 28, 28]" = split_with_sizes_3[1]
    getitem_41: "f32[8, 48, 28, 28]" = split_with_sizes_3[2];  split_with_sizes_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    expand_16: "f32[8, 8, 785, 16]" = torch.ops.aten.expand.default(getitem_36, [8, 8, 785, 16])
    clone_26: "f32[8, 8, 785, 16]" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
    view_71: "f32[64, 785, 16]" = torch.ops.aten.reshape.default(clone_26, [64, 785, 16]);  clone_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:123, code: k_softmax = k.softmax(dim=2)
    clone_24: "f32[8, 8, 785, 16]" = torch.ops.aten.clone.default(getitem_37, memory_format = torch.contiguous_format);  getitem_37 = None
    amax_3: "f32[8, 8, 1, 16]" = torch.ops.aten.amax.default(clone_24, [2], True)
    sub_12: "f32[8, 8, 785, 16]" = torch.ops.aten.sub.Tensor(clone_24, amax_3);  clone_24 = amax_3 = None
    exp_3: "f32[8, 8, 785, 16]" = torch.ops.aten.exp.default(sub_12);  sub_12 = None
    sum_4: "f32[8, 8, 1, 16]" = torch.ops.aten.sum.dim_IntList(exp_3, [2], True)
    div_3: "f32[8, 8, 785, 16]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    permute_40: "f32[8, 8, 16, 785]" = torch.ops.aten.permute.default(div_3, [0, 1, 3, 2]);  div_3 = None
    expand_14: "f32[8, 8, 16, 785]" = torch.ops.aten.expand.default(permute_40, [8, 8, 16, 785]);  permute_40 = None
    view_68: "f32[64, 16, 785]" = torch.ops.aten.reshape.default(expand_14, [64, 16, 785]);  expand_14 = None
    expand_15: "f32[8, 8, 785, 16]" = torch.ops.aten.expand.default(getitem_38, [8, 8, 785, 16]);  getitem_38 = None
    clone_25: "f32[8, 8, 785, 16]" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
    view_69: "f32[64, 785, 16]" = torch.ops.aten.reshape.default(clone_25, [64, 785, 16]);  clone_25 = None
    bmm_6: "f32[64, 16, 16]" = torch.ops.aten.bmm.default(view_68, view_69);  view_68 = view_69 = None
    view_70: "f32[8, 8, 16, 16]" = torch.ops.aten.reshape.default(bmm_6, [8, 8, 16, 16]);  bmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    expand_17: "f32[8, 8, 16, 16]" = torch.ops.aten.expand.default(view_70, [8, 8, 16, 16]);  view_70 = None
    view_72: "f32[64, 16, 16]" = torch.ops.aten.reshape.default(expand_17, [64, 16, 16]);  expand_17 = None
    bmm_7: "f32[64, 785, 16]" = torch.ops.aten.bmm.default(view_71, view_72);  view_71 = view_72 = None
    view_73: "f32[8, 8, 785, 16]" = torch.ops.aten.reshape.default(bmm_7, [8, 8, 785, 16]);  bmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:131, code: x = self.scale * factor_att + crpe
    mul_34: "f32[8, 8, 785, 16]" = torch.ops.aten.mul.Tensor(view_73, 0.25);  view_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:75, code: q_img = q[:, :, 1:, :]  # [B, h, H*W, Ch]
    slice_46: "f32[8, 8, 784, 16]" = torch.ops.aten.slice.Tensor(getitem_36, 2, 1, 9223372036854775807);  getitem_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    convolution_15: "f32[8, 32, 28, 28]" = torch.ops.aten.convolution.default(getitem_39, arg74_1, arg75_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 32);  getitem_39 = arg74_1 = arg75_1 = None
    convolution_16: "f32[8, 48, 28, 28]" = torch.ops.aten.convolution.default(getitem_40, arg76_1, arg77_1, [1, 1], [2, 2], [1, 1], False, [0, 0], 48);  getitem_40 = arg76_1 = arg77_1 = None
    convolution_17: "f32[8, 48, 28, 28]" = torch.ops.aten.convolution.default(getitem_41, arg78_1, arg79_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 48);  getitem_41 = arg78_1 = arg79_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:83, code: conv_v_img = torch.cat(conv_v_img_list, dim=1)
    cat_9: "f32[8, 128, 28, 28]" = torch.ops.aten.cat.default([convolution_15, convolution_16, convolution_17], 1);  convolution_15 = convolution_16 = convolution_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:84, code: conv_v_img = conv_v_img.reshape(B, num_heads, C, H * W).transpose(-1, -2)
    view_75: "f32[8, 8, 16, 784]" = torch.ops.aten.reshape.default(cat_9, [8, 8, 16, 784]);  cat_9 = None
    permute_42: "f32[8, 8, 784, 16]" = torch.ops.aten.permute.default(view_75, [0, 1, 3, 2]);  view_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:86, code: EV_hat = q_img * conv_v_img
    mul_33: "f32[8, 8, 784, 16]" = torch.ops.aten.mul.Tensor(slice_46, permute_42);  slice_46 = permute_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:87, code: EV_hat = F.pad(EV_hat, (0, 0, 1, 0, 0, 0))  # [B, h, N, Ch].
    constant_pad_nd_3: "f32[8, 8, 785, 16]" = torch.ops.aten.constant_pad_nd.default(mul_33, [0, 0, 1, 0, 0, 0], 0.0);  mul_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:131, code: x = self.scale * factor_att + crpe
    add_34: "f32[8, 8, 785, 16]" = torch.ops.aten.add.Tensor(mul_34, constant_pad_nd_3);  mul_34 = constant_pad_nd_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:132, code: x = x.transpose(1, 2).reshape(B, N, C)  # [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C]
    permute_43: "f32[8, 785, 8, 16]" = torch.ops.aten.permute.default(add_34, [0, 2, 1, 3]);  add_34 = None
    clone_27: "f32[8, 785, 8, 16]" = torch.ops.aten.clone.default(permute_43, memory_format = torch.contiguous_format);  permute_43 = None
    view_76: "f32[8, 785, 128]" = torch.ops.aten.reshape.default(clone_27, [8, 785, 128]);  clone_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:135, code: x = self.proj(x)
    view_77: "f32[6280, 128]" = torch.ops.aten.reshape.default(view_76, [6280, 128]);  view_76 = None
    permute_44: "f32[128, 128]" = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
    addmm_13: "f32[6280, 128]" = torch.ops.aten.addmm.default(arg89_1, view_77, permute_44);  arg89_1 = view_77 = permute_44 = None
    view_78: "f32[8, 785, 128]" = torch.ops.aten.reshape.default(addmm_13, [8, 785, 128]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:216, code: x = x + self.drop_path(cur)
    add_35: "f32[8, 785, 128]" = torch.ops.aten.add.Tensor(cat_8, view_78);  cat_8 = view_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_9 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
    getitem_42: "f32[8, 785, 1]" = var_mean_9[0]
    getitem_43: "f32[8, 785, 1]" = var_mean_9[1];  var_mean_9 = None
    sub_13: "f32[8, 785, 128]" = torch.ops.aten.sub.Tensor(add_35, getitem_43);  getitem_43 = None
    add_36: "f32[8, 785, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-06);  getitem_42 = None
    rsqrt_9: "f32[8, 785, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
    mul_35: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_9);  sub_13 = rsqrt_9 = None
    mul_36: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(mul_35, arg16_1);  mul_35 = arg16_1 = None
    add_37: "f32[8, 785, 128]" = torch.ops.aten.add.Tensor(mul_36, arg17_1);  mul_36 = arg17_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_79: "f32[6280, 128]" = torch.ops.aten.reshape.default(add_37, [6280, 128]);  add_37 = None
    permute_45: "f32[128, 1024]" = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
    addmm_14: "f32[6280, 1024]" = torch.ops.aten.addmm.default(arg91_1, view_79, permute_45);  arg91_1 = view_79 = permute_45 = None
    view_80: "f32[8, 785, 1024]" = torch.ops.aten.reshape.default(addmm_14, [8, 785, 1024]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_37: "f32[8, 785, 1024]" = torch.ops.aten.mul.Tensor(view_80, 0.5)
    mul_38: "f32[8, 785, 1024]" = torch.ops.aten.mul.Tensor(view_80, 0.7071067811865476);  view_80 = None
    erf_3: "f32[8, 785, 1024]" = torch.ops.aten.erf.default(mul_38);  mul_38 = None
    add_38: "f32[8, 785, 1024]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_39: "f32[8, 785, 1024]" = torch.ops.aten.mul.Tensor(mul_37, add_38);  mul_37 = add_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_81: "f32[6280, 1024]" = torch.ops.aten.reshape.default(mul_39, [6280, 1024]);  mul_39 = None
    permute_46: "f32[1024, 128]" = torch.ops.aten.permute.default(arg92_1, [1, 0]);  arg92_1 = None
    addmm_15: "f32[6280, 128]" = torch.ops.aten.addmm.default(arg93_1, view_81, permute_46);  arg93_1 = view_81 = permute_46 = None
    view_82: "f32[8, 785, 128]" = torch.ops.aten.reshape.default(addmm_15, [8, 785, 128]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:221, code: x = x + self.drop_path(cur)
    add_39: "f32[8, 785, 128]" = torch.ops.aten.add.Tensor(add_35, view_82);  add_35 = view_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:684, code: return x[:, 1:, :]
    slice_53: "f32[8, 784, 128]" = torch.ops.aten.slice.Tensor(add_39, 1, 1, 9223372036854775807);  add_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:587, code: x2_nocls = remove_cls(x2).reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()
    view_83: "f32[8, 28, 28, 128]" = torch.ops.aten.reshape.default(slice_53, [8, 28, 28, 128]);  slice_53 = None
    permute_47: "f32[8, 128, 28, 28]" = torch.ops.aten.permute.default(view_83, [0, 3, 1, 2]);  view_83 = None
    clone_31: "f32[8, 128, 28, 28]" = torch.ops.aten.clone.default(permute_47, memory_format = torch.contiguous_format);  permute_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution_18: "f32[8, 320, 14, 14]" = torch.ops.aten.convolution.default(clone_31, arg94_1, arg95_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  clone_31 = arg94_1 = arg95_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:89, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    view_84: "f32[8, 320, 196]" = torch.ops.aten.reshape.default(convolution_18, [8, 320, 196]);  convolution_18 = None
    permute_48: "f32[8, 196, 320]" = torch.ops.aten.permute.default(view_84, [0, 2, 1]);  view_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:92, code: x = self.norm(x)
    clone_32: "f32[8, 196, 320]" = torch.ops.aten.clone.default(permute_48, memory_format = torch.contiguous_format);  permute_48 = None
    var_mean_10 = torch.ops.aten.var_mean.correction(clone_32, [2], correction = 0, keepdim = True)
    getitem_44: "f32[8, 196, 1]" = var_mean_10[0]
    getitem_45: "f32[8, 196, 1]" = var_mean_10[1];  var_mean_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:677, code: cls_tokens = cls_token.expand(x.shape[0], -1, -1)
    expand_18: "f32[8, 1, 320]" = torch.ops.aten.expand.default(arg18_1, [8, -1, -1]);  arg18_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:92, code: x = self.norm(x)
    sub_14: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_32, getitem_45);  clone_32 = getitem_45 = None
    add_40: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
    rsqrt_10: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
    mul_40: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_10);  sub_14 = rsqrt_10 = None
    mul_41: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_40, arg96_1);  mul_40 = arg96_1 = None
    add_41: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_41, arg97_1);  mul_41 = arg97_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:678, code: x = torch.cat((cls_tokens, x), dim=1)
    cat_10: "f32[8, 197, 320]" = torch.ops.aten.cat.default([expand_18, add_41], 1);  expand_18 = add_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    slice_56: "f32[8, 1, 320]" = torch.ops.aten.slice.Tensor(cat_10, 1, 0, 1)
    slice_58: "f32[8, 196, 320]" = torch.ops.aten.slice.Tensor(cat_10, 1, 1, 9223372036854775807);  cat_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:158, code: feat = img_tokens.transpose(1, 2).view(B, C, H, W)
    permute_49: "f32[8, 320, 196]" = torch.ops.aten.permute.default(slice_58, [0, 2, 1]);  slice_58 = None
    view_85: "f32[8, 320, 14, 14]" = torch.ops.aten.reshape.default(permute_49, [8, 320, 14, 14]);  permute_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:159, code: x = self.proj(feat) + feat
    convolution_19: "f32[8, 320, 14, 14]" = torch.ops.aten.convolution.default(view_85, arg98_1, arg99_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 320)
    add_42: "f32[8, 320, 14, 14]" = torch.ops.aten.add.Tensor(convolution_19, view_85);  convolution_19 = view_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:160, code: x = x.flatten(2).transpose(1, 2)
    view_86: "f32[8, 320, 196]" = torch.ops.aten.reshape.default(add_42, [8, 320, 196]);  add_42 = None
    permute_50: "f32[8, 196, 320]" = torch.ops.aten.permute.default(view_86, [0, 2, 1]);  view_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:163, code: x = torch.cat((cls_token, x), dim=1)
    cat_11: "f32[8, 197, 320]" = torch.ops.aten.cat.default([slice_56, permute_50], 1);  slice_56 = permute_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_11 = torch.ops.aten.var_mean.correction(cat_11, [2], correction = 0, keepdim = True)
    getitem_46: "f32[8, 197, 1]" = var_mean_11[0]
    getitem_47: "f32[8, 197, 1]" = var_mean_11[1];  var_mean_11 = None
    sub_15: "f32[8, 197, 320]" = torch.ops.aten.sub.Tensor(cat_11, getitem_47);  getitem_47 = None
    add_43: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-06);  getitem_46 = None
    rsqrt_11: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
    mul_42: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_11);  sub_15 = rsqrt_11 = None
    mul_43: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(mul_42, arg19_1);  mul_42 = arg19_1 = None
    add_44: "f32[8, 197, 320]" = torch.ops.aten.add.Tensor(mul_43, arg20_1);  mul_43 = arg20_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:119, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_87: "f32[1576, 320]" = torch.ops.aten.reshape.default(add_44, [1576, 320]);  add_44 = None
    permute_51: "f32[320, 960]" = torch.ops.aten.permute.default(arg100_1, [1, 0]);  arg100_1 = None
    addmm_16: "f32[1576, 960]" = torch.ops.aten.addmm.default(arg101_1, view_87, permute_51);  arg101_1 = view_87 = permute_51 = None
    view_88: "f32[8, 197, 960]" = torch.ops.aten.reshape.default(addmm_16, [8, 197, 960]);  addmm_16 = None
    view_89: "f32[8, 197, 3, 8, 40]" = torch.ops.aten.reshape.default(view_88, [8, 197, 3, 8, 40]);  view_88 = None
    permute_52: "f32[3, 8, 8, 197, 40]" = torch.ops.aten.permute.default(view_89, [2, 0, 3, 1, 4]);  view_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:120, code: q, k, v = qkv.unbind(0)  # [B, h, N, Ch]
    unbind_4 = torch.ops.aten.unbind.int(permute_52);  permute_52 = None
    getitem_48: "f32[8, 8, 197, 40]" = unbind_4[0]
    getitem_49: "f32[8, 8, 197, 40]" = unbind_4[1]
    getitem_50: "f32[8, 8, 197, 40]" = unbind_4[2];  unbind_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:76, code: v_img = v[:, :, 1:, :]  # [B, h, H*W, Ch]
    slice_65: "f32[8, 8, 196, 40]" = torch.ops.aten.slice.Tensor(getitem_50, 2, 1, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:78, code: v_img = v_img.transpose(-1, -2).reshape(B, num_heads * C, H, W)
    permute_54: "f32[8, 8, 40, 196]" = torch.ops.aten.permute.default(slice_65, [0, 1, 3, 2]);  slice_65 = None
    view_96: "f32[8, 320, 14, 14]" = torch.ops.aten.reshape.default(permute_54, [8, 320, 14, 14]);  permute_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:79, code: v_img_list = torch.split(v_img, self.channel_splits, dim=1)  # Split according to channels
    split_with_sizes_4 = torch.ops.aten.split_with_sizes.default(view_96, [80, 120, 120], 1);  view_96 = None
    getitem_51: "f32[8, 80, 14, 14]" = split_with_sizes_4[0]
    getitem_52: "f32[8, 120, 14, 14]" = split_with_sizes_4[1]
    getitem_53: "f32[8, 120, 14, 14]" = split_with_sizes_4[2];  split_with_sizes_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    expand_21: "f32[8, 8, 197, 40]" = torch.ops.aten.expand.default(getitem_48, [8, 8, 197, 40])
    clone_35: "f32[8, 8, 197, 40]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
    view_93: "f32[64, 197, 40]" = torch.ops.aten.reshape.default(clone_35, [64, 197, 40]);  clone_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:123, code: k_softmax = k.softmax(dim=2)
    clone_33: "f32[8, 8, 197, 40]" = torch.ops.aten.clone.default(getitem_49, memory_format = torch.contiguous_format);  getitem_49 = None
    amax_4: "f32[8, 8, 1, 40]" = torch.ops.aten.amax.default(clone_33, [2], True)
    sub_16: "f32[8, 8, 197, 40]" = torch.ops.aten.sub.Tensor(clone_33, amax_4);  clone_33 = amax_4 = None
    exp_4: "f32[8, 8, 197, 40]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
    sum_5: "f32[8, 8, 1, 40]" = torch.ops.aten.sum.dim_IntList(exp_4, [2], True)
    div_4: "f32[8, 8, 197, 40]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    permute_53: "f32[8, 8, 40, 197]" = torch.ops.aten.permute.default(div_4, [0, 1, 3, 2]);  div_4 = None
    expand_19: "f32[8, 8, 40, 197]" = torch.ops.aten.expand.default(permute_53, [8, 8, 40, 197]);  permute_53 = None
    view_90: "f32[64, 40, 197]" = torch.ops.aten.reshape.default(expand_19, [64, 40, 197]);  expand_19 = None
    expand_20: "f32[8, 8, 197, 40]" = torch.ops.aten.expand.default(getitem_50, [8, 8, 197, 40]);  getitem_50 = None
    clone_34: "f32[8, 8, 197, 40]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
    view_91: "f32[64, 197, 40]" = torch.ops.aten.reshape.default(clone_34, [64, 197, 40]);  clone_34 = None
    bmm_8: "f32[64, 40, 40]" = torch.ops.aten.bmm.default(view_90, view_91);  view_90 = view_91 = None
    view_92: "f32[8, 8, 40, 40]" = torch.ops.aten.reshape.default(bmm_8, [8, 8, 40, 40]);  bmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    expand_22: "f32[8, 8, 40, 40]" = torch.ops.aten.expand.default(view_92, [8, 8, 40, 40]);  view_92 = None
    view_94: "f32[64, 40, 40]" = torch.ops.aten.reshape.default(expand_22, [64, 40, 40]);  expand_22 = None
    bmm_9: "f32[64, 197, 40]" = torch.ops.aten.bmm.default(view_93, view_94);  view_93 = view_94 = None
    view_95: "f32[8, 8, 197, 40]" = torch.ops.aten.reshape.default(bmm_9, [8, 8, 197, 40]);  bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:131, code: x = self.scale * factor_att + crpe
    mul_45: "f32[8, 8, 197, 40]" = torch.ops.aten.mul.Tensor(view_95, 0.15811388300841897);  view_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:75, code: q_img = q[:, :, 1:, :]  # [B, h, H*W, Ch]
    slice_61: "f32[8, 8, 196, 40]" = torch.ops.aten.slice.Tensor(getitem_48, 2, 1, 9223372036854775807);  getitem_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    convolution_20: "f32[8, 80, 14, 14]" = torch.ops.aten.convolution.default(getitem_51, arg102_1, arg103_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 80);  getitem_51 = None
    convolution_21: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(getitem_52, arg104_1, arg105_1, [1, 1], [2, 2], [1, 1], False, [0, 0], 120);  getitem_52 = None
    convolution_22: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(getitem_53, arg106_1, arg107_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 120);  getitem_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:83, code: conv_v_img = torch.cat(conv_v_img_list, dim=1)
    cat_12: "f32[8, 320, 14, 14]" = torch.ops.aten.cat.default([convolution_20, convolution_21, convolution_22], 1);  convolution_20 = convolution_21 = convolution_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:84, code: conv_v_img = conv_v_img.reshape(B, num_heads, C, H * W).transpose(-1, -2)
    view_97: "f32[8, 8, 40, 196]" = torch.ops.aten.reshape.default(cat_12, [8, 8, 40, 196]);  cat_12 = None
    permute_55: "f32[8, 8, 196, 40]" = torch.ops.aten.permute.default(view_97, [0, 1, 3, 2]);  view_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:86, code: EV_hat = q_img * conv_v_img
    mul_44: "f32[8, 8, 196, 40]" = torch.ops.aten.mul.Tensor(slice_61, permute_55);  slice_61 = permute_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:87, code: EV_hat = F.pad(EV_hat, (0, 0, 1, 0, 0, 0))  # [B, h, N, Ch].
    constant_pad_nd_4: "f32[8, 8, 197, 40]" = torch.ops.aten.constant_pad_nd.default(mul_44, [0, 0, 1, 0, 0, 0], 0.0);  mul_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:131, code: x = self.scale * factor_att + crpe
    add_45: "f32[8, 8, 197, 40]" = torch.ops.aten.add.Tensor(mul_45, constant_pad_nd_4);  mul_45 = constant_pad_nd_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:132, code: x = x.transpose(1, 2).reshape(B, N, C)  # [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C]
    permute_56: "f32[8, 197, 8, 40]" = torch.ops.aten.permute.default(add_45, [0, 2, 1, 3]);  add_45 = None
    clone_36: "f32[8, 197, 8, 40]" = torch.ops.aten.clone.default(permute_56, memory_format = torch.contiguous_format);  permute_56 = None
    view_98: "f32[8, 197, 320]" = torch.ops.aten.reshape.default(clone_36, [8, 197, 320]);  clone_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:135, code: x = self.proj(x)
    view_99: "f32[1576, 320]" = torch.ops.aten.reshape.default(view_98, [1576, 320]);  view_98 = None
    permute_57: "f32[320, 320]" = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
    addmm_17: "f32[1576, 320]" = torch.ops.aten.addmm.default(arg109_1, view_99, permute_57);  arg109_1 = view_99 = permute_57 = None
    view_100: "f32[8, 197, 320]" = torch.ops.aten.reshape.default(addmm_17, [8, 197, 320]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:216, code: x = x + self.drop_path(cur)
    add_46: "f32[8, 197, 320]" = torch.ops.aten.add.Tensor(cat_11, view_100);  cat_11 = view_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_12 = torch.ops.aten.var_mean.correction(add_46, [2], correction = 0, keepdim = True)
    getitem_54: "f32[8, 197, 1]" = var_mean_12[0]
    getitem_55: "f32[8, 197, 1]" = var_mean_12[1];  var_mean_12 = None
    sub_17: "f32[8, 197, 320]" = torch.ops.aten.sub.Tensor(add_46, getitem_55);  getitem_55 = None
    add_47: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-06);  getitem_54 = None
    rsqrt_12: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
    mul_46: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_12);  sub_17 = rsqrt_12 = None
    mul_47: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(mul_46, arg21_1);  mul_46 = arg21_1 = None
    add_48: "f32[8, 197, 320]" = torch.ops.aten.add.Tensor(mul_47, arg22_1);  mul_47 = arg22_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_101: "f32[1576, 320]" = torch.ops.aten.reshape.default(add_48, [1576, 320]);  add_48 = None
    permute_58: "f32[320, 1280]" = torch.ops.aten.permute.default(arg110_1, [1, 0]);  arg110_1 = None
    addmm_18: "f32[1576, 1280]" = torch.ops.aten.addmm.default(arg111_1, view_101, permute_58);  arg111_1 = view_101 = permute_58 = None
    view_102: "f32[8, 197, 1280]" = torch.ops.aten.reshape.default(addmm_18, [8, 197, 1280]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_48: "f32[8, 197, 1280]" = torch.ops.aten.mul.Tensor(view_102, 0.5)
    mul_49: "f32[8, 197, 1280]" = torch.ops.aten.mul.Tensor(view_102, 0.7071067811865476);  view_102 = None
    erf_4: "f32[8, 197, 1280]" = torch.ops.aten.erf.default(mul_49);  mul_49 = None
    add_49: "f32[8, 197, 1280]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_50: "f32[8, 197, 1280]" = torch.ops.aten.mul.Tensor(mul_48, add_49);  mul_48 = add_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_103: "f32[1576, 1280]" = torch.ops.aten.reshape.default(mul_50, [1576, 1280]);  mul_50 = None
    permute_59: "f32[1280, 320]" = torch.ops.aten.permute.default(arg112_1, [1, 0]);  arg112_1 = None
    addmm_19: "f32[1576, 320]" = torch.ops.aten.addmm.default(arg113_1, view_103, permute_59);  arg113_1 = view_103 = permute_59 = None
    view_104: "f32[8, 197, 320]" = torch.ops.aten.reshape.default(addmm_19, [8, 197, 320]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:221, code: x = x + self.drop_path(cur)
    add_50: "f32[8, 197, 320]" = torch.ops.aten.add.Tensor(add_46, view_104);  add_46 = view_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    slice_68: "f32[8, 1, 320]" = torch.ops.aten.slice.Tensor(add_50, 1, 0, 1)
    slice_70: "f32[8, 196, 320]" = torch.ops.aten.slice.Tensor(add_50, 1, 1, 9223372036854775807);  add_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:158, code: feat = img_tokens.transpose(1, 2).view(B, C, H, W)
    permute_60: "f32[8, 320, 196]" = torch.ops.aten.permute.default(slice_70, [0, 2, 1]);  slice_70 = None
    view_105: "f32[8, 320, 14, 14]" = torch.ops.aten.reshape.default(permute_60, [8, 320, 14, 14]);  permute_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:159, code: x = self.proj(feat) + feat
    convolution_23: "f32[8, 320, 14, 14]" = torch.ops.aten.convolution.default(view_105, arg98_1, arg99_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 320);  arg98_1 = arg99_1 = None
    add_51: "f32[8, 320, 14, 14]" = torch.ops.aten.add.Tensor(convolution_23, view_105);  convolution_23 = view_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:160, code: x = x.flatten(2).transpose(1, 2)
    view_106: "f32[8, 320, 196]" = torch.ops.aten.reshape.default(add_51, [8, 320, 196]);  add_51 = None
    permute_61: "f32[8, 196, 320]" = torch.ops.aten.permute.default(view_106, [0, 2, 1]);  view_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:163, code: x = torch.cat((cls_token, x), dim=1)
    cat_13: "f32[8, 197, 320]" = torch.ops.aten.cat.default([slice_68, permute_61], 1);  slice_68 = permute_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_13 = torch.ops.aten.var_mean.correction(cat_13, [2], correction = 0, keepdim = True)
    getitem_56: "f32[8, 197, 1]" = var_mean_13[0]
    getitem_57: "f32[8, 197, 1]" = var_mean_13[1];  var_mean_13 = None
    sub_18: "f32[8, 197, 320]" = torch.ops.aten.sub.Tensor(cat_13, getitem_57);  getitem_57 = None
    add_52: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-06);  getitem_56 = None
    rsqrt_13: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
    mul_51: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_13);  sub_18 = rsqrt_13 = None
    mul_52: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(mul_51, arg23_1);  mul_51 = arg23_1 = None
    add_53: "f32[8, 197, 320]" = torch.ops.aten.add.Tensor(mul_52, arg24_1);  mul_52 = arg24_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:119, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_107: "f32[1576, 320]" = torch.ops.aten.reshape.default(add_53, [1576, 320]);  add_53 = None
    permute_62: "f32[320, 960]" = torch.ops.aten.permute.default(arg114_1, [1, 0]);  arg114_1 = None
    addmm_20: "f32[1576, 960]" = torch.ops.aten.addmm.default(arg115_1, view_107, permute_62);  arg115_1 = view_107 = permute_62 = None
    view_108: "f32[8, 197, 960]" = torch.ops.aten.reshape.default(addmm_20, [8, 197, 960]);  addmm_20 = None
    view_109: "f32[8, 197, 3, 8, 40]" = torch.ops.aten.reshape.default(view_108, [8, 197, 3, 8, 40]);  view_108 = None
    permute_63: "f32[3, 8, 8, 197, 40]" = torch.ops.aten.permute.default(view_109, [2, 0, 3, 1, 4]);  view_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:120, code: q, k, v = qkv.unbind(0)  # [B, h, N, Ch]
    unbind_5 = torch.ops.aten.unbind.int(permute_63);  permute_63 = None
    getitem_58: "f32[8, 8, 197, 40]" = unbind_5[0]
    getitem_59: "f32[8, 8, 197, 40]" = unbind_5[1]
    getitem_60: "f32[8, 8, 197, 40]" = unbind_5[2];  unbind_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:76, code: v_img = v[:, :, 1:, :]  # [B, h, H*W, Ch]
    slice_77: "f32[8, 8, 196, 40]" = torch.ops.aten.slice.Tensor(getitem_60, 2, 1, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:78, code: v_img = v_img.transpose(-1, -2).reshape(B, num_heads * C, H, W)
    permute_65: "f32[8, 8, 40, 196]" = torch.ops.aten.permute.default(slice_77, [0, 1, 3, 2]);  slice_77 = None
    view_116: "f32[8, 320, 14, 14]" = torch.ops.aten.reshape.default(permute_65, [8, 320, 14, 14]);  permute_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:79, code: v_img_list = torch.split(v_img, self.channel_splits, dim=1)  # Split according to channels
    split_with_sizes_5 = torch.ops.aten.split_with_sizes.default(view_116, [80, 120, 120], 1);  view_116 = None
    getitem_61: "f32[8, 80, 14, 14]" = split_with_sizes_5[0]
    getitem_62: "f32[8, 120, 14, 14]" = split_with_sizes_5[1]
    getitem_63: "f32[8, 120, 14, 14]" = split_with_sizes_5[2];  split_with_sizes_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    expand_25: "f32[8, 8, 197, 40]" = torch.ops.aten.expand.default(getitem_58, [8, 8, 197, 40])
    clone_42: "f32[8, 8, 197, 40]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
    view_113: "f32[64, 197, 40]" = torch.ops.aten.reshape.default(clone_42, [64, 197, 40]);  clone_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:123, code: k_softmax = k.softmax(dim=2)
    clone_40: "f32[8, 8, 197, 40]" = torch.ops.aten.clone.default(getitem_59, memory_format = torch.contiguous_format);  getitem_59 = None
    amax_5: "f32[8, 8, 1, 40]" = torch.ops.aten.amax.default(clone_40, [2], True)
    sub_19: "f32[8, 8, 197, 40]" = torch.ops.aten.sub.Tensor(clone_40, amax_5);  clone_40 = amax_5 = None
    exp_5: "f32[8, 8, 197, 40]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
    sum_6: "f32[8, 8, 1, 40]" = torch.ops.aten.sum.dim_IntList(exp_5, [2], True)
    div_5: "f32[8, 8, 197, 40]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    permute_64: "f32[8, 8, 40, 197]" = torch.ops.aten.permute.default(div_5, [0, 1, 3, 2]);  div_5 = None
    expand_23: "f32[8, 8, 40, 197]" = torch.ops.aten.expand.default(permute_64, [8, 8, 40, 197]);  permute_64 = None
    view_110: "f32[64, 40, 197]" = torch.ops.aten.reshape.default(expand_23, [64, 40, 197]);  expand_23 = None
    expand_24: "f32[8, 8, 197, 40]" = torch.ops.aten.expand.default(getitem_60, [8, 8, 197, 40]);  getitem_60 = None
    clone_41: "f32[8, 8, 197, 40]" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
    view_111: "f32[64, 197, 40]" = torch.ops.aten.reshape.default(clone_41, [64, 197, 40]);  clone_41 = None
    bmm_10: "f32[64, 40, 40]" = torch.ops.aten.bmm.default(view_110, view_111);  view_110 = view_111 = None
    view_112: "f32[8, 8, 40, 40]" = torch.ops.aten.reshape.default(bmm_10, [8, 8, 40, 40]);  bmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    expand_26: "f32[8, 8, 40, 40]" = torch.ops.aten.expand.default(view_112, [8, 8, 40, 40]);  view_112 = None
    view_114: "f32[64, 40, 40]" = torch.ops.aten.reshape.default(expand_26, [64, 40, 40]);  expand_26 = None
    bmm_11: "f32[64, 197, 40]" = torch.ops.aten.bmm.default(view_113, view_114);  view_113 = view_114 = None
    view_115: "f32[8, 8, 197, 40]" = torch.ops.aten.reshape.default(bmm_11, [8, 8, 197, 40]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:131, code: x = self.scale * factor_att + crpe
    mul_54: "f32[8, 8, 197, 40]" = torch.ops.aten.mul.Tensor(view_115, 0.15811388300841897);  view_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:75, code: q_img = q[:, :, 1:, :]  # [B, h, H*W, Ch]
    slice_73: "f32[8, 8, 196, 40]" = torch.ops.aten.slice.Tensor(getitem_58, 2, 1, 9223372036854775807);  getitem_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    convolution_24: "f32[8, 80, 14, 14]" = torch.ops.aten.convolution.default(getitem_61, arg102_1, arg103_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 80);  getitem_61 = arg102_1 = arg103_1 = None
    convolution_25: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(getitem_62, arg104_1, arg105_1, [1, 1], [2, 2], [1, 1], False, [0, 0], 120);  getitem_62 = arg104_1 = arg105_1 = None
    convolution_26: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(getitem_63, arg106_1, arg107_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 120);  getitem_63 = arg106_1 = arg107_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:83, code: conv_v_img = torch.cat(conv_v_img_list, dim=1)
    cat_14: "f32[8, 320, 14, 14]" = torch.ops.aten.cat.default([convolution_24, convolution_25, convolution_26], 1);  convolution_24 = convolution_25 = convolution_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:84, code: conv_v_img = conv_v_img.reshape(B, num_heads, C, H * W).transpose(-1, -2)
    view_117: "f32[8, 8, 40, 196]" = torch.ops.aten.reshape.default(cat_14, [8, 8, 40, 196]);  cat_14 = None
    permute_66: "f32[8, 8, 196, 40]" = torch.ops.aten.permute.default(view_117, [0, 1, 3, 2]);  view_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:86, code: EV_hat = q_img * conv_v_img
    mul_53: "f32[8, 8, 196, 40]" = torch.ops.aten.mul.Tensor(slice_73, permute_66);  slice_73 = permute_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:87, code: EV_hat = F.pad(EV_hat, (0, 0, 1, 0, 0, 0))  # [B, h, N, Ch].
    constant_pad_nd_5: "f32[8, 8, 197, 40]" = torch.ops.aten.constant_pad_nd.default(mul_53, [0, 0, 1, 0, 0, 0], 0.0);  mul_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:131, code: x = self.scale * factor_att + crpe
    add_54: "f32[8, 8, 197, 40]" = torch.ops.aten.add.Tensor(mul_54, constant_pad_nd_5);  mul_54 = constant_pad_nd_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:132, code: x = x.transpose(1, 2).reshape(B, N, C)  # [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C]
    permute_67: "f32[8, 197, 8, 40]" = torch.ops.aten.permute.default(add_54, [0, 2, 1, 3]);  add_54 = None
    clone_43: "f32[8, 197, 8, 40]" = torch.ops.aten.clone.default(permute_67, memory_format = torch.contiguous_format);  permute_67 = None
    view_118: "f32[8, 197, 320]" = torch.ops.aten.reshape.default(clone_43, [8, 197, 320]);  clone_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:135, code: x = self.proj(x)
    view_119: "f32[1576, 320]" = torch.ops.aten.reshape.default(view_118, [1576, 320]);  view_118 = None
    permute_68: "f32[320, 320]" = torch.ops.aten.permute.default(arg116_1, [1, 0]);  arg116_1 = None
    addmm_21: "f32[1576, 320]" = torch.ops.aten.addmm.default(arg117_1, view_119, permute_68);  arg117_1 = view_119 = permute_68 = None
    view_120: "f32[8, 197, 320]" = torch.ops.aten.reshape.default(addmm_21, [8, 197, 320]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:216, code: x = x + self.drop_path(cur)
    add_55: "f32[8, 197, 320]" = torch.ops.aten.add.Tensor(cat_13, view_120);  cat_13 = view_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_14 = torch.ops.aten.var_mean.correction(add_55, [2], correction = 0, keepdim = True)
    getitem_64: "f32[8, 197, 1]" = var_mean_14[0]
    getitem_65: "f32[8, 197, 1]" = var_mean_14[1];  var_mean_14 = None
    sub_20: "f32[8, 197, 320]" = torch.ops.aten.sub.Tensor(add_55, getitem_65);  getitem_65 = None
    add_56: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-06);  getitem_64 = None
    rsqrt_14: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
    mul_55: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_14);  sub_20 = rsqrt_14 = None
    mul_56: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(mul_55, arg25_1);  mul_55 = arg25_1 = None
    add_57: "f32[8, 197, 320]" = torch.ops.aten.add.Tensor(mul_56, arg26_1);  mul_56 = arg26_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_121: "f32[1576, 320]" = torch.ops.aten.reshape.default(add_57, [1576, 320]);  add_57 = None
    permute_69: "f32[320, 1280]" = torch.ops.aten.permute.default(arg118_1, [1, 0]);  arg118_1 = None
    addmm_22: "f32[1576, 1280]" = torch.ops.aten.addmm.default(arg119_1, view_121, permute_69);  arg119_1 = view_121 = permute_69 = None
    view_122: "f32[8, 197, 1280]" = torch.ops.aten.reshape.default(addmm_22, [8, 197, 1280]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_57: "f32[8, 197, 1280]" = torch.ops.aten.mul.Tensor(view_122, 0.5)
    mul_58: "f32[8, 197, 1280]" = torch.ops.aten.mul.Tensor(view_122, 0.7071067811865476);  view_122 = None
    erf_5: "f32[8, 197, 1280]" = torch.ops.aten.erf.default(mul_58);  mul_58 = None
    add_58: "f32[8, 197, 1280]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_59: "f32[8, 197, 1280]" = torch.ops.aten.mul.Tensor(mul_57, add_58);  mul_57 = add_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_123: "f32[1576, 1280]" = torch.ops.aten.reshape.default(mul_59, [1576, 1280]);  mul_59 = None
    permute_70: "f32[1280, 320]" = torch.ops.aten.permute.default(arg120_1, [1, 0]);  arg120_1 = None
    addmm_23: "f32[1576, 320]" = torch.ops.aten.addmm.default(arg121_1, view_123, permute_70);  arg121_1 = view_123 = permute_70 = None
    view_124: "f32[8, 197, 320]" = torch.ops.aten.reshape.default(addmm_23, [8, 197, 320]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:221, code: x = x + self.drop_path(cur)
    add_59: "f32[8, 197, 320]" = torch.ops.aten.add.Tensor(add_55, view_124);  add_55 = view_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:684, code: return x[:, 1:, :]
    slice_80: "f32[8, 196, 320]" = torch.ops.aten.slice.Tensor(add_59, 1, 1, 9223372036854775807);  add_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:595, code: x3_nocls = remove_cls(x3).reshape(B, H3, W3, -1).permute(0, 3, 1, 2).contiguous()
    view_125: "f32[8, 14, 14, 320]" = torch.ops.aten.reshape.default(slice_80, [8, 14, 14, 320]);  slice_80 = None
    permute_71: "f32[8, 320, 14, 14]" = torch.ops.aten.permute.default(view_125, [0, 3, 1, 2]);  view_125 = None
    clone_47: "f32[8, 320, 14, 14]" = torch.ops.aten.clone.default(permute_71, memory_format = torch.contiguous_format);  permute_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution_27: "f32[8, 512, 7, 7]" = torch.ops.aten.convolution.default(clone_47, arg122_1, arg123_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  clone_47 = arg122_1 = arg123_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:89, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    view_126: "f32[8, 512, 49]" = torch.ops.aten.reshape.default(convolution_27, [8, 512, 49]);  convolution_27 = None
    permute_72: "f32[8, 49, 512]" = torch.ops.aten.permute.default(view_126, [0, 2, 1]);  view_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:92, code: x = self.norm(x)
    clone_48: "f32[8, 49, 512]" = torch.ops.aten.clone.default(permute_72, memory_format = torch.contiguous_format);  permute_72 = None
    var_mean_15 = torch.ops.aten.var_mean.correction(clone_48, [2], correction = 0, keepdim = True)
    getitem_66: "f32[8, 49, 1]" = var_mean_15[0]
    getitem_67: "f32[8, 49, 1]" = var_mean_15[1];  var_mean_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:677, code: cls_tokens = cls_token.expand(x.shape[0], -1, -1)
    expand_27: "f32[8, 1, 512]" = torch.ops.aten.expand.default(arg27_1, [8, -1, -1]);  arg27_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:92, code: x = self.norm(x)
    sub_21: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(clone_48, getitem_67);  clone_48 = getitem_67 = None
    add_60: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
    rsqrt_15: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
    mul_60: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_15);  sub_21 = rsqrt_15 = None
    mul_61: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_60, arg124_1);  mul_60 = arg124_1 = None
    add_61: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(mul_61, arg125_1);  mul_61 = arg125_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:678, code: x = torch.cat((cls_tokens, x), dim=1)
    cat_15: "f32[8, 50, 512]" = torch.ops.aten.cat.default([expand_27, add_61], 1);  expand_27 = add_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    slice_83: "f32[8, 1, 512]" = torch.ops.aten.slice.Tensor(cat_15, 1, 0, 1)
    slice_85: "f32[8, 49, 512]" = torch.ops.aten.slice.Tensor(cat_15, 1, 1, 9223372036854775807);  cat_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:158, code: feat = img_tokens.transpose(1, 2).view(B, C, H, W)
    permute_73: "f32[8, 512, 49]" = torch.ops.aten.permute.default(slice_85, [0, 2, 1]);  slice_85 = None
    view_127: "f32[8, 512, 7, 7]" = torch.ops.aten.reshape.default(permute_73, [8, 512, 7, 7]);  permute_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:159, code: x = self.proj(feat) + feat
    convolution_28: "f32[8, 512, 7, 7]" = torch.ops.aten.convolution.default(view_127, arg126_1, arg127_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 512)
    add_62: "f32[8, 512, 7, 7]" = torch.ops.aten.add.Tensor(convolution_28, view_127);  convolution_28 = view_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:160, code: x = x.flatten(2).transpose(1, 2)
    view_128: "f32[8, 512, 49]" = torch.ops.aten.reshape.default(add_62, [8, 512, 49]);  add_62 = None
    permute_74: "f32[8, 49, 512]" = torch.ops.aten.permute.default(view_128, [0, 2, 1]);  view_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:163, code: x = torch.cat((cls_token, x), dim=1)
    cat_16: "f32[8, 50, 512]" = torch.ops.aten.cat.default([slice_83, permute_74], 1);  slice_83 = permute_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_16 = torch.ops.aten.var_mean.correction(cat_16, [2], correction = 0, keepdim = True)
    getitem_68: "f32[8, 50, 1]" = var_mean_16[0]
    getitem_69: "f32[8, 50, 1]" = var_mean_16[1];  var_mean_16 = None
    sub_22: "f32[8, 50, 512]" = torch.ops.aten.sub.Tensor(cat_16, getitem_69);  getitem_69 = None
    add_63: "f32[8, 50, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-06);  getitem_68 = None
    rsqrt_16: "f32[8, 50, 1]" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
    mul_62: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_16);  sub_22 = rsqrt_16 = None
    mul_63: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(mul_62, arg28_1);  mul_62 = arg28_1 = None
    add_64: "f32[8, 50, 512]" = torch.ops.aten.add.Tensor(mul_63, arg29_1);  mul_63 = arg29_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:119, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_129: "f32[400, 512]" = torch.ops.aten.reshape.default(add_64, [400, 512]);  add_64 = None
    permute_75: "f32[512, 1536]" = torch.ops.aten.permute.default(arg128_1, [1, 0]);  arg128_1 = None
    addmm_24: "f32[400, 1536]" = torch.ops.aten.addmm.default(arg129_1, view_129, permute_75);  arg129_1 = view_129 = permute_75 = None
    view_130: "f32[8, 50, 1536]" = torch.ops.aten.reshape.default(addmm_24, [8, 50, 1536]);  addmm_24 = None
    view_131: "f32[8, 50, 3, 8, 64]" = torch.ops.aten.reshape.default(view_130, [8, 50, 3, 8, 64]);  view_130 = None
    permute_76: "f32[3, 8, 8, 50, 64]" = torch.ops.aten.permute.default(view_131, [2, 0, 3, 1, 4]);  view_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:120, code: q, k, v = qkv.unbind(0)  # [B, h, N, Ch]
    unbind_6 = torch.ops.aten.unbind.int(permute_76);  permute_76 = None
    getitem_70: "f32[8, 8, 50, 64]" = unbind_6[0]
    getitem_71: "f32[8, 8, 50, 64]" = unbind_6[1]
    getitem_72: "f32[8, 8, 50, 64]" = unbind_6[2];  unbind_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:76, code: v_img = v[:, :, 1:, :]  # [B, h, H*W, Ch]
    slice_92: "f32[8, 8, 49, 64]" = torch.ops.aten.slice.Tensor(getitem_72, 2, 1, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:78, code: v_img = v_img.transpose(-1, -2).reshape(B, num_heads * C, H, W)
    permute_78: "f32[8, 8, 64, 49]" = torch.ops.aten.permute.default(slice_92, [0, 1, 3, 2]);  slice_92 = None
    view_138: "f32[8, 512, 7, 7]" = torch.ops.aten.reshape.default(permute_78, [8, 512, 7, 7]);  permute_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:79, code: v_img_list = torch.split(v_img, self.channel_splits, dim=1)  # Split according to channels
    split_with_sizes_6 = torch.ops.aten.split_with_sizes.default(view_138, [128, 192, 192], 1);  view_138 = None
    getitem_73: "f32[8, 128, 7, 7]" = split_with_sizes_6[0]
    getitem_74: "f32[8, 192, 7, 7]" = split_with_sizes_6[1]
    getitem_75: "f32[8, 192, 7, 7]" = split_with_sizes_6[2];  split_with_sizes_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    expand_30: "f32[8, 8, 50, 64]" = torch.ops.aten.expand.default(getitem_70, [8, 8, 50, 64])
    clone_51: "f32[8, 8, 50, 64]" = torch.ops.aten.clone.default(expand_30, memory_format = torch.contiguous_format);  expand_30 = None
    view_135: "f32[64, 50, 64]" = torch.ops.aten.reshape.default(clone_51, [64, 50, 64]);  clone_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:123, code: k_softmax = k.softmax(dim=2)
    clone_49: "f32[8, 8, 50, 64]" = torch.ops.aten.clone.default(getitem_71, memory_format = torch.contiguous_format);  getitem_71 = None
    amax_6: "f32[8, 8, 1, 64]" = torch.ops.aten.amax.default(clone_49, [2], True)
    sub_23: "f32[8, 8, 50, 64]" = torch.ops.aten.sub.Tensor(clone_49, amax_6);  clone_49 = amax_6 = None
    exp_6: "f32[8, 8, 50, 64]" = torch.ops.aten.exp.default(sub_23);  sub_23 = None
    sum_7: "f32[8, 8, 1, 64]" = torch.ops.aten.sum.dim_IntList(exp_6, [2], True)
    div_6: "f32[8, 8, 50, 64]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    permute_77: "f32[8, 8, 64, 50]" = torch.ops.aten.permute.default(div_6, [0, 1, 3, 2]);  div_6 = None
    expand_28: "f32[8, 8, 64, 50]" = torch.ops.aten.expand.default(permute_77, [8, 8, 64, 50]);  permute_77 = None
    view_132: "f32[64, 64, 50]" = torch.ops.aten.reshape.default(expand_28, [64, 64, 50]);  expand_28 = None
    expand_29: "f32[8, 8, 50, 64]" = torch.ops.aten.expand.default(getitem_72, [8, 8, 50, 64]);  getitem_72 = None
    clone_50: "f32[8, 8, 50, 64]" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
    view_133: "f32[64, 50, 64]" = torch.ops.aten.reshape.default(clone_50, [64, 50, 64]);  clone_50 = None
    bmm_12: "f32[64, 64, 64]" = torch.ops.aten.bmm.default(view_132, view_133);  view_132 = view_133 = None
    view_134: "f32[8, 8, 64, 64]" = torch.ops.aten.reshape.default(bmm_12, [8, 8, 64, 64]);  bmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    expand_31: "f32[8, 8, 64, 64]" = torch.ops.aten.expand.default(view_134, [8, 8, 64, 64]);  view_134 = None
    view_136: "f32[64, 64, 64]" = torch.ops.aten.reshape.default(expand_31, [64, 64, 64]);  expand_31 = None
    bmm_13: "f32[64, 50, 64]" = torch.ops.aten.bmm.default(view_135, view_136);  view_135 = view_136 = None
    view_137: "f32[8, 8, 50, 64]" = torch.ops.aten.reshape.default(bmm_13, [8, 8, 50, 64]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:131, code: x = self.scale * factor_att + crpe
    mul_65: "f32[8, 8, 50, 64]" = torch.ops.aten.mul.Tensor(view_137, 0.125);  view_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:75, code: q_img = q[:, :, 1:, :]  # [B, h, H*W, Ch]
    slice_88: "f32[8, 8, 49, 64]" = torch.ops.aten.slice.Tensor(getitem_70, 2, 1, 9223372036854775807);  getitem_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    convolution_29: "f32[8, 128, 7, 7]" = torch.ops.aten.convolution.default(getitem_73, arg130_1, arg131_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 128);  getitem_73 = None
    convolution_30: "f32[8, 192, 7, 7]" = torch.ops.aten.convolution.default(getitem_74, arg132_1, arg133_1, [1, 1], [2, 2], [1, 1], False, [0, 0], 192);  getitem_74 = None
    convolution_31: "f32[8, 192, 7, 7]" = torch.ops.aten.convolution.default(getitem_75, arg134_1, arg135_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 192);  getitem_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:83, code: conv_v_img = torch.cat(conv_v_img_list, dim=1)
    cat_17: "f32[8, 512, 7, 7]" = torch.ops.aten.cat.default([convolution_29, convolution_30, convolution_31], 1);  convolution_29 = convolution_30 = convolution_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:84, code: conv_v_img = conv_v_img.reshape(B, num_heads, C, H * W).transpose(-1, -2)
    view_139: "f32[8, 8, 64, 49]" = torch.ops.aten.reshape.default(cat_17, [8, 8, 64, 49]);  cat_17 = None
    permute_79: "f32[8, 8, 49, 64]" = torch.ops.aten.permute.default(view_139, [0, 1, 3, 2]);  view_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:86, code: EV_hat = q_img * conv_v_img
    mul_64: "f32[8, 8, 49, 64]" = torch.ops.aten.mul.Tensor(slice_88, permute_79);  slice_88 = permute_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:87, code: EV_hat = F.pad(EV_hat, (0, 0, 1, 0, 0, 0))  # [B, h, N, Ch].
    constant_pad_nd_6: "f32[8, 8, 50, 64]" = torch.ops.aten.constant_pad_nd.default(mul_64, [0, 0, 1, 0, 0, 0], 0.0);  mul_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:131, code: x = self.scale * factor_att + crpe
    add_65: "f32[8, 8, 50, 64]" = torch.ops.aten.add.Tensor(mul_65, constant_pad_nd_6);  mul_65 = constant_pad_nd_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:132, code: x = x.transpose(1, 2).reshape(B, N, C)  # [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C]
    permute_80: "f32[8, 50, 8, 64]" = torch.ops.aten.permute.default(add_65, [0, 2, 1, 3]);  add_65 = None
    clone_52: "f32[8, 50, 8, 64]" = torch.ops.aten.clone.default(permute_80, memory_format = torch.contiguous_format);  permute_80 = None
    view_140: "f32[8, 50, 512]" = torch.ops.aten.reshape.default(clone_52, [8, 50, 512]);  clone_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:135, code: x = self.proj(x)
    view_141: "f32[400, 512]" = torch.ops.aten.reshape.default(view_140, [400, 512]);  view_140 = None
    permute_81: "f32[512, 512]" = torch.ops.aten.permute.default(arg136_1, [1, 0]);  arg136_1 = None
    addmm_25: "f32[400, 512]" = torch.ops.aten.addmm.default(arg137_1, view_141, permute_81);  arg137_1 = view_141 = permute_81 = None
    view_142: "f32[8, 50, 512]" = torch.ops.aten.reshape.default(addmm_25, [8, 50, 512]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:216, code: x = x + self.drop_path(cur)
    add_66: "f32[8, 50, 512]" = torch.ops.aten.add.Tensor(cat_16, view_142);  cat_16 = view_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_17 = torch.ops.aten.var_mean.correction(add_66, [2], correction = 0, keepdim = True)
    getitem_76: "f32[8, 50, 1]" = var_mean_17[0]
    getitem_77: "f32[8, 50, 1]" = var_mean_17[1];  var_mean_17 = None
    sub_24: "f32[8, 50, 512]" = torch.ops.aten.sub.Tensor(add_66, getitem_77);  getitem_77 = None
    add_67: "f32[8, 50, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-06);  getitem_76 = None
    rsqrt_17: "f32[8, 50, 1]" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
    mul_66: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_17);  sub_24 = rsqrt_17 = None
    mul_67: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(mul_66, arg30_1);  mul_66 = arg30_1 = None
    add_68: "f32[8, 50, 512]" = torch.ops.aten.add.Tensor(mul_67, arg31_1);  mul_67 = arg31_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_143: "f32[400, 512]" = torch.ops.aten.reshape.default(add_68, [400, 512]);  add_68 = None
    permute_82: "f32[512, 2048]" = torch.ops.aten.permute.default(arg138_1, [1, 0]);  arg138_1 = None
    addmm_26: "f32[400, 2048]" = torch.ops.aten.addmm.default(arg139_1, view_143, permute_82);  arg139_1 = view_143 = permute_82 = None
    view_144: "f32[8, 50, 2048]" = torch.ops.aten.reshape.default(addmm_26, [8, 50, 2048]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_68: "f32[8, 50, 2048]" = torch.ops.aten.mul.Tensor(view_144, 0.5)
    mul_69: "f32[8, 50, 2048]" = torch.ops.aten.mul.Tensor(view_144, 0.7071067811865476);  view_144 = None
    erf_6: "f32[8, 50, 2048]" = torch.ops.aten.erf.default(mul_69);  mul_69 = None
    add_69: "f32[8, 50, 2048]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_70: "f32[8, 50, 2048]" = torch.ops.aten.mul.Tensor(mul_68, add_69);  mul_68 = add_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_145: "f32[400, 2048]" = torch.ops.aten.reshape.default(mul_70, [400, 2048]);  mul_70 = None
    permute_83: "f32[2048, 512]" = torch.ops.aten.permute.default(arg140_1, [1, 0]);  arg140_1 = None
    addmm_27: "f32[400, 512]" = torch.ops.aten.addmm.default(arg141_1, view_145, permute_83);  arg141_1 = view_145 = permute_83 = None
    view_146: "f32[8, 50, 512]" = torch.ops.aten.reshape.default(addmm_27, [8, 50, 512]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:221, code: x = x + self.drop_path(cur)
    add_70: "f32[8, 50, 512]" = torch.ops.aten.add.Tensor(add_66, view_146);  add_66 = view_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    slice_95: "f32[8, 1, 512]" = torch.ops.aten.slice.Tensor(add_70, 1, 0, 1)
    slice_97: "f32[8, 49, 512]" = torch.ops.aten.slice.Tensor(add_70, 1, 1, 9223372036854775807);  add_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:158, code: feat = img_tokens.transpose(1, 2).view(B, C, H, W)
    permute_84: "f32[8, 512, 49]" = torch.ops.aten.permute.default(slice_97, [0, 2, 1]);  slice_97 = None
    view_147: "f32[8, 512, 7, 7]" = torch.ops.aten.reshape.default(permute_84, [8, 512, 7, 7]);  permute_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:159, code: x = self.proj(feat) + feat
    convolution_32: "f32[8, 512, 7, 7]" = torch.ops.aten.convolution.default(view_147, arg126_1, arg127_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 512);  arg126_1 = arg127_1 = None
    add_71: "f32[8, 512, 7, 7]" = torch.ops.aten.add.Tensor(convolution_32, view_147);  convolution_32 = view_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:160, code: x = x.flatten(2).transpose(1, 2)
    view_148: "f32[8, 512, 49]" = torch.ops.aten.reshape.default(add_71, [8, 512, 49]);  add_71 = None
    permute_85: "f32[8, 49, 512]" = torch.ops.aten.permute.default(view_148, [0, 2, 1]);  view_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:163, code: x = torch.cat((cls_token, x), dim=1)
    cat_18: "f32[8, 50, 512]" = torch.ops.aten.cat.default([slice_95, permute_85], 1);  slice_95 = permute_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_18 = torch.ops.aten.var_mean.correction(cat_18, [2], correction = 0, keepdim = True)
    getitem_78: "f32[8, 50, 1]" = var_mean_18[0]
    getitem_79: "f32[8, 50, 1]" = var_mean_18[1];  var_mean_18 = None
    sub_25: "f32[8, 50, 512]" = torch.ops.aten.sub.Tensor(cat_18, getitem_79);  getitem_79 = None
    add_72: "f32[8, 50, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-06);  getitem_78 = None
    rsqrt_18: "f32[8, 50, 1]" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
    mul_71: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_18);  sub_25 = rsqrt_18 = None
    mul_72: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(mul_71, arg32_1);  mul_71 = arg32_1 = None
    add_73: "f32[8, 50, 512]" = torch.ops.aten.add.Tensor(mul_72, arg33_1);  mul_72 = arg33_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:119, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_149: "f32[400, 512]" = torch.ops.aten.reshape.default(add_73, [400, 512]);  add_73 = None
    permute_86: "f32[512, 1536]" = torch.ops.aten.permute.default(arg142_1, [1, 0]);  arg142_1 = None
    addmm_28: "f32[400, 1536]" = torch.ops.aten.addmm.default(arg143_1, view_149, permute_86);  arg143_1 = view_149 = permute_86 = None
    view_150: "f32[8, 50, 1536]" = torch.ops.aten.reshape.default(addmm_28, [8, 50, 1536]);  addmm_28 = None
    view_151: "f32[8, 50, 3, 8, 64]" = torch.ops.aten.reshape.default(view_150, [8, 50, 3, 8, 64]);  view_150 = None
    permute_87: "f32[3, 8, 8, 50, 64]" = torch.ops.aten.permute.default(view_151, [2, 0, 3, 1, 4]);  view_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:120, code: q, k, v = qkv.unbind(0)  # [B, h, N, Ch]
    unbind_7 = torch.ops.aten.unbind.int(permute_87);  permute_87 = None
    getitem_80: "f32[8, 8, 50, 64]" = unbind_7[0]
    getitem_81: "f32[8, 8, 50, 64]" = unbind_7[1]
    getitem_82: "f32[8, 8, 50, 64]" = unbind_7[2];  unbind_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:76, code: v_img = v[:, :, 1:, :]  # [B, h, H*W, Ch]
    slice_104: "f32[8, 8, 49, 64]" = torch.ops.aten.slice.Tensor(getitem_82, 2, 1, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:78, code: v_img = v_img.transpose(-1, -2).reshape(B, num_heads * C, H, W)
    permute_89: "f32[8, 8, 64, 49]" = torch.ops.aten.permute.default(slice_104, [0, 1, 3, 2]);  slice_104 = None
    view_158: "f32[8, 512, 7, 7]" = torch.ops.aten.reshape.default(permute_89, [8, 512, 7, 7]);  permute_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:79, code: v_img_list = torch.split(v_img, self.channel_splits, dim=1)  # Split according to channels
    split_with_sizes_7 = torch.ops.aten.split_with_sizes.default(view_158, [128, 192, 192], 1);  view_158 = None
    getitem_83: "f32[8, 128, 7, 7]" = split_with_sizes_7[0]
    getitem_84: "f32[8, 192, 7, 7]" = split_with_sizes_7[1]
    getitem_85: "f32[8, 192, 7, 7]" = split_with_sizes_7[2];  split_with_sizes_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    expand_34: "f32[8, 8, 50, 64]" = torch.ops.aten.expand.default(getitem_80, [8, 8, 50, 64])
    clone_58: "f32[8, 8, 50, 64]" = torch.ops.aten.clone.default(expand_34, memory_format = torch.contiguous_format);  expand_34 = None
    view_155: "f32[64, 50, 64]" = torch.ops.aten.reshape.default(clone_58, [64, 50, 64]);  clone_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:123, code: k_softmax = k.softmax(dim=2)
    clone_56: "f32[8, 8, 50, 64]" = torch.ops.aten.clone.default(getitem_81, memory_format = torch.contiguous_format);  getitem_81 = None
    amax_7: "f32[8, 8, 1, 64]" = torch.ops.aten.amax.default(clone_56, [2], True)
    sub_26: "f32[8, 8, 50, 64]" = torch.ops.aten.sub.Tensor(clone_56, amax_7);  clone_56 = amax_7 = None
    exp_7: "f32[8, 8, 50, 64]" = torch.ops.aten.exp.default(sub_26);  sub_26 = None
    sum_8: "f32[8, 8, 1, 64]" = torch.ops.aten.sum.dim_IntList(exp_7, [2], True)
    div_7: "f32[8, 8, 50, 64]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    permute_88: "f32[8, 8, 64, 50]" = torch.ops.aten.permute.default(div_7, [0, 1, 3, 2]);  div_7 = None
    expand_32: "f32[8, 8, 64, 50]" = torch.ops.aten.expand.default(permute_88, [8, 8, 64, 50]);  permute_88 = None
    view_152: "f32[64, 64, 50]" = torch.ops.aten.reshape.default(expand_32, [64, 64, 50]);  expand_32 = None
    expand_33: "f32[8, 8, 50, 64]" = torch.ops.aten.expand.default(getitem_82, [8, 8, 50, 64]);  getitem_82 = None
    clone_57: "f32[8, 8, 50, 64]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
    view_153: "f32[64, 50, 64]" = torch.ops.aten.reshape.default(clone_57, [64, 50, 64]);  clone_57 = None
    bmm_14: "f32[64, 64, 64]" = torch.ops.aten.bmm.default(view_152, view_153);  view_152 = view_153 = None
    view_154: "f32[8, 8, 64, 64]" = torch.ops.aten.reshape.default(bmm_14, [8, 8, 64, 64]);  bmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    expand_35: "f32[8, 8, 64, 64]" = torch.ops.aten.expand.default(view_154, [8, 8, 64, 64]);  view_154 = None
    view_156: "f32[64, 64, 64]" = torch.ops.aten.reshape.default(expand_35, [64, 64, 64]);  expand_35 = None
    bmm_15: "f32[64, 50, 64]" = torch.ops.aten.bmm.default(view_155, view_156);  view_155 = view_156 = None
    view_157: "f32[8, 8, 50, 64]" = torch.ops.aten.reshape.default(bmm_15, [8, 8, 50, 64]);  bmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:131, code: x = self.scale * factor_att + crpe
    mul_74: "f32[8, 8, 50, 64]" = torch.ops.aten.mul.Tensor(view_157, 0.125);  view_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:75, code: q_img = q[:, :, 1:, :]  # [B, h, H*W, Ch]
    slice_100: "f32[8, 8, 49, 64]" = torch.ops.aten.slice.Tensor(getitem_80, 2, 1, 9223372036854775807);  getitem_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    convolution_33: "f32[8, 128, 7, 7]" = torch.ops.aten.convolution.default(getitem_83, arg130_1, arg131_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 128);  getitem_83 = arg130_1 = arg131_1 = None
    convolution_34: "f32[8, 192, 7, 7]" = torch.ops.aten.convolution.default(getitem_84, arg132_1, arg133_1, [1, 1], [2, 2], [1, 1], False, [0, 0], 192);  getitem_84 = arg132_1 = arg133_1 = None
    convolution_35: "f32[8, 192, 7, 7]" = torch.ops.aten.convolution.default(getitem_85, arg134_1, arg135_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 192);  getitem_85 = arg134_1 = arg135_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:83, code: conv_v_img = torch.cat(conv_v_img_list, dim=1)
    cat_19: "f32[8, 512, 7, 7]" = torch.ops.aten.cat.default([convolution_33, convolution_34, convolution_35], 1);  convolution_33 = convolution_34 = convolution_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:84, code: conv_v_img = conv_v_img.reshape(B, num_heads, C, H * W).transpose(-1, -2)
    view_159: "f32[8, 8, 64, 49]" = torch.ops.aten.reshape.default(cat_19, [8, 8, 64, 49]);  cat_19 = None
    permute_90: "f32[8, 8, 49, 64]" = torch.ops.aten.permute.default(view_159, [0, 1, 3, 2]);  view_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:86, code: EV_hat = q_img * conv_v_img
    mul_73: "f32[8, 8, 49, 64]" = torch.ops.aten.mul.Tensor(slice_100, permute_90);  slice_100 = permute_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:87, code: EV_hat = F.pad(EV_hat, (0, 0, 1, 0, 0, 0))  # [B, h, N, Ch].
    constant_pad_nd_7: "f32[8, 8, 50, 64]" = torch.ops.aten.constant_pad_nd.default(mul_73, [0, 0, 1, 0, 0, 0], 0.0);  mul_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:131, code: x = self.scale * factor_att + crpe
    add_74: "f32[8, 8, 50, 64]" = torch.ops.aten.add.Tensor(mul_74, constant_pad_nd_7);  mul_74 = constant_pad_nd_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:132, code: x = x.transpose(1, 2).reshape(B, N, C)  # [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C]
    permute_91: "f32[8, 50, 8, 64]" = torch.ops.aten.permute.default(add_74, [0, 2, 1, 3]);  add_74 = None
    clone_59: "f32[8, 50, 8, 64]" = torch.ops.aten.clone.default(permute_91, memory_format = torch.contiguous_format);  permute_91 = None
    view_160: "f32[8, 50, 512]" = torch.ops.aten.reshape.default(clone_59, [8, 50, 512]);  clone_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:135, code: x = self.proj(x)
    view_161: "f32[400, 512]" = torch.ops.aten.reshape.default(view_160, [400, 512]);  view_160 = None
    permute_92: "f32[512, 512]" = torch.ops.aten.permute.default(arg144_1, [1, 0]);  arg144_1 = None
    addmm_29: "f32[400, 512]" = torch.ops.aten.addmm.default(arg145_1, view_161, permute_92);  arg145_1 = view_161 = permute_92 = None
    view_162: "f32[8, 50, 512]" = torch.ops.aten.reshape.default(addmm_29, [8, 50, 512]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:216, code: x = x + self.drop_path(cur)
    add_75: "f32[8, 50, 512]" = torch.ops.aten.add.Tensor(cat_18, view_162);  cat_18 = view_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_19 = torch.ops.aten.var_mean.correction(add_75, [2], correction = 0, keepdim = True)
    getitem_86: "f32[8, 50, 1]" = var_mean_19[0]
    getitem_87: "f32[8, 50, 1]" = var_mean_19[1];  var_mean_19 = None
    sub_27: "f32[8, 50, 512]" = torch.ops.aten.sub.Tensor(add_75, getitem_87);  getitem_87 = None
    add_76: "f32[8, 50, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-06);  getitem_86 = None
    rsqrt_19: "f32[8, 50, 1]" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
    mul_75: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_19);  sub_27 = rsqrt_19 = None
    mul_76: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(mul_75, arg34_1);  mul_75 = arg34_1 = None
    add_77: "f32[8, 50, 512]" = torch.ops.aten.add.Tensor(mul_76, arg35_1);  mul_76 = arg35_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_163: "f32[400, 512]" = torch.ops.aten.reshape.default(add_77, [400, 512]);  add_77 = None
    permute_93: "f32[512, 2048]" = torch.ops.aten.permute.default(arg146_1, [1, 0]);  arg146_1 = None
    addmm_30: "f32[400, 2048]" = torch.ops.aten.addmm.default(arg147_1, view_163, permute_93);  arg147_1 = view_163 = permute_93 = None
    view_164: "f32[8, 50, 2048]" = torch.ops.aten.reshape.default(addmm_30, [8, 50, 2048]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_77: "f32[8, 50, 2048]" = torch.ops.aten.mul.Tensor(view_164, 0.5)
    mul_78: "f32[8, 50, 2048]" = torch.ops.aten.mul.Tensor(view_164, 0.7071067811865476);  view_164 = None
    erf_7: "f32[8, 50, 2048]" = torch.ops.aten.erf.default(mul_78);  mul_78 = None
    add_78: "f32[8, 50, 2048]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_79: "f32[8, 50, 2048]" = torch.ops.aten.mul.Tensor(mul_77, add_78);  mul_77 = add_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_165: "f32[400, 2048]" = torch.ops.aten.reshape.default(mul_79, [400, 2048]);  mul_79 = None
    permute_94: "f32[2048, 512]" = torch.ops.aten.permute.default(arg148_1, [1, 0]);  arg148_1 = None
    addmm_31: "f32[400, 512]" = torch.ops.aten.addmm.default(arg149_1, view_165, permute_94);  arg149_1 = view_165 = permute_94 = None
    view_166: "f32[8, 50, 512]" = torch.ops.aten.reshape.default(addmm_31, [8, 50, 512]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:221, code: x = x + self.drop_path(cur)
    add_79: "f32[8, 50, 512]" = torch.ops.aten.add.Tensor(add_75, view_166);  add_75 = view_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_20 = torch.ops.aten.var_mean.correction(add_79, [2], correction = 0, keepdim = True)
    getitem_88: "f32[8, 50, 1]" = var_mean_20[0]
    getitem_89: "f32[8, 50, 1]" = var_mean_20[1];  var_mean_20 = None
    sub_28: "f32[8, 50, 512]" = torch.ops.aten.sub.Tensor(add_79, getitem_89);  add_79 = getitem_89 = None
    add_80: "f32[8, 50, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-06);  getitem_88 = None
    rsqrt_20: "f32[8, 50, 1]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
    mul_80: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_20);  sub_28 = rsqrt_20 = None
    mul_81: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(mul_80, arg36_1);  mul_80 = arg36_1 = None
    add_81: "f32[8, 50, 512]" = torch.ops.aten.add.Tensor(mul_81, arg37_1);  mul_81 = arg37_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:660, code: x = x_feat[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x_feat[:, 0]
    select: "f32[8, 512]" = torch.ops.aten.select.int(add_81, 1, 0);  add_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:661, code: x = self.head_drop(x)
    clone_64: "f32[8, 512]" = torch.ops.aten.clone.default(select);  select = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:662, code: return x if pre_logits else self.head(x)
    permute_96: "f32[512, 1000]" = torch.ops.aten.permute.default(arg150_1, [1, 0]);  arg150_1 = None
    addmm_32: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg151_1, clone_64, permute_96);  arg151_1 = clone_64 = permute_96 = None
    return (addmm_32,)
    