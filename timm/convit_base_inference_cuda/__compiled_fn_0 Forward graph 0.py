from __future__ import annotations



def forward(self, arg0_1: "f32[1, 196, 768]", arg1_1: "f32[1, 1, 768]", arg2_1: "f32[768]", arg3_1: "f32[768]", arg4_1: "f32[16]", arg5_1: "f32[768]", arg6_1: "f32[768]", arg7_1: "f32[768]", arg8_1: "f32[768]", arg9_1: "f32[16]", arg10_1: "f32[768]", arg11_1: "f32[768]", arg12_1: "f32[768]", arg13_1: "f32[768]", arg14_1: "f32[16]", arg15_1: "f32[768]", arg16_1: "f32[768]", arg17_1: "f32[768]", arg18_1: "f32[768]", arg19_1: "f32[16]", arg20_1: "f32[768]", arg21_1: "f32[768]", arg22_1: "f32[768]", arg23_1: "f32[768]", arg24_1: "f32[16]", arg25_1: "f32[768]", arg26_1: "f32[768]", arg27_1: "f32[768]", arg28_1: "f32[768]", arg29_1: "f32[16]", arg30_1: "f32[768]", arg31_1: "f32[768]", arg32_1: "f32[768]", arg33_1: "f32[768]", arg34_1: "f32[16]", arg35_1: "f32[768]", arg36_1: "f32[768]", arg37_1: "f32[768]", arg38_1: "f32[768]", arg39_1: "f32[16]", arg40_1: "f32[768]", arg41_1: "f32[768]", arg42_1: "f32[768]", arg43_1: "f32[768]", arg44_1: "f32[16]", arg45_1: "f32[768]", arg46_1: "f32[768]", arg47_1: "f32[768]", arg48_1: "f32[768]", arg49_1: "f32[16]", arg50_1: "f32[768]", arg51_1: "f32[768]", arg52_1: "f32[768]", arg53_1: "f32[768]", arg54_1: "f32[768]", arg55_1: "f32[768]", arg56_1: "f32[768]", arg57_1: "f32[768]", arg58_1: "f32[768]", arg59_1: "f32[768]", arg60_1: "f32[768]", arg61_1: "f32[768]", arg62_1: "f32[768, 3, 16, 16]", arg63_1: "f32[768]", arg64_1: "f32[1536, 768]", arg65_1: "f32[16, 3]", arg66_1: "f32[16]", arg67_1: "f32[768, 768]", arg68_1: "f32[768, 768]", arg69_1: "f32[768]", arg70_1: "f32[3072, 768]", arg71_1: "f32[3072]", arg72_1: "f32[768, 3072]", arg73_1: "f32[768]", arg74_1: "f32[1536, 768]", arg75_1: "f32[16, 3]", arg76_1: "f32[16]", arg77_1: "f32[768, 768]", arg78_1: "f32[768, 768]", arg79_1: "f32[768]", arg80_1: "f32[3072, 768]", arg81_1: "f32[3072]", arg82_1: "f32[768, 3072]", arg83_1: "f32[768]", arg84_1: "f32[1536, 768]", arg85_1: "f32[16, 3]", arg86_1: "f32[16]", arg87_1: "f32[768, 768]", arg88_1: "f32[768, 768]", arg89_1: "f32[768]", arg90_1: "f32[3072, 768]", arg91_1: "f32[3072]", arg92_1: "f32[768, 3072]", arg93_1: "f32[768]", arg94_1: "f32[1536, 768]", arg95_1: "f32[16, 3]", arg96_1: "f32[16]", arg97_1: "f32[768, 768]", arg98_1: "f32[768, 768]", arg99_1: "f32[768]", arg100_1: "f32[3072, 768]", arg101_1: "f32[3072]", arg102_1: "f32[768, 3072]", arg103_1: "f32[768]", arg104_1: "f32[1536, 768]", arg105_1: "f32[16, 3]", arg106_1: "f32[16]", arg107_1: "f32[768, 768]", arg108_1: "f32[768, 768]", arg109_1: "f32[768]", arg110_1: "f32[3072, 768]", arg111_1: "f32[3072]", arg112_1: "f32[768, 3072]", arg113_1: "f32[768]", arg114_1: "f32[1536, 768]", arg115_1: "f32[16, 3]", arg116_1: "f32[16]", arg117_1: "f32[768, 768]", arg118_1: "f32[768, 768]", arg119_1: "f32[768]", arg120_1: "f32[3072, 768]", arg121_1: "f32[3072]", arg122_1: "f32[768, 3072]", arg123_1: "f32[768]", arg124_1: "f32[1536, 768]", arg125_1: "f32[16, 3]", arg126_1: "f32[16]", arg127_1: "f32[768, 768]", arg128_1: "f32[768, 768]", arg129_1: "f32[768]", arg130_1: "f32[3072, 768]", arg131_1: "f32[3072]", arg132_1: "f32[768, 3072]", arg133_1: "f32[768]", arg134_1: "f32[1536, 768]", arg135_1: "f32[16, 3]", arg136_1: "f32[16]", arg137_1: "f32[768, 768]", arg138_1: "f32[768, 768]", arg139_1: "f32[768]", arg140_1: "f32[3072, 768]", arg141_1: "f32[3072]", arg142_1: "f32[768, 3072]", arg143_1: "f32[768]", arg144_1: "f32[1536, 768]", arg145_1: "f32[16, 3]", arg146_1: "f32[16]", arg147_1: "f32[768, 768]", arg148_1: "f32[768, 768]", arg149_1: "f32[768]", arg150_1: "f32[3072, 768]", arg151_1: "f32[3072]", arg152_1: "f32[768, 3072]", arg153_1: "f32[768]", arg154_1: "f32[1536, 768]", arg155_1: "f32[16, 3]", arg156_1: "f32[16]", arg157_1: "f32[768, 768]", arg158_1: "f32[768, 768]", arg159_1: "f32[768]", arg160_1: "f32[3072, 768]", arg161_1: "f32[3072]", arg162_1: "f32[768, 3072]", arg163_1: "f32[768]", arg164_1: "f32[2304, 768]", arg165_1: "f32[768, 768]", arg166_1: "f32[768]", arg167_1: "f32[3072, 768]", arg168_1: "f32[3072]", arg169_1: "f32[768, 3072]", arg170_1: "f32[768]", arg171_1: "f32[2304, 768]", arg172_1: "f32[768, 768]", arg173_1: "f32[768]", arg174_1: "f32[3072, 768]", arg175_1: "f32[3072]", arg176_1: "f32[768, 3072]", arg177_1: "f32[768]", arg178_1: "f32[1000, 768]", arg179_1: "f32[1000]", arg180_1: "f32[8, 3, 224, 224]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution: "f32[8, 768, 14, 14]" = torch.ops.aten.convolution.default(arg180_1, arg62_1, arg63_1, [16, 16], [0, 0], [1, 1], False, [0, 0], 1);  arg180_1 = arg62_1 = arg63_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:89, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    view: "f32[8, 768, 196]" = torch.ops.aten.view.default(convolution, [8, 768, 196]);  convolution = None
    permute: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:362, code: x = x + self.pos_embed
    add: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(permute, arg0_1);  permute = arg0_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:363, code: x = self.pos_drop(x)
    clone: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add);  add = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:364, code: cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
    expand: "f32[8, 1, 768]" = torch.ops.aten.expand.default(arg1_1, [8, -1, -1]);  arg1_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_1: "f32[8, 196, 768]" = torch.ops.aten.clone.default(clone, memory_format = torch.contiguous_format)
    var_mean = torch.ops.aten.var_mean.correction(clone_1, [2], correction = 0, keepdim = True)
    getitem: "f32[8, 196, 1]" = var_mean[0]
    getitem_1: "f32[8, 196, 1]" = var_mean[1];  var_mean = None
    add_1: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
    rsqrt: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    sub: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_1, getitem_1);  clone_1 = getitem_1 = None
    mul: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
    mul_1: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul, arg2_1);  mul = arg2_1 = None
    add_2: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_1, arg3_1);  mul_1 = arg3_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:121, code: rel_indices = torch.zeros(1, num_patches, num_patches, 3)
    full: "f32[1, 196, 196, 3]" = torch.ops.aten.full.default([1, 196, 196, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:122, code: ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
    iota: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_1: "i64[1, 14]" = torch.ops.aten.view.default(iota, [1, -1]);  iota = None
    iota_1: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_2: "i64[14, 1]" = torch.ops.aten.view.default(iota_1, [-1, 1]);  iota_1 = None
    sub_1: "i64[14, 14]" = torch.ops.aten.sub.Tensor(view_1, view_2);  view_1 = view_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:123, code: indx = ind.repeat(img_size, img_size)
    repeat: "i64[196, 196]" = torch.ops.aten.repeat.default(sub_1, [14, 14])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:124, code: indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
    unsqueeze: "i64[14, 1, 14]" = torch.ops.aten.unsqueeze.default(sub_1, 1);  sub_1 = None
    expand_1: "i64[14, 14, 14]" = torch.ops.aten.expand.default(unsqueeze, [14, 14, 14]);  unsqueeze = None
    clone_2: "i64[14, 14, 14]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
    view_3: "i64[196, 14]" = torch.ops.aten.view.default(clone_2, [196, 14]);  clone_2 = None
    unsqueeze_1: "i64[196, 14, 1]" = torch.ops.aten.unsqueeze.default(view_3, 2);  view_3 = None
    expand_2: "i64[196, 14, 14]" = torch.ops.aten.expand.default(unsqueeze_1, [196, 14, 14]);  unsqueeze_1 = None
    clone_3: "i64[196, 14, 14]" = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
    view_4: "i64[196, 196]" = torch.ops.aten.view.default(clone_3, [196, 196]);  clone_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:125, code: indd = indx ** 2 + indy ** 2
    pow_1: "i64[196, 196]" = torch.ops.aten.pow.Tensor_Scalar(repeat, 2)
    pow_2: "i64[196, 196]" = torch.ops.aten.pow.Tensor_Scalar(view_4, 2)
    add_3: "i64[196, 196]" = torch.ops.aten.add.Tensor(pow_1, pow_2);  pow_1 = pow_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:126, code: rel_indices[:, :, :, 2] = indd.unsqueeze(0)
    unsqueeze_2: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(add_3, 0);  add_3 = None
    slice_1: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(full, 0, 0, 9223372036854775807)
    slice_2: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_1, 1, 0, 9223372036854775807);  slice_1 = None
    slice_3: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_2, 2, 0, 9223372036854775807);  slice_2 = None
    select: "f32[1, 196, 196]" = torch.ops.aten.select.int(slice_3, 3, 2);  slice_3 = None
    copy: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select, unsqueeze_2);  select = unsqueeze_2 = None
    slice_4: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(full, 0, 0, 9223372036854775807)
    slice_5: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_4, 1, 0, 9223372036854775807)
    slice_6: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_5, 2, 0, 9223372036854775807)
    select_scatter: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(slice_6, copy, 3, 2);  slice_6 = copy = None
    slice_scatter: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_5, select_scatter, 2, 0, 9223372036854775807);  slice_5 = select_scatter = None
    slice_scatter_1: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_4, slice_scatter, 1, 0, 9223372036854775807);  slice_4 = slice_scatter = None
    slice_scatter_2: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(full, slice_scatter_1, 0, 0, 9223372036854775807);  full = slice_scatter_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:127, code: rel_indices[:, :, :, 1] = indy.unsqueeze(0)
    unsqueeze_3: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(view_4, 0);  view_4 = None
    slice_13: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_2, 0, 0, 9223372036854775807)
    slice_14: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_13, 1, 0, 9223372036854775807);  slice_13 = None
    slice_15: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_14, 2, 0, 9223372036854775807);  slice_14 = None
    select_3: "f32[1, 196, 196]" = torch.ops.aten.select.int(slice_15, 3, 1);  slice_15 = None
    copy_1: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_3, unsqueeze_3);  select_3 = unsqueeze_3 = None
    slice_16: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_2, 0, 0, 9223372036854775807)
    slice_17: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_16, 1, 0, 9223372036854775807)
    slice_18: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_17, 2, 0, 9223372036854775807)
    select_scatter_1: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(slice_18, copy_1, 3, 1);  slice_18 = copy_1 = None
    slice_scatter_3: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_17, select_scatter_1, 2, 0, 9223372036854775807);  slice_17 = select_scatter_1 = None
    slice_scatter_4: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_16, slice_scatter_3, 1, 0, 9223372036854775807);  slice_16 = slice_scatter_3 = None
    slice_scatter_5: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_scatter_2, slice_scatter_4, 0, 0, 9223372036854775807);  slice_scatter_2 = slice_scatter_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:128, code: rel_indices[:, :, :, 0] = indx.unsqueeze(0)
    unsqueeze_4: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(repeat, 0);  repeat = None
    slice_25: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_5, 0, 0, 9223372036854775807)
    slice_26: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_25, 1, 0, 9223372036854775807);  slice_25 = None
    slice_27: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_26, 2, 0, 9223372036854775807);  slice_26 = None
    select_6: "f32[1, 196, 196]" = torch.ops.aten.select.int(slice_27, 3, 0);  slice_27 = None
    copy_2: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_6, unsqueeze_4);  select_6 = unsqueeze_4 = None
    slice_28: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_5, 0, 0, 9223372036854775807)
    slice_29: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_28, 1, 0, 9223372036854775807)
    slice_30: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_29, 2, 0, 9223372036854775807)
    select_scatter_2: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(slice_30, copy_2, 3, 0);  slice_30 = copy_2 = None
    slice_scatter_6: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_29, select_scatter_2, 2, 0, 9223372036854775807);  slice_29 = select_scatter_2 = None
    slice_scatter_7: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_28, slice_scatter_6, 1, 0, 9223372036854775807);  slice_28 = slice_scatter_6 = None
    slice_scatter_8: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_scatter_5, slice_scatter_7, 0, 0, 9223372036854775807);  slice_scatter_5 = slice_scatter_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:130, code: return rel_indices.to(device)
    device_put: "f32[1, 196, 196, 3]" = torch.ops.prims.device_put.default(slice_scatter_8, device(type='cuda', index=0));  slice_scatter_8 = None
    convert_element_type: "f32[1, 196, 196, 3]" = torch.ops.prims.convert_element_type.default(device_put, torch.float32);  device_put = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_1: "f32[768, 1536]" = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
    view_5: "f32[1568, 768]" = torch.ops.aten.view.default(add_2, [1568, 768])
    mm: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_5, permute_1);  view_5 = permute_1 = None
    view_6: "f32[8, 196, 1536]" = torch.ops.aten.view.default(mm, [8, 196, 1536]);  mm = None
    view_7: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.view.default(view_6, [8, 196, 2, 16, 48]);  view_6 = None
    permute_2: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.permute.default(view_7, [2, 0, 3, 1, 4]);  view_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    select_8: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_2, 0, 0)
    select_9: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_2, 0, 1);  permute_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:84, code: pos_score = self.rel_indices.expand(B, -1, -1, -1)
    expand_3: "f32[8, 196, 196, 3]" = torch.ops.aten.expand.default(convert_element_type, [8, -1, -1, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    permute_3: "f32[3, 16]" = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
    clone_4: "f32[8, 196, 196, 3]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
    view_8: "f32[307328, 3]" = torch.ops.aten.view.default(clone_4, [307328, 3]);  clone_4 = None
    mm_1: "f32[307328, 16]" = torch.ops.aten.mm.default(view_8, permute_3);  view_8 = permute_3 = None
    view_9: "f32[8, 196, 196, 16]" = torch.ops.aten.view.default(mm_1, [8, 196, 196, 16]);  mm_1 = None
    add_4: "f32[8, 196, 196, 16]" = torch.ops.aten.add.Tensor(view_9, arg66_1);  view_9 = arg66_1 = None
    permute_4: "f32[8, 16, 196, 196]" = torch.ops.aten.permute.default(add_4, [0, 3, 1, 2]);  add_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    permute_5: "f32[8, 16, 48, 196]" = torch.ops.aten.permute.default(select_9, [0, 1, 3, 2]);  select_9 = None
    expand_4: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(select_8, [8, 16, 196, 48]);  select_8 = None
    clone_5: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
    view_10: "f32[128, 196, 48]" = torch.ops.aten.view.default(clone_5, [128, 196, 48]);  clone_5 = None
    expand_5: "f32[8, 16, 48, 196]" = torch.ops.aten.expand.default(permute_5, [8, 16, 48, 196]);  permute_5 = None
    clone_6: "f32[8, 16, 48, 196]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
    view_11: "f32[128, 48, 196]" = torch.ops.aten.view.default(clone_6, [128, 48, 196]);  clone_6 = None
    bmm: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_10, view_11);  view_10 = view_11 = None
    view_12: "f32[8, 16, 196, 196]" = torch.ops.aten.view.default(bmm, [8, 16, 196, 196]);  bmm = None
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
    view_13: "f32[1, 16, 1, 1]" = torch.ops.aten.view.default(arg4_1, [1, -1, 1, 1]);  arg4_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    sigmoid: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_13)
    sub_4: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1.0, sigmoid);  sigmoid = None
    mul_3: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_4, div);  sub_4 = div = None
    sigmoid_1: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_13);  view_13 = None
    mul_4: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sigmoid_1, div_1);  sigmoid_1 = div_1 = None
    add_5: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(mul_3, mul_4);  mul_3 = mul_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    sum_3: "f32[8, 16, 196]" = torch.ops.aten.sum.dim_IntList(add_5, [-1])
    unsqueeze_5: "f32[8, 16, 196, 1]" = torch.ops.aten.unsqueeze.default(sum_3, -1);  sum_3 = None
    div_2: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(add_5, unsqueeze_5);  add_5 = unsqueeze_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:93, code: attn = self.attn_drop(attn)
    clone_8: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(div_2);  div_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_6: "f32[768, 768]" = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
    view_14: "f32[1568, 768]" = torch.ops.aten.view.default(add_2, [1568, 768]);  add_2 = None
    mm_2: "f32[1568, 768]" = torch.ops.aten.mm.default(view_14, permute_6);  view_14 = permute_6 = None
    view_15: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_2, [8, 196, 768]);  mm_2 = None
    view_16: "f32[8, 196, 16, 48]" = torch.ops.aten.view.default(view_15, [8, 196, 16, 48]);  view_15 = None
    permute_7: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_6: "f32[8, 16, 196, 196]" = torch.ops.aten.expand.default(clone_8, [8, 16, 196, 196]);  clone_8 = None
    view_17: "f32[128, 196, 196]" = torch.ops.aten.view.default(expand_6, [128, 196, 196]);  expand_6 = None
    expand_7: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(permute_7, [8, 16, 196, 48]);  permute_7 = None
    clone_9: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
    view_18: "f32[128, 196, 48]" = torch.ops.aten.view.default(clone_9, [128, 196, 48]);  clone_9 = None
    bmm_1: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(view_17, view_18);  view_17 = view_18 = None
    view_19: "f32[8, 16, 196, 48]" = torch.ops.aten.view.default(bmm_1, [8, 16, 196, 48]);  bmm_1 = None
    permute_8: "f32[8, 196, 16, 48]" = torch.ops.aten.permute.default(view_19, [0, 2, 1, 3]);  view_19 = None
    clone_10: "f32[8, 196, 16, 48]" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None
    view_20: "f32[8, 196, 768]" = torch.ops.aten.view.default(clone_10, [8, 196, 768]);  clone_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_21: "f32[1568, 768]" = torch.ops.aten.view.default(view_20, [1568, 768]);  view_20 = None
    permute_9: "f32[768, 768]" = torch.ops.aten.permute.default(arg68_1, [1, 0]);  arg68_1 = None
    addmm: "f32[1568, 768]" = torch.ops.aten.addmm.default(arg69_1, view_21, permute_9);  arg69_1 = view_21 = permute_9 = None
    view_22: "f32[8, 196, 768]" = torch.ops.aten.view.default(addmm, [8, 196, 768]);  addmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:77, code: x = self.proj_drop(x)
    clone_11: "f32[8, 196, 768]" = torch.ops.aten.clone.default(view_22);  view_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:235, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_6: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(clone, clone_11);  clone = clone_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_12: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_6, memory_format = torch.contiguous_format)
    var_mean_1 = torch.ops.aten.var_mean.correction(clone_12, [2], correction = 0, keepdim = True)
    getitem_2: "f32[8, 196, 1]" = var_mean_1[0]
    getitem_3: "f32[8, 196, 1]" = var_mean_1[1];  var_mean_1 = None
    add_7: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-06);  getitem_2 = None
    rsqrt_1: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
    sub_5: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_12, getitem_3);  clone_12 = getitem_3 = None
    mul_5: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_1);  sub_5 = rsqrt_1 = None
    mul_6: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_5, arg5_1);  mul_5 = arg5_1 = None
    add_8: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_6, arg6_1);  mul_6 = arg6_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_23: "f32[1568, 768]" = torch.ops.aten.view.default(add_8, [1568, 768]);  add_8 = None
    permute_10: "f32[768, 3072]" = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
    addmm_1: "f32[1568, 3072]" = torch.ops.aten.addmm.default(arg71_1, view_23, permute_10);  arg71_1 = view_23 = permute_10 = None
    view_24: "f32[8, 196, 3072]" = torch.ops.aten.view.default(addmm_1, [8, 196, 3072]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_7: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_24, 0.5)
    mul_8: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_24, 0.7071067811865476);  view_24 = None
    erf: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_8);  mul_8 = None
    add_9: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_9: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_7, add_9);  mul_7 = add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_13: "f32[8, 196, 3072]" = torch.ops.aten.clone.default(mul_9);  mul_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_25: "f32[1568, 3072]" = torch.ops.aten.view.default(clone_13, [1568, 3072]);  clone_13 = None
    permute_11: "f32[3072, 768]" = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
    addmm_2: "f32[1568, 768]" = torch.ops.aten.addmm.default(arg73_1, view_25, permute_11);  arg73_1 = view_25 = permute_11 = None
    view_26: "f32[8, 196, 768]" = torch.ops.aten.view.default(addmm_2, [8, 196, 768]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_14: "f32[8, 196, 768]" = torch.ops.aten.clone.default(view_26);  view_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:236, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_10: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_6, clone_14);  add_6 = clone_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_15: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_10, memory_format = torch.contiguous_format)
    var_mean_2 = torch.ops.aten.var_mean.correction(clone_15, [2], correction = 0, keepdim = True)
    getitem_4: "f32[8, 196, 1]" = var_mean_2[0]
    getitem_5: "f32[8, 196, 1]" = var_mean_2[1];  var_mean_2 = None
    add_11: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-06);  getitem_4 = None
    rsqrt_2: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    sub_6: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_15, getitem_5);  clone_15 = getitem_5 = None
    mul_10: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_2);  sub_6 = rsqrt_2 = None
    mul_11: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_10, arg7_1);  mul_10 = arg7_1 = None
    add_12: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_11, arg8_1);  mul_11 = arg8_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:121, code: rel_indices = torch.zeros(1, num_patches, num_patches, 3)
    full_1: "f32[1, 196, 196, 3]" = torch.ops.aten.full.default([1, 196, 196, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:122, code: ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
    iota_2: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_27: "i64[1, 14]" = torch.ops.aten.view.default(iota_2, [1, -1]);  iota_2 = None
    iota_3: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_28: "i64[14, 1]" = torch.ops.aten.view.default(iota_3, [-1, 1]);  iota_3 = None
    sub_7: "i64[14, 14]" = torch.ops.aten.sub.Tensor(view_27, view_28);  view_27 = view_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:123, code: indx = ind.repeat(img_size, img_size)
    repeat_1: "i64[196, 196]" = torch.ops.aten.repeat.default(sub_7, [14, 14])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:124, code: indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
    unsqueeze_6: "i64[14, 1, 14]" = torch.ops.aten.unsqueeze.default(sub_7, 1);  sub_7 = None
    expand_8: "i64[14, 14, 14]" = torch.ops.aten.expand.default(unsqueeze_6, [14, 14, 14]);  unsqueeze_6 = None
    clone_16: "i64[14, 14, 14]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
    view_29: "i64[196, 14]" = torch.ops.aten.view.default(clone_16, [196, 14]);  clone_16 = None
    unsqueeze_7: "i64[196, 14, 1]" = torch.ops.aten.unsqueeze.default(view_29, 2);  view_29 = None
    expand_9: "i64[196, 14, 14]" = torch.ops.aten.expand.default(unsqueeze_7, [196, 14, 14]);  unsqueeze_7 = None
    clone_17: "i64[196, 14, 14]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
    view_30: "i64[196, 196]" = torch.ops.aten.view.default(clone_17, [196, 196]);  clone_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:125, code: indd = indx ** 2 + indy ** 2
    pow_3: "i64[196, 196]" = torch.ops.aten.pow.Tensor_Scalar(repeat_1, 2)
    pow_4: "i64[196, 196]" = torch.ops.aten.pow.Tensor_Scalar(view_30, 2)
    add_13: "i64[196, 196]" = torch.ops.aten.add.Tensor(pow_3, pow_4);  pow_3 = pow_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:126, code: rel_indices[:, :, :, 2] = indd.unsqueeze(0)
    unsqueeze_8: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(add_13, 0);  add_13 = None
    slice_34: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(full_1, 0, 0, 9223372036854775807)
    slice_35: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_34, 1, 0, 9223372036854775807);  slice_34 = None
    slice_36: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_35, 2, 0, 9223372036854775807);  slice_35 = None
    select_10: "f32[1, 196, 196]" = torch.ops.aten.select.int(slice_36, 3, 2);  slice_36 = None
    copy_3: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_10, unsqueeze_8);  select_10 = unsqueeze_8 = None
    slice_37: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(full_1, 0, 0, 9223372036854775807)
    slice_38: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_37, 1, 0, 9223372036854775807)
    slice_39: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_38, 2, 0, 9223372036854775807)
    select_scatter_3: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(slice_39, copy_3, 3, 2);  slice_39 = copy_3 = None
    slice_scatter_9: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_38, select_scatter_3, 2, 0, 9223372036854775807);  slice_38 = select_scatter_3 = None
    slice_scatter_10: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_37, slice_scatter_9, 1, 0, 9223372036854775807);  slice_37 = slice_scatter_9 = None
    slice_scatter_11: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(full_1, slice_scatter_10, 0, 0, 9223372036854775807);  full_1 = slice_scatter_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:127, code: rel_indices[:, :, :, 1] = indy.unsqueeze(0)
    unsqueeze_9: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(view_30, 0);  view_30 = None
    slice_46: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_11, 0, 0, 9223372036854775807)
    slice_47: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_46, 1, 0, 9223372036854775807);  slice_46 = None
    slice_48: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_47, 2, 0, 9223372036854775807);  slice_47 = None
    select_13: "f32[1, 196, 196]" = torch.ops.aten.select.int(slice_48, 3, 1);  slice_48 = None
    copy_4: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_13, unsqueeze_9);  select_13 = unsqueeze_9 = None
    slice_49: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_11, 0, 0, 9223372036854775807)
    slice_50: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_49, 1, 0, 9223372036854775807)
    slice_51: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_50, 2, 0, 9223372036854775807)
    select_scatter_4: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(slice_51, copy_4, 3, 1);  slice_51 = copy_4 = None
    slice_scatter_12: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_50, select_scatter_4, 2, 0, 9223372036854775807);  slice_50 = select_scatter_4 = None
    slice_scatter_13: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_49, slice_scatter_12, 1, 0, 9223372036854775807);  slice_49 = slice_scatter_12 = None
    slice_scatter_14: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_scatter_11, slice_scatter_13, 0, 0, 9223372036854775807);  slice_scatter_11 = slice_scatter_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:128, code: rel_indices[:, :, :, 0] = indx.unsqueeze(0)
    unsqueeze_10: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(repeat_1, 0);  repeat_1 = None
    slice_58: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_14, 0, 0, 9223372036854775807)
    slice_59: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_58, 1, 0, 9223372036854775807);  slice_58 = None
    slice_60: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_59, 2, 0, 9223372036854775807);  slice_59 = None
    select_16: "f32[1, 196, 196]" = torch.ops.aten.select.int(slice_60, 3, 0);  slice_60 = None
    copy_5: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_16, unsqueeze_10);  select_16 = unsqueeze_10 = None
    slice_61: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_14, 0, 0, 9223372036854775807)
    slice_62: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_61, 1, 0, 9223372036854775807)
    slice_63: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_62, 2, 0, 9223372036854775807)
    select_scatter_5: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(slice_63, copy_5, 3, 0);  slice_63 = copy_5 = None
    slice_scatter_15: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_62, select_scatter_5, 2, 0, 9223372036854775807);  slice_62 = select_scatter_5 = None
    slice_scatter_16: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_61, slice_scatter_15, 1, 0, 9223372036854775807);  slice_61 = slice_scatter_15 = None
    slice_scatter_17: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_scatter_14, slice_scatter_16, 0, 0, 9223372036854775807);  slice_scatter_14 = slice_scatter_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:130, code: return rel_indices.to(device)
    device_put_1: "f32[1, 196, 196, 3]" = torch.ops.prims.device_put.default(slice_scatter_17, device(type='cuda', index=0));  slice_scatter_17 = None
    convert_element_type_1: "f32[1, 196, 196, 3]" = torch.ops.prims.convert_element_type.default(device_put_1, torch.float32);  device_put_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_12: "f32[768, 1536]" = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
    view_31: "f32[1568, 768]" = torch.ops.aten.view.default(add_12, [1568, 768])
    mm_3: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_31, permute_12);  view_31 = permute_12 = None
    view_32: "f32[8, 196, 1536]" = torch.ops.aten.view.default(mm_3, [8, 196, 1536]);  mm_3 = None
    view_33: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.view.default(view_32, [8, 196, 2, 16, 48]);  view_32 = None
    permute_13: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.permute.default(view_33, [2, 0, 3, 1, 4]);  view_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    select_18: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_13, 0, 0)
    select_19: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_13, 0, 1);  permute_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:84, code: pos_score = self.rel_indices.expand(B, -1, -1, -1)
    expand_10: "f32[8, 196, 196, 3]" = torch.ops.aten.expand.default(convert_element_type_1, [8, -1, -1, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    permute_14: "f32[3, 16]" = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
    clone_18: "f32[8, 196, 196, 3]" = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
    view_34: "f32[307328, 3]" = torch.ops.aten.view.default(clone_18, [307328, 3]);  clone_18 = None
    mm_4: "f32[307328, 16]" = torch.ops.aten.mm.default(view_34, permute_14);  view_34 = permute_14 = None
    view_35: "f32[8, 196, 196, 16]" = torch.ops.aten.view.default(mm_4, [8, 196, 196, 16]);  mm_4 = None
    add_14: "f32[8, 196, 196, 16]" = torch.ops.aten.add.Tensor(view_35, arg76_1);  view_35 = arg76_1 = None
    permute_15: "f32[8, 16, 196, 196]" = torch.ops.aten.permute.default(add_14, [0, 3, 1, 2]);  add_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    permute_16: "f32[8, 16, 48, 196]" = torch.ops.aten.permute.default(select_19, [0, 1, 3, 2]);  select_19 = None
    expand_11: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(select_18, [8, 16, 196, 48]);  select_18 = None
    clone_19: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
    view_36: "f32[128, 196, 48]" = torch.ops.aten.view.default(clone_19, [128, 196, 48]);  clone_19 = None
    expand_12: "f32[8, 16, 48, 196]" = torch.ops.aten.expand.default(permute_16, [8, 16, 48, 196]);  permute_16 = None
    clone_20: "f32[8, 16, 48, 196]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
    view_37: "f32[128, 48, 196]" = torch.ops.aten.view.default(clone_20, [128, 48, 196]);  clone_20 = None
    bmm_2: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_36, view_37);  view_36 = view_37 = None
    view_38: "f32[8, 16, 196, 196]" = torch.ops.aten.view.default(bmm_2, [8, 16, 196, 196]);  bmm_2 = None
    mul_12: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(view_38, 0.14433756729740643);  view_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    amax_2: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(mul_12, [-1], True)
    sub_8: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_12, amax_2);  mul_12 = amax_2 = None
    exp_2: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_8);  sub_8 = None
    sum_4: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_3: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_2, sum_4);  exp_2 = sum_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    clone_21: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
    amax_3: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(clone_21, [-1], True)
    sub_9: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(clone_21, amax_3);  clone_21 = amax_3 = None
    exp_3: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_9);  sub_9 = None
    sum_5: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_4: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_3, sum_5);  exp_3 = sum_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_39: "f32[1, 16, 1, 1]" = torch.ops.aten.view.default(arg9_1, [1, -1, 1, 1]);  arg9_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    sigmoid_2: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_39)
    sub_10: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1.0, sigmoid_2);  sigmoid_2 = None
    mul_13: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_10, div_3);  sub_10 = div_3 = None
    sigmoid_3: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_39);  view_39 = None
    mul_14: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sigmoid_3, div_4);  sigmoid_3 = div_4 = None
    add_15: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(mul_13, mul_14);  mul_13 = mul_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    sum_6: "f32[8, 16, 196]" = torch.ops.aten.sum.dim_IntList(add_15, [-1])
    unsqueeze_11: "f32[8, 16, 196, 1]" = torch.ops.aten.unsqueeze.default(sum_6, -1);  sum_6 = None
    div_5: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(add_15, unsqueeze_11);  add_15 = unsqueeze_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:93, code: attn = self.attn_drop(attn)
    clone_22: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(div_5);  div_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_17: "f32[768, 768]" = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
    view_40: "f32[1568, 768]" = torch.ops.aten.view.default(add_12, [1568, 768]);  add_12 = None
    mm_5: "f32[1568, 768]" = torch.ops.aten.mm.default(view_40, permute_17);  view_40 = permute_17 = None
    view_41: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_5, [8, 196, 768]);  mm_5 = None
    view_42: "f32[8, 196, 16, 48]" = torch.ops.aten.view.default(view_41, [8, 196, 16, 48]);  view_41 = None
    permute_18: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_42, [0, 2, 1, 3]);  view_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_13: "f32[8, 16, 196, 196]" = torch.ops.aten.expand.default(clone_22, [8, 16, 196, 196]);  clone_22 = None
    view_43: "f32[128, 196, 196]" = torch.ops.aten.view.default(expand_13, [128, 196, 196]);  expand_13 = None
    expand_14: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(permute_18, [8, 16, 196, 48]);  permute_18 = None
    clone_23: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_14, memory_format = torch.contiguous_format);  expand_14 = None
    view_44: "f32[128, 196, 48]" = torch.ops.aten.view.default(clone_23, [128, 196, 48]);  clone_23 = None
    bmm_3: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(view_43, view_44);  view_43 = view_44 = None
    view_45: "f32[8, 16, 196, 48]" = torch.ops.aten.view.default(bmm_3, [8, 16, 196, 48]);  bmm_3 = None
    permute_19: "f32[8, 196, 16, 48]" = torch.ops.aten.permute.default(view_45, [0, 2, 1, 3]);  view_45 = None
    clone_24: "f32[8, 196, 16, 48]" = torch.ops.aten.clone.default(permute_19, memory_format = torch.contiguous_format);  permute_19 = None
    view_46: "f32[8, 196, 768]" = torch.ops.aten.view.default(clone_24, [8, 196, 768]);  clone_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_47: "f32[1568, 768]" = torch.ops.aten.view.default(view_46, [1568, 768]);  view_46 = None
    permute_20: "f32[768, 768]" = torch.ops.aten.permute.default(arg78_1, [1, 0]);  arg78_1 = None
    addmm_3: "f32[1568, 768]" = torch.ops.aten.addmm.default(arg79_1, view_47, permute_20);  arg79_1 = view_47 = permute_20 = None
    view_48: "f32[8, 196, 768]" = torch.ops.aten.view.default(addmm_3, [8, 196, 768]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:77, code: x = self.proj_drop(x)
    clone_25: "f32[8, 196, 768]" = torch.ops.aten.clone.default(view_48);  view_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:235, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_16: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_10, clone_25);  add_10 = clone_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_26: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_16, memory_format = torch.contiguous_format)
    var_mean_3 = torch.ops.aten.var_mean.correction(clone_26, [2], correction = 0, keepdim = True)
    getitem_6: "f32[8, 196, 1]" = var_mean_3[0]
    getitem_7: "f32[8, 196, 1]" = var_mean_3[1];  var_mean_3 = None
    add_17: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-06);  getitem_6 = None
    rsqrt_3: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
    sub_11: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_26, getitem_7);  clone_26 = getitem_7 = None
    mul_15: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_3);  sub_11 = rsqrt_3 = None
    mul_16: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_15, arg10_1);  mul_15 = arg10_1 = None
    add_18: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_16, arg11_1);  mul_16 = arg11_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_49: "f32[1568, 768]" = torch.ops.aten.view.default(add_18, [1568, 768]);  add_18 = None
    permute_21: "f32[768, 3072]" = torch.ops.aten.permute.default(arg80_1, [1, 0]);  arg80_1 = None
    addmm_4: "f32[1568, 3072]" = torch.ops.aten.addmm.default(arg81_1, view_49, permute_21);  arg81_1 = view_49 = permute_21 = None
    view_50: "f32[8, 196, 3072]" = torch.ops.aten.view.default(addmm_4, [8, 196, 3072]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_17: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_50, 0.5)
    mul_18: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_50, 0.7071067811865476);  view_50 = None
    erf_1: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_18);  mul_18 = None
    add_19: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_19: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_17, add_19);  mul_17 = add_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_27: "f32[8, 196, 3072]" = torch.ops.aten.clone.default(mul_19);  mul_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_51: "f32[1568, 3072]" = torch.ops.aten.view.default(clone_27, [1568, 3072]);  clone_27 = None
    permute_22: "f32[3072, 768]" = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
    addmm_5: "f32[1568, 768]" = torch.ops.aten.addmm.default(arg83_1, view_51, permute_22);  arg83_1 = view_51 = permute_22 = None
    view_52: "f32[8, 196, 768]" = torch.ops.aten.view.default(addmm_5, [8, 196, 768]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_28: "f32[8, 196, 768]" = torch.ops.aten.clone.default(view_52);  view_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:236, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_20: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_16, clone_28);  add_16 = clone_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_29: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_20, memory_format = torch.contiguous_format)
    var_mean_4 = torch.ops.aten.var_mean.correction(clone_29, [2], correction = 0, keepdim = True)
    getitem_8: "f32[8, 196, 1]" = var_mean_4[0]
    getitem_9: "f32[8, 196, 1]" = var_mean_4[1];  var_mean_4 = None
    add_21: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-06);  getitem_8 = None
    rsqrt_4: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
    sub_12: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_29, getitem_9);  clone_29 = getitem_9 = None
    mul_20: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_4);  sub_12 = rsqrt_4 = None
    mul_21: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_20, arg12_1);  mul_20 = arg12_1 = None
    add_22: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_21, arg13_1);  mul_21 = arg13_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:121, code: rel_indices = torch.zeros(1, num_patches, num_patches, 3)
    full_2: "f32[1, 196, 196, 3]" = torch.ops.aten.full.default([1, 196, 196, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:122, code: ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
    iota_4: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_53: "i64[1, 14]" = torch.ops.aten.view.default(iota_4, [1, -1]);  iota_4 = None
    iota_5: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_54: "i64[14, 1]" = torch.ops.aten.view.default(iota_5, [-1, 1]);  iota_5 = None
    sub_13: "i64[14, 14]" = torch.ops.aten.sub.Tensor(view_53, view_54);  view_53 = view_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:123, code: indx = ind.repeat(img_size, img_size)
    repeat_2: "i64[196, 196]" = torch.ops.aten.repeat.default(sub_13, [14, 14])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:124, code: indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
    unsqueeze_12: "i64[14, 1, 14]" = torch.ops.aten.unsqueeze.default(sub_13, 1);  sub_13 = None
    expand_15: "i64[14, 14, 14]" = torch.ops.aten.expand.default(unsqueeze_12, [14, 14, 14]);  unsqueeze_12 = None
    clone_30: "i64[14, 14, 14]" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
    view_55: "i64[196, 14]" = torch.ops.aten.view.default(clone_30, [196, 14]);  clone_30 = None
    unsqueeze_13: "i64[196, 14, 1]" = torch.ops.aten.unsqueeze.default(view_55, 2);  view_55 = None
    expand_16: "i64[196, 14, 14]" = torch.ops.aten.expand.default(unsqueeze_13, [196, 14, 14]);  unsqueeze_13 = None
    clone_31: "i64[196, 14, 14]" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
    view_56: "i64[196, 196]" = torch.ops.aten.view.default(clone_31, [196, 196]);  clone_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:125, code: indd = indx ** 2 + indy ** 2
    pow_5: "i64[196, 196]" = torch.ops.aten.pow.Tensor_Scalar(repeat_2, 2)
    pow_6: "i64[196, 196]" = torch.ops.aten.pow.Tensor_Scalar(view_56, 2)
    add_23: "i64[196, 196]" = torch.ops.aten.add.Tensor(pow_5, pow_6);  pow_5 = pow_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:126, code: rel_indices[:, :, :, 2] = indd.unsqueeze(0)
    unsqueeze_14: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(add_23, 0);  add_23 = None
    slice_67: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(full_2, 0, 0, 9223372036854775807)
    slice_68: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_67, 1, 0, 9223372036854775807);  slice_67 = None
    slice_69: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_68, 2, 0, 9223372036854775807);  slice_68 = None
    select_20: "f32[1, 196, 196]" = torch.ops.aten.select.int(slice_69, 3, 2);  slice_69 = None
    copy_6: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_20, unsqueeze_14);  select_20 = unsqueeze_14 = None
    slice_70: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(full_2, 0, 0, 9223372036854775807)
    slice_71: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_70, 1, 0, 9223372036854775807)
    slice_72: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_71, 2, 0, 9223372036854775807)
    select_scatter_6: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(slice_72, copy_6, 3, 2);  slice_72 = copy_6 = None
    slice_scatter_18: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_71, select_scatter_6, 2, 0, 9223372036854775807);  slice_71 = select_scatter_6 = None
    slice_scatter_19: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_70, slice_scatter_18, 1, 0, 9223372036854775807);  slice_70 = slice_scatter_18 = None
    slice_scatter_20: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(full_2, slice_scatter_19, 0, 0, 9223372036854775807);  full_2 = slice_scatter_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:127, code: rel_indices[:, :, :, 1] = indy.unsqueeze(0)
    unsqueeze_15: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(view_56, 0);  view_56 = None
    slice_79: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_20, 0, 0, 9223372036854775807)
    slice_80: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_79, 1, 0, 9223372036854775807);  slice_79 = None
    slice_81: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_80, 2, 0, 9223372036854775807);  slice_80 = None
    select_23: "f32[1, 196, 196]" = torch.ops.aten.select.int(slice_81, 3, 1);  slice_81 = None
    copy_7: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_23, unsqueeze_15);  select_23 = unsqueeze_15 = None
    slice_82: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_20, 0, 0, 9223372036854775807)
    slice_83: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_82, 1, 0, 9223372036854775807)
    slice_84: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_83, 2, 0, 9223372036854775807)
    select_scatter_7: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(slice_84, copy_7, 3, 1);  slice_84 = copy_7 = None
    slice_scatter_21: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_83, select_scatter_7, 2, 0, 9223372036854775807);  slice_83 = select_scatter_7 = None
    slice_scatter_22: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_82, slice_scatter_21, 1, 0, 9223372036854775807);  slice_82 = slice_scatter_21 = None
    slice_scatter_23: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_scatter_20, slice_scatter_22, 0, 0, 9223372036854775807);  slice_scatter_20 = slice_scatter_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:128, code: rel_indices[:, :, :, 0] = indx.unsqueeze(0)
    unsqueeze_16: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(repeat_2, 0);  repeat_2 = None
    slice_91: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_23, 0, 0, 9223372036854775807)
    slice_92: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_91, 1, 0, 9223372036854775807);  slice_91 = None
    slice_93: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_92, 2, 0, 9223372036854775807);  slice_92 = None
    select_26: "f32[1, 196, 196]" = torch.ops.aten.select.int(slice_93, 3, 0);  slice_93 = None
    copy_8: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_26, unsqueeze_16);  select_26 = unsqueeze_16 = None
    slice_94: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_23, 0, 0, 9223372036854775807)
    slice_95: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_94, 1, 0, 9223372036854775807)
    slice_96: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_95, 2, 0, 9223372036854775807)
    select_scatter_8: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(slice_96, copy_8, 3, 0);  slice_96 = copy_8 = None
    slice_scatter_24: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_95, select_scatter_8, 2, 0, 9223372036854775807);  slice_95 = select_scatter_8 = None
    slice_scatter_25: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_94, slice_scatter_24, 1, 0, 9223372036854775807);  slice_94 = slice_scatter_24 = None
    slice_scatter_26: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_scatter_23, slice_scatter_25, 0, 0, 9223372036854775807);  slice_scatter_23 = slice_scatter_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:130, code: return rel_indices.to(device)
    device_put_2: "f32[1, 196, 196, 3]" = torch.ops.prims.device_put.default(slice_scatter_26, device(type='cuda', index=0));  slice_scatter_26 = None
    convert_element_type_2: "f32[1, 196, 196, 3]" = torch.ops.prims.convert_element_type.default(device_put_2, torch.float32);  device_put_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_23: "f32[768, 1536]" = torch.ops.aten.permute.default(arg84_1, [1, 0]);  arg84_1 = None
    view_57: "f32[1568, 768]" = torch.ops.aten.view.default(add_22, [1568, 768])
    mm_6: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_57, permute_23);  view_57 = permute_23 = None
    view_58: "f32[8, 196, 1536]" = torch.ops.aten.view.default(mm_6, [8, 196, 1536]);  mm_6 = None
    view_59: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.view.default(view_58, [8, 196, 2, 16, 48]);  view_58 = None
    permute_24: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.permute.default(view_59, [2, 0, 3, 1, 4]);  view_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    select_28: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_24, 0, 0)
    select_29: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_24, 0, 1);  permute_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:84, code: pos_score = self.rel_indices.expand(B, -1, -1, -1)
    expand_17: "f32[8, 196, 196, 3]" = torch.ops.aten.expand.default(convert_element_type_2, [8, -1, -1, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    permute_25: "f32[3, 16]" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
    clone_32: "f32[8, 196, 196, 3]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
    view_60: "f32[307328, 3]" = torch.ops.aten.view.default(clone_32, [307328, 3]);  clone_32 = None
    mm_7: "f32[307328, 16]" = torch.ops.aten.mm.default(view_60, permute_25);  view_60 = permute_25 = None
    view_61: "f32[8, 196, 196, 16]" = torch.ops.aten.view.default(mm_7, [8, 196, 196, 16]);  mm_7 = None
    add_24: "f32[8, 196, 196, 16]" = torch.ops.aten.add.Tensor(view_61, arg86_1);  view_61 = arg86_1 = None
    permute_26: "f32[8, 16, 196, 196]" = torch.ops.aten.permute.default(add_24, [0, 3, 1, 2]);  add_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    permute_27: "f32[8, 16, 48, 196]" = torch.ops.aten.permute.default(select_29, [0, 1, 3, 2]);  select_29 = None
    expand_18: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(select_28, [8, 16, 196, 48]);  select_28 = None
    clone_33: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
    view_62: "f32[128, 196, 48]" = torch.ops.aten.view.default(clone_33, [128, 196, 48]);  clone_33 = None
    expand_19: "f32[8, 16, 48, 196]" = torch.ops.aten.expand.default(permute_27, [8, 16, 48, 196]);  permute_27 = None
    clone_34: "f32[8, 16, 48, 196]" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
    view_63: "f32[128, 48, 196]" = torch.ops.aten.view.default(clone_34, [128, 48, 196]);  clone_34 = None
    bmm_4: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_62, view_63);  view_62 = view_63 = None
    view_64: "f32[8, 16, 196, 196]" = torch.ops.aten.view.default(bmm_4, [8, 16, 196, 196]);  bmm_4 = None
    mul_22: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(view_64, 0.14433756729740643);  view_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    amax_4: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(mul_22, [-1], True)
    sub_14: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_22, amax_4);  mul_22 = amax_4 = None
    exp_4: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_14);  sub_14 = None
    sum_7: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_6: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_4, sum_7);  exp_4 = sum_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    clone_35: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(permute_26, memory_format = torch.contiguous_format);  permute_26 = None
    amax_5: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(clone_35, [-1], True)
    sub_15: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(clone_35, amax_5);  clone_35 = amax_5 = None
    exp_5: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_15);  sub_15 = None
    sum_8: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_7: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_5, sum_8);  exp_5 = sum_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_65: "f32[1, 16, 1, 1]" = torch.ops.aten.view.default(arg14_1, [1, -1, 1, 1]);  arg14_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    sigmoid_4: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_65)
    sub_16: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1.0, sigmoid_4);  sigmoid_4 = None
    mul_23: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_16, div_6);  sub_16 = div_6 = None
    sigmoid_5: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_65);  view_65 = None
    mul_24: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sigmoid_5, div_7);  sigmoid_5 = div_7 = None
    add_25: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(mul_23, mul_24);  mul_23 = mul_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    sum_9: "f32[8, 16, 196]" = torch.ops.aten.sum.dim_IntList(add_25, [-1])
    unsqueeze_17: "f32[8, 16, 196, 1]" = torch.ops.aten.unsqueeze.default(sum_9, -1);  sum_9 = None
    div_8: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(add_25, unsqueeze_17);  add_25 = unsqueeze_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:93, code: attn = self.attn_drop(attn)
    clone_36: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(div_8);  div_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_28: "f32[768, 768]" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
    view_66: "f32[1568, 768]" = torch.ops.aten.view.default(add_22, [1568, 768]);  add_22 = None
    mm_8: "f32[1568, 768]" = torch.ops.aten.mm.default(view_66, permute_28);  view_66 = permute_28 = None
    view_67: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_8, [8, 196, 768]);  mm_8 = None
    view_68: "f32[8, 196, 16, 48]" = torch.ops.aten.view.default(view_67, [8, 196, 16, 48]);  view_67 = None
    permute_29: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_68, [0, 2, 1, 3]);  view_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_20: "f32[8, 16, 196, 196]" = torch.ops.aten.expand.default(clone_36, [8, 16, 196, 196]);  clone_36 = None
    view_69: "f32[128, 196, 196]" = torch.ops.aten.view.default(expand_20, [128, 196, 196]);  expand_20 = None
    expand_21: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(permute_29, [8, 16, 196, 48]);  permute_29 = None
    clone_37: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
    view_70: "f32[128, 196, 48]" = torch.ops.aten.view.default(clone_37, [128, 196, 48]);  clone_37 = None
    bmm_5: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(view_69, view_70);  view_69 = view_70 = None
    view_71: "f32[8, 16, 196, 48]" = torch.ops.aten.view.default(bmm_5, [8, 16, 196, 48]);  bmm_5 = None
    permute_30: "f32[8, 196, 16, 48]" = torch.ops.aten.permute.default(view_71, [0, 2, 1, 3]);  view_71 = None
    clone_38: "f32[8, 196, 16, 48]" = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None
    view_72: "f32[8, 196, 768]" = torch.ops.aten.view.default(clone_38, [8, 196, 768]);  clone_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_73: "f32[1568, 768]" = torch.ops.aten.view.default(view_72, [1568, 768]);  view_72 = None
    permute_31: "f32[768, 768]" = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
    addmm_6: "f32[1568, 768]" = torch.ops.aten.addmm.default(arg89_1, view_73, permute_31);  arg89_1 = view_73 = permute_31 = None
    view_74: "f32[8, 196, 768]" = torch.ops.aten.view.default(addmm_6, [8, 196, 768]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:77, code: x = self.proj_drop(x)
    clone_39: "f32[8, 196, 768]" = torch.ops.aten.clone.default(view_74);  view_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:235, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_26: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_20, clone_39);  add_20 = clone_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_40: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_26, memory_format = torch.contiguous_format)
    var_mean_5 = torch.ops.aten.var_mean.correction(clone_40, [2], correction = 0, keepdim = True)
    getitem_10: "f32[8, 196, 1]" = var_mean_5[0]
    getitem_11: "f32[8, 196, 1]" = var_mean_5[1];  var_mean_5 = None
    add_27: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-06);  getitem_10 = None
    rsqrt_5: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
    sub_17: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_40, getitem_11);  clone_40 = getitem_11 = None
    mul_25: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_5);  sub_17 = rsqrt_5 = None
    mul_26: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_25, arg15_1);  mul_25 = arg15_1 = None
    add_28: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_26, arg16_1);  mul_26 = arg16_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_75: "f32[1568, 768]" = torch.ops.aten.view.default(add_28, [1568, 768]);  add_28 = None
    permute_32: "f32[768, 3072]" = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
    addmm_7: "f32[1568, 3072]" = torch.ops.aten.addmm.default(arg91_1, view_75, permute_32);  arg91_1 = view_75 = permute_32 = None
    view_76: "f32[8, 196, 3072]" = torch.ops.aten.view.default(addmm_7, [8, 196, 3072]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_27: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_76, 0.5)
    mul_28: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_76, 0.7071067811865476);  view_76 = None
    erf_2: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_28);  mul_28 = None
    add_29: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_29: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_27, add_29);  mul_27 = add_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_41: "f32[8, 196, 3072]" = torch.ops.aten.clone.default(mul_29);  mul_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_77: "f32[1568, 3072]" = torch.ops.aten.view.default(clone_41, [1568, 3072]);  clone_41 = None
    permute_33: "f32[3072, 768]" = torch.ops.aten.permute.default(arg92_1, [1, 0]);  arg92_1 = None
    addmm_8: "f32[1568, 768]" = torch.ops.aten.addmm.default(arg93_1, view_77, permute_33);  arg93_1 = view_77 = permute_33 = None
    view_78: "f32[8, 196, 768]" = torch.ops.aten.view.default(addmm_8, [8, 196, 768]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_42: "f32[8, 196, 768]" = torch.ops.aten.clone.default(view_78);  view_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:236, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_30: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_26, clone_42);  add_26 = clone_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_43: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_30, memory_format = torch.contiguous_format)
    var_mean_6 = torch.ops.aten.var_mean.correction(clone_43, [2], correction = 0, keepdim = True)
    getitem_12: "f32[8, 196, 1]" = var_mean_6[0]
    getitem_13: "f32[8, 196, 1]" = var_mean_6[1];  var_mean_6 = None
    add_31: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-06);  getitem_12 = None
    rsqrt_6: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
    sub_18: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_43, getitem_13);  clone_43 = getitem_13 = None
    mul_30: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_6);  sub_18 = rsqrt_6 = None
    mul_31: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_30, arg17_1);  mul_30 = arg17_1 = None
    add_32: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_31, arg18_1);  mul_31 = arg18_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:121, code: rel_indices = torch.zeros(1, num_patches, num_patches, 3)
    full_3: "f32[1, 196, 196, 3]" = torch.ops.aten.full.default([1, 196, 196, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:122, code: ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
    iota_6: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_79: "i64[1, 14]" = torch.ops.aten.view.default(iota_6, [1, -1]);  iota_6 = None
    iota_7: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_80: "i64[14, 1]" = torch.ops.aten.view.default(iota_7, [-1, 1]);  iota_7 = None
    sub_19: "i64[14, 14]" = torch.ops.aten.sub.Tensor(view_79, view_80);  view_79 = view_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:123, code: indx = ind.repeat(img_size, img_size)
    repeat_3: "i64[196, 196]" = torch.ops.aten.repeat.default(sub_19, [14, 14])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:124, code: indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
    unsqueeze_18: "i64[14, 1, 14]" = torch.ops.aten.unsqueeze.default(sub_19, 1);  sub_19 = None
    expand_22: "i64[14, 14, 14]" = torch.ops.aten.expand.default(unsqueeze_18, [14, 14, 14]);  unsqueeze_18 = None
    clone_44: "i64[14, 14, 14]" = torch.ops.aten.clone.default(expand_22, memory_format = torch.contiguous_format);  expand_22 = None
    view_81: "i64[196, 14]" = torch.ops.aten.view.default(clone_44, [196, 14]);  clone_44 = None
    unsqueeze_19: "i64[196, 14, 1]" = torch.ops.aten.unsqueeze.default(view_81, 2);  view_81 = None
    expand_23: "i64[196, 14, 14]" = torch.ops.aten.expand.default(unsqueeze_19, [196, 14, 14]);  unsqueeze_19 = None
    clone_45: "i64[196, 14, 14]" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
    view_82: "i64[196, 196]" = torch.ops.aten.view.default(clone_45, [196, 196]);  clone_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:125, code: indd = indx ** 2 + indy ** 2
    pow_7: "i64[196, 196]" = torch.ops.aten.pow.Tensor_Scalar(repeat_3, 2)
    pow_8: "i64[196, 196]" = torch.ops.aten.pow.Tensor_Scalar(view_82, 2)
    add_33: "i64[196, 196]" = torch.ops.aten.add.Tensor(pow_7, pow_8);  pow_7 = pow_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:126, code: rel_indices[:, :, :, 2] = indd.unsqueeze(0)
    unsqueeze_20: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(add_33, 0);  add_33 = None
    slice_100: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(full_3, 0, 0, 9223372036854775807)
    slice_101: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_100, 1, 0, 9223372036854775807);  slice_100 = None
    slice_102: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_101, 2, 0, 9223372036854775807);  slice_101 = None
    select_30: "f32[1, 196, 196]" = torch.ops.aten.select.int(slice_102, 3, 2);  slice_102 = None
    copy_9: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_30, unsqueeze_20);  select_30 = unsqueeze_20 = None
    slice_103: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(full_3, 0, 0, 9223372036854775807)
    slice_104: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_103, 1, 0, 9223372036854775807)
    slice_105: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_104, 2, 0, 9223372036854775807)
    select_scatter_9: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(slice_105, copy_9, 3, 2);  slice_105 = copy_9 = None
    slice_scatter_27: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_104, select_scatter_9, 2, 0, 9223372036854775807);  slice_104 = select_scatter_9 = None
    slice_scatter_28: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_103, slice_scatter_27, 1, 0, 9223372036854775807);  slice_103 = slice_scatter_27 = None
    slice_scatter_29: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(full_3, slice_scatter_28, 0, 0, 9223372036854775807);  full_3 = slice_scatter_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:127, code: rel_indices[:, :, :, 1] = indy.unsqueeze(0)
    unsqueeze_21: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(view_82, 0);  view_82 = None
    slice_112: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_29, 0, 0, 9223372036854775807)
    slice_113: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_112, 1, 0, 9223372036854775807);  slice_112 = None
    slice_114: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_113, 2, 0, 9223372036854775807);  slice_113 = None
    select_33: "f32[1, 196, 196]" = torch.ops.aten.select.int(slice_114, 3, 1);  slice_114 = None
    copy_10: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_33, unsqueeze_21);  select_33 = unsqueeze_21 = None
    slice_115: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_29, 0, 0, 9223372036854775807)
    slice_116: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_115, 1, 0, 9223372036854775807)
    slice_117: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_116, 2, 0, 9223372036854775807)
    select_scatter_10: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(slice_117, copy_10, 3, 1);  slice_117 = copy_10 = None
    slice_scatter_30: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_116, select_scatter_10, 2, 0, 9223372036854775807);  slice_116 = select_scatter_10 = None
    slice_scatter_31: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_115, slice_scatter_30, 1, 0, 9223372036854775807);  slice_115 = slice_scatter_30 = None
    slice_scatter_32: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_scatter_29, slice_scatter_31, 0, 0, 9223372036854775807);  slice_scatter_29 = slice_scatter_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:128, code: rel_indices[:, :, :, 0] = indx.unsqueeze(0)
    unsqueeze_22: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(repeat_3, 0);  repeat_3 = None
    slice_124: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_32, 0, 0, 9223372036854775807)
    slice_125: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_124, 1, 0, 9223372036854775807);  slice_124 = None
    slice_126: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_125, 2, 0, 9223372036854775807);  slice_125 = None
    select_36: "f32[1, 196, 196]" = torch.ops.aten.select.int(slice_126, 3, 0);  slice_126 = None
    copy_11: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_36, unsqueeze_22);  select_36 = unsqueeze_22 = None
    slice_127: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_32, 0, 0, 9223372036854775807)
    slice_128: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_127, 1, 0, 9223372036854775807)
    slice_129: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_128, 2, 0, 9223372036854775807)
    select_scatter_11: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(slice_129, copy_11, 3, 0);  slice_129 = copy_11 = None
    slice_scatter_33: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_128, select_scatter_11, 2, 0, 9223372036854775807);  slice_128 = select_scatter_11 = None
    slice_scatter_34: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_127, slice_scatter_33, 1, 0, 9223372036854775807);  slice_127 = slice_scatter_33 = None
    slice_scatter_35: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_scatter_32, slice_scatter_34, 0, 0, 9223372036854775807);  slice_scatter_32 = slice_scatter_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:130, code: return rel_indices.to(device)
    device_put_3: "f32[1, 196, 196, 3]" = torch.ops.prims.device_put.default(slice_scatter_35, device(type='cuda', index=0));  slice_scatter_35 = None
    convert_element_type_3: "f32[1, 196, 196, 3]" = torch.ops.prims.convert_element_type.default(device_put_3, torch.float32);  device_put_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_34: "f32[768, 1536]" = torch.ops.aten.permute.default(arg94_1, [1, 0]);  arg94_1 = None
    view_83: "f32[1568, 768]" = torch.ops.aten.view.default(add_32, [1568, 768])
    mm_9: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_83, permute_34);  view_83 = permute_34 = None
    view_84: "f32[8, 196, 1536]" = torch.ops.aten.view.default(mm_9, [8, 196, 1536]);  mm_9 = None
    view_85: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.view.default(view_84, [8, 196, 2, 16, 48]);  view_84 = None
    permute_35: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.permute.default(view_85, [2, 0, 3, 1, 4]);  view_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    select_38: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_35, 0, 0)
    select_39: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_35, 0, 1);  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:84, code: pos_score = self.rel_indices.expand(B, -1, -1, -1)
    expand_24: "f32[8, 196, 196, 3]" = torch.ops.aten.expand.default(convert_element_type_3, [8, -1, -1, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    permute_36: "f32[3, 16]" = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
    clone_46: "f32[8, 196, 196, 3]" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
    view_86: "f32[307328, 3]" = torch.ops.aten.view.default(clone_46, [307328, 3]);  clone_46 = None
    mm_10: "f32[307328, 16]" = torch.ops.aten.mm.default(view_86, permute_36);  view_86 = permute_36 = None
    view_87: "f32[8, 196, 196, 16]" = torch.ops.aten.view.default(mm_10, [8, 196, 196, 16]);  mm_10 = None
    add_34: "f32[8, 196, 196, 16]" = torch.ops.aten.add.Tensor(view_87, arg96_1);  view_87 = arg96_1 = None
    permute_37: "f32[8, 16, 196, 196]" = torch.ops.aten.permute.default(add_34, [0, 3, 1, 2]);  add_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    permute_38: "f32[8, 16, 48, 196]" = torch.ops.aten.permute.default(select_39, [0, 1, 3, 2]);  select_39 = None
    expand_25: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(select_38, [8, 16, 196, 48]);  select_38 = None
    clone_47: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
    view_88: "f32[128, 196, 48]" = torch.ops.aten.view.default(clone_47, [128, 196, 48]);  clone_47 = None
    expand_26: "f32[8, 16, 48, 196]" = torch.ops.aten.expand.default(permute_38, [8, 16, 48, 196]);  permute_38 = None
    clone_48: "f32[8, 16, 48, 196]" = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
    view_89: "f32[128, 48, 196]" = torch.ops.aten.view.default(clone_48, [128, 48, 196]);  clone_48 = None
    bmm_6: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_88, view_89);  view_88 = view_89 = None
    view_90: "f32[8, 16, 196, 196]" = torch.ops.aten.view.default(bmm_6, [8, 16, 196, 196]);  bmm_6 = None
    mul_32: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(view_90, 0.14433756729740643);  view_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    amax_6: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(mul_32, [-1], True)
    sub_20: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_32, amax_6);  mul_32 = amax_6 = None
    exp_6: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_20);  sub_20 = None
    sum_10: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_9: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_6, sum_10);  exp_6 = sum_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    clone_49: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
    amax_7: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(clone_49, [-1], True)
    sub_21: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(clone_49, amax_7);  clone_49 = amax_7 = None
    exp_7: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_21);  sub_21 = None
    sum_11: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_10: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_7, sum_11);  exp_7 = sum_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_91: "f32[1, 16, 1, 1]" = torch.ops.aten.view.default(arg19_1, [1, -1, 1, 1]);  arg19_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    sigmoid_6: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_91)
    sub_22: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1.0, sigmoid_6);  sigmoid_6 = None
    mul_33: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_22, div_9);  sub_22 = div_9 = None
    sigmoid_7: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_91);  view_91 = None
    mul_34: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sigmoid_7, div_10);  sigmoid_7 = div_10 = None
    add_35: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(mul_33, mul_34);  mul_33 = mul_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    sum_12: "f32[8, 16, 196]" = torch.ops.aten.sum.dim_IntList(add_35, [-1])
    unsqueeze_23: "f32[8, 16, 196, 1]" = torch.ops.aten.unsqueeze.default(sum_12, -1);  sum_12 = None
    div_11: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(add_35, unsqueeze_23);  add_35 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:93, code: attn = self.attn_drop(attn)
    clone_50: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(div_11);  div_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_39: "f32[768, 768]" = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
    view_92: "f32[1568, 768]" = torch.ops.aten.view.default(add_32, [1568, 768]);  add_32 = None
    mm_11: "f32[1568, 768]" = torch.ops.aten.mm.default(view_92, permute_39);  view_92 = permute_39 = None
    view_93: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_11, [8, 196, 768]);  mm_11 = None
    view_94: "f32[8, 196, 16, 48]" = torch.ops.aten.view.default(view_93, [8, 196, 16, 48]);  view_93 = None
    permute_40: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_27: "f32[8, 16, 196, 196]" = torch.ops.aten.expand.default(clone_50, [8, 16, 196, 196]);  clone_50 = None
    view_95: "f32[128, 196, 196]" = torch.ops.aten.view.default(expand_27, [128, 196, 196]);  expand_27 = None
    expand_28: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(permute_40, [8, 16, 196, 48]);  permute_40 = None
    clone_51: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
    view_96: "f32[128, 196, 48]" = torch.ops.aten.view.default(clone_51, [128, 196, 48]);  clone_51 = None
    bmm_7: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(view_95, view_96);  view_95 = view_96 = None
    view_97: "f32[8, 16, 196, 48]" = torch.ops.aten.view.default(bmm_7, [8, 16, 196, 48]);  bmm_7 = None
    permute_41: "f32[8, 196, 16, 48]" = torch.ops.aten.permute.default(view_97, [0, 2, 1, 3]);  view_97 = None
    clone_52: "f32[8, 196, 16, 48]" = torch.ops.aten.clone.default(permute_41, memory_format = torch.contiguous_format);  permute_41 = None
    view_98: "f32[8, 196, 768]" = torch.ops.aten.view.default(clone_52, [8, 196, 768]);  clone_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_99: "f32[1568, 768]" = torch.ops.aten.view.default(view_98, [1568, 768]);  view_98 = None
    permute_42: "f32[768, 768]" = torch.ops.aten.permute.default(arg98_1, [1, 0]);  arg98_1 = None
    addmm_9: "f32[1568, 768]" = torch.ops.aten.addmm.default(arg99_1, view_99, permute_42);  arg99_1 = view_99 = permute_42 = None
    view_100: "f32[8, 196, 768]" = torch.ops.aten.view.default(addmm_9, [8, 196, 768]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:77, code: x = self.proj_drop(x)
    clone_53: "f32[8, 196, 768]" = torch.ops.aten.clone.default(view_100);  view_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:235, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_36: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_30, clone_53);  add_30 = clone_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_54: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_36, memory_format = torch.contiguous_format)
    var_mean_7 = torch.ops.aten.var_mean.correction(clone_54, [2], correction = 0, keepdim = True)
    getitem_14: "f32[8, 196, 1]" = var_mean_7[0]
    getitem_15: "f32[8, 196, 1]" = var_mean_7[1];  var_mean_7 = None
    add_37: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-06);  getitem_14 = None
    rsqrt_7: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
    sub_23: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_54, getitem_15);  clone_54 = getitem_15 = None
    mul_35: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_7);  sub_23 = rsqrt_7 = None
    mul_36: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_35, arg20_1);  mul_35 = arg20_1 = None
    add_38: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_36, arg21_1);  mul_36 = arg21_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_101: "f32[1568, 768]" = torch.ops.aten.view.default(add_38, [1568, 768]);  add_38 = None
    permute_43: "f32[768, 3072]" = torch.ops.aten.permute.default(arg100_1, [1, 0]);  arg100_1 = None
    addmm_10: "f32[1568, 3072]" = torch.ops.aten.addmm.default(arg101_1, view_101, permute_43);  arg101_1 = view_101 = permute_43 = None
    view_102: "f32[8, 196, 3072]" = torch.ops.aten.view.default(addmm_10, [8, 196, 3072]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_37: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_102, 0.5)
    mul_38: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_102, 0.7071067811865476);  view_102 = None
    erf_3: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_38);  mul_38 = None
    add_39: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_39: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_37, add_39);  mul_37 = add_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_55: "f32[8, 196, 3072]" = torch.ops.aten.clone.default(mul_39);  mul_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_103: "f32[1568, 3072]" = torch.ops.aten.view.default(clone_55, [1568, 3072]);  clone_55 = None
    permute_44: "f32[3072, 768]" = torch.ops.aten.permute.default(arg102_1, [1, 0]);  arg102_1 = None
    addmm_11: "f32[1568, 768]" = torch.ops.aten.addmm.default(arg103_1, view_103, permute_44);  arg103_1 = view_103 = permute_44 = None
    view_104: "f32[8, 196, 768]" = torch.ops.aten.view.default(addmm_11, [8, 196, 768]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_56: "f32[8, 196, 768]" = torch.ops.aten.clone.default(view_104);  view_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:236, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_40: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_36, clone_56);  add_36 = clone_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_57: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_40, memory_format = torch.contiguous_format)
    var_mean_8 = torch.ops.aten.var_mean.correction(clone_57, [2], correction = 0, keepdim = True)
    getitem_16: "f32[8, 196, 1]" = var_mean_8[0]
    getitem_17: "f32[8, 196, 1]" = var_mean_8[1];  var_mean_8 = None
    add_41: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-06);  getitem_16 = None
    rsqrt_8: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_41);  add_41 = None
    sub_24: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_57, getitem_17);  clone_57 = getitem_17 = None
    mul_40: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_8);  sub_24 = rsqrt_8 = None
    mul_41: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_40, arg22_1);  mul_40 = arg22_1 = None
    add_42: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_41, arg23_1);  mul_41 = arg23_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:121, code: rel_indices = torch.zeros(1, num_patches, num_patches, 3)
    full_4: "f32[1, 196, 196, 3]" = torch.ops.aten.full.default([1, 196, 196, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:122, code: ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
    iota_8: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_105: "i64[1, 14]" = torch.ops.aten.view.default(iota_8, [1, -1]);  iota_8 = None
    iota_9: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_106: "i64[14, 1]" = torch.ops.aten.view.default(iota_9, [-1, 1]);  iota_9 = None
    sub_25: "i64[14, 14]" = torch.ops.aten.sub.Tensor(view_105, view_106);  view_105 = view_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:123, code: indx = ind.repeat(img_size, img_size)
    repeat_4: "i64[196, 196]" = torch.ops.aten.repeat.default(sub_25, [14, 14])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:124, code: indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
    unsqueeze_24: "i64[14, 1, 14]" = torch.ops.aten.unsqueeze.default(sub_25, 1);  sub_25 = None
    expand_29: "i64[14, 14, 14]" = torch.ops.aten.expand.default(unsqueeze_24, [14, 14, 14]);  unsqueeze_24 = None
    clone_58: "i64[14, 14, 14]" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
    view_107: "i64[196, 14]" = torch.ops.aten.view.default(clone_58, [196, 14]);  clone_58 = None
    unsqueeze_25: "i64[196, 14, 1]" = torch.ops.aten.unsqueeze.default(view_107, 2);  view_107 = None
    expand_30: "i64[196, 14, 14]" = torch.ops.aten.expand.default(unsqueeze_25, [196, 14, 14]);  unsqueeze_25 = None
    clone_59: "i64[196, 14, 14]" = torch.ops.aten.clone.default(expand_30, memory_format = torch.contiguous_format);  expand_30 = None
    view_108: "i64[196, 196]" = torch.ops.aten.view.default(clone_59, [196, 196]);  clone_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:125, code: indd = indx ** 2 + indy ** 2
    pow_9: "i64[196, 196]" = torch.ops.aten.pow.Tensor_Scalar(repeat_4, 2)
    pow_10: "i64[196, 196]" = torch.ops.aten.pow.Tensor_Scalar(view_108, 2)
    add_43: "i64[196, 196]" = torch.ops.aten.add.Tensor(pow_9, pow_10);  pow_9 = pow_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:126, code: rel_indices[:, :, :, 2] = indd.unsqueeze(0)
    unsqueeze_26: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(add_43, 0);  add_43 = None
    slice_133: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(full_4, 0, 0, 9223372036854775807)
    slice_134: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_133, 1, 0, 9223372036854775807);  slice_133 = None
    slice_135: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_134, 2, 0, 9223372036854775807);  slice_134 = None
    select_40: "f32[1, 196, 196]" = torch.ops.aten.select.int(slice_135, 3, 2);  slice_135 = None
    copy_12: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_40, unsqueeze_26);  select_40 = unsqueeze_26 = None
    slice_136: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(full_4, 0, 0, 9223372036854775807)
    slice_137: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_136, 1, 0, 9223372036854775807)
    slice_138: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_137, 2, 0, 9223372036854775807)
    select_scatter_12: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(slice_138, copy_12, 3, 2);  slice_138 = copy_12 = None
    slice_scatter_36: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_137, select_scatter_12, 2, 0, 9223372036854775807);  slice_137 = select_scatter_12 = None
    slice_scatter_37: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_136, slice_scatter_36, 1, 0, 9223372036854775807);  slice_136 = slice_scatter_36 = None
    slice_scatter_38: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(full_4, slice_scatter_37, 0, 0, 9223372036854775807);  full_4 = slice_scatter_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:127, code: rel_indices[:, :, :, 1] = indy.unsqueeze(0)
    unsqueeze_27: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(view_108, 0);  view_108 = None
    slice_145: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_38, 0, 0, 9223372036854775807)
    slice_146: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_145, 1, 0, 9223372036854775807);  slice_145 = None
    slice_147: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_146, 2, 0, 9223372036854775807);  slice_146 = None
    select_43: "f32[1, 196, 196]" = torch.ops.aten.select.int(slice_147, 3, 1);  slice_147 = None
    copy_13: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_43, unsqueeze_27);  select_43 = unsqueeze_27 = None
    slice_148: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_38, 0, 0, 9223372036854775807)
    slice_149: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_148, 1, 0, 9223372036854775807)
    slice_150: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_149, 2, 0, 9223372036854775807)
    select_scatter_13: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(slice_150, copy_13, 3, 1);  slice_150 = copy_13 = None
    slice_scatter_39: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_149, select_scatter_13, 2, 0, 9223372036854775807);  slice_149 = select_scatter_13 = None
    slice_scatter_40: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_148, slice_scatter_39, 1, 0, 9223372036854775807);  slice_148 = slice_scatter_39 = None
    slice_scatter_41: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_scatter_38, slice_scatter_40, 0, 0, 9223372036854775807);  slice_scatter_38 = slice_scatter_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:128, code: rel_indices[:, :, :, 0] = indx.unsqueeze(0)
    unsqueeze_28: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(repeat_4, 0);  repeat_4 = None
    slice_157: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_41, 0, 0, 9223372036854775807)
    slice_158: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_157, 1, 0, 9223372036854775807);  slice_157 = None
    slice_159: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_158, 2, 0, 9223372036854775807);  slice_158 = None
    select_46: "f32[1, 196, 196]" = torch.ops.aten.select.int(slice_159, 3, 0);  slice_159 = None
    copy_14: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_46, unsqueeze_28);  select_46 = unsqueeze_28 = None
    slice_160: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_41, 0, 0, 9223372036854775807)
    slice_161: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_160, 1, 0, 9223372036854775807)
    slice_162: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_161, 2, 0, 9223372036854775807)
    select_scatter_14: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(slice_162, copy_14, 3, 0);  slice_162 = copy_14 = None
    slice_scatter_42: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_161, select_scatter_14, 2, 0, 9223372036854775807);  slice_161 = select_scatter_14 = None
    slice_scatter_43: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_160, slice_scatter_42, 1, 0, 9223372036854775807);  slice_160 = slice_scatter_42 = None
    slice_scatter_44: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_scatter_41, slice_scatter_43, 0, 0, 9223372036854775807);  slice_scatter_41 = slice_scatter_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:130, code: return rel_indices.to(device)
    device_put_4: "f32[1, 196, 196, 3]" = torch.ops.prims.device_put.default(slice_scatter_44, device(type='cuda', index=0));  slice_scatter_44 = None
    convert_element_type_4: "f32[1, 196, 196, 3]" = torch.ops.prims.convert_element_type.default(device_put_4, torch.float32);  device_put_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_45: "f32[768, 1536]" = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
    view_109: "f32[1568, 768]" = torch.ops.aten.view.default(add_42, [1568, 768])
    mm_12: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_109, permute_45);  view_109 = permute_45 = None
    view_110: "f32[8, 196, 1536]" = torch.ops.aten.view.default(mm_12, [8, 196, 1536]);  mm_12 = None
    view_111: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.view.default(view_110, [8, 196, 2, 16, 48]);  view_110 = None
    permute_46: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.permute.default(view_111, [2, 0, 3, 1, 4]);  view_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    select_48: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_46, 0, 0)
    select_49: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_46, 0, 1);  permute_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:84, code: pos_score = self.rel_indices.expand(B, -1, -1, -1)
    expand_31: "f32[8, 196, 196, 3]" = torch.ops.aten.expand.default(convert_element_type_4, [8, -1, -1, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    permute_47: "f32[3, 16]" = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
    clone_60: "f32[8, 196, 196, 3]" = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
    view_112: "f32[307328, 3]" = torch.ops.aten.view.default(clone_60, [307328, 3]);  clone_60 = None
    mm_13: "f32[307328, 16]" = torch.ops.aten.mm.default(view_112, permute_47);  view_112 = permute_47 = None
    view_113: "f32[8, 196, 196, 16]" = torch.ops.aten.view.default(mm_13, [8, 196, 196, 16]);  mm_13 = None
    add_44: "f32[8, 196, 196, 16]" = torch.ops.aten.add.Tensor(view_113, arg106_1);  view_113 = arg106_1 = None
    permute_48: "f32[8, 16, 196, 196]" = torch.ops.aten.permute.default(add_44, [0, 3, 1, 2]);  add_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    permute_49: "f32[8, 16, 48, 196]" = torch.ops.aten.permute.default(select_49, [0, 1, 3, 2]);  select_49 = None
    expand_32: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(select_48, [8, 16, 196, 48]);  select_48 = None
    clone_61: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
    view_114: "f32[128, 196, 48]" = torch.ops.aten.view.default(clone_61, [128, 196, 48]);  clone_61 = None
    expand_33: "f32[8, 16, 48, 196]" = torch.ops.aten.expand.default(permute_49, [8, 16, 48, 196]);  permute_49 = None
    clone_62: "f32[8, 16, 48, 196]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
    view_115: "f32[128, 48, 196]" = torch.ops.aten.view.default(clone_62, [128, 48, 196]);  clone_62 = None
    bmm_8: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_114, view_115);  view_114 = view_115 = None
    view_116: "f32[8, 16, 196, 196]" = torch.ops.aten.view.default(bmm_8, [8, 16, 196, 196]);  bmm_8 = None
    mul_42: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(view_116, 0.14433756729740643);  view_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    amax_8: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(mul_42, [-1], True)
    sub_26: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_42, amax_8);  mul_42 = amax_8 = None
    exp_8: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_26);  sub_26 = None
    sum_13: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_12: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_8, sum_13);  exp_8 = sum_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    clone_63: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(permute_48, memory_format = torch.contiguous_format);  permute_48 = None
    amax_9: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(clone_63, [-1], True)
    sub_27: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(clone_63, amax_9);  clone_63 = amax_9 = None
    exp_9: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_27);  sub_27 = None
    sum_14: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_13: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_9, sum_14);  exp_9 = sum_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_117: "f32[1, 16, 1, 1]" = torch.ops.aten.view.default(arg24_1, [1, -1, 1, 1]);  arg24_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    sigmoid_8: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_117)
    sub_28: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1.0, sigmoid_8);  sigmoid_8 = None
    mul_43: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_28, div_12);  sub_28 = div_12 = None
    sigmoid_9: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_117);  view_117 = None
    mul_44: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sigmoid_9, div_13);  sigmoid_9 = div_13 = None
    add_45: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(mul_43, mul_44);  mul_43 = mul_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    sum_15: "f32[8, 16, 196]" = torch.ops.aten.sum.dim_IntList(add_45, [-1])
    unsqueeze_29: "f32[8, 16, 196, 1]" = torch.ops.aten.unsqueeze.default(sum_15, -1);  sum_15 = None
    div_14: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(add_45, unsqueeze_29);  add_45 = unsqueeze_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:93, code: attn = self.attn_drop(attn)
    clone_64: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(div_14);  div_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_50: "f32[768, 768]" = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
    view_118: "f32[1568, 768]" = torch.ops.aten.view.default(add_42, [1568, 768]);  add_42 = None
    mm_14: "f32[1568, 768]" = torch.ops.aten.mm.default(view_118, permute_50);  view_118 = permute_50 = None
    view_119: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_14, [8, 196, 768]);  mm_14 = None
    view_120: "f32[8, 196, 16, 48]" = torch.ops.aten.view.default(view_119, [8, 196, 16, 48]);  view_119 = None
    permute_51: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_34: "f32[8, 16, 196, 196]" = torch.ops.aten.expand.default(clone_64, [8, 16, 196, 196]);  clone_64 = None
    view_121: "f32[128, 196, 196]" = torch.ops.aten.view.default(expand_34, [128, 196, 196]);  expand_34 = None
    expand_35: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(permute_51, [8, 16, 196, 48]);  permute_51 = None
    clone_65: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
    view_122: "f32[128, 196, 48]" = torch.ops.aten.view.default(clone_65, [128, 196, 48]);  clone_65 = None
    bmm_9: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(view_121, view_122);  view_121 = view_122 = None
    view_123: "f32[8, 16, 196, 48]" = torch.ops.aten.view.default(bmm_9, [8, 16, 196, 48]);  bmm_9 = None
    permute_52: "f32[8, 196, 16, 48]" = torch.ops.aten.permute.default(view_123, [0, 2, 1, 3]);  view_123 = None
    clone_66: "f32[8, 196, 16, 48]" = torch.ops.aten.clone.default(permute_52, memory_format = torch.contiguous_format);  permute_52 = None
    view_124: "f32[8, 196, 768]" = torch.ops.aten.view.default(clone_66, [8, 196, 768]);  clone_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_125: "f32[1568, 768]" = torch.ops.aten.view.default(view_124, [1568, 768]);  view_124 = None
    permute_53: "f32[768, 768]" = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
    addmm_12: "f32[1568, 768]" = torch.ops.aten.addmm.default(arg109_1, view_125, permute_53);  arg109_1 = view_125 = permute_53 = None
    view_126: "f32[8, 196, 768]" = torch.ops.aten.view.default(addmm_12, [8, 196, 768]);  addmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:77, code: x = self.proj_drop(x)
    clone_67: "f32[8, 196, 768]" = torch.ops.aten.clone.default(view_126);  view_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:235, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_46: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_40, clone_67);  add_40 = clone_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_68: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_46, memory_format = torch.contiguous_format)
    var_mean_9 = torch.ops.aten.var_mean.correction(clone_68, [2], correction = 0, keepdim = True)
    getitem_18: "f32[8, 196, 1]" = var_mean_9[0]
    getitem_19: "f32[8, 196, 1]" = var_mean_9[1];  var_mean_9 = None
    add_47: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-06);  getitem_18 = None
    rsqrt_9: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
    sub_29: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_68, getitem_19);  clone_68 = getitem_19 = None
    mul_45: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_9);  sub_29 = rsqrt_9 = None
    mul_46: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_45, arg25_1);  mul_45 = arg25_1 = None
    add_48: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_46, arg26_1);  mul_46 = arg26_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_127: "f32[1568, 768]" = torch.ops.aten.view.default(add_48, [1568, 768]);  add_48 = None
    permute_54: "f32[768, 3072]" = torch.ops.aten.permute.default(arg110_1, [1, 0]);  arg110_1 = None
    addmm_13: "f32[1568, 3072]" = torch.ops.aten.addmm.default(arg111_1, view_127, permute_54);  arg111_1 = view_127 = permute_54 = None
    view_128: "f32[8, 196, 3072]" = torch.ops.aten.view.default(addmm_13, [8, 196, 3072]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_47: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_128, 0.5)
    mul_48: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_128, 0.7071067811865476);  view_128 = None
    erf_4: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_48);  mul_48 = None
    add_49: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_49: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_47, add_49);  mul_47 = add_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_69: "f32[8, 196, 3072]" = torch.ops.aten.clone.default(mul_49);  mul_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_129: "f32[1568, 3072]" = torch.ops.aten.view.default(clone_69, [1568, 3072]);  clone_69 = None
    permute_55: "f32[3072, 768]" = torch.ops.aten.permute.default(arg112_1, [1, 0]);  arg112_1 = None
    addmm_14: "f32[1568, 768]" = torch.ops.aten.addmm.default(arg113_1, view_129, permute_55);  arg113_1 = view_129 = permute_55 = None
    view_130: "f32[8, 196, 768]" = torch.ops.aten.view.default(addmm_14, [8, 196, 768]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_70: "f32[8, 196, 768]" = torch.ops.aten.clone.default(view_130);  view_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:236, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_50: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_46, clone_70);  add_46 = clone_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_71: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_50, memory_format = torch.contiguous_format)
    var_mean_10 = torch.ops.aten.var_mean.correction(clone_71, [2], correction = 0, keepdim = True)
    getitem_20: "f32[8, 196, 1]" = var_mean_10[0]
    getitem_21: "f32[8, 196, 1]" = var_mean_10[1];  var_mean_10 = None
    add_51: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-06);  getitem_20 = None
    rsqrt_10: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
    sub_30: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_71, getitem_21);  clone_71 = getitem_21 = None
    mul_50: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_10);  sub_30 = rsqrt_10 = None
    mul_51: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_50, arg27_1);  mul_50 = arg27_1 = None
    add_52: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_51, arg28_1);  mul_51 = arg28_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:121, code: rel_indices = torch.zeros(1, num_patches, num_patches, 3)
    full_5: "f32[1, 196, 196, 3]" = torch.ops.aten.full.default([1, 196, 196, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:122, code: ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
    iota_10: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_131: "i64[1, 14]" = torch.ops.aten.view.default(iota_10, [1, -1]);  iota_10 = None
    iota_11: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_132: "i64[14, 1]" = torch.ops.aten.view.default(iota_11, [-1, 1]);  iota_11 = None
    sub_31: "i64[14, 14]" = torch.ops.aten.sub.Tensor(view_131, view_132);  view_131 = view_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:123, code: indx = ind.repeat(img_size, img_size)
    repeat_5: "i64[196, 196]" = torch.ops.aten.repeat.default(sub_31, [14, 14])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:124, code: indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
    unsqueeze_30: "i64[14, 1, 14]" = torch.ops.aten.unsqueeze.default(sub_31, 1);  sub_31 = None
    expand_36: "i64[14, 14, 14]" = torch.ops.aten.expand.default(unsqueeze_30, [14, 14, 14]);  unsqueeze_30 = None
    clone_72: "i64[14, 14, 14]" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
    view_133: "i64[196, 14]" = torch.ops.aten.view.default(clone_72, [196, 14]);  clone_72 = None
    unsqueeze_31: "i64[196, 14, 1]" = torch.ops.aten.unsqueeze.default(view_133, 2);  view_133 = None
    expand_37: "i64[196, 14, 14]" = torch.ops.aten.expand.default(unsqueeze_31, [196, 14, 14]);  unsqueeze_31 = None
    clone_73: "i64[196, 14, 14]" = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
    view_134: "i64[196, 196]" = torch.ops.aten.view.default(clone_73, [196, 196]);  clone_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:125, code: indd = indx ** 2 + indy ** 2
    pow_11: "i64[196, 196]" = torch.ops.aten.pow.Tensor_Scalar(repeat_5, 2)
    pow_12: "i64[196, 196]" = torch.ops.aten.pow.Tensor_Scalar(view_134, 2)
    add_53: "i64[196, 196]" = torch.ops.aten.add.Tensor(pow_11, pow_12);  pow_11 = pow_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:126, code: rel_indices[:, :, :, 2] = indd.unsqueeze(0)
    unsqueeze_32: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(add_53, 0);  add_53 = None
    slice_166: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(full_5, 0, 0, 9223372036854775807)
    slice_167: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_166, 1, 0, 9223372036854775807);  slice_166 = None
    slice_168: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_167, 2, 0, 9223372036854775807);  slice_167 = None
    select_50: "f32[1, 196, 196]" = torch.ops.aten.select.int(slice_168, 3, 2);  slice_168 = None
    copy_15: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_50, unsqueeze_32);  select_50 = unsqueeze_32 = None
    slice_169: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(full_5, 0, 0, 9223372036854775807)
    slice_170: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_169, 1, 0, 9223372036854775807)
    slice_171: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_170, 2, 0, 9223372036854775807)
    select_scatter_15: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(slice_171, copy_15, 3, 2);  slice_171 = copy_15 = None
    slice_scatter_45: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_170, select_scatter_15, 2, 0, 9223372036854775807);  slice_170 = select_scatter_15 = None
    slice_scatter_46: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_169, slice_scatter_45, 1, 0, 9223372036854775807);  slice_169 = slice_scatter_45 = None
    slice_scatter_47: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(full_5, slice_scatter_46, 0, 0, 9223372036854775807);  full_5 = slice_scatter_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:127, code: rel_indices[:, :, :, 1] = indy.unsqueeze(0)
    unsqueeze_33: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(view_134, 0);  view_134 = None
    slice_178: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_47, 0, 0, 9223372036854775807)
    slice_179: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_178, 1, 0, 9223372036854775807);  slice_178 = None
    slice_180: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_179, 2, 0, 9223372036854775807);  slice_179 = None
    select_53: "f32[1, 196, 196]" = torch.ops.aten.select.int(slice_180, 3, 1);  slice_180 = None
    copy_16: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_53, unsqueeze_33);  select_53 = unsqueeze_33 = None
    slice_181: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_47, 0, 0, 9223372036854775807)
    slice_182: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_181, 1, 0, 9223372036854775807)
    slice_183: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_182, 2, 0, 9223372036854775807)
    select_scatter_16: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(slice_183, copy_16, 3, 1);  slice_183 = copy_16 = None
    slice_scatter_48: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_182, select_scatter_16, 2, 0, 9223372036854775807);  slice_182 = select_scatter_16 = None
    slice_scatter_49: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_181, slice_scatter_48, 1, 0, 9223372036854775807);  slice_181 = slice_scatter_48 = None
    slice_scatter_50: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_scatter_47, slice_scatter_49, 0, 0, 9223372036854775807);  slice_scatter_47 = slice_scatter_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:128, code: rel_indices[:, :, :, 0] = indx.unsqueeze(0)
    unsqueeze_34: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(repeat_5, 0);  repeat_5 = None
    slice_190: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_50, 0, 0, 9223372036854775807)
    slice_191: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_190, 1, 0, 9223372036854775807);  slice_190 = None
    slice_192: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_191, 2, 0, 9223372036854775807);  slice_191 = None
    select_56: "f32[1, 196, 196]" = torch.ops.aten.select.int(slice_192, 3, 0);  slice_192 = None
    copy_17: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_56, unsqueeze_34);  select_56 = unsqueeze_34 = None
    slice_193: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_50, 0, 0, 9223372036854775807)
    slice_194: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_193, 1, 0, 9223372036854775807)
    slice_195: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_194, 2, 0, 9223372036854775807)
    select_scatter_17: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(slice_195, copy_17, 3, 0);  slice_195 = copy_17 = None
    slice_scatter_51: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_194, select_scatter_17, 2, 0, 9223372036854775807);  slice_194 = select_scatter_17 = None
    slice_scatter_52: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_193, slice_scatter_51, 1, 0, 9223372036854775807);  slice_193 = slice_scatter_51 = None
    slice_scatter_53: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_scatter_50, slice_scatter_52, 0, 0, 9223372036854775807);  slice_scatter_50 = slice_scatter_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:130, code: return rel_indices.to(device)
    device_put_5: "f32[1, 196, 196, 3]" = torch.ops.prims.device_put.default(slice_scatter_53, device(type='cuda', index=0));  slice_scatter_53 = None
    convert_element_type_5: "f32[1, 196, 196, 3]" = torch.ops.prims.convert_element_type.default(device_put_5, torch.float32);  device_put_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_56: "f32[768, 1536]" = torch.ops.aten.permute.default(arg114_1, [1, 0]);  arg114_1 = None
    view_135: "f32[1568, 768]" = torch.ops.aten.view.default(add_52, [1568, 768])
    mm_15: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_135, permute_56);  view_135 = permute_56 = None
    view_136: "f32[8, 196, 1536]" = torch.ops.aten.view.default(mm_15, [8, 196, 1536]);  mm_15 = None
    view_137: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.view.default(view_136, [8, 196, 2, 16, 48]);  view_136 = None
    permute_57: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.permute.default(view_137, [2, 0, 3, 1, 4]);  view_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    select_58: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_57, 0, 0)
    select_59: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_57, 0, 1);  permute_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:84, code: pos_score = self.rel_indices.expand(B, -1, -1, -1)
    expand_38: "f32[8, 196, 196, 3]" = torch.ops.aten.expand.default(convert_element_type_5, [8, -1, -1, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    permute_58: "f32[3, 16]" = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
    clone_74: "f32[8, 196, 196, 3]" = torch.ops.aten.clone.default(expand_38, memory_format = torch.contiguous_format);  expand_38 = None
    view_138: "f32[307328, 3]" = torch.ops.aten.view.default(clone_74, [307328, 3]);  clone_74 = None
    mm_16: "f32[307328, 16]" = torch.ops.aten.mm.default(view_138, permute_58);  view_138 = permute_58 = None
    view_139: "f32[8, 196, 196, 16]" = torch.ops.aten.view.default(mm_16, [8, 196, 196, 16]);  mm_16 = None
    add_54: "f32[8, 196, 196, 16]" = torch.ops.aten.add.Tensor(view_139, arg116_1);  view_139 = arg116_1 = None
    permute_59: "f32[8, 16, 196, 196]" = torch.ops.aten.permute.default(add_54, [0, 3, 1, 2]);  add_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    permute_60: "f32[8, 16, 48, 196]" = torch.ops.aten.permute.default(select_59, [0, 1, 3, 2]);  select_59 = None
    expand_39: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(select_58, [8, 16, 196, 48]);  select_58 = None
    clone_75: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
    view_140: "f32[128, 196, 48]" = torch.ops.aten.view.default(clone_75, [128, 196, 48]);  clone_75 = None
    expand_40: "f32[8, 16, 48, 196]" = torch.ops.aten.expand.default(permute_60, [8, 16, 48, 196]);  permute_60 = None
    clone_76: "f32[8, 16, 48, 196]" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
    view_141: "f32[128, 48, 196]" = torch.ops.aten.view.default(clone_76, [128, 48, 196]);  clone_76 = None
    bmm_10: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_140, view_141);  view_140 = view_141 = None
    view_142: "f32[8, 16, 196, 196]" = torch.ops.aten.view.default(bmm_10, [8, 16, 196, 196]);  bmm_10 = None
    mul_52: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(view_142, 0.14433756729740643);  view_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    amax_10: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(mul_52, [-1], True)
    sub_32: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_52, amax_10);  mul_52 = amax_10 = None
    exp_10: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_32);  sub_32 = None
    sum_16: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_15: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_10, sum_16);  exp_10 = sum_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    clone_77: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None
    amax_11: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(clone_77, [-1], True)
    sub_33: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(clone_77, amax_11);  clone_77 = amax_11 = None
    exp_11: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_33);  sub_33 = None
    sum_17: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_16: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_11, sum_17);  exp_11 = sum_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_143: "f32[1, 16, 1, 1]" = torch.ops.aten.view.default(arg29_1, [1, -1, 1, 1]);  arg29_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    sigmoid_10: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_143)
    sub_34: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1.0, sigmoid_10);  sigmoid_10 = None
    mul_53: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_34, div_15);  sub_34 = div_15 = None
    sigmoid_11: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_143);  view_143 = None
    mul_54: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sigmoid_11, div_16);  sigmoid_11 = div_16 = None
    add_55: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(mul_53, mul_54);  mul_53 = mul_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    sum_18: "f32[8, 16, 196]" = torch.ops.aten.sum.dim_IntList(add_55, [-1])
    unsqueeze_35: "f32[8, 16, 196, 1]" = torch.ops.aten.unsqueeze.default(sum_18, -1);  sum_18 = None
    div_17: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(add_55, unsqueeze_35);  add_55 = unsqueeze_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:93, code: attn = self.attn_drop(attn)
    clone_78: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(div_17);  div_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_61: "f32[768, 768]" = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
    view_144: "f32[1568, 768]" = torch.ops.aten.view.default(add_52, [1568, 768]);  add_52 = None
    mm_17: "f32[1568, 768]" = torch.ops.aten.mm.default(view_144, permute_61);  view_144 = permute_61 = None
    view_145: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_17, [8, 196, 768]);  mm_17 = None
    view_146: "f32[8, 196, 16, 48]" = torch.ops.aten.view.default(view_145, [8, 196, 16, 48]);  view_145 = None
    permute_62: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_146, [0, 2, 1, 3]);  view_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_41: "f32[8, 16, 196, 196]" = torch.ops.aten.expand.default(clone_78, [8, 16, 196, 196]);  clone_78 = None
    view_147: "f32[128, 196, 196]" = torch.ops.aten.view.default(expand_41, [128, 196, 196]);  expand_41 = None
    expand_42: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(permute_62, [8, 16, 196, 48]);  permute_62 = None
    clone_79: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
    view_148: "f32[128, 196, 48]" = torch.ops.aten.view.default(clone_79, [128, 196, 48]);  clone_79 = None
    bmm_11: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(view_147, view_148);  view_147 = view_148 = None
    view_149: "f32[8, 16, 196, 48]" = torch.ops.aten.view.default(bmm_11, [8, 16, 196, 48]);  bmm_11 = None
    permute_63: "f32[8, 196, 16, 48]" = torch.ops.aten.permute.default(view_149, [0, 2, 1, 3]);  view_149 = None
    clone_80: "f32[8, 196, 16, 48]" = torch.ops.aten.clone.default(permute_63, memory_format = torch.contiguous_format);  permute_63 = None
    view_150: "f32[8, 196, 768]" = torch.ops.aten.view.default(clone_80, [8, 196, 768]);  clone_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_151: "f32[1568, 768]" = torch.ops.aten.view.default(view_150, [1568, 768]);  view_150 = None
    permute_64: "f32[768, 768]" = torch.ops.aten.permute.default(arg118_1, [1, 0]);  arg118_1 = None
    addmm_15: "f32[1568, 768]" = torch.ops.aten.addmm.default(arg119_1, view_151, permute_64);  arg119_1 = view_151 = permute_64 = None
    view_152: "f32[8, 196, 768]" = torch.ops.aten.view.default(addmm_15, [8, 196, 768]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:77, code: x = self.proj_drop(x)
    clone_81: "f32[8, 196, 768]" = torch.ops.aten.clone.default(view_152);  view_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:235, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_56: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_50, clone_81);  add_50 = clone_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_82: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_56, memory_format = torch.contiguous_format)
    var_mean_11 = torch.ops.aten.var_mean.correction(clone_82, [2], correction = 0, keepdim = True)
    getitem_22: "f32[8, 196, 1]" = var_mean_11[0]
    getitem_23: "f32[8, 196, 1]" = var_mean_11[1];  var_mean_11 = None
    add_57: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-06);  getitem_22 = None
    rsqrt_11: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
    sub_35: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_82, getitem_23);  clone_82 = getitem_23 = None
    mul_55: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_11);  sub_35 = rsqrt_11 = None
    mul_56: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_55, arg30_1);  mul_55 = arg30_1 = None
    add_58: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_56, arg31_1);  mul_56 = arg31_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_153: "f32[1568, 768]" = torch.ops.aten.view.default(add_58, [1568, 768]);  add_58 = None
    permute_65: "f32[768, 3072]" = torch.ops.aten.permute.default(arg120_1, [1, 0]);  arg120_1 = None
    addmm_16: "f32[1568, 3072]" = torch.ops.aten.addmm.default(arg121_1, view_153, permute_65);  arg121_1 = view_153 = permute_65 = None
    view_154: "f32[8, 196, 3072]" = torch.ops.aten.view.default(addmm_16, [8, 196, 3072]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_57: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_154, 0.5)
    mul_58: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_154, 0.7071067811865476);  view_154 = None
    erf_5: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_58);  mul_58 = None
    add_59: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_59: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_57, add_59);  mul_57 = add_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_83: "f32[8, 196, 3072]" = torch.ops.aten.clone.default(mul_59);  mul_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_155: "f32[1568, 3072]" = torch.ops.aten.view.default(clone_83, [1568, 3072]);  clone_83 = None
    permute_66: "f32[3072, 768]" = torch.ops.aten.permute.default(arg122_1, [1, 0]);  arg122_1 = None
    addmm_17: "f32[1568, 768]" = torch.ops.aten.addmm.default(arg123_1, view_155, permute_66);  arg123_1 = view_155 = permute_66 = None
    view_156: "f32[8, 196, 768]" = torch.ops.aten.view.default(addmm_17, [8, 196, 768]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_84: "f32[8, 196, 768]" = torch.ops.aten.clone.default(view_156);  view_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:236, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_60: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_56, clone_84);  add_56 = clone_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_85: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_60, memory_format = torch.contiguous_format)
    var_mean_12 = torch.ops.aten.var_mean.correction(clone_85, [2], correction = 0, keepdim = True)
    getitem_24: "f32[8, 196, 1]" = var_mean_12[0]
    getitem_25: "f32[8, 196, 1]" = var_mean_12[1];  var_mean_12 = None
    add_61: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-06);  getitem_24 = None
    rsqrt_12: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
    sub_36: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_85, getitem_25);  clone_85 = getitem_25 = None
    mul_60: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_12);  sub_36 = rsqrt_12 = None
    mul_61: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_60, arg32_1);  mul_60 = arg32_1 = None
    add_62: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_61, arg33_1);  mul_61 = arg33_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:121, code: rel_indices = torch.zeros(1, num_patches, num_patches, 3)
    full_6: "f32[1, 196, 196, 3]" = torch.ops.aten.full.default([1, 196, 196, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:122, code: ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
    iota_12: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_157: "i64[1, 14]" = torch.ops.aten.view.default(iota_12, [1, -1]);  iota_12 = None
    iota_13: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_158: "i64[14, 1]" = torch.ops.aten.view.default(iota_13, [-1, 1]);  iota_13 = None
    sub_37: "i64[14, 14]" = torch.ops.aten.sub.Tensor(view_157, view_158);  view_157 = view_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:123, code: indx = ind.repeat(img_size, img_size)
    repeat_6: "i64[196, 196]" = torch.ops.aten.repeat.default(sub_37, [14, 14])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:124, code: indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
    unsqueeze_36: "i64[14, 1, 14]" = torch.ops.aten.unsqueeze.default(sub_37, 1);  sub_37 = None
    expand_43: "i64[14, 14, 14]" = torch.ops.aten.expand.default(unsqueeze_36, [14, 14, 14]);  unsqueeze_36 = None
    clone_86: "i64[14, 14, 14]" = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
    view_159: "i64[196, 14]" = torch.ops.aten.view.default(clone_86, [196, 14]);  clone_86 = None
    unsqueeze_37: "i64[196, 14, 1]" = torch.ops.aten.unsqueeze.default(view_159, 2);  view_159 = None
    expand_44: "i64[196, 14, 14]" = torch.ops.aten.expand.default(unsqueeze_37, [196, 14, 14]);  unsqueeze_37 = None
    clone_87: "i64[196, 14, 14]" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
    view_160: "i64[196, 196]" = torch.ops.aten.view.default(clone_87, [196, 196]);  clone_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:125, code: indd = indx ** 2 + indy ** 2
    pow_13: "i64[196, 196]" = torch.ops.aten.pow.Tensor_Scalar(repeat_6, 2)
    pow_14: "i64[196, 196]" = torch.ops.aten.pow.Tensor_Scalar(view_160, 2)
    add_63: "i64[196, 196]" = torch.ops.aten.add.Tensor(pow_13, pow_14);  pow_13 = pow_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:126, code: rel_indices[:, :, :, 2] = indd.unsqueeze(0)
    unsqueeze_38: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(add_63, 0);  add_63 = None
    slice_199: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(full_6, 0, 0, 9223372036854775807)
    slice_200: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_199, 1, 0, 9223372036854775807);  slice_199 = None
    slice_201: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_200, 2, 0, 9223372036854775807);  slice_200 = None
    select_60: "f32[1, 196, 196]" = torch.ops.aten.select.int(slice_201, 3, 2);  slice_201 = None
    copy_18: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_60, unsqueeze_38);  select_60 = unsqueeze_38 = None
    slice_202: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(full_6, 0, 0, 9223372036854775807)
    slice_203: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_202, 1, 0, 9223372036854775807)
    slice_204: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_203, 2, 0, 9223372036854775807)
    select_scatter_18: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(slice_204, copy_18, 3, 2);  slice_204 = copy_18 = None
    slice_scatter_54: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_203, select_scatter_18, 2, 0, 9223372036854775807);  slice_203 = select_scatter_18 = None
    slice_scatter_55: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_202, slice_scatter_54, 1, 0, 9223372036854775807);  slice_202 = slice_scatter_54 = None
    slice_scatter_56: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(full_6, slice_scatter_55, 0, 0, 9223372036854775807);  full_6 = slice_scatter_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:127, code: rel_indices[:, :, :, 1] = indy.unsqueeze(0)
    unsqueeze_39: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(view_160, 0);  view_160 = None
    slice_211: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_56, 0, 0, 9223372036854775807)
    slice_212: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_211, 1, 0, 9223372036854775807);  slice_211 = None
    slice_213: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_212, 2, 0, 9223372036854775807);  slice_212 = None
    select_63: "f32[1, 196, 196]" = torch.ops.aten.select.int(slice_213, 3, 1);  slice_213 = None
    copy_19: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_63, unsqueeze_39);  select_63 = unsqueeze_39 = None
    slice_214: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_56, 0, 0, 9223372036854775807)
    slice_215: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_214, 1, 0, 9223372036854775807)
    slice_216: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_215, 2, 0, 9223372036854775807)
    select_scatter_19: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(slice_216, copy_19, 3, 1);  slice_216 = copy_19 = None
    slice_scatter_57: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_215, select_scatter_19, 2, 0, 9223372036854775807);  slice_215 = select_scatter_19 = None
    slice_scatter_58: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_214, slice_scatter_57, 1, 0, 9223372036854775807);  slice_214 = slice_scatter_57 = None
    slice_scatter_59: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_scatter_56, slice_scatter_58, 0, 0, 9223372036854775807);  slice_scatter_56 = slice_scatter_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:128, code: rel_indices[:, :, :, 0] = indx.unsqueeze(0)
    unsqueeze_40: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(repeat_6, 0);  repeat_6 = None
    slice_223: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_59, 0, 0, 9223372036854775807)
    slice_224: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_223, 1, 0, 9223372036854775807);  slice_223 = None
    slice_225: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_224, 2, 0, 9223372036854775807);  slice_224 = None
    select_66: "f32[1, 196, 196]" = torch.ops.aten.select.int(slice_225, 3, 0);  slice_225 = None
    copy_20: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_66, unsqueeze_40);  select_66 = unsqueeze_40 = None
    slice_226: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_59, 0, 0, 9223372036854775807)
    slice_227: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_226, 1, 0, 9223372036854775807)
    slice_228: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_227, 2, 0, 9223372036854775807)
    select_scatter_20: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(slice_228, copy_20, 3, 0);  slice_228 = copy_20 = None
    slice_scatter_60: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_227, select_scatter_20, 2, 0, 9223372036854775807);  slice_227 = select_scatter_20 = None
    slice_scatter_61: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_226, slice_scatter_60, 1, 0, 9223372036854775807);  slice_226 = slice_scatter_60 = None
    slice_scatter_62: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_scatter_59, slice_scatter_61, 0, 0, 9223372036854775807);  slice_scatter_59 = slice_scatter_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:130, code: return rel_indices.to(device)
    device_put_6: "f32[1, 196, 196, 3]" = torch.ops.prims.device_put.default(slice_scatter_62, device(type='cuda', index=0));  slice_scatter_62 = None
    convert_element_type_6: "f32[1, 196, 196, 3]" = torch.ops.prims.convert_element_type.default(device_put_6, torch.float32);  device_put_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_67: "f32[768, 1536]" = torch.ops.aten.permute.default(arg124_1, [1, 0]);  arg124_1 = None
    view_161: "f32[1568, 768]" = torch.ops.aten.view.default(add_62, [1568, 768])
    mm_18: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_161, permute_67);  view_161 = permute_67 = None
    view_162: "f32[8, 196, 1536]" = torch.ops.aten.view.default(mm_18, [8, 196, 1536]);  mm_18 = None
    view_163: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.view.default(view_162, [8, 196, 2, 16, 48]);  view_162 = None
    permute_68: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.permute.default(view_163, [2, 0, 3, 1, 4]);  view_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    select_68: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_68, 0, 0)
    select_69: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_68, 0, 1);  permute_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:84, code: pos_score = self.rel_indices.expand(B, -1, -1, -1)
    expand_45: "f32[8, 196, 196, 3]" = torch.ops.aten.expand.default(convert_element_type_6, [8, -1, -1, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    permute_69: "f32[3, 16]" = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
    clone_88: "f32[8, 196, 196, 3]" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
    view_164: "f32[307328, 3]" = torch.ops.aten.view.default(clone_88, [307328, 3]);  clone_88 = None
    mm_19: "f32[307328, 16]" = torch.ops.aten.mm.default(view_164, permute_69);  view_164 = permute_69 = None
    view_165: "f32[8, 196, 196, 16]" = torch.ops.aten.view.default(mm_19, [8, 196, 196, 16]);  mm_19 = None
    add_64: "f32[8, 196, 196, 16]" = torch.ops.aten.add.Tensor(view_165, arg126_1);  view_165 = arg126_1 = None
    permute_70: "f32[8, 16, 196, 196]" = torch.ops.aten.permute.default(add_64, [0, 3, 1, 2]);  add_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    permute_71: "f32[8, 16, 48, 196]" = torch.ops.aten.permute.default(select_69, [0, 1, 3, 2]);  select_69 = None
    expand_46: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(select_68, [8, 16, 196, 48]);  select_68 = None
    clone_89: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_46, memory_format = torch.contiguous_format);  expand_46 = None
    view_166: "f32[128, 196, 48]" = torch.ops.aten.view.default(clone_89, [128, 196, 48]);  clone_89 = None
    expand_47: "f32[8, 16, 48, 196]" = torch.ops.aten.expand.default(permute_71, [8, 16, 48, 196]);  permute_71 = None
    clone_90: "f32[8, 16, 48, 196]" = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
    view_167: "f32[128, 48, 196]" = torch.ops.aten.view.default(clone_90, [128, 48, 196]);  clone_90 = None
    bmm_12: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_166, view_167);  view_166 = view_167 = None
    view_168: "f32[8, 16, 196, 196]" = torch.ops.aten.view.default(bmm_12, [8, 16, 196, 196]);  bmm_12 = None
    mul_62: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(view_168, 0.14433756729740643);  view_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    amax_12: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(mul_62, [-1], True)
    sub_38: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_62, amax_12);  mul_62 = amax_12 = None
    exp_12: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_38);  sub_38 = None
    sum_19: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
    div_18: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_12, sum_19);  exp_12 = sum_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    clone_91: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(permute_70, memory_format = torch.contiguous_format);  permute_70 = None
    amax_13: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(clone_91, [-1], True)
    sub_39: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(clone_91, amax_13);  clone_91 = amax_13 = None
    exp_13: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_39);  sub_39 = None
    sum_20: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
    div_19: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_13, sum_20);  exp_13 = sum_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_169: "f32[1, 16, 1, 1]" = torch.ops.aten.view.default(arg34_1, [1, -1, 1, 1]);  arg34_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    sigmoid_12: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_169)
    sub_40: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1.0, sigmoid_12);  sigmoid_12 = None
    mul_63: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_40, div_18);  sub_40 = div_18 = None
    sigmoid_13: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_169);  view_169 = None
    mul_64: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sigmoid_13, div_19);  sigmoid_13 = div_19 = None
    add_65: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(mul_63, mul_64);  mul_63 = mul_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    sum_21: "f32[8, 16, 196]" = torch.ops.aten.sum.dim_IntList(add_65, [-1])
    unsqueeze_41: "f32[8, 16, 196, 1]" = torch.ops.aten.unsqueeze.default(sum_21, -1);  sum_21 = None
    div_20: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(add_65, unsqueeze_41);  add_65 = unsqueeze_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:93, code: attn = self.attn_drop(attn)
    clone_92: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(div_20);  div_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_72: "f32[768, 768]" = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
    view_170: "f32[1568, 768]" = torch.ops.aten.view.default(add_62, [1568, 768]);  add_62 = None
    mm_20: "f32[1568, 768]" = torch.ops.aten.mm.default(view_170, permute_72);  view_170 = permute_72 = None
    view_171: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_20, [8, 196, 768]);  mm_20 = None
    view_172: "f32[8, 196, 16, 48]" = torch.ops.aten.view.default(view_171, [8, 196, 16, 48]);  view_171 = None
    permute_73: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_172, [0, 2, 1, 3]);  view_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_48: "f32[8, 16, 196, 196]" = torch.ops.aten.expand.default(clone_92, [8, 16, 196, 196]);  clone_92 = None
    view_173: "f32[128, 196, 196]" = torch.ops.aten.view.default(expand_48, [128, 196, 196]);  expand_48 = None
    expand_49: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(permute_73, [8, 16, 196, 48]);  permute_73 = None
    clone_93: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
    view_174: "f32[128, 196, 48]" = torch.ops.aten.view.default(clone_93, [128, 196, 48]);  clone_93 = None
    bmm_13: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(view_173, view_174);  view_173 = view_174 = None
    view_175: "f32[8, 16, 196, 48]" = torch.ops.aten.view.default(bmm_13, [8, 16, 196, 48]);  bmm_13 = None
    permute_74: "f32[8, 196, 16, 48]" = torch.ops.aten.permute.default(view_175, [0, 2, 1, 3]);  view_175 = None
    clone_94: "f32[8, 196, 16, 48]" = torch.ops.aten.clone.default(permute_74, memory_format = torch.contiguous_format);  permute_74 = None
    view_176: "f32[8, 196, 768]" = torch.ops.aten.view.default(clone_94, [8, 196, 768]);  clone_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_177: "f32[1568, 768]" = torch.ops.aten.view.default(view_176, [1568, 768]);  view_176 = None
    permute_75: "f32[768, 768]" = torch.ops.aten.permute.default(arg128_1, [1, 0]);  arg128_1 = None
    addmm_18: "f32[1568, 768]" = torch.ops.aten.addmm.default(arg129_1, view_177, permute_75);  arg129_1 = view_177 = permute_75 = None
    view_178: "f32[8, 196, 768]" = torch.ops.aten.view.default(addmm_18, [8, 196, 768]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:77, code: x = self.proj_drop(x)
    clone_95: "f32[8, 196, 768]" = torch.ops.aten.clone.default(view_178);  view_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:235, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_66: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_60, clone_95);  add_60 = clone_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_96: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_66, memory_format = torch.contiguous_format)
    var_mean_13 = torch.ops.aten.var_mean.correction(clone_96, [2], correction = 0, keepdim = True)
    getitem_26: "f32[8, 196, 1]" = var_mean_13[0]
    getitem_27: "f32[8, 196, 1]" = var_mean_13[1];  var_mean_13 = None
    add_67: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-06);  getitem_26 = None
    rsqrt_13: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
    sub_41: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_96, getitem_27);  clone_96 = getitem_27 = None
    mul_65: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_13);  sub_41 = rsqrt_13 = None
    mul_66: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_65, arg35_1);  mul_65 = arg35_1 = None
    add_68: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_66, arg36_1);  mul_66 = arg36_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_179: "f32[1568, 768]" = torch.ops.aten.view.default(add_68, [1568, 768]);  add_68 = None
    permute_76: "f32[768, 3072]" = torch.ops.aten.permute.default(arg130_1, [1, 0]);  arg130_1 = None
    addmm_19: "f32[1568, 3072]" = torch.ops.aten.addmm.default(arg131_1, view_179, permute_76);  arg131_1 = view_179 = permute_76 = None
    view_180: "f32[8, 196, 3072]" = torch.ops.aten.view.default(addmm_19, [8, 196, 3072]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_67: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_180, 0.5)
    mul_68: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_180, 0.7071067811865476);  view_180 = None
    erf_6: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_68);  mul_68 = None
    add_69: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_69: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_67, add_69);  mul_67 = add_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_97: "f32[8, 196, 3072]" = torch.ops.aten.clone.default(mul_69);  mul_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_181: "f32[1568, 3072]" = torch.ops.aten.view.default(clone_97, [1568, 3072]);  clone_97 = None
    permute_77: "f32[3072, 768]" = torch.ops.aten.permute.default(arg132_1, [1, 0]);  arg132_1 = None
    addmm_20: "f32[1568, 768]" = torch.ops.aten.addmm.default(arg133_1, view_181, permute_77);  arg133_1 = view_181 = permute_77 = None
    view_182: "f32[8, 196, 768]" = torch.ops.aten.view.default(addmm_20, [8, 196, 768]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_98: "f32[8, 196, 768]" = torch.ops.aten.clone.default(view_182);  view_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:236, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_70: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_66, clone_98);  add_66 = clone_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_99: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_70, memory_format = torch.contiguous_format)
    var_mean_14 = torch.ops.aten.var_mean.correction(clone_99, [2], correction = 0, keepdim = True)
    getitem_28: "f32[8, 196, 1]" = var_mean_14[0]
    getitem_29: "f32[8, 196, 1]" = var_mean_14[1];  var_mean_14 = None
    add_71: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-06);  getitem_28 = None
    rsqrt_14: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
    sub_42: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_99, getitem_29);  clone_99 = getitem_29 = None
    mul_70: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_14);  sub_42 = rsqrt_14 = None
    mul_71: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_70, arg37_1);  mul_70 = arg37_1 = None
    add_72: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_71, arg38_1);  mul_71 = arg38_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:121, code: rel_indices = torch.zeros(1, num_patches, num_patches, 3)
    full_7: "f32[1, 196, 196, 3]" = torch.ops.aten.full.default([1, 196, 196, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:122, code: ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
    iota_14: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_183: "i64[1, 14]" = torch.ops.aten.view.default(iota_14, [1, -1]);  iota_14 = None
    iota_15: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_184: "i64[14, 1]" = torch.ops.aten.view.default(iota_15, [-1, 1]);  iota_15 = None
    sub_43: "i64[14, 14]" = torch.ops.aten.sub.Tensor(view_183, view_184);  view_183 = view_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:123, code: indx = ind.repeat(img_size, img_size)
    repeat_7: "i64[196, 196]" = torch.ops.aten.repeat.default(sub_43, [14, 14])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:124, code: indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
    unsqueeze_42: "i64[14, 1, 14]" = torch.ops.aten.unsqueeze.default(sub_43, 1);  sub_43 = None
    expand_50: "i64[14, 14, 14]" = torch.ops.aten.expand.default(unsqueeze_42, [14, 14, 14]);  unsqueeze_42 = None
    clone_100: "i64[14, 14, 14]" = torch.ops.aten.clone.default(expand_50, memory_format = torch.contiguous_format);  expand_50 = None
    view_185: "i64[196, 14]" = torch.ops.aten.view.default(clone_100, [196, 14]);  clone_100 = None
    unsqueeze_43: "i64[196, 14, 1]" = torch.ops.aten.unsqueeze.default(view_185, 2);  view_185 = None
    expand_51: "i64[196, 14, 14]" = torch.ops.aten.expand.default(unsqueeze_43, [196, 14, 14]);  unsqueeze_43 = None
    clone_101: "i64[196, 14, 14]" = torch.ops.aten.clone.default(expand_51, memory_format = torch.contiguous_format);  expand_51 = None
    view_186: "i64[196, 196]" = torch.ops.aten.view.default(clone_101, [196, 196]);  clone_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:125, code: indd = indx ** 2 + indy ** 2
    pow_15: "i64[196, 196]" = torch.ops.aten.pow.Tensor_Scalar(repeat_7, 2)
    pow_16: "i64[196, 196]" = torch.ops.aten.pow.Tensor_Scalar(view_186, 2)
    add_73: "i64[196, 196]" = torch.ops.aten.add.Tensor(pow_15, pow_16);  pow_15 = pow_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:126, code: rel_indices[:, :, :, 2] = indd.unsqueeze(0)
    unsqueeze_44: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(add_73, 0);  add_73 = None
    slice_232: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(full_7, 0, 0, 9223372036854775807)
    slice_233: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_232, 1, 0, 9223372036854775807);  slice_232 = None
    slice_234: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_233, 2, 0, 9223372036854775807);  slice_233 = None
    select_70: "f32[1, 196, 196]" = torch.ops.aten.select.int(slice_234, 3, 2);  slice_234 = None
    copy_21: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_70, unsqueeze_44);  select_70 = unsqueeze_44 = None
    slice_235: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(full_7, 0, 0, 9223372036854775807)
    slice_236: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_235, 1, 0, 9223372036854775807)
    slice_237: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_236, 2, 0, 9223372036854775807)
    select_scatter_21: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(slice_237, copy_21, 3, 2);  slice_237 = copy_21 = None
    slice_scatter_63: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_236, select_scatter_21, 2, 0, 9223372036854775807);  slice_236 = select_scatter_21 = None
    slice_scatter_64: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_235, slice_scatter_63, 1, 0, 9223372036854775807);  slice_235 = slice_scatter_63 = None
    slice_scatter_65: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(full_7, slice_scatter_64, 0, 0, 9223372036854775807);  full_7 = slice_scatter_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:127, code: rel_indices[:, :, :, 1] = indy.unsqueeze(0)
    unsqueeze_45: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(view_186, 0);  view_186 = None
    slice_244: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_65, 0, 0, 9223372036854775807)
    slice_245: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_244, 1, 0, 9223372036854775807);  slice_244 = None
    slice_246: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_245, 2, 0, 9223372036854775807);  slice_245 = None
    select_73: "f32[1, 196, 196]" = torch.ops.aten.select.int(slice_246, 3, 1);  slice_246 = None
    copy_22: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_73, unsqueeze_45);  select_73 = unsqueeze_45 = None
    slice_247: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_65, 0, 0, 9223372036854775807)
    slice_248: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_247, 1, 0, 9223372036854775807)
    slice_249: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_248, 2, 0, 9223372036854775807)
    select_scatter_22: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(slice_249, copy_22, 3, 1);  slice_249 = copy_22 = None
    slice_scatter_66: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_248, select_scatter_22, 2, 0, 9223372036854775807);  slice_248 = select_scatter_22 = None
    slice_scatter_67: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_247, slice_scatter_66, 1, 0, 9223372036854775807);  slice_247 = slice_scatter_66 = None
    slice_scatter_68: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_scatter_65, slice_scatter_67, 0, 0, 9223372036854775807);  slice_scatter_65 = slice_scatter_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:128, code: rel_indices[:, :, :, 0] = indx.unsqueeze(0)
    unsqueeze_46: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(repeat_7, 0);  repeat_7 = None
    slice_256: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_68, 0, 0, 9223372036854775807)
    slice_257: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_256, 1, 0, 9223372036854775807);  slice_256 = None
    slice_258: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_257, 2, 0, 9223372036854775807);  slice_257 = None
    select_76: "f32[1, 196, 196]" = torch.ops.aten.select.int(slice_258, 3, 0);  slice_258 = None
    copy_23: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_76, unsqueeze_46);  select_76 = unsqueeze_46 = None
    slice_259: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_68, 0, 0, 9223372036854775807)
    slice_260: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_259, 1, 0, 9223372036854775807)
    slice_261: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_260, 2, 0, 9223372036854775807)
    select_scatter_23: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(slice_261, copy_23, 3, 0);  slice_261 = copy_23 = None
    slice_scatter_69: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_260, select_scatter_23, 2, 0, 9223372036854775807);  slice_260 = select_scatter_23 = None
    slice_scatter_70: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_259, slice_scatter_69, 1, 0, 9223372036854775807);  slice_259 = slice_scatter_69 = None
    slice_scatter_71: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_scatter_68, slice_scatter_70, 0, 0, 9223372036854775807);  slice_scatter_68 = slice_scatter_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:130, code: return rel_indices.to(device)
    device_put_7: "f32[1, 196, 196, 3]" = torch.ops.prims.device_put.default(slice_scatter_71, device(type='cuda', index=0));  slice_scatter_71 = None
    convert_element_type_7: "f32[1, 196, 196, 3]" = torch.ops.prims.convert_element_type.default(device_put_7, torch.float32);  device_put_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_78: "f32[768, 1536]" = torch.ops.aten.permute.default(arg134_1, [1, 0]);  arg134_1 = None
    view_187: "f32[1568, 768]" = torch.ops.aten.view.default(add_72, [1568, 768])
    mm_21: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_187, permute_78);  view_187 = permute_78 = None
    view_188: "f32[8, 196, 1536]" = torch.ops.aten.view.default(mm_21, [8, 196, 1536]);  mm_21 = None
    view_189: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.view.default(view_188, [8, 196, 2, 16, 48]);  view_188 = None
    permute_79: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.permute.default(view_189, [2, 0, 3, 1, 4]);  view_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    select_78: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_79, 0, 0)
    select_79: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_79, 0, 1);  permute_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:84, code: pos_score = self.rel_indices.expand(B, -1, -1, -1)
    expand_52: "f32[8, 196, 196, 3]" = torch.ops.aten.expand.default(convert_element_type_7, [8, -1, -1, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    permute_80: "f32[3, 16]" = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
    clone_102: "f32[8, 196, 196, 3]" = torch.ops.aten.clone.default(expand_52, memory_format = torch.contiguous_format);  expand_52 = None
    view_190: "f32[307328, 3]" = torch.ops.aten.view.default(clone_102, [307328, 3]);  clone_102 = None
    mm_22: "f32[307328, 16]" = torch.ops.aten.mm.default(view_190, permute_80);  view_190 = permute_80 = None
    view_191: "f32[8, 196, 196, 16]" = torch.ops.aten.view.default(mm_22, [8, 196, 196, 16]);  mm_22 = None
    add_74: "f32[8, 196, 196, 16]" = torch.ops.aten.add.Tensor(view_191, arg136_1);  view_191 = arg136_1 = None
    permute_81: "f32[8, 16, 196, 196]" = torch.ops.aten.permute.default(add_74, [0, 3, 1, 2]);  add_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    permute_82: "f32[8, 16, 48, 196]" = torch.ops.aten.permute.default(select_79, [0, 1, 3, 2]);  select_79 = None
    expand_53: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(select_78, [8, 16, 196, 48]);  select_78 = None
    clone_103: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_53, memory_format = torch.contiguous_format);  expand_53 = None
    view_192: "f32[128, 196, 48]" = torch.ops.aten.view.default(clone_103, [128, 196, 48]);  clone_103 = None
    expand_54: "f32[8, 16, 48, 196]" = torch.ops.aten.expand.default(permute_82, [8, 16, 48, 196]);  permute_82 = None
    clone_104: "f32[8, 16, 48, 196]" = torch.ops.aten.clone.default(expand_54, memory_format = torch.contiguous_format);  expand_54 = None
    view_193: "f32[128, 48, 196]" = torch.ops.aten.view.default(clone_104, [128, 48, 196]);  clone_104 = None
    bmm_14: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_192, view_193);  view_192 = view_193 = None
    view_194: "f32[8, 16, 196, 196]" = torch.ops.aten.view.default(bmm_14, [8, 16, 196, 196]);  bmm_14 = None
    mul_72: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(view_194, 0.14433756729740643);  view_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    amax_14: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(mul_72, [-1], True)
    sub_44: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_72, amax_14);  mul_72 = amax_14 = None
    exp_14: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_44);  sub_44 = None
    sum_22: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
    div_21: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_14, sum_22);  exp_14 = sum_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    clone_105: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(permute_81, memory_format = torch.contiguous_format);  permute_81 = None
    amax_15: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(clone_105, [-1], True)
    sub_45: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(clone_105, amax_15);  clone_105 = amax_15 = None
    exp_15: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_45);  sub_45 = None
    sum_23: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
    div_22: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_15, sum_23);  exp_15 = sum_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_195: "f32[1, 16, 1, 1]" = torch.ops.aten.view.default(arg39_1, [1, -1, 1, 1]);  arg39_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    sigmoid_14: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_195)
    sub_46: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1.0, sigmoid_14);  sigmoid_14 = None
    mul_73: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_46, div_21);  sub_46 = div_21 = None
    sigmoid_15: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_195);  view_195 = None
    mul_74: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sigmoid_15, div_22);  sigmoid_15 = div_22 = None
    add_75: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(mul_73, mul_74);  mul_73 = mul_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    sum_24: "f32[8, 16, 196]" = torch.ops.aten.sum.dim_IntList(add_75, [-1])
    unsqueeze_47: "f32[8, 16, 196, 1]" = torch.ops.aten.unsqueeze.default(sum_24, -1);  sum_24 = None
    div_23: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(add_75, unsqueeze_47);  add_75 = unsqueeze_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:93, code: attn = self.attn_drop(attn)
    clone_106: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(div_23);  div_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_83: "f32[768, 768]" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
    view_196: "f32[1568, 768]" = torch.ops.aten.view.default(add_72, [1568, 768]);  add_72 = None
    mm_23: "f32[1568, 768]" = torch.ops.aten.mm.default(view_196, permute_83);  view_196 = permute_83 = None
    view_197: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_23, [8, 196, 768]);  mm_23 = None
    view_198: "f32[8, 196, 16, 48]" = torch.ops.aten.view.default(view_197, [8, 196, 16, 48]);  view_197 = None
    permute_84: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_198, [0, 2, 1, 3]);  view_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_55: "f32[8, 16, 196, 196]" = torch.ops.aten.expand.default(clone_106, [8, 16, 196, 196]);  clone_106 = None
    view_199: "f32[128, 196, 196]" = torch.ops.aten.view.default(expand_55, [128, 196, 196]);  expand_55 = None
    expand_56: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(permute_84, [8, 16, 196, 48]);  permute_84 = None
    clone_107: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
    view_200: "f32[128, 196, 48]" = torch.ops.aten.view.default(clone_107, [128, 196, 48]);  clone_107 = None
    bmm_15: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(view_199, view_200);  view_199 = view_200 = None
    view_201: "f32[8, 16, 196, 48]" = torch.ops.aten.view.default(bmm_15, [8, 16, 196, 48]);  bmm_15 = None
    permute_85: "f32[8, 196, 16, 48]" = torch.ops.aten.permute.default(view_201, [0, 2, 1, 3]);  view_201 = None
    clone_108: "f32[8, 196, 16, 48]" = torch.ops.aten.clone.default(permute_85, memory_format = torch.contiguous_format);  permute_85 = None
    view_202: "f32[8, 196, 768]" = torch.ops.aten.view.default(clone_108, [8, 196, 768]);  clone_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_203: "f32[1568, 768]" = torch.ops.aten.view.default(view_202, [1568, 768]);  view_202 = None
    permute_86: "f32[768, 768]" = torch.ops.aten.permute.default(arg138_1, [1, 0]);  arg138_1 = None
    addmm_21: "f32[1568, 768]" = torch.ops.aten.addmm.default(arg139_1, view_203, permute_86);  arg139_1 = view_203 = permute_86 = None
    view_204: "f32[8, 196, 768]" = torch.ops.aten.view.default(addmm_21, [8, 196, 768]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:77, code: x = self.proj_drop(x)
    clone_109: "f32[8, 196, 768]" = torch.ops.aten.clone.default(view_204);  view_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:235, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_76: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_70, clone_109);  add_70 = clone_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_110: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_76, memory_format = torch.contiguous_format)
    var_mean_15 = torch.ops.aten.var_mean.correction(clone_110, [2], correction = 0, keepdim = True)
    getitem_30: "f32[8, 196, 1]" = var_mean_15[0]
    getitem_31: "f32[8, 196, 1]" = var_mean_15[1];  var_mean_15 = None
    add_77: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-06);  getitem_30 = None
    rsqrt_15: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
    sub_47: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_110, getitem_31);  clone_110 = getitem_31 = None
    mul_75: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_15);  sub_47 = rsqrt_15 = None
    mul_76: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_75, arg40_1);  mul_75 = arg40_1 = None
    add_78: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_76, arg41_1);  mul_76 = arg41_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_205: "f32[1568, 768]" = torch.ops.aten.view.default(add_78, [1568, 768]);  add_78 = None
    permute_87: "f32[768, 3072]" = torch.ops.aten.permute.default(arg140_1, [1, 0]);  arg140_1 = None
    addmm_22: "f32[1568, 3072]" = torch.ops.aten.addmm.default(arg141_1, view_205, permute_87);  arg141_1 = view_205 = permute_87 = None
    view_206: "f32[8, 196, 3072]" = torch.ops.aten.view.default(addmm_22, [8, 196, 3072]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_77: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_206, 0.5)
    mul_78: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_206, 0.7071067811865476);  view_206 = None
    erf_7: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_78);  mul_78 = None
    add_79: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_79: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_77, add_79);  mul_77 = add_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_111: "f32[8, 196, 3072]" = torch.ops.aten.clone.default(mul_79);  mul_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_207: "f32[1568, 3072]" = torch.ops.aten.view.default(clone_111, [1568, 3072]);  clone_111 = None
    permute_88: "f32[3072, 768]" = torch.ops.aten.permute.default(arg142_1, [1, 0]);  arg142_1 = None
    addmm_23: "f32[1568, 768]" = torch.ops.aten.addmm.default(arg143_1, view_207, permute_88);  arg143_1 = view_207 = permute_88 = None
    view_208: "f32[8, 196, 768]" = torch.ops.aten.view.default(addmm_23, [8, 196, 768]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_112: "f32[8, 196, 768]" = torch.ops.aten.clone.default(view_208);  view_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:236, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_80: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_76, clone_112);  add_76 = clone_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_113: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_80, memory_format = torch.contiguous_format)
    var_mean_16 = torch.ops.aten.var_mean.correction(clone_113, [2], correction = 0, keepdim = True)
    getitem_32: "f32[8, 196, 1]" = var_mean_16[0]
    getitem_33: "f32[8, 196, 1]" = var_mean_16[1];  var_mean_16 = None
    add_81: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-06);  getitem_32 = None
    rsqrt_16: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
    sub_48: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_113, getitem_33);  clone_113 = getitem_33 = None
    mul_80: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_16);  sub_48 = rsqrt_16 = None
    mul_81: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_80, arg42_1);  mul_80 = arg42_1 = None
    add_82: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_81, arg43_1);  mul_81 = arg43_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:121, code: rel_indices = torch.zeros(1, num_patches, num_patches, 3)
    full_8: "f32[1, 196, 196, 3]" = torch.ops.aten.full.default([1, 196, 196, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:122, code: ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
    iota_16: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_209: "i64[1, 14]" = torch.ops.aten.view.default(iota_16, [1, -1]);  iota_16 = None
    iota_17: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_210: "i64[14, 1]" = torch.ops.aten.view.default(iota_17, [-1, 1]);  iota_17 = None
    sub_49: "i64[14, 14]" = torch.ops.aten.sub.Tensor(view_209, view_210);  view_209 = view_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:123, code: indx = ind.repeat(img_size, img_size)
    repeat_8: "i64[196, 196]" = torch.ops.aten.repeat.default(sub_49, [14, 14])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:124, code: indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
    unsqueeze_48: "i64[14, 1, 14]" = torch.ops.aten.unsqueeze.default(sub_49, 1);  sub_49 = None
    expand_57: "i64[14, 14, 14]" = torch.ops.aten.expand.default(unsqueeze_48, [14, 14, 14]);  unsqueeze_48 = None
    clone_114: "i64[14, 14, 14]" = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
    view_211: "i64[196, 14]" = torch.ops.aten.view.default(clone_114, [196, 14]);  clone_114 = None
    unsqueeze_49: "i64[196, 14, 1]" = torch.ops.aten.unsqueeze.default(view_211, 2);  view_211 = None
    expand_58: "i64[196, 14, 14]" = torch.ops.aten.expand.default(unsqueeze_49, [196, 14, 14]);  unsqueeze_49 = None
    clone_115: "i64[196, 14, 14]" = torch.ops.aten.clone.default(expand_58, memory_format = torch.contiguous_format);  expand_58 = None
    view_212: "i64[196, 196]" = torch.ops.aten.view.default(clone_115, [196, 196]);  clone_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:125, code: indd = indx ** 2 + indy ** 2
    pow_17: "i64[196, 196]" = torch.ops.aten.pow.Tensor_Scalar(repeat_8, 2)
    pow_18: "i64[196, 196]" = torch.ops.aten.pow.Tensor_Scalar(view_212, 2)
    add_83: "i64[196, 196]" = torch.ops.aten.add.Tensor(pow_17, pow_18);  pow_17 = pow_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:126, code: rel_indices[:, :, :, 2] = indd.unsqueeze(0)
    unsqueeze_50: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(add_83, 0);  add_83 = None
    slice_265: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(full_8, 0, 0, 9223372036854775807)
    slice_266: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_265, 1, 0, 9223372036854775807);  slice_265 = None
    slice_267: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_266, 2, 0, 9223372036854775807);  slice_266 = None
    select_80: "f32[1, 196, 196]" = torch.ops.aten.select.int(slice_267, 3, 2);  slice_267 = None
    copy_24: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_80, unsqueeze_50);  select_80 = unsqueeze_50 = None
    slice_268: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(full_8, 0, 0, 9223372036854775807)
    slice_269: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_268, 1, 0, 9223372036854775807)
    slice_270: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_269, 2, 0, 9223372036854775807)
    select_scatter_24: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(slice_270, copy_24, 3, 2);  slice_270 = copy_24 = None
    slice_scatter_72: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_269, select_scatter_24, 2, 0, 9223372036854775807);  slice_269 = select_scatter_24 = None
    slice_scatter_73: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_268, slice_scatter_72, 1, 0, 9223372036854775807);  slice_268 = slice_scatter_72 = None
    slice_scatter_74: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(full_8, slice_scatter_73, 0, 0, 9223372036854775807);  full_8 = slice_scatter_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:127, code: rel_indices[:, :, :, 1] = indy.unsqueeze(0)
    unsqueeze_51: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(view_212, 0);  view_212 = None
    slice_277: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_74, 0, 0, 9223372036854775807)
    slice_278: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_277, 1, 0, 9223372036854775807);  slice_277 = None
    slice_279: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_278, 2, 0, 9223372036854775807);  slice_278 = None
    select_83: "f32[1, 196, 196]" = torch.ops.aten.select.int(slice_279, 3, 1);  slice_279 = None
    copy_25: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_83, unsqueeze_51);  select_83 = unsqueeze_51 = None
    slice_280: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_74, 0, 0, 9223372036854775807)
    slice_281: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_280, 1, 0, 9223372036854775807)
    slice_282: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_281, 2, 0, 9223372036854775807)
    select_scatter_25: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(slice_282, copy_25, 3, 1);  slice_282 = copy_25 = None
    slice_scatter_75: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_281, select_scatter_25, 2, 0, 9223372036854775807);  slice_281 = select_scatter_25 = None
    slice_scatter_76: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_280, slice_scatter_75, 1, 0, 9223372036854775807);  slice_280 = slice_scatter_75 = None
    slice_scatter_77: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_scatter_74, slice_scatter_76, 0, 0, 9223372036854775807);  slice_scatter_74 = slice_scatter_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:128, code: rel_indices[:, :, :, 0] = indx.unsqueeze(0)
    unsqueeze_52: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(repeat_8, 0);  repeat_8 = None
    slice_289: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_77, 0, 0, 9223372036854775807)
    slice_290: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_289, 1, 0, 9223372036854775807);  slice_289 = None
    slice_291: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_290, 2, 0, 9223372036854775807);  slice_290 = None
    select_86: "f32[1, 196, 196]" = torch.ops.aten.select.int(slice_291, 3, 0);  slice_291 = None
    copy_26: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_86, unsqueeze_52);  select_86 = unsqueeze_52 = None
    slice_292: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_77, 0, 0, 9223372036854775807)
    slice_293: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_292, 1, 0, 9223372036854775807)
    slice_294: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_293, 2, 0, 9223372036854775807)
    select_scatter_26: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(slice_294, copy_26, 3, 0);  slice_294 = copy_26 = None
    slice_scatter_78: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_293, select_scatter_26, 2, 0, 9223372036854775807);  slice_293 = select_scatter_26 = None
    slice_scatter_79: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_292, slice_scatter_78, 1, 0, 9223372036854775807);  slice_292 = slice_scatter_78 = None
    slice_scatter_80: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_scatter_77, slice_scatter_79, 0, 0, 9223372036854775807);  slice_scatter_77 = slice_scatter_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:130, code: return rel_indices.to(device)
    device_put_8: "f32[1, 196, 196, 3]" = torch.ops.prims.device_put.default(slice_scatter_80, device(type='cuda', index=0));  slice_scatter_80 = None
    convert_element_type_8: "f32[1, 196, 196, 3]" = torch.ops.prims.convert_element_type.default(device_put_8, torch.float32);  device_put_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_89: "f32[768, 1536]" = torch.ops.aten.permute.default(arg144_1, [1, 0]);  arg144_1 = None
    view_213: "f32[1568, 768]" = torch.ops.aten.view.default(add_82, [1568, 768])
    mm_24: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_213, permute_89);  view_213 = permute_89 = None
    view_214: "f32[8, 196, 1536]" = torch.ops.aten.view.default(mm_24, [8, 196, 1536]);  mm_24 = None
    view_215: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.view.default(view_214, [8, 196, 2, 16, 48]);  view_214 = None
    permute_90: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.permute.default(view_215, [2, 0, 3, 1, 4]);  view_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    select_88: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_90, 0, 0)
    select_89: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_90, 0, 1);  permute_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:84, code: pos_score = self.rel_indices.expand(B, -1, -1, -1)
    expand_59: "f32[8, 196, 196, 3]" = torch.ops.aten.expand.default(convert_element_type_8, [8, -1, -1, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    permute_91: "f32[3, 16]" = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
    clone_116: "f32[8, 196, 196, 3]" = torch.ops.aten.clone.default(expand_59, memory_format = torch.contiguous_format);  expand_59 = None
    view_216: "f32[307328, 3]" = torch.ops.aten.view.default(clone_116, [307328, 3]);  clone_116 = None
    mm_25: "f32[307328, 16]" = torch.ops.aten.mm.default(view_216, permute_91);  view_216 = permute_91 = None
    view_217: "f32[8, 196, 196, 16]" = torch.ops.aten.view.default(mm_25, [8, 196, 196, 16]);  mm_25 = None
    add_84: "f32[8, 196, 196, 16]" = torch.ops.aten.add.Tensor(view_217, arg146_1);  view_217 = arg146_1 = None
    permute_92: "f32[8, 16, 196, 196]" = torch.ops.aten.permute.default(add_84, [0, 3, 1, 2]);  add_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    permute_93: "f32[8, 16, 48, 196]" = torch.ops.aten.permute.default(select_89, [0, 1, 3, 2]);  select_89 = None
    expand_60: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(select_88, [8, 16, 196, 48]);  select_88 = None
    clone_117: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_60, memory_format = torch.contiguous_format);  expand_60 = None
    view_218: "f32[128, 196, 48]" = torch.ops.aten.view.default(clone_117, [128, 196, 48]);  clone_117 = None
    expand_61: "f32[8, 16, 48, 196]" = torch.ops.aten.expand.default(permute_93, [8, 16, 48, 196]);  permute_93 = None
    clone_118: "f32[8, 16, 48, 196]" = torch.ops.aten.clone.default(expand_61, memory_format = torch.contiguous_format);  expand_61 = None
    view_219: "f32[128, 48, 196]" = torch.ops.aten.view.default(clone_118, [128, 48, 196]);  clone_118 = None
    bmm_16: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_218, view_219);  view_218 = view_219 = None
    view_220: "f32[8, 16, 196, 196]" = torch.ops.aten.view.default(bmm_16, [8, 16, 196, 196]);  bmm_16 = None
    mul_82: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(view_220, 0.14433756729740643);  view_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    amax_16: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(mul_82, [-1], True)
    sub_50: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_82, amax_16);  mul_82 = amax_16 = None
    exp_16: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_50);  sub_50 = None
    sum_25: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
    div_24: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_16, sum_25);  exp_16 = sum_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    clone_119: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(permute_92, memory_format = torch.contiguous_format);  permute_92 = None
    amax_17: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(clone_119, [-1], True)
    sub_51: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(clone_119, amax_17);  clone_119 = amax_17 = None
    exp_17: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_51);  sub_51 = None
    sum_26: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
    div_25: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_17, sum_26);  exp_17 = sum_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_221: "f32[1, 16, 1, 1]" = torch.ops.aten.view.default(arg44_1, [1, -1, 1, 1]);  arg44_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    sigmoid_16: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_221)
    sub_52: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1.0, sigmoid_16);  sigmoid_16 = None
    mul_83: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_52, div_24);  sub_52 = div_24 = None
    sigmoid_17: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_221);  view_221 = None
    mul_84: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sigmoid_17, div_25);  sigmoid_17 = div_25 = None
    add_85: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(mul_83, mul_84);  mul_83 = mul_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    sum_27: "f32[8, 16, 196]" = torch.ops.aten.sum.dim_IntList(add_85, [-1])
    unsqueeze_53: "f32[8, 16, 196, 1]" = torch.ops.aten.unsqueeze.default(sum_27, -1);  sum_27 = None
    div_26: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(add_85, unsqueeze_53);  add_85 = unsqueeze_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:93, code: attn = self.attn_drop(attn)
    clone_120: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(div_26);  div_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_94: "f32[768, 768]" = torch.ops.aten.permute.default(arg147_1, [1, 0]);  arg147_1 = None
    view_222: "f32[1568, 768]" = torch.ops.aten.view.default(add_82, [1568, 768]);  add_82 = None
    mm_26: "f32[1568, 768]" = torch.ops.aten.mm.default(view_222, permute_94);  view_222 = permute_94 = None
    view_223: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_26, [8, 196, 768]);  mm_26 = None
    view_224: "f32[8, 196, 16, 48]" = torch.ops.aten.view.default(view_223, [8, 196, 16, 48]);  view_223 = None
    permute_95: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_224, [0, 2, 1, 3]);  view_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_62: "f32[8, 16, 196, 196]" = torch.ops.aten.expand.default(clone_120, [8, 16, 196, 196]);  clone_120 = None
    view_225: "f32[128, 196, 196]" = torch.ops.aten.view.default(expand_62, [128, 196, 196]);  expand_62 = None
    expand_63: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(permute_95, [8, 16, 196, 48]);  permute_95 = None
    clone_121: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_63, memory_format = torch.contiguous_format);  expand_63 = None
    view_226: "f32[128, 196, 48]" = torch.ops.aten.view.default(clone_121, [128, 196, 48]);  clone_121 = None
    bmm_17: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(view_225, view_226);  view_225 = view_226 = None
    view_227: "f32[8, 16, 196, 48]" = torch.ops.aten.view.default(bmm_17, [8, 16, 196, 48]);  bmm_17 = None
    permute_96: "f32[8, 196, 16, 48]" = torch.ops.aten.permute.default(view_227, [0, 2, 1, 3]);  view_227 = None
    clone_122: "f32[8, 196, 16, 48]" = torch.ops.aten.clone.default(permute_96, memory_format = torch.contiguous_format);  permute_96 = None
    view_228: "f32[8, 196, 768]" = torch.ops.aten.view.default(clone_122, [8, 196, 768]);  clone_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_229: "f32[1568, 768]" = torch.ops.aten.view.default(view_228, [1568, 768]);  view_228 = None
    permute_97: "f32[768, 768]" = torch.ops.aten.permute.default(arg148_1, [1, 0]);  arg148_1 = None
    addmm_24: "f32[1568, 768]" = torch.ops.aten.addmm.default(arg149_1, view_229, permute_97);  arg149_1 = view_229 = permute_97 = None
    view_230: "f32[8, 196, 768]" = torch.ops.aten.view.default(addmm_24, [8, 196, 768]);  addmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:77, code: x = self.proj_drop(x)
    clone_123: "f32[8, 196, 768]" = torch.ops.aten.clone.default(view_230);  view_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:235, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_86: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_80, clone_123);  add_80 = clone_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_124: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_86, memory_format = torch.contiguous_format)
    var_mean_17 = torch.ops.aten.var_mean.correction(clone_124, [2], correction = 0, keepdim = True)
    getitem_34: "f32[8, 196, 1]" = var_mean_17[0]
    getitem_35: "f32[8, 196, 1]" = var_mean_17[1];  var_mean_17 = None
    add_87: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-06);  getitem_34 = None
    rsqrt_17: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
    sub_53: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_124, getitem_35);  clone_124 = getitem_35 = None
    mul_85: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_17);  sub_53 = rsqrt_17 = None
    mul_86: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_85, arg45_1);  mul_85 = arg45_1 = None
    add_88: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_86, arg46_1);  mul_86 = arg46_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_231: "f32[1568, 768]" = torch.ops.aten.view.default(add_88, [1568, 768]);  add_88 = None
    permute_98: "f32[768, 3072]" = torch.ops.aten.permute.default(arg150_1, [1, 0]);  arg150_1 = None
    addmm_25: "f32[1568, 3072]" = torch.ops.aten.addmm.default(arg151_1, view_231, permute_98);  arg151_1 = view_231 = permute_98 = None
    view_232: "f32[8, 196, 3072]" = torch.ops.aten.view.default(addmm_25, [8, 196, 3072]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_87: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_232, 0.5)
    mul_88: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_232, 0.7071067811865476);  view_232 = None
    erf_8: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_88);  mul_88 = None
    add_89: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_89: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_87, add_89);  mul_87 = add_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_125: "f32[8, 196, 3072]" = torch.ops.aten.clone.default(mul_89);  mul_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_233: "f32[1568, 3072]" = torch.ops.aten.view.default(clone_125, [1568, 3072]);  clone_125 = None
    permute_99: "f32[3072, 768]" = torch.ops.aten.permute.default(arg152_1, [1, 0]);  arg152_1 = None
    addmm_26: "f32[1568, 768]" = torch.ops.aten.addmm.default(arg153_1, view_233, permute_99);  arg153_1 = view_233 = permute_99 = None
    view_234: "f32[8, 196, 768]" = torch.ops.aten.view.default(addmm_26, [8, 196, 768]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_126: "f32[8, 196, 768]" = torch.ops.aten.clone.default(view_234);  view_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:236, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_90: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_86, clone_126);  add_86 = clone_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_127: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_90, memory_format = torch.contiguous_format)
    var_mean_18 = torch.ops.aten.var_mean.correction(clone_127, [2], correction = 0, keepdim = True)
    getitem_36: "f32[8, 196, 1]" = var_mean_18[0]
    getitem_37: "f32[8, 196, 1]" = var_mean_18[1];  var_mean_18 = None
    add_91: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-06);  getitem_36 = None
    rsqrt_18: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_91);  add_91 = None
    sub_54: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_127, getitem_37);  clone_127 = getitem_37 = None
    mul_90: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_18);  sub_54 = rsqrt_18 = None
    mul_91: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_90, arg47_1);  mul_90 = arg47_1 = None
    add_92: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_91, arg48_1);  mul_91 = arg48_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:121, code: rel_indices = torch.zeros(1, num_patches, num_patches, 3)
    full_9: "f32[1, 196, 196, 3]" = torch.ops.aten.full.default([1, 196, 196, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:122, code: ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
    iota_18: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_235: "i64[1, 14]" = torch.ops.aten.view.default(iota_18, [1, -1]);  iota_18 = None
    iota_19: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_236: "i64[14, 1]" = torch.ops.aten.view.default(iota_19, [-1, 1]);  iota_19 = None
    sub_55: "i64[14, 14]" = torch.ops.aten.sub.Tensor(view_235, view_236);  view_235 = view_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:123, code: indx = ind.repeat(img_size, img_size)
    repeat_9: "i64[196, 196]" = torch.ops.aten.repeat.default(sub_55, [14, 14])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:124, code: indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
    unsqueeze_54: "i64[14, 1, 14]" = torch.ops.aten.unsqueeze.default(sub_55, 1);  sub_55 = None
    expand_64: "i64[14, 14, 14]" = torch.ops.aten.expand.default(unsqueeze_54, [14, 14, 14]);  unsqueeze_54 = None
    clone_128: "i64[14, 14, 14]" = torch.ops.aten.clone.default(expand_64, memory_format = torch.contiguous_format);  expand_64 = None
    view_237: "i64[196, 14]" = torch.ops.aten.view.default(clone_128, [196, 14]);  clone_128 = None
    unsqueeze_55: "i64[196, 14, 1]" = torch.ops.aten.unsqueeze.default(view_237, 2);  view_237 = None
    expand_65: "i64[196, 14, 14]" = torch.ops.aten.expand.default(unsqueeze_55, [196, 14, 14]);  unsqueeze_55 = None
    clone_129: "i64[196, 14, 14]" = torch.ops.aten.clone.default(expand_65, memory_format = torch.contiguous_format);  expand_65 = None
    view_238: "i64[196, 196]" = torch.ops.aten.view.default(clone_129, [196, 196]);  clone_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:125, code: indd = indx ** 2 + indy ** 2
    pow_19: "i64[196, 196]" = torch.ops.aten.pow.Tensor_Scalar(repeat_9, 2)
    pow_20: "i64[196, 196]" = torch.ops.aten.pow.Tensor_Scalar(view_238, 2)
    add_93: "i64[196, 196]" = torch.ops.aten.add.Tensor(pow_19, pow_20);  pow_19 = pow_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:126, code: rel_indices[:, :, :, 2] = indd.unsqueeze(0)
    unsqueeze_56: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(add_93, 0);  add_93 = None
    slice_298: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(full_9, 0, 0, 9223372036854775807)
    slice_299: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_298, 1, 0, 9223372036854775807);  slice_298 = None
    slice_300: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_299, 2, 0, 9223372036854775807);  slice_299 = None
    select_90: "f32[1, 196, 196]" = torch.ops.aten.select.int(slice_300, 3, 2);  slice_300 = None
    copy_27: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_90, unsqueeze_56);  select_90 = unsqueeze_56 = None
    slice_301: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(full_9, 0, 0, 9223372036854775807)
    slice_302: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_301, 1, 0, 9223372036854775807)
    slice_303: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_302, 2, 0, 9223372036854775807)
    select_scatter_27: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(slice_303, copy_27, 3, 2);  slice_303 = copy_27 = None
    slice_scatter_81: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_302, select_scatter_27, 2, 0, 9223372036854775807);  slice_302 = select_scatter_27 = None
    slice_scatter_82: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_301, slice_scatter_81, 1, 0, 9223372036854775807);  slice_301 = slice_scatter_81 = None
    slice_scatter_83: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(full_9, slice_scatter_82, 0, 0, 9223372036854775807);  full_9 = slice_scatter_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:127, code: rel_indices[:, :, :, 1] = indy.unsqueeze(0)
    unsqueeze_57: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(view_238, 0);  view_238 = None
    slice_310: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_83, 0, 0, 9223372036854775807)
    slice_311: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_310, 1, 0, 9223372036854775807);  slice_310 = None
    slice_312: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_311, 2, 0, 9223372036854775807);  slice_311 = None
    select_93: "f32[1, 196, 196]" = torch.ops.aten.select.int(slice_312, 3, 1);  slice_312 = None
    copy_28: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_93, unsqueeze_57);  select_93 = unsqueeze_57 = None
    slice_313: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_83, 0, 0, 9223372036854775807)
    slice_314: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_313, 1, 0, 9223372036854775807)
    slice_315: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_314, 2, 0, 9223372036854775807)
    select_scatter_28: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(slice_315, copy_28, 3, 1);  slice_315 = copy_28 = None
    slice_scatter_84: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_314, select_scatter_28, 2, 0, 9223372036854775807);  slice_314 = select_scatter_28 = None
    slice_scatter_85: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_313, slice_scatter_84, 1, 0, 9223372036854775807);  slice_313 = slice_scatter_84 = None
    slice_scatter_86: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_scatter_83, slice_scatter_85, 0, 0, 9223372036854775807);  slice_scatter_83 = slice_scatter_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:128, code: rel_indices[:, :, :, 0] = indx.unsqueeze(0)
    unsqueeze_58: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(repeat_9, 0);  repeat_9 = None
    slice_322: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_86, 0, 0, 9223372036854775807)
    slice_323: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_322, 1, 0, 9223372036854775807);  slice_322 = None
    slice_324: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_323, 2, 0, 9223372036854775807);  slice_323 = None
    select_96: "f32[1, 196, 196]" = torch.ops.aten.select.int(slice_324, 3, 0);  slice_324 = None
    copy_29: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_96, unsqueeze_58);  select_96 = unsqueeze_58 = None
    slice_325: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_86, 0, 0, 9223372036854775807)
    slice_326: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_325, 1, 0, 9223372036854775807)
    slice_327: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_326, 2, 0, 9223372036854775807)
    select_scatter_29: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(slice_327, copy_29, 3, 0);  slice_327 = copy_29 = None
    slice_scatter_87: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_326, select_scatter_29, 2, 0, 9223372036854775807);  slice_326 = select_scatter_29 = None
    slice_scatter_88: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_325, slice_scatter_87, 1, 0, 9223372036854775807);  slice_325 = slice_scatter_87 = None
    slice_scatter_89: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_scatter_86, slice_scatter_88, 0, 0, 9223372036854775807);  slice_scatter_86 = slice_scatter_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:130, code: return rel_indices.to(device)
    device_put_9: "f32[1, 196, 196, 3]" = torch.ops.prims.device_put.default(slice_scatter_89, device(type='cuda', index=0));  slice_scatter_89 = None
    convert_element_type_9: "f32[1, 196, 196, 3]" = torch.ops.prims.convert_element_type.default(device_put_9, torch.float32);  device_put_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_100: "f32[768, 1536]" = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
    view_239: "f32[1568, 768]" = torch.ops.aten.view.default(add_92, [1568, 768])
    mm_27: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_239, permute_100);  view_239 = permute_100 = None
    view_240: "f32[8, 196, 1536]" = torch.ops.aten.view.default(mm_27, [8, 196, 1536]);  mm_27 = None
    view_241: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.view.default(view_240, [8, 196, 2, 16, 48]);  view_240 = None
    permute_101: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.permute.default(view_241, [2, 0, 3, 1, 4]);  view_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    select_98: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_101, 0, 0)
    select_99: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_101, 0, 1);  permute_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:84, code: pos_score = self.rel_indices.expand(B, -1, -1, -1)
    expand_66: "f32[8, 196, 196, 3]" = torch.ops.aten.expand.default(convert_element_type_9, [8, -1, -1, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    permute_102: "f32[3, 16]" = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
    clone_130: "f32[8, 196, 196, 3]" = torch.ops.aten.clone.default(expand_66, memory_format = torch.contiguous_format);  expand_66 = None
    view_242: "f32[307328, 3]" = torch.ops.aten.view.default(clone_130, [307328, 3]);  clone_130 = None
    mm_28: "f32[307328, 16]" = torch.ops.aten.mm.default(view_242, permute_102);  view_242 = permute_102 = None
    view_243: "f32[8, 196, 196, 16]" = torch.ops.aten.view.default(mm_28, [8, 196, 196, 16]);  mm_28 = None
    add_94: "f32[8, 196, 196, 16]" = torch.ops.aten.add.Tensor(view_243, arg156_1);  view_243 = arg156_1 = None
    permute_103: "f32[8, 16, 196, 196]" = torch.ops.aten.permute.default(add_94, [0, 3, 1, 2]);  add_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    permute_104: "f32[8, 16, 48, 196]" = torch.ops.aten.permute.default(select_99, [0, 1, 3, 2]);  select_99 = None
    expand_67: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(select_98, [8, 16, 196, 48]);  select_98 = None
    clone_131: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_67, memory_format = torch.contiguous_format);  expand_67 = None
    view_244: "f32[128, 196, 48]" = torch.ops.aten.view.default(clone_131, [128, 196, 48]);  clone_131 = None
    expand_68: "f32[8, 16, 48, 196]" = torch.ops.aten.expand.default(permute_104, [8, 16, 48, 196]);  permute_104 = None
    clone_132: "f32[8, 16, 48, 196]" = torch.ops.aten.clone.default(expand_68, memory_format = torch.contiguous_format);  expand_68 = None
    view_245: "f32[128, 48, 196]" = torch.ops.aten.view.default(clone_132, [128, 48, 196]);  clone_132 = None
    bmm_18: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_244, view_245);  view_244 = view_245 = None
    view_246: "f32[8, 16, 196, 196]" = torch.ops.aten.view.default(bmm_18, [8, 16, 196, 196]);  bmm_18 = None
    mul_92: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(view_246, 0.14433756729740643);  view_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    amax_18: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(mul_92, [-1], True)
    sub_56: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_92, amax_18);  mul_92 = amax_18 = None
    exp_18: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_56);  sub_56 = None
    sum_28: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
    div_27: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_18, sum_28);  exp_18 = sum_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    clone_133: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(permute_103, memory_format = torch.contiguous_format);  permute_103 = None
    amax_19: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(clone_133, [-1], True)
    sub_57: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(clone_133, amax_19);  clone_133 = amax_19 = None
    exp_19: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_57);  sub_57 = None
    sum_29: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
    div_28: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_19, sum_29);  exp_19 = sum_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_247: "f32[1, 16, 1, 1]" = torch.ops.aten.view.default(arg49_1, [1, -1, 1, 1]);  arg49_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    sigmoid_18: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_247)
    sub_58: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1.0, sigmoid_18);  sigmoid_18 = None
    mul_93: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_58, div_27);  sub_58 = div_27 = None
    sigmoid_19: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_247);  view_247 = None
    mul_94: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sigmoid_19, div_28);  sigmoid_19 = div_28 = None
    add_95: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(mul_93, mul_94);  mul_93 = mul_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    sum_30: "f32[8, 16, 196]" = torch.ops.aten.sum.dim_IntList(add_95, [-1])
    unsqueeze_59: "f32[8, 16, 196, 1]" = torch.ops.aten.unsqueeze.default(sum_30, -1);  sum_30 = None
    div_29: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(add_95, unsqueeze_59);  add_95 = unsqueeze_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:93, code: attn = self.attn_drop(attn)
    clone_134: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(div_29);  div_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_105: "f32[768, 768]" = torch.ops.aten.permute.default(arg157_1, [1, 0]);  arg157_1 = None
    view_248: "f32[1568, 768]" = torch.ops.aten.view.default(add_92, [1568, 768]);  add_92 = None
    mm_29: "f32[1568, 768]" = torch.ops.aten.mm.default(view_248, permute_105);  view_248 = permute_105 = None
    view_249: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_29, [8, 196, 768]);  mm_29 = None
    view_250: "f32[8, 196, 16, 48]" = torch.ops.aten.view.default(view_249, [8, 196, 16, 48]);  view_249 = None
    permute_106: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_250, [0, 2, 1, 3]);  view_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_69: "f32[8, 16, 196, 196]" = torch.ops.aten.expand.default(clone_134, [8, 16, 196, 196]);  clone_134 = None
    view_251: "f32[128, 196, 196]" = torch.ops.aten.view.default(expand_69, [128, 196, 196]);  expand_69 = None
    expand_70: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(permute_106, [8, 16, 196, 48]);  permute_106 = None
    clone_135: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_70, memory_format = torch.contiguous_format);  expand_70 = None
    view_252: "f32[128, 196, 48]" = torch.ops.aten.view.default(clone_135, [128, 196, 48]);  clone_135 = None
    bmm_19: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(view_251, view_252);  view_251 = view_252 = None
    view_253: "f32[8, 16, 196, 48]" = torch.ops.aten.view.default(bmm_19, [8, 16, 196, 48]);  bmm_19 = None
    permute_107: "f32[8, 196, 16, 48]" = torch.ops.aten.permute.default(view_253, [0, 2, 1, 3]);  view_253 = None
    clone_136: "f32[8, 196, 16, 48]" = torch.ops.aten.clone.default(permute_107, memory_format = torch.contiguous_format);  permute_107 = None
    view_254: "f32[8, 196, 768]" = torch.ops.aten.view.default(clone_136, [8, 196, 768]);  clone_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_255: "f32[1568, 768]" = torch.ops.aten.view.default(view_254, [1568, 768]);  view_254 = None
    permute_108: "f32[768, 768]" = torch.ops.aten.permute.default(arg158_1, [1, 0]);  arg158_1 = None
    addmm_27: "f32[1568, 768]" = torch.ops.aten.addmm.default(arg159_1, view_255, permute_108);  arg159_1 = view_255 = permute_108 = None
    view_256: "f32[8, 196, 768]" = torch.ops.aten.view.default(addmm_27, [8, 196, 768]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:77, code: x = self.proj_drop(x)
    clone_137: "f32[8, 196, 768]" = torch.ops.aten.clone.default(view_256);  view_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:235, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_96: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_90, clone_137);  add_90 = clone_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_138: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_96, memory_format = torch.contiguous_format)
    var_mean_19 = torch.ops.aten.var_mean.correction(clone_138, [2], correction = 0, keepdim = True)
    getitem_38: "f32[8, 196, 1]" = var_mean_19[0]
    getitem_39: "f32[8, 196, 1]" = var_mean_19[1];  var_mean_19 = None
    add_97: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-06);  getitem_38 = None
    rsqrt_19: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_97);  add_97 = None
    sub_59: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_138, getitem_39);  clone_138 = getitem_39 = None
    mul_95: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_19);  sub_59 = rsqrt_19 = None
    mul_96: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_95, arg50_1);  mul_95 = arg50_1 = None
    add_98: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_96, arg51_1);  mul_96 = arg51_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_257: "f32[1568, 768]" = torch.ops.aten.view.default(add_98, [1568, 768]);  add_98 = None
    permute_109: "f32[768, 3072]" = torch.ops.aten.permute.default(arg160_1, [1, 0]);  arg160_1 = None
    addmm_28: "f32[1568, 3072]" = torch.ops.aten.addmm.default(arg161_1, view_257, permute_109);  arg161_1 = view_257 = permute_109 = None
    view_258: "f32[8, 196, 3072]" = torch.ops.aten.view.default(addmm_28, [8, 196, 3072]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_97: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_258, 0.5)
    mul_98: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_258, 0.7071067811865476);  view_258 = None
    erf_9: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_98);  mul_98 = None
    add_99: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_99: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_97, add_99);  mul_97 = add_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_139: "f32[8, 196, 3072]" = torch.ops.aten.clone.default(mul_99);  mul_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_259: "f32[1568, 3072]" = torch.ops.aten.view.default(clone_139, [1568, 3072]);  clone_139 = None
    permute_110: "f32[3072, 768]" = torch.ops.aten.permute.default(arg162_1, [1, 0]);  arg162_1 = None
    addmm_29: "f32[1568, 768]" = torch.ops.aten.addmm.default(arg163_1, view_259, permute_110);  arg163_1 = view_259 = permute_110 = None
    view_260: "f32[8, 196, 768]" = torch.ops.aten.view.default(addmm_29, [8, 196, 768]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_140: "f32[8, 196, 768]" = torch.ops.aten.clone.default(view_260);  view_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:236, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_100: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_96, clone_140);  add_96 = clone_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:367, code: x = torch.cat((cls_tokens, x), dim=1)
    cat: "f32[8, 197, 768]" = torch.ops.aten.cat.default([expand, add_100], 1);  expand = add_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_20 = torch.ops.aten.var_mean.correction(cat, [2], correction = 0, keepdim = True)
    getitem_40: "f32[8, 197, 1]" = var_mean_20[0]
    getitem_41: "f32[8, 197, 1]" = var_mean_20[1];  var_mean_20 = None
    add_101: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-06);  getitem_40 = None
    rsqrt_20: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_101);  add_101 = None
    sub_60: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(cat, getitem_41);  getitem_41 = None
    mul_100: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_20);  sub_60 = rsqrt_20 = None
    mul_101: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_100, arg52_1);  mul_100 = arg52_1 = None
    add_102: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_101, arg53_1);  mul_101 = arg53_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:175, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_111: "f32[768, 2304]" = torch.ops.aten.permute.default(arg164_1, [1, 0]);  arg164_1 = None
    view_261: "f32[1576, 768]" = torch.ops.aten.view.default(add_102, [1576, 768]);  add_102 = None
    mm_30: "f32[1576, 2304]" = torch.ops.aten.mm.default(view_261, permute_111);  view_261 = permute_111 = None
    view_262: "f32[8, 197, 2304]" = torch.ops.aten.view.default(mm_30, [8, 197, 2304]);  mm_30 = None
    view_263: "f32[8, 197, 3, 16, 48]" = torch.ops.aten.view.default(view_262, [8, 197, 3, 16, 48]);  view_262 = None
    permute_112: "f32[3, 8, 16, 197, 48]" = torch.ops.aten.permute.default(view_263, [2, 0, 3, 1, 4]);  view_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:176, code: q, k, v = qkv.unbind(0)
    unbind = torch.ops.aten.unbind.int(permute_112);  permute_112 = None
    getitem_42: "f32[8, 16, 197, 48]" = unbind[0]
    getitem_43: "f32[8, 16, 197, 48]" = unbind[1]
    getitem_44: "f32[8, 16, 197, 48]" = unbind[2];  unbind = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:178, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_113: "f32[8, 16, 48, 197]" = torch.ops.aten.permute.default(getitem_43, [0, 1, 3, 2]);  getitem_43 = None
    expand_71: "f32[8, 16, 197, 48]" = torch.ops.aten.expand.default(getitem_42, [8, 16, 197, 48]);  getitem_42 = None
    clone_141: "f32[8, 16, 197, 48]" = torch.ops.aten.clone.default(expand_71, memory_format = torch.contiguous_format);  expand_71 = None
    view_264: "f32[128, 197, 48]" = torch.ops.aten.view.default(clone_141, [128, 197, 48]);  clone_141 = None
    expand_72: "f32[8, 16, 48, 197]" = torch.ops.aten.expand.default(permute_113, [8, 16, 48, 197]);  permute_113 = None
    clone_142: "f32[8, 16, 48, 197]" = torch.ops.aten.clone.default(expand_72, memory_format = torch.contiguous_format);  expand_72 = None
    view_265: "f32[128, 48, 197]" = torch.ops.aten.view.default(clone_142, [128, 48, 197]);  clone_142 = None
    bmm_20: "f32[128, 197, 197]" = torch.ops.aten.bmm.default(view_264, view_265);  view_264 = view_265 = None
    view_266: "f32[8, 16, 197, 197]" = torch.ops.aten.view.default(bmm_20, [8, 16, 197, 197]);  bmm_20 = None
    mul_102: "f32[8, 16, 197, 197]" = torch.ops.aten.mul.Tensor(view_266, 0.14433756729740643);  view_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:179, code: attn = attn.softmax(dim=-1)
    amax_20: "f32[8, 16, 197, 1]" = torch.ops.aten.amax.default(mul_102, [-1], True)
    sub_61: "f32[8, 16, 197, 197]" = torch.ops.aten.sub.Tensor(mul_102, amax_20);  mul_102 = amax_20 = None
    exp_20: "f32[8, 16, 197, 197]" = torch.ops.aten.exp.default(sub_61);  sub_61 = None
    sum_31: "f32[8, 16, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
    div_30: "f32[8, 16, 197, 197]" = torch.ops.aten.div.Tensor(exp_20, sum_31);  exp_20 = sum_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:180, code: attn = self.attn_drop(attn)
    clone_143: "f32[8, 16, 197, 197]" = torch.ops.aten.clone.default(div_30);  div_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:182, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_73: "f32[8, 16, 197, 197]" = torch.ops.aten.expand.default(clone_143, [8, 16, 197, 197]);  clone_143 = None
    view_267: "f32[128, 197, 197]" = torch.ops.aten.view.default(expand_73, [128, 197, 197]);  expand_73 = None
    expand_74: "f32[8, 16, 197, 48]" = torch.ops.aten.expand.default(getitem_44, [8, 16, 197, 48]);  getitem_44 = None
    clone_144: "f32[8, 16, 197, 48]" = torch.ops.aten.clone.default(expand_74, memory_format = torch.contiguous_format);  expand_74 = None
    view_268: "f32[128, 197, 48]" = torch.ops.aten.view.default(clone_144, [128, 197, 48]);  clone_144 = None
    bmm_21: "f32[128, 197, 48]" = torch.ops.aten.bmm.default(view_267, view_268);  view_267 = view_268 = None
    view_269: "f32[8, 16, 197, 48]" = torch.ops.aten.view.default(bmm_21, [8, 16, 197, 48]);  bmm_21 = None
    permute_114: "f32[8, 197, 16, 48]" = torch.ops.aten.permute.default(view_269, [0, 2, 1, 3]);  view_269 = None
    clone_145: "f32[8, 197, 16, 48]" = torch.ops.aten.clone.default(permute_114, memory_format = torch.contiguous_format);  permute_114 = None
    view_270: "f32[8, 197, 768]" = torch.ops.aten.view.default(clone_145, [8, 197, 768]);  clone_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:183, code: x = self.proj(x)
    view_271: "f32[1576, 768]" = torch.ops.aten.view.default(view_270, [1576, 768]);  view_270 = None
    permute_115: "f32[768, 768]" = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
    addmm_30: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg166_1, view_271, permute_115);  arg166_1 = view_271 = permute_115 = None
    view_272: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_30, [8, 197, 768]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:184, code: x = self.proj_drop(x)
    clone_146: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_272);  view_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:235, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_103: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(cat, clone_146);  cat = clone_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_21 = torch.ops.aten.var_mean.correction(add_103, [2], correction = 0, keepdim = True)
    getitem_45: "f32[8, 197, 1]" = var_mean_21[0]
    getitem_46: "f32[8, 197, 1]" = var_mean_21[1];  var_mean_21 = None
    add_104: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_45, 1e-06);  getitem_45 = None
    rsqrt_21: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_104);  add_104 = None
    sub_62: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_103, getitem_46);  getitem_46 = None
    mul_103: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_21);  sub_62 = rsqrt_21 = None
    mul_104: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_103, arg54_1);  mul_103 = arg54_1 = None
    add_105: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_104, arg55_1);  mul_104 = arg55_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_273: "f32[1576, 768]" = torch.ops.aten.view.default(add_105, [1576, 768]);  add_105 = None
    permute_116: "f32[768, 3072]" = torch.ops.aten.permute.default(arg167_1, [1, 0]);  arg167_1 = None
    addmm_31: "f32[1576, 3072]" = torch.ops.aten.addmm.default(arg168_1, view_273, permute_116);  arg168_1 = view_273 = permute_116 = None
    view_274: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_31, [8, 197, 3072]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_105: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_274, 0.5)
    mul_106: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_274, 0.7071067811865476);  view_274 = None
    erf_10: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_106);  mul_106 = None
    add_106: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_107: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_105, add_106);  mul_105 = add_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_147: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_107);  mul_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_275: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_147, [1576, 3072]);  clone_147 = None
    permute_117: "f32[3072, 768]" = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
    addmm_32: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg170_1, view_275, permute_117);  arg170_1 = view_275 = permute_117 = None
    view_276: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_32, [8, 197, 768]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_148: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_276);  view_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:236, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_107: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_103, clone_148);  add_103 = clone_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_22 = torch.ops.aten.var_mean.correction(add_107, [2], correction = 0, keepdim = True)
    getitem_47: "f32[8, 197, 1]" = var_mean_22[0]
    getitem_48: "f32[8, 197, 1]" = var_mean_22[1];  var_mean_22 = None
    add_108: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_47, 1e-06);  getitem_47 = None
    rsqrt_22: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
    sub_63: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_107, getitem_48);  getitem_48 = None
    mul_108: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_22);  sub_63 = rsqrt_22 = None
    mul_109: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_108, arg56_1);  mul_108 = arg56_1 = None
    add_109: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_109, arg57_1);  mul_109 = arg57_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:175, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_118: "f32[768, 2304]" = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
    view_277: "f32[1576, 768]" = torch.ops.aten.view.default(add_109, [1576, 768]);  add_109 = None
    mm_31: "f32[1576, 2304]" = torch.ops.aten.mm.default(view_277, permute_118);  view_277 = permute_118 = None
    view_278: "f32[8, 197, 2304]" = torch.ops.aten.view.default(mm_31, [8, 197, 2304]);  mm_31 = None
    view_279: "f32[8, 197, 3, 16, 48]" = torch.ops.aten.view.default(view_278, [8, 197, 3, 16, 48]);  view_278 = None
    permute_119: "f32[3, 8, 16, 197, 48]" = torch.ops.aten.permute.default(view_279, [2, 0, 3, 1, 4]);  view_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:176, code: q, k, v = qkv.unbind(0)
    unbind_1 = torch.ops.aten.unbind.int(permute_119);  permute_119 = None
    getitem_49: "f32[8, 16, 197, 48]" = unbind_1[0]
    getitem_50: "f32[8, 16, 197, 48]" = unbind_1[1]
    getitem_51: "f32[8, 16, 197, 48]" = unbind_1[2];  unbind_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:178, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_120: "f32[8, 16, 48, 197]" = torch.ops.aten.permute.default(getitem_50, [0, 1, 3, 2]);  getitem_50 = None
    expand_75: "f32[8, 16, 197, 48]" = torch.ops.aten.expand.default(getitem_49, [8, 16, 197, 48]);  getitem_49 = None
    clone_149: "f32[8, 16, 197, 48]" = torch.ops.aten.clone.default(expand_75, memory_format = torch.contiguous_format);  expand_75 = None
    view_280: "f32[128, 197, 48]" = torch.ops.aten.view.default(clone_149, [128, 197, 48]);  clone_149 = None
    expand_76: "f32[8, 16, 48, 197]" = torch.ops.aten.expand.default(permute_120, [8, 16, 48, 197]);  permute_120 = None
    clone_150: "f32[8, 16, 48, 197]" = torch.ops.aten.clone.default(expand_76, memory_format = torch.contiguous_format);  expand_76 = None
    view_281: "f32[128, 48, 197]" = torch.ops.aten.view.default(clone_150, [128, 48, 197]);  clone_150 = None
    bmm_22: "f32[128, 197, 197]" = torch.ops.aten.bmm.default(view_280, view_281);  view_280 = view_281 = None
    view_282: "f32[8, 16, 197, 197]" = torch.ops.aten.view.default(bmm_22, [8, 16, 197, 197]);  bmm_22 = None
    mul_110: "f32[8, 16, 197, 197]" = torch.ops.aten.mul.Tensor(view_282, 0.14433756729740643);  view_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:179, code: attn = attn.softmax(dim=-1)
    amax_21: "f32[8, 16, 197, 1]" = torch.ops.aten.amax.default(mul_110, [-1], True)
    sub_64: "f32[8, 16, 197, 197]" = torch.ops.aten.sub.Tensor(mul_110, amax_21);  mul_110 = amax_21 = None
    exp_21: "f32[8, 16, 197, 197]" = torch.ops.aten.exp.default(sub_64);  sub_64 = None
    sum_32: "f32[8, 16, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
    div_31: "f32[8, 16, 197, 197]" = torch.ops.aten.div.Tensor(exp_21, sum_32);  exp_21 = sum_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:180, code: attn = self.attn_drop(attn)
    clone_151: "f32[8, 16, 197, 197]" = torch.ops.aten.clone.default(div_31);  div_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:182, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_77: "f32[8, 16, 197, 197]" = torch.ops.aten.expand.default(clone_151, [8, 16, 197, 197]);  clone_151 = None
    view_283: "f32[128, 197, 197]" = torch.ops.aten.view.default(expand_77, [128, 197, 197]);  expand_77 = None
    expand_78: "f32[8, 16, 197, 48]" = torch.ops.aten.expand.default(getitem_51, [8, 16, 197, 48]);  getitem_51 = None
    clone_152: "f32[8, 16, 197, 48]" = torch.ops.aten.clone.default(expand_78, memory_format = torch.contiguous_format);  expand_78 = None
    view_284: "f32[128, 197, 48]" = torch.ops.aten.view.default(clone_152, [128, 197, 48]);  clone_152 = None
    bmm_23: "f32[128, 197, 48]" = torch.ops.aten.bmm.default(view_283, view_284);  view_283 = view_284 = None
    view_285: "f32[8, 16, 197, 48]" = torch.ops.aten.view.default(bmm_23, [8, 16, 197, 48]);  bmm_23 = None
    permute_121: "f32[8, 197, 16, 48]" = torch.ops.aten.permute.default(view_285, [0, 2, 1, 3]);  view_285 = None
    clone_153: "f32[8, 197, 16, 48]" = torch.ops.aten.clone.default(permute_121, memory_format = torch.contiguous_format);  permute_121 = None
    view_286: "f32[8, 197, 768]" = torch.ops.aten.view.default(clone_153, [8, 197, 768]);  clone_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:183, code: x = self.proj(x)
    view_287: "f32[1576, 768]" = torch.ops.aten.view.default(view_286, [1576, 768]);  view_286 = None
    permute_122: "f32[768, 768]" = torch.ops.aten.permute.default(arg172_1, [1, 0]);  arg172_1 = None
    addmm_33: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg173_1, view_287, permute_122);  arg173_1 = view_287 = permute_122 = None
    view_288: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_33, [8, 197, 768]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:184, code: x = self.proj_drop(x)
    clone_154: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_288);  view_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:235, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_110: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_107, clone_154);  add_107 = clone_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_23 = torch.ops.aten.var_mean.correction(add_110, [2], correction = 0, keepdim = True)
    getitem_52: "f32[8, 197, 1]" = var_mean_23[0]
    getitem_53: "f32[8, 197, 1]" = var_mean_23[1];  var_mean_23 = None
    add_111: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-06);  getitem_52 = None
    rsqrt_23: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
    sub_65: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_110, getitem_53);  getitem_53 = None
    mul_111: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_23);  sub_65 = rsqrt_23 = None
    mul_112: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_111, arg58_1);  mul_111 = arg58_1 = None
    add_112: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_112, arg59_1);  mul_112 = arg59_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_289: "f32[1576, 768]" = torch.ops.aten.view.default(add_112, [1576, 768]);  add_112 = None
    permute_123: "f32[768, 3072]" = torch.ops.aten.permute.default(arg174_1, [1, 0]);  arg174_1 = None
    addmm_34: "f32[1576, 3072]" = torch.ops.aten.addmm.default(arg175_1, view_289, permute_123);  arg175_1 = view_289 = permute_123 = None
    view_290: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_34, [8, 197, 3072]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_113: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_290, 0.5)
    mul_114: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_290, 0.7071067811865476);  view_290 = None
    erf_11: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_114);  mul_114 = None
    add_113: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_115: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_113, add_113);  mul_113 = add_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_155: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_115);  mul_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_291: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_155, [1576, 3072]);  clone_155 = None
    permute_124: "f32[3072, 768]" = torch.ops.aten.permute.default(arg176_1, [1, 0]);  arg176_1 = None
    addmm_35: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg177_1, view_291, permute_124);  arg177_1 = view_291 = permute_124 = None
    view_292: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_35, [8, 197, 768]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_156: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_292);  view_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:236, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_114: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_110, clone_156);  add_110 = clone_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_24 = torch.ops.aten.var_mean.correction(add_114, [2], correction = 0, keepdim = True)
    getitem_54: "f32[8, 197, 1]" = var_mean_24[0]
    getitem_55: "f32[8, 197, 1]" = var_mean_24[1];  var_mean_24 = None
    add_115: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-06);  getitem_54 = None
    rsqrt_24: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
    sub_66: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_114, getitem_55);  add_114 = getitem_55 = None
    mul_116: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_24);  sub_66 = rsqrt_24 = None
    mul_117: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_116, arg60_1);  mul_116 = arg60_1 = None
    add_116: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_117, arg61_1);  mul_117 = arg61_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:374, code: x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
    slice_331: "f32[8, 197, 768]" = torch.ops.aten.slice.Tensor(add_116, 0, 0, 9223372036854775807);  add_116 = None
    select_100: "f32[8, 768]" = torch.ops.aten.select.int(slice_331, 1, 0);  slice_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:375, code: x = self.head_drop(x)
    clone_157: "f32[8, 768]" = torch.ops.aten.clone.default(select_100);  select_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:376, code: return x if pre_logits else self.head(x)
    permute_125: "f32[768, 1000]" = torch.ops.aten.permute.default(arg178_1, [1, 0]);  arg178_1 = None
    addmm_36: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg179_1, clone_157, permute_125);  arg179_1 = clone_157 = permute_125 = None
    return (addmm_36, convert_element_type, convert_element_type_1, convert_element_type_2, convert_element_type_3, convert_element_type_4, convert_element_type_5, convert_element_type_6, convert_element_type_7, convert_element_type_8, convert_element_type_9)
    