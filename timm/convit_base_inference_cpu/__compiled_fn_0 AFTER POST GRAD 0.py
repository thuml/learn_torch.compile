from __future__ import annotations



def forward(self, arg0_1: "f32[1, 196, 768]", arg1_1: "f32[1, 1, 768]", arg2_1: "f32[768]", arg3_1: "f32[768]", arg4_1: "f32[16]", arg5_1: "f32[768]", arg6_1: "f32[768]", arg7_1: "f32[768]", arg8_1: "f32[768]", arg9_1: "f32[16]", arg10_1: "f32[768]", arg11_1: "f32[768]", arg12_1: "f32[768]", arg13_1: "f32[768]", arg14_1: "f32[16]", arg15_1: "f32[768]", arg16_1: "f32[768]", arg17_1: "f32[768]", arg18_1: "f32[768]", arg19_1: "f32[16]", arg20_1: "f32[768]", arg21_1: "f32[768]", arg22_1: "f32[768]", arg23_1: "f32[768]", arg24_1: "f32[16]", arg25_1: "f32[768]", arg26_1: "f32[768]", arg27_1: "f32[768]", arg28_1: "f32[768]", arg29_1: "f32[16]", arg30_1: "f32[768]", arg31_1: "f32[768]", arg32_1: "f32[768]", arg33_1: "f32[768]", arg34_1: "f32[16]", arg35_1: "f32[768]", arg36_1: "f32[768]", arg37_1: "f32[768]", arg38_1: "f32[768]", arg39_1: "f32[16]", arg40_1: "f32[768]", arg41_1: "f32[768]", arg42_1: "f32[768]", arg43_1: "f32[768]", arg44_1: "f32[16]", arg45_1: "f32[768]", arg46_1: "f32[768]", arg47_1: "f32[768]", arg48_1: "f32[768]", arg49_1: "f32[16]", arg50_1: "f32[768]", arg51_1: "f32[768]", arg52_1: "f32[768]", arg53_1: "f32[768]", arg54_1: "f32[768]", arg55_1: "f32[768]", arg56_1: "f32[768]", arg57_1: "f32[768]", arg58_1: "f32[768]", arg59_1: "f32[768]", arg60_1: "f32[768]", arg61_1: "f32[768]", arg62_1: "f32[768, 3, 16, 16]", arg63_1: "f32[768]", arg64_1: "f32[1536, 768]", arg65_1: "f32[16, 3]", arg66_1: "f32[16]", arg67_1: "f32[768, 768]", arg68_1: "f32[768, 768]", arg69_1: "f32[768]", arg70_1: "f32[3072, 768]", arg71_1: "f32[3072]", arg72_1: "f32[768, 3072]", arg73_1: "f32[768]", arg74_1: "f32[1536, 768]", arg75_1: "f32[16, 3]", arg76_1: "f32[16]", arg77_1: "f32[768, 768]", arg78_1: "f32[768, 768]", arg79_1: "f32[768]", arg80_1: "f32[3072, 768]", arg81_1: "f32[3072]", arg82_1: "f32[768, 3072]", arg83_1: "f32[768]", arg84_1: "f32[1536, 768]", arg85_1: "f32[16, 3]", arg86_1: "f32[16]", arg87_1: "f32[768, 768]", arg88_1: "f32[768, 768]", arg89_1: "f32[768]", arg90_1: "f32[3072, 768]", arg91_1: "f32[3072]", arg92_1: "f32[768, 3072]", arg93_1: "f32[768]", arg94_1: "f32[1536, 768]", arg95_1: "f32[16, 3]", arg96_1: "f32[16]", arg97_1: "f32[768, 768]", arg98_1: "f32[768, 768]", arg99_1: "f32[768]", arg100_1: "f32[3072, 768]", arg101_1: "f32[3072]", arg102_1: "f32[768, 3072]", arg103_1: "f32[768]", arg104_1: "f32[1536, 768]", arg105_1: "f32[16, 3]", arg106_1: "f32[16]", arg107_1: "f32[768, 768]", arg108_1: "f32[768, 768]", arg109_1: "f32[768]", arg110_1: "f32[3072, 768]", arg111_1: "f32[3072]", arg112_1: "f32[768, 3072]", arg113_1: "f32[768]", arg114_1: "f32[1536, 768]", arg115_1: "f32[16, 3]", arg116_1: "f32[16]", arg117_1: "f32[768, 768]", arg118_1: "f32[768, 768]", arg119_1: "f32[768]", arg120_1: "f32[3072, 768]", arg121_1: "f32[3072]", arg122_1: "f32[768, 3072]", arg123_1: "f32[768]", arg124_1: "f32[1536, 768]", arg125_1: "f32[16, 3]", arg126_1: "f32[16]", arg127_1: "f32[768, 768]", arg128_1: "f32[768, 768]", arg129_1: "f32[768]", arg130_1: "f32[3072, 768]", arg131_1: "f32[3072]", arg132_1: "f32[768, 3072]", arg133_1: "f32[768]", arg134_1: "f32[1536, 768]", arg135_1: "f32[16, 3]", arg136_1: "f32[16]", arg137_1: "f32[768, 768]", arg138_1: "f32[768, 768]", arg139_1: "f32[768]", arg140_1: "f32[3072, 768]", arg141_1: "f32[3072]", arg142_1: "f32[768, 3072]", arg143_1: "f32[768]", arg144_1: "f32[1536, 768]", arg145_1: "f32[16, 3]", arg146_1: "f32[16]", arg147_1: "f32[768, 768]", arg148_1: "f32[768, 768]", arg149_1: "f32[768]", arg150_1: "f32[3072, 768]", arg151_1: "f32[3072]", arg152_1: "f32[768, 3072]", arg153_1: "f32[768]", arg154_1: "f32[1536, 768]", arg155_1: "f32[16, 3]", arg156_1: "f32[16]", arg157_1: "f32[768, 768]", arg158_1: "f32[768, 768]", arg159_1: "f32[768]", arg160_1: "f32[3072, 768]", arg161_1: "f32[3072]", arg162_1: "f32[768, 3072]", arg163_1: "f32[768]", arg164_1: "f32[2304, 768]", arg165_1: "f32[768, 768]", arg166_1: "f32[768]", arg167_1: "f32[3072, 768]", arg168_1: "f32[3072]", arg169_1: "f32[768, 3072]", arg170_1: "f32[768]", arg171_1: "f32[2304, 768]", arg172_1: "f32[768, 768]", arg173_1: "f32[768]", arg174_1: "f32[3072, 768]", arg175_1: "f32[3072]", arg176_1: "f32[768, 3072]", arg177_1: "f32[768]", arg178_1: "f32[1000, 768]", arg179_1: "f32[1000]", arg180_1: "f32[8, 3, 224, 224]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution: "f32[8, 768, 14, 14]" = torch.ops.aten.convolution.default(arg180_1, arg62_1, arg63_1, [16, 16], [0, 0], [1, 1], False, [0, 0], 1);  arg180_1 = arg62_1 = arg63_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:89, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    view: "f32[8, 768, 196]" = torch.ops.aten.reshape.default(convolution, [8, 768, 196]);  convolution = None
    permute: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:362, code: x = x + self.pos_embed
    add: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(permute, arg0_1);  permute = arg0_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_1: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add, memory_format = torch.contiguous_format)
    var_mean = torch.ops.aten.var_mean.correction(clone_1, [2], correction = 0, keepdim = True)
    getitem: "f32[8, 196, 1]" = var_mean[0]
    getitem_1: "f32[8, 196, 1]" = var_mean[1];  var_mean = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_13: "f32[1, 16, 1, 1]" = torch.ops.aten.reshape.default(arg4_1, [1, -1, 1, 1]);  arg4_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    sigmoid: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_13)
    sub_4: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1.0, sigmoid);  sigmoid = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_1, getitem_1);  clone_1 = getitem_1 = None
    add_1: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
    rsqrt: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    mul: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
    mul_1: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul, arg2_1);  mul = arg2_1 = None
    add_2: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_1, arg3_1);  mul_1 = arg3_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_5: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_2, [1568, 768])
    permute_1: "f32[768, 1536]" = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
    mm: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_5, permute_1);  view_5 = permute_1 = None
    view_6: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(mm, [8, 196, 1536]);  mm = None
    view_7: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.reshape.default(view_6, [8, 196, 2, 16, 48]);  view_6 = None
    permute_2: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.permute.default(view_7, [2, 0, 3, 1, 4]);  view_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    select_8: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_2, 0, 0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    expand_5: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(select_8, [8, 16, 196, 48]);  select_8 = None
    clone_5: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
    view_10: "f32[128, 196, 48]" = torch.ops.aten.reshape.default(clone_5, [128, 196, 48]);  clone_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    select_9: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_2, 0, 1);  permute_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    permute_5: "f32[8, 16, 48, 196]" = torch.ops.aten.permute.default(select_9, [0, 1, 3, 2]);  select_9 = None
    expand_6: "f32[8, 16, 48, 196]" = torch.ops.aten.expand.default(permute_5, [8, 16, 48, 196]);  permute_5 = None
    clone_6: "f32[8, 16, 48, 196]" = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format);  expand_6 = None
    view_11: "f32[128, 48, 196]" = torch.ops.aten.reshape.default(clone_6, [128, 48, 196]);  clone_6 = None
    bmm: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_10, view_11);  view_10 = view_11 = None
    view_12: "f32[8, 16, 196, 196]" = torch.ops.aten.reshape.default(bmm, [8, 16, 196, 196]);  bmm = None
    mul_2: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(view_12, 0.14433756729740643);  view_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    amax: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(mul_2, [-1], True)
    sub_2: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_2, amax);  mul_2 = amax = None
    exp: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
    sum_1: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    mul_3: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_4, div);  sub_4 = div = None
    sigmoid_1: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_13);  view_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:121, code: rel_indices = torch.zeros(1, num_patches, num_patches, 3)
    full: "f32[1, 196, 196, 3]" = torch.ops.aten.full.default([1, 196, 196, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:126, code: rel_indices[:, :, :, 2] = indd.unsqueeze(0)
    select: "f32[1, 196, 196]" = torch.ops.aten.select.int(full, 3, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:122, code: ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
    iota: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_1: "i64[1, 14]" = torch.ops.aten.reshape.default(iota, [1, -1]);  iota = None
    iota_1: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_2: "i64[14, 1]" = torch.ops.aten.reshape.default(iota_1, [-1, 1]);  iota_1 = None
    sub_1: "i64[14, 14]" = torch.ops.aten.sub.Tensor(view_1, view_2);  view_1 = view_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:123, code: indx = ind.repeat(img_size, img_size)
    repeat: "i64[196, 196]" = torch.ops.aten.repeat.default(sub_1, [14, 14])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:125, code: indd = indx ** 2 + indy ** 2
    pow_1: "i64[196, 196]" = torch.ops.aten.pow.Tensor_Scalar(repeat, 2)
    
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
    pow_2: "i64[196, 196]" = torch.ops.aten.pow.Tensor_Scalar(view_4, 2)
    add_3: "i64[196, 196]" = torch.ops.aten.add.Tensor(pow_1, pow_2);  pow_1 = pow_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:126, code: rel_indices[:, :, :, 2] = indd.unsqueeze(0)
    unsqueeze_2: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(add_3, 0);  add_3 = None
    copy: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select, unsqueeze_2);  select = unsqueeze_2 = None
    select_scatter: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(full, copy, 3, 2);  full = copy = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:127, code: rel_indices[:, :, :, 1] = indy.unsqueeze(0)
    select_3: "f32[1, 196, 196]" = torch.ops.aten.select.int(select_scatter, 3, 1)
    unsqueeze_3: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(view_4, 0);  view_4 = None
    copy_1: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_3, unsqueeze_3);  select_3 = unsqueeze_3 = None
    select_scatter_1: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(select_scatter, copy_1, 3, 1);  select_scatter = copy_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:128, code: rel_indices[:, :, :, 0] = indx.unsqueeze(0)
    select_6: "f32[1, 196, 196]" = torch.ops.aten.select.int(select_scatter_1, 3, 0)
    unsqueeze_4: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(repeat, 0);  repeat = None
    copy_2: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_6, unsqueeze_4);  select_6 = unsqueeze_4 = None
    select_scatter_2: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(select_scatter_1, copy_2, 3, 0);  select_scatter_1 = copy_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    expand_4: "f32[8, 196, 196, 3]" = torch.ops.aten.expand.default(select_scatter_2, [8, -1, -1, -1])
    clone_4: "f32[8, 196, 196, 3]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
    view_8: "f32[307328, 3]" = torch.ops.aten.reshape.default(clone_4, [307328, 3]);  clone_4 = None
    permute_3: "f32[3, 16]" = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
    mm_1: "f32[307328, 16]" = torch.ops.aten.mm.default(view_8, permute_3);  view_8 = permute_3 = None
    view_9: "f32[8, 196, 196, 16]" = torch.ops.aten.reshape.default(mm_1, [8, 196, 196, 16]);  mm_1 = None
    add_4: "f32[8, 196, 196, 16]" = torch.ops.aten.add.Tensor(view_9, arg66_1);  view_9 = arg66_1 = None
    permute_4: "f32[8, 16, 196, 196]" = torch.ops.aten.permute.default(add_4, [0, 3, 1, 2]);  add_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    clone_7: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
    amax_1: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(clone_7, [-1], True)
    sub_3: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(clone_7, amax_1);  clone_7 = amax_1 = None
    exp_1: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_3);  sub_3 = None
    sum_2: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_1: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    mul_4: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sigmoid_1, div_1);  sigmoid_1 = div_1 = None
    add_5: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(mul_3, mul_4);  mul_3 = mul_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    sum_3: "f32[8, 16, 196]" = torch.ops.aten.sum.dim_IntList(add_5, [-1])
    unsqueeze_5: "f32[8, 16, 196, 1]" = torch.ops.aten.unsqueeze.default(sum_3, -1);  sum_3 = None
    div_2: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(add_5, unsqueeze_5);  add_5 = unsqueeze_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_7: "f32[8, 16, 196, 196]" = torch.ops.aten.expand.default(div_2, [8, 16, 196, 196]);  div_2 = None
    view_17: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(expand_7, [128, 196, 196]);  expand_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_14: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_2, [1568, 768]);  add_2 = None
    permute_6: "f32[768, 768]" = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
    mm_2: "f32[1568, 768]" = torch.ops.aten.mm.default(view_14, permute_6);  view_14 = permute_6 = None
    view_15: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_2, [8, 196, 768]);  mm_2 = None
    view_16: "f32[8, 196, 16, 48]" = torch.ops.aten.reshape.default(view_15, [8, 196, 16, 48]);  view_15 = None
    permute_7: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_8: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(permute_7, [8, 16, 196, 48]);  permute_7 = None
    clone_9: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
    view_18: "f32[128, 196, 48]" = torch.ops.aten.reshape.default(clone_9, [128, 196, 48]);  clone_9 = None
    bmm_1: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(view_17, view_18);  view_17 = view_18 = None
    view_19: "f32[8, 16, 196, 48]" = torch.ops.aten.reshape.default(bmm_1, [8, 16, 196, 48]);  bmm_1 = None
    permute_8: "f32[8, 196, 16, 48]" = torch.ops.aten.permute.default(view_19, [0, 2, 1, 3]);  view_19 = None
    clone_10: "f32[8, 196, 16, 48]" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None
    view_20: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(clone_10, [8, 196, 768]);  clone_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_21: "f32[1568, 768]" = torch.ops.aten.reshape.default(view_20, [1568, 768]);  view_20 = None
    permute_9: "f32[768, 768]" = torch.ops.aten.permute.default(arg68_1, [1, 0]);  arg68_1 = None
    addmm: "f32[1568, 768]" = torch.ops.aten.addmm.default(arg69_1, view_21, permute_9);  arg69_1 = view_21 = permute_9 = None
    view_22: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(addmm, [8, 196, 768]);  addmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:235, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_6: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add, view_22);  add = view_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_12: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_6, memory_format = torch.contiguous_format)
    var_mean_1 = torch.ops.aten.var_mean.correction(clone_12, [2], correction = 0, keepdim = True)
    getitem_2: "f32[8, 196, 1]" = var_mean_1[0]
    getitem_3: "f32[8, 196, 1]" = var_mean_1[1];  var_mean_1 = None
    sub_5: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_12, getitem_3);  clone_12 = getitem_3 = None
    add_7: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-06);  getitem_2 = None
    rsqrt_1: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
    mul_5: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_1);  sub_5 = rsqrt_1 = None
    mul_6: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_5, arg5_1);  mul_5 = arg5_1 = None
    add_8: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_6, arg6_1);  mul_6 = arg6_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_23: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_8, [1568, 768]);  add_8 = None
    permute_10: "f32[768, 3072]" = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
    addmm_1: "f32[1568, 3072]" = torch.ops.aten.addmm.default(arg71_1, view_23, permute_10);  arg71_1 = view_23 = permute_10 = None
    view_24: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(addmm_1, [8, 196, 3072]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_7: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_24, 0.5)
    mul_8: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_24, 0.7071067811865476);  view_24 = None
    erf: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_8);  mul_8 = None
    add_9: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_9: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_7, add_9);  mul_7 = add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_25: "f32[1568, 3072]" = torch.ops.aten.reshape.default(mul_9, [1568, 3072]);  mul_9 = None
    permute_11: "f32[3072, 768]" = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
    addmm_2: "f32[1568, 768]" = torch.ops.aten.addmm.default(arg73_1, view_25, permute_11);  arg73_1 = view_25 = permute_11 = None
    view_26: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(addmm_2, [8, 196, 768]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:236, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_10: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_6, view_26);  add_6 = view_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_15: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_10, memory_format = torch.contiguous_format)
    var_mean_2 = torch.ops.aten.var_mean.correction(clone_15, [2], correction = 0, keepdim = True)
    getitem_4: "f32[8, 196, 1]" = var_mean_2[0]
    getitem_5: "f32[8, 196, 1]" = var_mean_2[1];  var_mean_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_39: "f32[1, 16, 1, 1]" = torch.ops.aten.reshape.default(arg9_1, [1, -1, 1, 1]);  arg9_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    sigmoid_2: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_39)
    sub_10: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1.0, sigmoid_2);  sigmoid_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_6: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_15, getitem_5);  clone_15 = getitem_5 = None
    add_11: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-06);  getitem_4 = None
    rsqrt_2: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    mul_10: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_2);  sub_6 = rsqrt_2 = None
    mul_11: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_10, arg7_1);  mul_10 = arg7_1 = None
    add_12: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_11, arg8_1);  mul_11 = arg8_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_31: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_12, [1568, 768])
    permute_12: "f32[768, 1536]" = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
    mm_3: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_31, permute_12);  view_31 = permute_12 = None
    view_32: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(mm_3, [8, 196, 1536]);  mm_3 = None
    view_33: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.reshape.default(view_32, [8, 196, 2, 16, 48]);  view_32 = None
    permute_13: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.permute.default(view_33, [2, 0, 3, 1, 4]);  view_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    select_18: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_13, 0, 0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    expand_13: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(select_18, [8, 16, 196, 48]);  select_18 = None
    clone_19: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
    view_36: "f32[128, 196, 48]" = torch.ops.aten.reshape.default(clone_19, [128, 196, 48]);  clone_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    select_19: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_13, 0, 1);  permute_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    permute_16: "f32[8, 16, 48, 196]" = torch.ops.aten.permute.default(select_19, [0, 1, 3, 2]);  select_19 = None
    expand_14: "f32[8, 16, 48, 196]" = torch.ops.aten.expand.default(permute_16, [8, 16, 48, 196]);  permute_16 = None
    clone_20: "f32[8, 16, 48, 196]" = torch.ops.aten.clone.default(expand_14, memory_format = torch.contiguous_format);  expand_14 = None
    view_37: "f32[128, 48, 196]" = torch.ops.aten.reshape.default(clone_20, [128, 48, 196]);  clone_20 = None
    bmm_2: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_36, view_37);  view_36 = view_37 = None
    view_38: "f32[8, 16, 196, 196]" = torch.ops.aten.reshape.default(bmm_2, [8, 16, 196, 196]);  bmm_2 = None
    mul_12: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(view_38, 0.14433756729740643);  view_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    amax_2: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(mul_12, [-1], True)
    sub_8: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_12, amax_2);  mul_12 = amax_2 = None
    exp_2: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_8);  sub_8 = None
    sum_4: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_3: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_2, sum_4);  exp_2 = sum_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    mul_13: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_10, div_3);  sub_10 = div_3 = None
    sigmoid_3: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_39);  view_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:121, code: rel_indices = torch.zeros(1, num_patches, num_patches, 3)
    full_1: "f32[1, 196, 196, 3]" = torch.ops.aten.full.default([1, 196, 196, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:126, code: rel_indices[:, :, :, 2] = indd.unsqueeze(0)
    select_10: "f32[1, 196, 196]" = torch.ops.aten.select.int(full_1, 3, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:122, code: ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
    iota_2: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_27: "i64[1, 14]" = torch.ops.aten.reshape.default(iota_2, [1, -1]);  iota_2 = None
    iota_3: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_28: "i64[14, 1]" = torch.ops.aten.reshape.default(iota_3, [-1, 1]);  iota_3 = None
    sub_7: "i64[14, 14]" = torch.ops.aten.sub.Tensor(view_27, view_28);  view_27 = view_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:123, code: indx = ind.repeat(img_size, img_size)
    repeat_1: "i64[196, 196]" = torch.ops.aten.repeat.default(sub_7, [14, 14])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:125, code: indd = indx ** 2 + indy ** 2
    pow_3: "i64[196, 196]" = torch.ops.aten.pow.Tensor_Scalar(repeat_1, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:124, code: indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
    unsqueeze_6: "i64[14, 1, 14]" = torch.ops.aten.unsqueeze.default(sub_7, 1);  sub_7 = None
    expand_9: "i64[14, 14, 14]" = torch.ops.aten.expand.default(unsqueeze_6, [14, 14, 14]);  unsqueeze_6 = None
    clone_16: "i64[14, 14, 14]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
    view_29: "i64[196, 14]" = torch.ops.aten.reshape.default(clone_16, [196, 14]);  clone_16 = None
    unsqueeze_7: "i64[196, 14, 1]" = torch.ops.aten.unsqueeze.default(view_29, 2);  view_29 = None
    expand_10: "i64[196, 14, 14]" = torch.ops.aten.expand.default(unsqueeze_7, [196, 14, 14]);  unsqueeze_7 = None
    clone_17: "i64[196, 14, 14]" = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
    view_30: "i64[196, 196]" = torch.ops.aten.reshape.default(clone_17, [196, 196]);  clone_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:125, code: indd = indx ** 2 + indy ** 2
    pow_4: "i64[196, 196]" = torch.ops.aten.pow.Tensor_Scalar(view_30, 2)
    add_13: "i64[196, 196]" = torch.ops.aten.add.Tensor(pow_3, pow_4);  pow_3 = pow_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:126, code: rel_indices[:, :, :, 2] = indd.unsqueeze(0)
    unsqueeze_8: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(add_13, 0);  add_13 = None
    copy_3: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_10, unsqueeze_8);  select_10 = unsqueeze_8 = None
    select_scatter_3: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(full_1, copy_3, 3, 2);  full_1 = copy_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:127, code: rel_indices[:, :, :, 1] = indy.unsqueeze(0)
    select_13: "f32[1, 196, 196]" = torch.ops.aten.select.int(select_scatter_3, 3, 1)
    unsqueeze_9: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(view_30, 0);  view_30 = None
    copy_4: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_13, unsqueeze_9);  select_13 = unsqueeze_9 = None
    select_scatter_4: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(select_scatter_3, copy_4, 3, 1);  select_scatter_3 = copy_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:128, code: rel_indices[:, :, :, 0] = indx.unsqueeze(0)
    select_16: "f32[1, 196, 196]" = torch.ops.aten.select.int(select_scatter_4, 3, 0)
    unsqueeze_10: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(repeat_1, 0);  repeat_1 = None
    copy_5: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_16, unsqueeze_10);  select_16 = unsqueeze_10 = None
    select_scatter_5: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(select_scatter_4, copy_5, 3, 0);  select_scatter_4 = copy_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    expand_12: "f32[8, 196, 196, 3]" = torch.ops.aten.expand.default(select_scatter_5, [8, -1, -1, -1])
    clone_18: "f32[8, 196, 196, 3]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
    view_34: "f32[307328, 3]" = torch.ops.aten.reshape.default(clone_18, [307328, 3]);  clone_18 = None
    permute_14: "f32[3, 16]" = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
    mm_4: "f32[307328, 16]" = torch.ops.aten.mm.default(view_34, permute_14);  view_34 = permute_14 = None
    view_35: "f32[8, 196, 196, 16]" = torch.ops.aten.reshape.default(mm_4, [8, 196, 196, 16]);  mm_4 = None
    add_14: "f32[8, 196, 196, 16]" = torch.ops.aten.add.Tensor(view_35, arg76_1);  view_35 = arg76_1 = None
    permute_15: "f32[8, 16, 196, 196]" = torch.ops.aten.permute.default(add_14, [0, 3, 1, 2]);  add_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    clone_21: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
    amax_3: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(clone_21, [-1], True)
    sub_9: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(clone_21, amax_3);  clone_21 = amax_3 = None
    exp_3: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_9);  sub_9 = None
    sum_5: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_4: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_3, sum_5);  exp_3 = sum_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    mul_14: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sigmoid_3, div_4);  sigmoid_3 = div_4 = None
    add_15: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(mul_13, mul_14);  mul_13 = mul_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    sum_6: "f32[8, 16, 196]" = torch.ops.aten.sum.dim_IntList(add_15, [-1])
    unsqueeze_11: "f32[8, 16, 196, 1]" = torch.ops.aten.unsqueeze.default(sum_6, -1);  sum_6 = None
    div_5: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(add_15, unsqueeze_11);  add_15 = unsqueeze_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_15: "f32[8, 16, 196, 196]" = torch.ops.aten.expand.default(div_5, [8, 16, 196, 196]);  div_5 = None
    view_43: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(expand_15, [128, 196, 196]);  expand_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_40: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_12, [1568, 768]);  add_12 = None
    permute_17: "f32[768, 768]" = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
    mm_5: "f32[1568, 768]" = torch.ops.aten.mm.default(view_40, permute_17);  view_40 = permute_17 = None
    view_41: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_5, [8, 196, 768]);  mm_5 = None
    view_42: "f32[8, 196, 16, 48]" = torch.ops.aten.reshape.default(view_41, [8, 196, 16, 48]);  view_41 = None
    permute_18: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_42, [0, 2, 1, 3]);  view_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_16: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(permute_18, [8, 16, 196, 48]);  permute_18 = None
    clone_23: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
    view_44: "f32[128, 196, 48]" = torch.ops.aten.reshape.default(clone_23, [128, 196, 48]);  clone_23 = None
    bmm_3: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(view_43, view_44);  view_43 = view_44 = None
    view_45: "f32[8, 16, 196, 48]" = torch.ops.aten.reshape.default(bmm_3, [8, 16, 196, 48]);  bmm_3 = None
    permute_19: "f32[8, 196, 16, 48]" = torch.ops.aten.permute.default(view_45, [0, 2, 1, 3]);  view_45 = None
    clone_24: "f32[8, 196, 16, 48]" = torch.ops.aten.clone.default(permute_19, memory_format = torch.contiguous_format);  permute_19 = None
    view_46: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(clone_24, [8, 196, 768]);  clone_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_47: "f32[1568, 768]" = torch.ops.aten.reshape.default(view_46, [1568, 768]);  view_46 = None
    permute_20: "f32[768, 768]" = torch.ops.aten.permute.default(arg78_1, [1, 0]);  arg78_1 = None
    addmm_3: "f32[1568, 768]" = torch.ops.aten.addmm.default(arg79_1, view_47, permute_20);  arg79_1 = view_47 = permute_20 = None
    view_48: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(addmm_3, [8, 196, 768]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:235, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_16: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_10, view_48);  add_10 = view_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_26: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_16, memory_format = torch.contiguous_format)
    var_mean_3 = torch.ops.aten.var_mean.correction(clone_26, [2], correction = 0, keepdim = True)
    getitem_6: "f32[8, 196, 1]" = var_mean_3[0]
    getitem_7: "f32[8, 196, 1]" = var_mean_3[1];  var_mean_3 = None
    sub_11: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_26, getitem_7);  clone_26 = getitem_7 = None
    add_17: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-06);  getitem_6 = None
    rsqrt_3: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
    mul_15: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_3);  sub_11 = rsqrt_3 = None
    mul_16: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_15, arg10_1);  mul_15 = arg10_1 = None
    add_18: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_16, arg11_1);  mul_16 = arg11_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_49: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_18, [1568, 768]);  add_18 = None
    permute_21: "f32[768, 3072]" = torch.ops.aten.permute.default(arg80_1, [1, 0]);  arg80_1 = None
    addmm_4: "f32[1568, 3072]" = torch.ops.aten.addmm.default(arg81_1, view_49, permute_21);  arg81_1 = view_49 = permute_21 = None
    view_50: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(addmm_4, [8, 196, 3072]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_17: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_50, 0.5)
    mul_18: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_50, 0.7071067811865476);  view_50 = None
    erf_1: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_18);  mul_18 = None
    add_19: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_19: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_17, add_19);  mul_17 = add_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_51: "f32[1568, 3072]" = torch.ops.aten.reshape.default(mul_19, [1568, 3072]);  mul_19 = None
    permute_22: "f32[3072, 768]" = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
    addmm_5: "f32[1568, 768]" = torch.ops.aten.addmm.default(arg83_1, view_51, permute_22);  arg83_1 = view_51 = permute_22 = None
    view_52: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(addmm_5, [8, 196, 768]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:236, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_20: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_16, view_52);  add_16 = view_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_29: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_20, memory_format = torch.contiguous_format)
    var_mean_4 = torch.ops.aten.var_mean.correction(clone_29, [2], correction = 0, keepdim = True)
    getitem_8: "f32[8, 196, 1]" = var_mean_4[0]
    getitem_9: "f32[8, 196, 1]" = var_mean_4[1];  var_mean_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_65: "f32[1, 16, 1, 1]" = torch.ops.aten.reshape.default(arg14_1, [1, -1, 1, 1]);  arg14_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    sigmoid_4: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_65)
    sub_16: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1.0, sigmoid_4);  sigmoid_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_12: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_29, getitem_9);  clone_29 = getitem_9 = None
    add_21: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-06);  getitem_8 = None
    rsqrt_4: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
    mul_20: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_4);  sub_12 = rsqrt_4 = None
    mul_21: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_20, arg12_1);  mul_20 = arg12_1 = None
    add_22: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_21, arg13_1);  mul_21 = arg13_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_57: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_22, [1568, 768])
    permute_23: "f32[768, 1536]" = torch.ops.aten.permute.default(arg84_1, [1, 0]);  arg84_1 = None
    mm_6: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_57, permute_23);  view_57 = permute_23 = None
    view_58: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(mm_6, [8, 196, 1536]);  mm_6 = None
    view_59: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.reshape.default(view_58, [8, 196, 2, 16, 48]);  view_58 = None
    permute_24: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.permute.default(view_59, [2, 0, 3, 1, 4]);  view_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    select_28: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_24, 0, 0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    expand_21: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(select_28, [8, 16, 196, 48]);  select_28 = None
    clone_33: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
    view_62: "f32[128, 196, 48]" = torch.ops.aten.reshape.default(clone_33, [128, 196, 48]);  clone_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    select_29: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_24, 0, 1);  permute_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    permute_27: "f32[8, 16, 48, 196]" = torch.ops.aten.permute.default(select_29, [0, 1, 3, 2]);  select_29 = None
    expand_22: "f32[8, 16, 48, 196]" = torch.ops.aten.expand.default(permute_27, [8, 16, 48, 196]);  permute_27 = None
    clone_34: "f32[8, 16, 48, 196]" = torch.ops.aten.clone.default(expand_22, memory_format = torch.contiguous_format);  expand_22 = None
    view_63: "f32[128, 48, 196]" = torch.ops.aten.reshape.default(clone_34, [128, 48, 196]);  clone_34 = None
    bmm_4: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_62, view_63);  view_62 = view_63 = None
    view_64: "f32[8, 16, 196, 196]" = torch.ops.aten.reshape.default(bmm_4, [8, 16, 196, 196]);  bmm_4 = None
    mul_22: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(view_64, 0.14433756729740643);  view_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    amax_4: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(mul_22, [-1], True)
    sub_14: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_22, amax_4);  mul_22 = amax_4 = None
    exp_4: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_14);  sub_14 = None
    sum_7: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_6: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_4, sum_7);  exp_4 = sum_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    mul_23: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_16, div_6);  sub_16 = div_6 = None
    sigmoid_5: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_65);  view_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:121, code: rel_indices = torch.zeros(1, num_patches, num_patches, 3)
    full_2: "f32[1, 196, 196, 3]" = torch.ops.aten.full.default([1, 196, 196, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:126, code: rel_indices[:, :, :, 2] = indd.unsqueeze(0)
    select_20: "f32[1, 196, 196]" = torch.ops.aten.select.int(full_2, 3, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:122, code: ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
    iota_4: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_53: "i64[1, 14]" = torch.ops.aten.reshape.default(iota_4, [1, -1]);  iota_4 = None
    iota_5: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_54: "i64[14, 1]" = torch.ops.aten.reshape.default(iota_5, [-1, 1]);  iota_5 = None
    sub_13: "i64[14, 14]" = torch.ops.aten.sub.Tensor(view_53, view_54);  view_53 = view_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:123, code: indx = ind.repeat(img_size, img_size)
    repeat_2: "i64[196, 196]" = torch.ops.aten.repeat.default(sub_13, [14, 14])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:125, code: indd = indx ** 2 + indy ** 2
    pow_5: "i64[196, 196]" = torch.ops.aten.pow.Tensor_Scalar(repeat_2, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:124, code: indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
    unsqueeze_12: "i64[14, 1, 14]" = torch.ops.aten.unsqueeze.default(sub_13, 1);  sub_13 = None
    expand_17: "i64[14, 14, 14]" = torch.ops.aten.expand.default(unsqueeze_12, [14, 14, 14]);  unsqueeze_12 = None
    clone_30: "i64[14, 14, 14]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
    view_55: "i64[196, 14]" = torch.ops.aten.reshape.default(clone_30, [196, 14]);  clone_30 = None
    unsqueeze_13: "i64[196, 14, 1]" = torch.ops.aten.unsqueeze.default(view_55, 2);  view_55 = None
    expand_18: "i64[196, 14, 14]" = torch.ops.aten.expand.default(unsqueeze_13, [196, 14, 14]);  unsqueeze_13 = None
    clone_31: "i64[196, 14, 14]" = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
    view_56: "i64[196, 196]" = torch.ops.aten.reshape.default(clone_31, [196, 196]);  clone_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:125, code: indd = indx ** 2 + indy ** 2
    pow_6: "i64[196, 196]" = torch.ops.aten.pow.Tensor_Scalar(view_56, 2)
    add_23: "i64[196, 196]" = torch.ops.aten.add.Tensor(pow_5, pow_6);  pow_5 = pow_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:126, code: rel_indices[:, :, :, 2] = indd.unsqueeze(0)
    unsqueeze_14: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(add_23, 0);  add_23 = None
    copy_6: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_20, unsqueeze_14);  select_20 = unsqueeze_14 = None
    select_scatter_6: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(full_2, copy_6, 3, 2);  full_2 = copy_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:127, code: rel_indices[:, :, :, 1] = indy.unsqueeze(0)
    select_23: "f32[1, 196, 196]" = torch.ops.aten.select.int(select_scatter_6, 3, 1)
    unsqueeze_15: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(view_56, 0);  view_56 = None
    copy_7: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_23, unsqueeze_15);  select_23 = unsqueeze_15 = None
    select_scatter_7: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(select_scatter_6, copy_7, 3, 1);  select_scatter_6 = copy_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:128, code: rel_indices[:, :, :, 0] = indx.unsqueeze(0)
    select_26: "f32[1, 196, 196]" = torch.ops.aten.select.int(select_scatter_7, 3, 0)
    unsqueeze_16: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(repeat_2, 0);  repeat_2 = None
    copy_8: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_26, unsqueeze_16);  select_26 = unsqueeze_16 = None
    select_scatter_8: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(select_scatter_7, copy_8, 3, 0);  select_scatter_7 = copy_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    expand_20: "f32[8, 196, 196, 3]" = torch.ops.aten.expand.default(select_scatter_8, [8, -1, -1, -1])
    clone_32: "f32[8, 196, 196, 3]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
    view_60: "f32[307328, 3]" = torch.ops.aten.reshape.default(clone_32, [307328, 3]);  clone_32 = None
    permute_25: "f32[3, 16]" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
    mm_7: "f32[307328, 16]" = torch.ops.aten.mm.default(view_60, permute_25);  view_60 = permute_25 = None
    view_61: "f32[8, 196, 196, 16]" = torch.ops.aten.reshape.default(mm_7, [8, 196, 196, 16]);  mm_7 = None
    add_24: "f32[8, 196, 196, 16]" = torch.ops.aten.add.Tensor(view_61, arg86_1);  view_61 = arg86_1 = None
    permute_26: "f32[8, 16, 196, 196]" = torch.ops.aten.permute.default(add_24, [0, 3, 1, 2]);  add_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    clone_35: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(permute_26, memory_format = torch.contiguous_format);  permute_26 = None
    amax_5: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(clone_35, [-1], True)
    sub_15: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(clone_35, amax_5);  clone_35 = amax_5 = None
    exp_5: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_15);  sub_15 = None
    sum_8: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_7: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_5, sum_8);  exp_5 = sum_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    mul_24: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sigmoid_5, div_7);  sigmoid_5 = div_7 = None
    add_25: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(mul_23, mul_24);  mul_23 = mul_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    sum_9: "f32[8, 16, 196]" = torch.ops.aten.sum.dim_IntList(add_25, [-1])
    unsqueeze_17: "f32[8, 16, 196, 1]" = torch.ops.aten.unsqueeze.default(sum_9, -1);  sum_9 = None
    div_8: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(add_25, unsqueeze_17);  add_25 = unsqueeze_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_23: "f32[8, 16, 196, 196]" = torch.ops.aten.expand.default(div_8, [8, 16, 196, 196]);  div_8 = None
    view_69: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(expand_23, [128, 196, 196]);  expand_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_66: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_22, [1568, 768]);  add_22 = None
    permute_28: "f32[768, 768]" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
    mm_8: "f32[1568, 768]" = torch.ops.aten.mm.default(view_66, permute_28);  view_66 = permute_28 = None
    view_67: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_8, [8, 196, 768]);  mm_8 = None
    view_68: "f32[8, 196, 16, 48]" = torch.ops.aten.reshape.default(view_67, [8, 196, 16, 48]);  view_67 = None
    permute_29: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_68, [0, 2, 1, 3]);  view_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_24: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(permute_29, [8, 16, 196, 48]);  permute_29 = None
    clone_37: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
    view_70: "f32[128, 196, 48]" = torch.ops.aten.reshape.default(clone_37, [128, 196, 48]);  clone_37 = None
    bmm_5: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(view_69, view_70);  view_69 = view_70 = None
    view_71: "f32[8, 16, 196, 48]" = torch.ops.aten.reshape.default(bmm_5, [8, 16, 196, 48]);  bmm_5 = None
    permute_30: "f32[8, 196, 16, 48]" = torch.ops.aten.permute.default(view_71, [0, 2, 1, 3]);  view_71 = None
    clone_38: "f32[8, 196, 16, 48]" = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None
    view_72: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(clone_38, [8, 196, 768]);  clone_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_73: "f32[1568, 768]" = torch.ops.aten.reshape.default(view_72, [1568, 768]);  view_72 = None
    permute_31: "f32[768, 768]" = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
    addmm_6: "f32[1568, 768]" = torch.ops.aten.addmm.default(arg89_1, view_73, permute_31);  arg89_1 = view_73 = permute_31 = None
    view_74: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(addmm_6, [8, 196, 768]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:235, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_26: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_20, view_74);  add_20 = view_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_40: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_26, memory_format = torch.contiguous_format)
    var_mean_5 = torch.ops.aten.var_mean.correction(clone_40, [2], correction = 0, keepdim = True)
    getitem_10: "f32[8, 196, 1]" = var_mean_5[0]
    getitem_11: "f32[8, 196, 1]" = var_mean_5[1];  var_mean_5 = None
    sub_17: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_40, getitem_11);  clone_40 = getitem_11 = None
    add_27: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-06);  getitem_10 = None
    rsqrt_5: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
    mul_25: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_5);  sub_17 = rsqrt_5 = None
    mul_26: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_25, arg15_1);  mul_25 = arg15_1 = None
    add_28: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_26, arg16_1);  mul_26 = arg16_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_75: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_28, [1568, 768]);  add_28 = None
    permute_32: "f32[768, 3072]" = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
    addmm_7: "f32[1568, 3072]" = torch.ops.aten.addmm.default(arg91_1, view_75, permute_32);  arg91_1 = view_75 = permute_32 = None
    view_76: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(addmm_7, [8, 196, 3072]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_27: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_76, 0.5)
    mul_28: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_76, 0.7071067811865476);  view_76 = None
    erf_2: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_28);  mul_28 = None
    add_29: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_29: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_27, add_29);  mul_27 = add_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_77: "f32[1568, 3072]" = torch.ops.aten.reshape.default(mul_29, [1568, 3072]);  mul_29 = None
    permute_33: "f32[3072, 768]" = torch.ops.aten.permute.default(arg92_1, [1, 0]);  arg92_1 = None
    addmm_8: "f32[1568, 768]" = torch.ops.aten.addmm.default(arg93_1, view_77, permute_33);  arg93_1 = view_77 = permute_33 = None
    view_78: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(addmm_8, [8, 196, 768]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:236, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_30: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_26, view_78);  add_26 = view_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_43: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_30, memory_format = torch.contiguous_format)
    var_mean_6 = torch.ops.aten.var_mean.correction(clone_43, [2], correction = 0, keepdim = True)
    getitem_12: "f32[8, 196, 1]" = var_mean_6[0]
    getitem_13: "f32[8, 196, 1]" = var_mean_6[1];  var_mean_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_91: "f32[1, 16, 1, 1]" = torch.ops.aten.reshape.default(arg19_1, [1, -1, 1, 1]);  arg19_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    sigmoid_6: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_91)
    sub_22: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1.0, sigmoid_6);  sigmoid_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_18: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_43, getitem_13);  clone_43 = getitem_13 = None
    add_31: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-06);  getitem_12 = None
    rsqrt_6: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
    mul_30: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_6);  sub_18 = rsqrt_6 = None
    mul_31: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_30, arg17_1);  mul_30 = arg17_1 = None
    add_32: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_31, arg18_1);  mul_31 = arg18_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_83: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_32, [1568, 768])
    permute_34: "f32[768, 1536]" = torch.ops.aten.permute.default(arg94_1, [1, 0]);  arg94_1 = None
    mm_9: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_83, permute_34);  view_83 = permute_34 = None
    view_84: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(mm_9, [8, 196, 1536]);  mm_9 = None
    view_85: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.reshape.default(view_84, [8, 196, 2, 16, 48]);  view_84 = None
    permute_35: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.permute.default(view_85, [2, 0, 3, 1, 4]);  view_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    select_38: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_35, 0, 0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    expand_29: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(select_38, [8, 16, 196, 48]);  select_38 = None
    clone_47: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
    view_88: "f32[128, 196, 48]" = torch.ops.aten.reshape.default(clone_47, [128, 196, 48]);  clone_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    select_39: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_35, 0, 1);  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    permute_38: "f32[8, 16, 48, 196]" = torch.ops.aten.permute.default(select_39, [0, 1, 3, 2]);  select_39 = None
    expand_30: "f32[8, 16, 48, 196]" = torch.ops.aten.expand.default(permute_38, [8, 16, 48, 196]);  permute_38 = None
    clone_48: "f32[8, 16, 48, 196]" = torch.ops.aten.clone.default(expand_30, memory_format = torch.contiguous_format);  expand_30 = None
    view_89: "f32[128, 48, 196]" = torch.ops.aten.reshape.default(clone_48, [128, 48, 196]);  clone_48 = None
    bmm_6: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_88, view_89);  view_88 = view_89 = None
    view_90: "f32[8, 16, 196, 196]" = torch.ops.aten.reshape.default(bmm_6, [8, 16, 196, 196]);  bmm_6 = None
    mul_32: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(view_90, 0.14433756729740643);  view_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    amax_6: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(mul_32, [-1], True)
    sub_20: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_32, amax_6);  mul_32 = amax_6 = None
    exp_6: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_20);  sub_20 = None
    sum_10: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_9: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_6, sum_10);  exp_6 = sum_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    mul_33: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_22, div_9);  sub_22 = div_9 = None
    sigmoid_7: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_91);  view_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:121, code: rel_indices = torch.zeros(1, num_patches, num_patches, 3)
    full_3: "f32[1, 196, 196, 3]" = torch.ops.aten.full.default([1, 196, 196, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:126, code: rel_indices[:, :, :, 2] = indd.unsqueeze(0)
    select_30: "f32[1, 196, 196]" = torch.ops.aten.select.int(full_3, 3, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:122, code: ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
    iota_6: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_79: "i64[1, 14]" = torch.ops.aten.reshape.default(iota_6, [1, -1]);  iota_6 = None
    iota_7: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_80: "i64[14, 1]" = torch.ops.aten.reshape.default(iota_7, [-1, 1]);  iota_7 = None
    sub_19: "i64[14, 14]" = torch.ops.aten.sub.Tensor(view_79, view_80);  view_79 = view_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:123, code: indx = ind.repeat(img_size, img_size)
    repeat_3: "i64[196, 196]" = torch.ops.aten.repeat.default(sub_19, [14, 14])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:125, code: indd = indx ** 2 + indy ** 2
    pow_7: "i64[196, 196]" = torch.ops.aten.pow.Tensor_Scalar(repeat_3, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:124, code: indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
    unsqueeze_18: "i64[14, 1, 14]" = torch.ops.aten.unsqueeze.default(sub_19, 1);  sub_19 = None
    expand_25: "i64[14, 14, 14]" = torch.ops.aten.expand.default(unsqueeze_18, [14, 14, 14]);  unsqueeze_18 = None
    clone_44: "i64[14, 14, 14]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
    view_81: "i64[196, 14]" = torch.ops.aten.reshape.default(clone_44, [196, 14]);  clone_44 = None
    unsqueeze_19: "i64[196, 14, 1]" = torch.ops.aten.unsqueeze.default(view_81, 2);  view_81 = None
    expand_26: "i64[196, 14, 14]" = torch.ops.aten.expand.default(unsqueeze_19, [196, 14, 14]);  unsqueeze_19 = None
    clone_45: "i64[196, 14, 14]" = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
    view_82: "i64[196, 196]" = torch.ops.aten.reshape.default(clone_45, [196, 196]);  clone_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:125, code: indd = indx ** 2 + indy ** 2
    pow_8: "i64[196, 196]" = torch.ops.aten.pow.Tensor_Scalar(view_82, 2)
    add_33: "i64[196, 196]" = torch.ops.aten.add.Tensor(pow_7, pow_8);  pow_7 = pow_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:126, code: rel_indices[:, :, :, 2] = indd.unsqueeze(0)
    unsqueeze_20: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(add_33, 0);  add_33 = None
    copy_9: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_30, unsqueeze_20);  select_30 = unsqueeze_20 = None
    select_scatter_9: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(full_3, copy_9, 3, 2);  full_3 = copy_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:127, code: rel_indices[:, :, :, 1] = indy.unsqueeze(0)
    select_33: "f32[1, 196, 196]" = torch.ops.aten.select.int(select_scatter_9, 3, 1)
    unsqueeze_21: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(view_82, 0);  view_82 = None
    copy_10: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_33, unsqueeze_21);  select_33 = unsqueeze_21 = None
    select_scatter_10: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(select_scatter_9, copy_10, 3, 1);  select_scatter_9 = copy_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:128, code: rel_indices[:, :, :, 0] = indx.unsqueeze(0)
    select_36: "f32[1, 196, 196]" = torch.ops.aten.select.int(select_scatter_10, 3, 0)
    unsqueeze_22: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(repeat_3, 0);  repeat_3 = None
    copy_11: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_36, unsqueeze_22);  select_36 = unsqueeze_22 = None
    select_scatter_11: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(select_scatter_10, copy_11, 3, 0);  select_scatter_10 = copy_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    expand_28: "f32[8, 196, 196, 3]" = torch.ops.aten.expand.default(select_scatter_11, [8, -1, -1, -1])
    clone_46: "f32[8, 196, 196, 3]" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
    view_86: "f32[307328, 3]" = torch.ops.aten.reshape.default(clone_46, [307328, 3]);  clone_46 = None
    permute_36: "f32[3, 16]" = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
    mm_10: "f32[307328, 16]" = torch.ops.aten.mm.default(view_86, permute_36);  view_86 = permute_36 = None
    view_87: "f32[8, 196, 196, 16]" = torch.ops.aten.reshape.default(mm_10, [8, 196, 196, 16]);  mm_10 = None
    add_34: "f32[8, 196, 196, 16]" = torch.ops.aten.add.Tensor(view_87, arg96_1);  view_87 = arg96_1 = None
    permute_37: "f32[8, 16, 196, 196]" = torch.ops.aten.permute.default(add_34, [0, 3, 1, 2]);  add_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    clone_49: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
    amax_7: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(clone_49, [-1], True)
    sub_21: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(clone_49, amax_7);  clone_49 = amax_7 = None
    exp_7: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_21);  sub_21 = None
    sum_11: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_10: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_7, sum_11);  exp_7 = sum_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    mul_34: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sigmoid_7, div_10);  sigmoid_7 = div_10 = None
    add_35: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(mul_33, mul_34);  mul_33 = mul_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    sum_12: "f32[8, 16, 196]" = torch.ops.aten.sum.dim_IntList(add_35, [-1])
    unsqueeze_23: "f32[8, 16, 196, 1]" = torch.ops.aten.unsqueeze.default(sum_12, -1);  sum_12 = None
    div_11: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(add_35, unsqueeze_23);  add_35 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_31: "f32[8, 16, 196, 196]" = torch.ops.aten.expand.default(div_11, [8, 16, 196, 196]);  div_11 = None
    view_95: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(expand_31, [128, 196, 196]);  expand_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_92: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_32, [1568, 768]);  add_32 = None
    permute_39: "f32[768, 768]" = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
    mm_11: "f32[1568, 768]" = torch.ops.aten.mm.default(view_92, permute_39);  view_92 = permute_39 = None
    view_93: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_11, [8, 196, 768]);  mm_11 = None
    view_94: "f32[8, 196, 16, 48]" = torch.ops.aten.reshape.default(view_93, [8, 196, 16, 48]);  view_93 = None
    permute_40: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_32: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(permute_40, [8, 16, 196, 48]);  permute_40 = None
    clone_51: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
    view_96: "f32[128, 196, 48]" = torch.ops.aten.reshape.default(clone_51, [128, 196, 48]);  clone_51 = None
    bmm_7: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(view_95, view_96);  view_95 = view_96 = None
    view_97: "f32[8, 16, 196, 48]" = torch.ops.aten.reshape.default(bmm_7, [8, 16, 196, 48]);  bmm_7 = None
    permute_41: "f32[8, 196, 16, 48]" = torch.ops.aten.permute.default(view_97, [0, 2, 1, 3]);  view_97 = None
    clone_52: "f32[8, 196, 16, 48]" = torch.ops.aten.clone.default(permute_41, memory_format = torch.contiguous_format);  permute_41 = None
    view_98: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(clone_52, [8, 196, 768]);  clone_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_99: "f32[1568, 768]" = torch.ops.aten.reshape.default(view_98, [1568, 768]);  view_98 = None
    permute_42: "f32[768, 768]" = torch.ops.aten.permute.default(arg98_1, [1, 0]);  arg98_1 = None
    addmm_9: "f32[1568, 768]" = torch.ops.aten.addmm.default(arg99_1, view_99, permute_42);  arg99_1 = view_99 = permute_42 = None
    view_100: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(addmm_9, [8, 196, 768]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:235, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_36: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_30, view_100);  add_30 = view_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_54: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_36, memory_format = torch.contiguous_format)
    var_mean_7 = torch.ops.aten.var_mean.correction(clone_54, [2], correction = 0, keepdim = True)
    getitem_14: "f32[8, 196, 1]" = var_mean_7[0]
    getitem_15: "f32[8, 196, 1]" = var_mean_7[1];  var_mean_7 = None
    sub_23: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_54, getitem_15);  clone_54 = getitem_15 = None
    add_37: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-06);  getitem_14 = None
    rsqrt_7: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
    mul_35: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_7);  sub_23 = rsqrt_7 = None
    mul_36: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_35, arg20_1);  mul_35 = arg20_1 = None
    add_38: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_36, arg21_1);  mul_36 = arg21_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_101: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_38, [1568, 768]);  add_38 = None
    permute_43: "f32[768, 3072]" = torch.ops.aten.permute.default(arg100_1, [1, 0]);  arg100_1 = None
    addmm_10: "f32[1568, 3072]" = torch.ops.aten.addmm.default(arg101_1, view_101, permute_43);  arg101_1 = view_101 = permute_43 = None
    view_102: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(addmm_10, [8, 196, 3072]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_37: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_102, 0.5)
    mul_38: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_102, 0.7071067811865476);  view_102 = None
    erf_3: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_38);  mul_38 = None
    add_39: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_39: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_37, add_39);  mul_37 = add_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_103: "f32[1568, 3072]" = torch.ops.aten.reshape.default(mul_39, [1568, 3072]);  mul_39 = None
    permute_44: "f32[3072, 768]" = torch.ops.aten.permute.default(arg102_1, [1, 0]);  arg102_1 = None
    addmm_11: "f32[1568, 768]" = torch.ops.aten.addmm.default(arg103_1, view_103, permute_44);  arg103_1 = view_103 = permute_44 = None
    view_104: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(addmm_11, [8, 196, 768]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:236, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_40: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_36, view_104);  add_36 = view_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_57: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_40, memory_format = torch.contiguous_format)
    var_mean_8 = torch.ops.aten.var_mean.correction(clone_57, [2], correction = 0, keepdim = True)
    getitem_16: "f32[8, 196, 1]" = var_mean_8[0]
    getitem_17: "f32[8, 196, 1]" = var_mean_8[1];  var_mean_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_117: "f32[1, 16, 1, 1]" = torch.ops.aten.reshape.default(arg24_1, [1, -1, 1, 1]);  arg24_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    sigmoid_8: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_117)
    sub_28: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1.0, sigmoid_8);  sigmoid_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_24: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_57, getitem_17);  clone_57 = getitem_17 = None
    add_41: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-06);  getitem_16 = None
    rsqrt_8: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_41);  add_41 = None
    mul_40: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_8);  sub_24 = rsqrt_8 = None
    mul_41: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_40, arg22_1);  mul_40 = arg22_1 = None
    add_42: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_41, arg23_1);  mul_41 = arg23_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_109: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_42, [1568, 768])
    permute_45: "f32[768, 1536]" = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
    mm_12: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_109, permute_45);  view_109 = permute_45 = None
    view_110: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(mm_12, [8, 196, 1536]);  mm_12 = None
    view_111: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.reshape.default(view_110, [8, 196, 2, 16, 48]);  view_110 = None
    permute_46: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.permute.default(view_111, [2, 0, 3, 1, 4]);  view_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    select_48: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_46, 0, 0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    expand_37: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(select_48, [8, 16, 196, 48]);  select_48 = None
    clone_61: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
    view_114: "f32[128, 196, 48]" = torch.ops.aten.reshape.default(clone_61, [128, 196, 48]);  clone_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    select_49: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_46, 0, 1);  permute_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    permute_49: "f32[8, 16, 48, 196]" = torch.ops.aten.permute.default(select_49, [0, 1, 3, 2]);  select_49 = None
    expand_38: "f32[8, 16, 48, 196]" = torch.ops.aten.expand.default(permute_49, [8, 16, 48, 196]);  permute_49 = None
    clone_62: "f32[8, 16, 48, 196]" = torch.ops.aten.clone.default(expand_38, memory_format = torch.contiguous_format);  expand_38 = None
    view_115: "f32[128, 48, 196]" = torch.ops.aten.reshape.default(clone_62, [128, 48, 196]);  clone_62 = None
    bmm_8: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_114, view_115);  view_114 = view_115 = None
    view_116: "f32[8, 16, 196, 196]" = torch.ops.aten.reshape.default(bmm_8, [8, 16, 196, 196]);  bmm_8 = None
    mul_42: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(view_116, 0.14433756729740643);  view_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    amax_8: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(mul_42, [-1], True)
    sub_26: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_42, amax_8);  mul_42 = amax_8 = None
    exp_8: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_26);  sub_26 = None
    sum_13: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_12: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_8, sum_13);  exp_8 = sum_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    mul_43: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_28, div_12);  sub_28 = div_12 = None
    sigmoid_9: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_117);  view_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:121, code: rel_indices = torch.zeros(1, num_patches, num_patches, 3)
    full_4: "f32[1, 196, 196, 3]" = torch.ops.aten.full.default([1, 196, 196, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:126, code: rel_indices[:, :, :, 2] = indd.unsqueeze(0)
    select_40: "f32[1, 196, 196]" = torch.ops.aten.select.int(full_4, 3, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:122, code: ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
    iota_8: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_105: "i64[1, 14]" = torch.ops.aten.reshape.default(iota_8, [1, -1]);  iota_8 = None
    iota_9: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_106: "i64[14, 1]" = torch.ops.aten.reshape.default(iota_9, [-1, 1]);  iota_9 = None
    sub_25: "i64[14, 14]" = torch.ops.aten.sub.Tensor(view_105, view_106);  view_105 = view_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:123, code: indx = ind.repeat(img_size, img_size)
    repeat_4: "i64[196, 196]" = torch.ops.aten.repeat.default(sub_25, [14, 14])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:125, code: indd = indx ** 2 + indy ** 2
    pow_9: "i64[196, 196]" = torch.ops.aten.pow.Tensor_Scalar(repeat_4, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:124, code: indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
    unsqueeze_24: "i64[14, 1, 14]" = torch.ops.aten.unsqueeze.default(sub_25, 1);  sub_25 = None
    expand_33: "i64[14, 14, 14]" = torch.ops.aten.expand.default(unsqueeze_24, [14, 14, 14]);  unsqueeze_24 = None
    clone_58: "i64[14, 14, 14]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
    view_107: "i64[196, 14]" = torch.ops.aten.reshape.default(clone_58, [196, 14]);  clone_58 = None
    unsqueeze_25: "i64[196, 14, 1]" = torch.ops.aten.unsqueeze.default(view_107, 2);  view_107 = None
    expand_34: "i64[196, 14, 14]" = torch.ops.aten.expand.default(unsqueeze_25, [196, 14, 14]);  unsqueeze_25 = None
    clone_59: "i64[196, 14, 14]" = torch.ops.aten.clone.default(expand_34, memory_format = torch.contiguous_format);  expand_34 = None
    view_108: "i64[196, 196]" = torch.ops.aten.reshape.default(clone_59, [196, 196]);  clone_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:125, code: indd = indx ** 2 + indy ** 2
    pow_10: "i64[196, 196]" = torch.ops.aten.pow.Tensor_Scalar(view_108, 2)
    add_43: "i64[196, 196]" = torch.ops.aten.add.Tensor(pow_9, pow_10);  pow_9 = pow_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:126, code: rel_indices[:, :, :, 2] = indd.unsqueeze(0)
    unsqueeze_26: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(add_43, 0);  add_43 = None
    copy_12: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_40, unsqueeze_26);  select_40 = unsqueeze_26 = None
    select_scatter_12: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(full_4, copy_12, 3, 2);  full_4 = copy_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:127, code: rel_indices[:, :, :, 1] = indy.unsqueeze(0)
    select_43: "f32[1, 196, 196]" = torch.ops.aten.select.int(select_scatter_12, 3, 1)
    unsqueeze_27: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(view_108, 0);  view_108 = None
    copy_13: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_43, unsqueeze_27);  select_43 = unsqueeze_27 = None
    select_scatter_13: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(select_scatter_12, copy_13, 3, 1);  select_scatter_12 = copy_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:128, code: rel_indices[:, :, :, 0] = indx.unsqueeze(0)
    select_46: "f32[1, 196, 196]" = torch.ops.aten.select.int(select_scatter_13, 3, 0)
    unsqueeze_28: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(repeat_4, 0);  repeat_4 = None
    copy_14: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_46, unsqueeze_28);  select_46 = unsqueeze_28 = None
    select_scatter_14: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(select_scatter_13, copy_14, 3, 0);  select_scatter_13 = copy_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    expand_36: "f32[8, 196, 196, 3]" = torch.ops.aten.expand.default(select_scatter_14, [8, -1, -1, -1])
    clone_60: "f32[8, 196, 196, 3]" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
    view_112: "f32[307328, 3]" = torch.ops.aten.reshape.default(clone_60, [307328, 3]);  clone_60 = None
    permute_47: "f32[3, 16]" = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
    mm_13: "f32[307328, 16]" = torch.ops.aten.mm.default(view_112, permute_47);  view_112 = permute_47 = None
    view_113: "f32[8, 196, 196, 16]" = torch.ops.aten.reshape.default(mm_13, [8, 196, 196, 16]);  mm_13 = None
    add_44: "f32[8, 196, 196, 16]" = torch.ops.aten.add.Tensor(view_113, arg106_1);  view_113 = arg106_1 = None
    permute_48: "f32[8, 16, 196, 196]" = torch.ops.aten.permute.default(add_44, [0, 3, 1, 2]);  add_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    clone_63: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(permute_48, memory_format = torch.contiguous_format);  permute_48 = None
    amax_9: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(clone_63, [-1], True)
    sub_27: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(clone_63, amax_9);  clone_63 = amax_9 = None
    exp_9: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_27);  sub_27 = None
    sum_14: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_13: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_9, sum_14);  exp_9 = sum_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    mul_44: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sigmoid_9, div_13);  sigmoid_9 = div_13 = None
    add_45: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(mul_43, mul_44);  mul_43 = mul_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    sum_15: "f32[8, 16, 196]" = torch.ops.aten.sum.dim_IntList(add_45, [-1])
    unsqueeze_29: "f32[8, 16, 196, 1]" = torch.ops.aten.unsqueeze.default(sum_15, -1);  sum_15 = None
    div_14: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(add_45, unsqueeze_29);  add_45 = unsqueeze_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_39: "f32[8, 16, 196, 196]" = torch.ops.aten.expand.default(div_14, [8, 16, 196, 196]);  div_14 = None
    view_121: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(expand_39, [128, 196, 196]);  expand_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_118: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_42, [1568, 768]);  add_42 = None
    permute_50: "f32[768, 768]" = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
    mm_14: "f32[1568, 768]" = torch.ops.aten.mm.default(view_118, permute_50);  view_118 = permute_50 = None
    view_119: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_14, [8, 196, 768]);  mm_14 = None
    view_120: "f32[8, 196, 16, 48]" = torch.ops.aten.reshape.default(view_119, [8, 196, 16, 48]);  view_119 = None
    permute_51: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_40: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(permute_51, [8, 16, 196, 48]);  permute_51 = None
    clone_65: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
    view_122: "f32[128, 196, 48]" = torch.ops.aten.reshape.default(clone_65, [128, 196, 48]);  clone_65 = None
    bmm_9: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(view_121, view_122);  view_121 = view_122 = None
    view_123: "f32[8, 16, 196, 48]" = torch.ops.aten.reshape.default(bmm_9, [8, 16, 196, 48]);  bmm_9 = None
    permute_52: "f32[8, 196, 16, 48]" = torch.ops.aten.permute.default(view_123, [0, 2, 1, 3]);  view_123 = None
    clone_66: "f32[8, 196, 16, 48]" = torch.ops.aten.clone.default(permute_52, memory_format = torch.contiguous_format);  permute_52 = None
    view_124: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(clone_66, [8, 196, 768]);  clone_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_125: "f32[1568, 768]" = torch.ops.aten.reshape.default(view_124, [1568, 768]);  view_124 = None
    permute_53: "f32[768, 768]" = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
    addmm_12: "f32[1568, 768]" = torch.ops.aten.addmm.default(arg109_1, view_125, permute_53);  arg109_1 = view_125 = permute_53 = None
    view_126: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(addmm_12, [8, 196, 768]);  addmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:235, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_46: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_40, view_126);  add_40 = view_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_68: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_46, memory_format = torch.contiguous_format)
    var_mean_9 = torch.ops.aten.var_mean.correction(clone_68, [2], correction = 0, keepdim = True)
    getitem_18: "f32[8, 196, 1]" = var_mean_9[0]
    getitem_19: "f32[8, 196, 1]" = var_mean_9[1];  var_mean_9 = None
    sub_29: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_68, getitem_19);  clone_68 = getitem_19 = None
    add_47: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-06);  getitem_18 = None
    rsqrt_9: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
    mul_45: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_9);  sub_29 = rsqrt_9 = None
    mul_46: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_45, arg25_1);  mul_45 = arg25_1 = None
    add_48: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_46, arg26_1);  mul_46 = arg26_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_127: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_48, [1568, 768]);  add_48 = None
    permute_54: "f32[768, 3072]" = torch.ops.aten.permute.default(arg110_1, [1, 0]);  arg110_1 = None
    addmm_13: "f32[1568, 3072]" = torch.ops.aten.addmm.default(arg111_1, view_127, permute_54);  arg111_1 = view_127 = permute_54 = None
    view_128: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(addmm_13, [8, 196, 3072]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_47: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_128, 0.5)
    mul_48: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_128, 0.7071067811865476);  view_128 = None
    erf_4: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_48);  mul_48 = None
    add_49: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_49: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_47, add_49);  mul_47 = add_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_129: "f32[1568, 3072]" = torch.ops.aten.reshape.default(mul_49, [1568, 3072]);  mul_49 = None
    permute_55: "f32[3072, 768]" = torch.ops.aten.permute.default(arg112_1, [1, 0]);  arg112_1 = None
    addmm_14: "f32[1568, 768]" = torch.ops.aten.addmm.default(arg113_1, view_129, permute_55);  arg113_1 = view_129 = permute_55 = None
    view_130: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(addmm_14, [8, 196, 768]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:236, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_50: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_46, view_130);  add_46 = view_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_71: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_50, memory_format = torch.contiguous_format)
    var_mean_10 = torch.ops.aten.var_mean.correction(clone_71, [2], correction = 0, keepdim = True)
    getitem_20: "f32[8, 196, 1]" = var_mean_10[0]
    getitem_21: "f32[8, 196, 1]" = var_mean_10[1];  var_mean_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_143: "f32[1, 16, 1, 1]" = torch.ops.aten.reshape.default(arg29_1, [1, -1, 1, 1]);  arg29_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    sigmoid_10: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_143)
    sub_34: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1.0, sigmoid_10);  sigmoid_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_30: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_71, getitem_21);  clone_71 = getitem_21 = None
    add_51: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-06);  getitem_20 = None
    rsqrt_10: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
    mul_50: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_10);  sub_30 = rsqrt_10 = None
    mul_51: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_50, arg27_1);  mul_50 = arg27_1 = None
    add_52: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_51, arg28_1);  mul_51 = arg28_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_135: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_52, [1568, 768])
    permute_56: "f32[768, 1536]" = torch.ops.aten.permute.default(arg114_1, [1, 0]);  arg114_1 = None
    mm_15: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_135, permute_56);  view_135 = permute_56 = None
    view_136: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(mm_15, [8, 196, 1536]);  mm_15 = None
    view_137: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.reshape.default(view_136, [8, 196, 2, 16, 48]);  view_136 = None
    permute_57: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.permute.default(view_137, [2, 0, 3, 1, 4]);  view_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    select_58: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_57, 0, 0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    expand_45: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(select_58, [8, 16, 196, 48]);  select_58 = None
    clone_75: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
    view_140: "f32[128, 196, 48]" = torch.ops.aten.reshape.default(clone_75, [128, 196, 48]);  clone_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    select_59: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_57, 0, 1);  permute_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    permute_60: "f32[8, 16, 48, 196]" = torch.ops.aten.permute.default(select_59, [0, 1, 3, 2]);  select_59 = None
    expand_46: "f32[8, 16, 48, 196]" = torch.ops.aten.expand.default(permute_60, [8, 16, 48, 196]);  permute_60 = None
    clone_76: "f32[8, 16, 48, 196]" = torch.ops.aten.clone.default(expand_46, memory_format = torch.contiguous_format);  expand_46 = None
    view_141: "f32[128, 48, 196]" = torch.ops.aten.reshape.default(clone_76, [128, 48, 196]);  clone_76 = None
    bmm_10: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_140, view_141);  view_140 = view_141 = None
    view_142: "f32[8, 16, 196, 196]" = torch.ops.aten.reshape.default(bmm_10, [8, 16, 196, 196]);  bmm_10 = None
    mul_52: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(view_142, 0.14433756729740643);  view_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    amax_10: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(mul_52, [-1], True)
    sub_32: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_52, amax_10);  mul_52 = amax_10 = None
    exp_10: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_32);  sub_32 = None
    sum_16: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_15: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_10, sum_16);  exp_10 = sum_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    mul_53: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_34, div_15);  sub_34 = div_15 = None
    sigmoid_11: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_143);  view_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:121, code: rel_indices = torch.zeros(1, num_patches, num_patches, 3)
    full_5: "f32[1, 196, 196, 3]" = torch.ops.aten.full.default([1, 196, 196, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:126, code: rel_indices[:, :, :, 2] = indd.unsqueeze(0)
    select_50: "f32[1, 196, 196]" = torch.ops.aten.select.int(full_5, 3, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:122, code: ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
    iota_10: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_131: "i64[1, 14]" = torch.ops.aten.reshape.default(iota_10, [1, -1]);  iota_10 = None
    iota_11: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_132: "i64[14, 1]" = torch.ops.aten.reshape.default(iota_11, [-1, 1]);  iota_11 = None
    sub_31: "i64[14, 14]" = torch.ops.aten.sub.Tensor(view_131, view_132);  view_131 = view_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:123, code: indx = ind.repeat(img_size, img_size)
    repeat_5: "i64[196, 196]" = torch.ops.aten.repeat.default(sub_31, [14, 14])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:125, code: indd = indx ** 2 + indy ** 2
    pow_11: "i64[196, 196]" = torch.ops.aten.pow.Tensor_Scalar(repeat_5, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:124, code: indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
    unsqueeze_30: "i64[14, 1, 14]" = torch.ops.aten.unsqueeze.default(sub_31, 1);  sub_31 = None
    expand_41: "i64[14, 14, 14]" = torch.ops.aten.expand.default(unsqueeze_30, [14, 14, 14]);  unsqueeze_30 = None
    clone_72: "i64[14, 14, 14]" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
    view_133: "i64[196, 14]" = torch.ops.aten.reshape.default(clone_72, [196, 14]);  clone_72 = None
    unsqueeze_31: "i64[196, 14, 1]" = torch.ops.aten.unsqueeze.default(view_133, 2);  view_133 = None
    expand_42: "i64[196, 14, 14]" = torch.ops.aten.expand.default(unsqueeze_31, [196, 14, 14]);  unsqueeze_31 = None
    clone_73: "i64[196, 14, 14]" = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
    view_134: "i64[196, 196]" = torch.ops.aten.reshape.default(clone_73, [196, 196]);  clone_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:125, code: indd = indx ** 2 + indy ** 2
    pow_12: "i64[196, 196]" = torch.ops.aten.pow.Tensor_Scalar(view_134, 2)
    add_53: "i64[196, 196]" = torch.ops.aten.add.Tensor(pow_11, pow_12);  pow_11 = pow_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:126, code: rel_indices[:, :, :, 2] = indd.unsqueeze(0)
    unsqueeze_32: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(add_53, 0);  add_53 = None
    copy_15: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_50, unsqueeze_32);  select_50 = unsqueeze_32 = None
    select_scatter_15: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(full_5, copy_15, 3, 2);  full_5 = copy_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:127, code: rel_indices[:, :, :, 1] = indy.unsqueeze(0)
    select_53: "f32[1, 196, 196]" = torch.ops.aten.select.int(select_scatter_15, 3, 1)
    unsqueeze_33: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(view_134, 0);  view_134 = None
    copy_16: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_53, unsqueeze_33);  select_53 = unsqueeze_33 = None
    select_scatter_16: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(select_scatter_15, copy_16, 3, 1);  select_scatter_15 = copy_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:128, code: rel_indices[:, :, :, 0] = indx.unsqueeze(0)
    select_56: "f32[1, 196, 196]" = torch.ops.aten.select.int(select_scatter_16, 3, 0)
    unsqueeze_34: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(repeat_5, 0);  repeat_5 = None
    copy_17: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_56, unsqueeze_34);  select_56 = unsqueeze_34 = None
    select_scatter_17: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(select_scatter_16, copy_17, 3, 0);  select_scatter_16 = copy_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    expand_44: "f32[8, 196, 196, 3]" = torch.ops.aten.expand.default(select_scatter_17, [8, -1, -1, -1])
    clone_74: "f32[8, 196, 196, 3]" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
    view_138: "f32[307328, 3]" = torch.ops.aten.reshape.default(clone_74, [307328, 3]);  clone_74 = None
    permute_58: "f32[3, 16]" = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
    mm_16: "f32[307328, 16]" = torch.ops.aten.mm.default(view_138, permute_58);  view_138 = permute_58 = None
    view_139: "f32[8, 196, 196, 16]" = torch.ops.aten.reshape.default(mm_16, [8, 196, 196, 16]);  mm_16 = None
    add_54: "f32[8, 196, 196, 16]" = torch.ops.aten.add.Tensor(view_139, arg116_1);  view_139 = arg116_1 = None
    permute_59: "f32[8, 16, 196, 196]" = torch.ops.aten.permute.default(add_54, [0, 3, 1, 2]);  add_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    clone_77: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None
    amax_11: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(clone_77, [-1], True)
    sub_33: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(clone_77, amax_11);  clone_77 = amax_11 = None
    exp_11: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_33);  sub_33 = None
    sum_17: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_16: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_11, sum_17);  exp_11 = sum_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    mul_54: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sigmoid_11, div_16);  sigmoid_11 = div_16 = None
    add_55: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(mul_53, mul_54);  mul_53 = mul_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    sum_18: "f32[8, 16, 196]" = torch.ops.aten.sum.dim_IntList(add_55, [-1])
    unsqueeze_35: "f32[8, 16, 196, 1]" = torch.ops.aten.unsqueeze.default(sum_18, -1);  sum_18 = None
    div_17: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(add_55, unsqueeze_35);  add_55 = unsqueeze_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_47: "f32[8, 16, 196, 196]" = torch.ops.aten.expand.default(div_17, [8, 16, 196, 196]);  div_17 = None
    view_147: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(expand_47, [128, 196, 196]);  expand_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_144: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_52, [1568, 768]);  add_52 = None
    permute_61: "f32[768, 768]" = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
    mm_17: "f32[1568, 768]" = torch.ops.aten.mm.default(view_144, permute_61);  view_144 = permute_61 = None
    view_145: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_17, [8, 196, 768]);  mm_17 = None
    view_146: "f32[8, 196, 16, 48]" = torch.ops.aten.reshape.default(view_145, [8, 196, 16, 48]);  view_145 = None
    permute_62: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_146, [0, 2, 1, 3]);  view_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_48: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(permute_62, [8, 16, 196, 48]);  permute_62 = None
    clone_79: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
    view_148: "f32[128, 196, 48]" = torch.ops.aten.reshape.default(clone_79, [128, 196, 48]);  clone_79 = None
    bmm_11: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(view_147, view_148);  view_147 = view_148 = None
    view_149: "f32[8, 16, 196, 48]" = torch.ops.aten.reshape.default(bmm_11, [8, 16, 196, 48]);  bmm_11 = None
    permute_63: "f32[8, 196, 16, 48]" = torch.ops.aten.permute.default(view_149, [0, 2, 1, 3]);  view_149 = None
    clone_80: "f32[8, 196, 16, 48]" = torch.ops.aten.clone.default(permute_63, memory_format = torch.contiguous_format);  permute_63 = None
    view_150: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(clone_80, [8, 196, 768]);  clone_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_151: "f32[1568, 768]" = torch.ops.aten.reshape.default(view_150, [1568, 768]);  view_150 = None
    permute_64: "f32[768, 768]" = torch.ops.aten.permute.default(arg118_1, [1, 0]);  arg118_1 = None
    addmm_15: "f32[1568, 768]" = torch.ops.aten.addmm.default(arg119_1, view_151, permute_64);  arg119_1 = view_151 = permute_64 = None
    view_152: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(addmm_15, [8, 196, 768]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:235, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_56: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_50, view_152);  add_50 = view_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_82: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_56, memory_format = torch.contiguous_format)
    var_mean_11 = torch.ops.aten.var_mean.correction(clone_82, [2], correction = 0, keepdim = True)
    getitem_22: "f32[8, 196, 1]" = var_mean_11[0]
    getitem_23: "f32[8, 196, 1]" = var_mean_11[1];  var_mean_11 = None
    sub_35: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_82, getitem_23);  clone_82 = getitem_23 = None
    add_57: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-06);  getitem_22 = None
    rsqrt_11: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
    mul_55: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_11);  sub_35 = rsqrt_11 = None
    mul_56: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_55, arg30_1);  mul_55 = arg30_1 = None
    add_58: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_56, arg31_1);  mul_56 = arg31_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_153: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_58, [1568, 768]);  add_58 = None
    permute_65: "f32[768, 3072]" = torch.ops.aten.permute.default(arg120_1, [1, 0]);  arg120_1 = None
    addmm_16: "f32[1568, 3072]" = torch.ops.aten.addmm.default(arg121_1, view_153, permute_65);  arg121_1 = view_153 = permute_65 = None
    view_154: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(addmm_16, [8, 196, 3072]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_57: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_154, 0.5)
    mul_58: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_154, 0.7071067811865476);  view_154 = None
    erf_5: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_58);  mul_58 = None
    add_59: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_59: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_57, add_59);  mul_57 = add_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_155: "f32[1568, 3072]" = torch.ops.aten.reshape.default(mul_59, [1568, 3072]);  mul_59 = None
    permute_66: "f32[3072, 768]" = torch.ops.aten.permute.default(arg122_1, [1, 0]);  arg122_1 = None
    addmm_17: "f32[1568, 768]" = torch.ops.aten.addmm.default(arg123_1, view_155, permute_66);  arg123_1 = view_155 = permute_66 = None
    view_156: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(addmm_17, [8, 196, 768]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:236, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_60: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_56, view_156);  add_56 = view_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_85: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_60, memory_format = torch.contiguous_format)
    var_mean_12 = torch.ops.aten.var_mean.correction(clone_85, [2], correction = 0, keepdim = True)
    getitem_24: "f32[8, 196, 1]" = var_mean_12[0]
    getitem_25: "f32[8, 196, 1]" = var_mean_12[1];  var_mean_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_169: "f32[1, 16, 1, 1]" = torch.ops.aten.reshape.default(arg34_1, [1, -1, 1, 1]);  arg34_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    sigmoid_12: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_169)
    sub_40: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1.0, sigmoid_12);  sigmoid_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_36: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_85, getitem_25);  clone_85 = getitem_25 = None
    add_61: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-06);  getitem_24 = None
    rsqrt_12: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
    mul_60: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_12);  sub_36 = rsqrt_12 = None
    mul_61: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_60, arg32_1);  mul_60 = arg32_1 = None
    add_62: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_61, arg33_1);  mul_61 = arg33_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_161: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_62, [1568, 768])
    permute_67: "f32[768, 1536]" = torch.ops.aten.permute.default(arg124_1, [1, 0]);  arg124_1 = None
    mm_18: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_161, permute_67);  view_161 = permute_67 = None
    view_162: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(mm_18, [8, 196, 1536]);  mm_18 = None
    view_163: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.reshape.default(view_162, [8, 196, 2, 16, 48]);  view_162 = None
    permute_68: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.permute.default(view_163, [2, 0, 3, 1, 4]);  view_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    select_68: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_68, 0, 0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    expand_53: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(select_68, [8, 16, 196, 48]);  select_68 = None
    clone_89: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_53, memory_format = torch.contiguous_format);  expand_53 = None
    view_166: "f32[128, 196, 48]" = torch.ops.aten.reshape.default(clone_89, [128, 196, 48]);  clone_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    select_69: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_68, 0, 1);  permute_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    permute_71: "f32[8, 16, 48, 196]" = torch.ops.aten.permute.default(select_69, [0, 1, 3, 2]);  select_69 = None
    expand_54: "f32[8, 16, 48, 196]" = torch.ops.aten.expand.default(permute_71, [8, 16, 48, 196]);  permute_71 = None
    clone_90: "f32[8, 16, 48, 196]" = torch.ops.aten.clone.default(expand_54, memory_format = torch.contiguous_format);  expand_54 = None
    view_167: "f32[128, 48, 196]" = torch.ops.aten.reshape.default(clone_90, [128, 48, 196]);  clone_90 = None
    bmm_12: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_166, view_167);  view_166 = view_167 = None
    view_168: "f32[8, 16, 196, 196]" = torch.ops.aten.reshape.default(bmm_12, [8, 16, 196, 196]);  bmm_12 = None
    mul_62: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(view_168, 0.14433756729740643);  view_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    amax_12: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(mul_62, [-1], True)
    sub_38: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_62, amax_12);  mul_62 = amax_12 = None
    exp_12: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_38);  sub_38 = None
    sum_19: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
    div_18: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_12, sum_19);  exp_12 = sum_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    mul_63: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_40, div_18);  sub_40 = div_18 = None
    sigmoid_13: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_169);  view_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:121, code: rel_indices = torch.zeros(1, num_patches, num_patches, 3)
    full_6: "f32[1, 196, 196, 3]" = torch.ops.aten.full.default([1, 196, 196, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:126, code: rel_indices[:, :, :, 2] = indd.unsqueeze(0)
    select_60: "f32[1, 196, 196]" = torch.ops.aten.select.int(full_6, 3, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:122, code: ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
    iota_12: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_157: "i64[1, 14]" = torch.ops.aten.reshape.default(iota_12, [1, -1]);  iota_12 = None
    iota_13: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_158: "i64[14, 1]" = torch.ops.aten.reshape.default(iota_13, [-1, 1]);  iota_13 = None
    sub_37: "i64[14, 14]" = torch.ops.aten.sub.Tensor(view_157, view_158);  view_157 = view_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:123, code: indx = ind.repeat(img_size, img_size)
    repeat_6: "i64[196, 196]" = torch.ops.aten.repeat.default(sub_37, [14, 14])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:125, code: indd = indx ** 2 + indy ** 2
    pow_13: "i64[196, 196]" = torch.ops.aten.pow.Tensor_Scalar(repeat_6, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:124, code: indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
    unsqueeze_36: "i64[14, 1, 14]" = torch.ops.aten.unsqueeze.default(sub_37, 1);  sub_37 = None
    expand_49: "i64[14, 14, 14]" = torch.ops.aten.expand.default(unsqueeze_36, [14, 14, 14]);  unsqueeze_36 = None
    clone_86: "i64[14, 14, 14]" = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
    view_159: "i64[196, 14]" = torch.ops.aten.reshape.default(clone_86, [196, 14]);  clone_86 = None
    unsqueeze_37: "i64[196, 14, 1]" = torch.ops.aten.unsqueeze.default(view_159, 2);  view_159 = None
    expand_50: "i64[196, 14, 14]" = torch.ops.aten.expand.default(unsqueeze_37, [196, 14, 14]);  unsqueeze_37 = None
    clone_87: "i64[196, 14, 14]" = torch.ops.aten.clone.default(expand_50, memory_format = torch.contiguous_format);  expand_50 = None
    view_160: "i64[196, 196]" = torch.ops.aten.reshape.default(clone_87, [196, 196]);  clone_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:125, code: indd = indx ** 2 + indy ** 2
    pow_14: "i64[196, 196]" = torch.ops.aten.pow.Tensor_Scalar(view_160, 2)
    add_63: "i64[196, 196]" = torch.ops.aten.add.Tensor(pow_13, pow_14);  pow_13 = pow_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:126, code: rel_indices[:, :, :, 2] = indd.unsqueeze(0)
    unsqueeze_38: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(add_63, 0);  add_63 = None
    copy_18: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_60, unsqueeze_38);  select_60 = unsqueeze_38 = None
    select_scatter_18: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(full_6, copy_18, 3, 2);  full_6 = copy_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:127, code: rel_indices[:, :, :, 1] = indy.unsqueeze(0)
    select_63: "f32[1, 196, 196]" = torch.ops.aten.select.int(select_scatter_18, 3, 1)
    unsqueeze_39: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(view_160, 0);  view_160 = None
    copy_19: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_63, unsqueeze_39);  select_63 = unsqueeze_39 = None
    select_scatter_19: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(select_scatter_18, copy_19, 3, 1);  select_scatter_18 = copy_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:128, code: rel_indices[:, :, :, 0] = indx.unsqueeze(0)
    select_66: "f32[1, 196, 196]" = torch.ops.aten.select.int(select_scatter_19, 3, 0)
    unsqueeze_40: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(repeat_6, 0);  repeat_6 = None
    copy_20: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_66, unsqueeze_40);  select_66 = unsqueeze_40 = None
    select_scatter_20: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(select_scatter_19, copy_20, 3, 0);  select_scatter_19 = copy_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    expand_52: "f32[8, 196, 196, 3]" = torch.ops.aten.expand.default(select_scatter_20, [8, -1, -1, -1])
    clone_88: "f32[8, 196, 196, 3]" = torch.ops.aten.clone.default(expand_52, memory_format = torch.contiguous_format);  expand_52 = None
    view_164: "f32[307328, 3]" = torch.ops.aten.reshape.default(clone_88, [307328, 3]);  clone_88 = None
    permute_69: "f32[3, 16]" = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
    mm_19: "f32[307328, 16]" = torch.ops.aten.mm.default(view_164, permute_69);  view_164 = permute_69 = None
    view_165: "f32[8, 196, 196, 16]" = torch.ops.aten.reshape.default(mm_19, [8, 196, 196, 16]);  mm_19 = None
    add_64: "f32[8, 196, 196, 16]" = torch.ops.aten.add.Tensor(view_165, arg126_1);  view_165 = arg126_1 = None
    permute_70: "f32[8, 16, 196, 196]" = torch.ops.aten.permute.default(add_64, [0, 3, 1, 2]);  add_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    clone_91: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(permute_70, memory_format = torch.contiguous_format);  permute_70 = None
    amax_13: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(clone_91, [-1], True)
    sub_39: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(clone_91, amax_13);  clone_91 = amax_13 = None
    exp_13: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_39);  sub_39 = None
    sum_20: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
    div_19: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_13, sum_20);  exp_13 = sum_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    mul_64: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sigmoid_13, div_19);  sigmoid_13 = div_19 = None
    add_65: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(mul_63, mul_64);  mul_63 = mul_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    sum_21: "f32[8, 16, 196]" = torch.ops.aten.sum.dim_IntList(add_65, [-1])
    unsqueeze_41: "f32[8, 16, 196, 1]" = torch.ops.aten.unsqueeze.default(sum_21, -1);  sum_21 = None
    div_20: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(add_65, unsqueeze_41);  add_65 = unsqueeze_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_55: "f32[8, 16, 196, 196]" = torch.ops.aten.expand.default(div_20, [8, 16, 196, 196]);  div_20 = None
    view_173: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(expand_55, [128, 196, 196]);  expand_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_170: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_62, [1568, 768]);  add_62 = None
    permute_72: "f32[768, 768]" = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
    mm_20: "f32[1568, 768]" = torch.ops.aten.mm.default(view_170, permute_72);  view_170 = permute_72 = None
    view_171: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_20, [8, 196, 768]);  mm_20 = None
    view_172: "f32[8, 196, 16, 48]" = torch.ops.aten.reshape.default(view_171, [8, 196, 16, 48]);  view_171 = None
    permute_73: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_172, [0, 2, 1, 3]);  view_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_56: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(permute_73, [8, 16, 196, 48]);  permute_73 = None
    clone_93: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
    view_174: "f32[128, 196, 48]" = torch.ops.aten.reshape.default(clone_93, [128, 196, 48]);  clone_93 = None
    bmm_13: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(view_173, view_174);  view_173 = view_174 = None
    view_175: "f32[8, 16, 196, 48]" = torch.ops.aten.reshape.default(bmm_13, [8, 16, 196, 48]);  bmm_13 = None
    permute_74: "f32[8, 196, 16, 48]" = torch.ops.aten.permute.default(view_175, [0, 2, 1, 3]);  view_175 = None
    clone_94: "f32[8, 196, 16, 48]" = torch.ops.aten.clone.default(permute_74, memory_format = torch.contiguous_format);  permute_74 = None
    view_176: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(clone_94, [8, 196, 768]);  clone_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_177: "f32[1568, 768]" = torch.ops.aten.reshape.default(view_176, [1568, 768]);  view_176 = None
    permute_75: "f32[768, 768]" = torch.ops.aten.permute.default(arg128_1, [1, 0]);  arg128_1 = None
    addmm_18: "f32[1568, 768]" = torch.ops.aten.addmm.default(arg129_1, view_177, permute_75);  arg129_1 = view_177 = permute_75 = None
    view_178: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(addmm_18, [8, 196, 768]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:235, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_66: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_60, view_178);  add_60 = view_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_96: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_66, memory_format = torch.contiguous_format)
    var_mean_13 = torch.ops.aten.var_mean.correction(clone_96, [2], correction = 0, keepdim = True)
    getitem_26: "f32[8, 196, 1]" = var_mean_13[0]
    getitem_27: "f32[8, 196, 1]" = var_mean_13[1];  var_mean_13 = None
    sub_41: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_96, getitem_27);  clone_96 = getitem_27 = None
    add_67: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-06);  getitem_26 = None
    rsqrt_13: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
    mul_65: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_13);  sub_41 = rsqrt_13 = None
    mul_66: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_65, arg35_1);  mul_65 = arg35_1 = None
    add_68: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_66, arg36_1);  mul_66 = arg36_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_179: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_68, [1568, 768]);  add_68 = None
    permute_76: "f32[768, 3072]" = torch.ops.aten.permute.default(arg130_1, [1, 0]);  arg130_1 = None
    addmm_19: "f32[1568, 3072]" = torch.ops.aten.addmm.default(arg131_1, view_179, permute_76);  arg131_1 = view_179 = permute_76 = None
    view_180: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(addmm_19, [8, 196, 3072]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_67: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_180, 0.5)
    mul_68: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_180, 0.7071067811865476);  view_180 = None
    erf_6: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_68);  mul_68 = None
    add_69: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_69: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_67, add_69);  mul_67 = add_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_181: "f32[1568, 3072]" = torch.ops.aten.reshape.default(mul_69, [1568, 3072]);  mul_69 = None
    permute_77: "f32[3072, 768]" = torch.ops.aten.permute.default(arg132_1, [1, 0]);  arg132_1 = None
    addmm_20: "f32[1568, 768]" = torch.ops.aten.addmm.default(arg133_1, view_181, permute_77);  arg133_1 = view_181 = permute_77 = None
    view_182: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(addmm_20, [8, 196, 768]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:236, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_70: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_66, view_182);  add_66 = view_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_99: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_70, memory_format = torch.contiguous_format)
    var_mean_14 = torch.ops.aten.var_mean.correction(clone_99, [2], correction = 0, keepdim = True)
    getitem_28: "f32[8, 196, 1]" = var_mean_14[0]
    getitem_29: "f32[8, 196, 1]" = var_mean_14[1];  var_mean_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_195: "f32[1, 16, 1, 1]" = torch.ops.aten.reshape.default(arg39_1, [1, -1, 1, 1]);  arg39_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    sigmoid_14: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_195)
    sub_46: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1.0, sigmoid_14);  sigmoid_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_42: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_99, getitem_29);  clone_99 = getitem_29 = None
    add_71: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-06);  getitem_28 = None
    rsqrt_14: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
    mul_70: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_14);  sub_42 = rsqrt_14 = None
    mul_71: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_70, arg37_1);  mul_70 = arg37_1 = None
    add_72: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_71, arg38_1);  mul_71 = arg38_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_187: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_72, [1568, 768])
    permute_78: "f32[768, 1536]" = torch.ops.aten.permute.default(arg134_1, [1, 0]);  arg134_1 = None
    mm_21: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_187, permute_78);  view_187 = permute_78 = None
    view_188: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(mm_21, [8, 196, 1536]);  mm_21 = None
    view_189: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.reshape.default(view_188, [8, 196, 2, 16, 48]);  view_188 = None
    permute_79: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.permute.default(view_189, [2, 0, 3, 1, 4]);  view_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    select_78: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_79, 0, 0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    expand_61: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(select_78, [8, 16, 196, 48]);  select_78 = None
    clone_103: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_61, memory_format = torch.contiguous_format);  expand_61 = None
    view_192: "f32[128, 196, 48]" = torch.ops.aten.reshape.default(clone_103, [128, 196, 48]);  clone_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    select_79: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_79, 0, 1);  permute_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    permute_82: "f32[8, 16, 48, 196]" = torch.ops.aten.permute.default(select_79, [0, 1, 3, 2]);  select_79 = None
    expand_62: "f32[8, 16, 48, 196]" = torch.ops.aten.expand.default(permute_82, [8, 16, 48, 196]);  permute_82 = None
    clone_104: "f32[8, 16, 48, 196]" = torch.ops.aten.clone.default(expand_62, memory_format = torch.contiguous_format);  expand_62 = None
    view_193: "f32[128, 48, 196]" = torch.ops.aten.reshape.default(clone_104, [128, 48, 196]);  clone_104 = None
    bmm_14: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_192, view_193);  view_192 = view_193 = None
    view_194: "f32[8, 16, 196, 196]" = torch.ops.aten.reshape.default(bmm_14, [8, 16, 196, 196]);  bmm_14 = None
    mul_72: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(view_194, 0.14433756729740643);  view_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    amax_14: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(mul_72, [-1], True)
    sub_44: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_72, amax_14);  mul_72 = amax_14 = None
    exp_14: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_44);  sub_44 = None
    sum_22: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
    div_21: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_14, sum_22);  exp_14 = sum_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    mul_73: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_46, div_21);  sub_46 = div_21 = None
    sigmoid_15: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_195);  view_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:121, code: rel_indices = torch.zeros(1, num_patches, num_patches, 3)
    full_7: "f32[1, 196, 196, 3]" = torch.ops.aten.full.default([1, 196, 196, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:126, code: rel_indices[:, :, :, 2] = indd.unsqueeze(0)
    select_70: "f32[1, 196, 196]" = torch.ops.aten.select.int(full_7, 3, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:122, code: ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
    iota_14: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_183: "i64[1, 14]" = torch.ops.aten.reshape.default(iota_14, [1, -1]);  iota_14 = None
    iota_15: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_184: "i64[14, 1]" = torch.ops.aten.reshape.default(iota_15, [-1, 1]);  iota_15 = None
    sub_43: "i64[14, 14]" = torch.ops.aten.sub.Tensor(view_183, view_184);  view_183 = view_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:123, code: indx = ind.repeat(img_size, img_size)
    repeat_7: "i64[196, 196]" = torch.ops.aten.repeat.default(sub_43, [14, 14])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:125, code: indd = indx ** 2 + indy ** 2
    pow_15: "i64[196, 196]" = torch.ops.aten.pow.Tensor_Scalar(repeat_7, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:124, code: indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
    unsqueeze_42: "i64[14, 1, 14]" = torch.ops.aten.unsqueeze.default(sub_43, 1);  sub_43 = None
    expand_57: "i64[14, 14, 14]" = torch.ops.aten.expand.default(unsqueeze_42, [14, 14, 14]);  unsqueeze_42 = None
    clone_100: "i64[14, 14, 14]" = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
    view_185: "i64[196, 14]" = torch.ops.aten.reshape.default(clone_100, [196, 14]);  clone_100 = None
    unsqueeze_43: "i64[196, 14, 1]" = torch.ops.aten.unsqueeze.default(view_185, 2);  view_185 = None
    expand_58: "i64[196, 14, 14]" = torch.ops.aten.expand.default(unsqueeze_43, [196, 14, 14]);  unsqueeze_43 = None
    clone_101: "i64[196, 14, 14]" = torch.ops.aten.clone.default(expand_58, memory_format = torch.contiguous_format);  expand_58 = None
    view_186: "i64[196, 196]" = torch.ops.aten.reshape.default(clone_101, [196, 196]);  clone_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:125, code: indd = indx ** 2 + indy ** 2
    pow_16: "i64[196, 196]" = torch.ops.aten.pow.Tensor_Scalar(view_186, 2)
    add_73: "i64[196, 196]" = torch.ops.aten.add.Tensor(pow_15, pow_16);  pow_15 = pow_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:126, code: rel_indices[:, :, :, 2] = indd.unsqueeze(0)
    unsqueeze_44: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(add_73, 0);  add_73 = None
    copy_21: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_70, unsqueeze_44);  select_70 = unsqueeze_44 = None
    select_scatter_21: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(full_7, copy_21, 3, 2);  full_7 = copy_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:127, code: rel_indices[:, :, :, 1] = indy.unsqueeze(0)
    select_73: "f32[1, 196, 196]" = torch.ops.aten.select.int(select_scatter_21, 3, 1)
    unsqueeze_45: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(view_186, 0);  view_186 = None
    copy_22: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_73, unsqueeze_45);  select_73 = unsqueeze_45 = None
    select_scatter_22: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(select_scatter_21, copy_22, 3, 1);  select_scatter_21 = copy_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:128, code: rel_indices[:, :, :, 0] = indx.unsqueeze(0)
    select_76: "f32[1, 196, 196]" = torch.ops.aten.select.int(select_scatter_22, 3, 0)
    unsqueeze_46: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(repeat_7, 0);  repeat_7 = None
    copy_23: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_76, unsqueeze_46);  select_76 = unsqueeze_46 = None
    select_scatter_23: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(select_scatter_22, copy_23, 3, 0);  select_scatter_22 = copy_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    expand_60: "f32[8, 196, 196, 3]" = torch.ops.aten.expand.default(select_scatter_23, [8, -1, -1, -1])
    clone_102: "f32[8, 196, 196, 3]" = torch.ops.aten.clone.default(expand_60, memory_format = torch.contiguous_format);  expand_60 = None
    view_190: "f32[307328, 3]" = torch.ops.aten.reshape.default(clone_102, [307328, 3]);  clone_102 = None
    permute_80: "f32[3, 16]" = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
    mm_22: "f32[307328, 16]" = torch.ops.aten.mm.default(view_190, permute_80);  view_190 = permute_80 = None
    view_191: "f32[8, 196, 196, 16]" = torch.ops.aten.reshape.default(mm_22, [8, 196, 196, 16]);  mm_22 = None
    add_74: "f32[8, 196, 196, 16]" = torch.ops.aten.add.Tensor(view_191, arg136_1);  view_191 = arg136_1 = None
    permute_81: "f32[8, 16, 196, 196]" = torch.ops.aten.permute.default(add_74, [0, 3, 1, 2]);  add_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    clone_105: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(permute_81, memory_format = torch.contiguous_format);  permute_81 = None
    amax_15: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(clone_105, [-1], True)
    sub_45: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(clone_105, amax_15);  clone_105 = amax_15 = None
    exp_15: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_45);  sub_45 = None
    sum_23: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
    div_22: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_15, sum_23);  exp_15 = sum_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    mul_74: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sigmoid_15, div_22);  sigmoid_15 = div_22 = None
    add_75: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(mul_73, mul_74);  mul_73 = mul_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    sum_24: "f32[8, 16, 196]" = torch.ops.aten.sum.dim_IntList(add_75, [-1])
    unsqueeze_47: "f32[8, 16, 196, 1]" = torch.ops.aten.unsqueeze.default(sum_24, -1);  sum_24 = None
    div_23: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(add_75, unsqueeze_47);  add_75 = unsqueeze_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_63: "f32[8, 16, 196, 196]" = torch.ops.aten.expand.default(div_23, [8, 16, 196, 196]);  div_23 = None
    view_199: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(expand_63, [128, 196, 196]);  expand_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_196: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_72, [1568, 768]);  add_72 = None
    permute_83: "f32[768, 768]" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
    mm_23: "f32[1568, 768]" = torch.ops.aten.mm.default(view_196, permute_83);  view_196 = permute_83 = None
    view_197: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_23, [8, 196, 768]);  mm_23 = None
    view_198: "f32[8, 196, 16, 48]" = torch.ops.aten.reshape.default(view_197, [8, 196, 16, 48]);  view_197 = None
    permute_84: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_198, [0, 2, 1, 3]);  view_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_64: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(permute_84, [8, 16, 196, 48]);  permute_84 = None
    clone_107: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_64, memory_format = torch.contiguous_format);  expand_64 = None
    view_200: "f32[128, 196, 48]" = torch.ops.aten.reshape.default(clone_107, [128, 196, 48]);  clone_107 = None
    bmm_15: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(view_199, view_200);  view_199 = view_200 = None
    view_201: "f32[8, 16, 196, 48]" = torch.ops.aten.reshape.default(bmm_15, [8, 16, 196, 48]);  bmm_15 = None
    permute_85: "f32[8, 196, 16, 48]" = torch.ops.aten.permute.default(view_201, [0, 2, 1, 3]);  view_201 = None
    clone_108: "f32[8, 196, 16, 48]" = torch.ops.aten.clone.default(permute_85, memory_format = torch.contiguous_format);  permute_85 = None
    view_202: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(clone_108, [8, 196, 768]);  clone_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_203: "f32[1568, 768]" = torch.ops.aten.reshape.default(view_202, [1568, 768]);  view_202 = None
    permute_86: "f32[768, 768]" = torch.ops.aten.permute.default(arg138_1, [1, 0]);  arg138_1 = None
    addmm_21: "f32[1568, 768]" = torch.ops.aten.addmm.default(arg139_1, view_203, permute_86);  arg139_1 = view_203 = permute_86 = None
    view_204: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(addmm_21, [8, 196, 768]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:235, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_76: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_70, view_204);  add_70 = view_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_110: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_76, memory_format = torch.contiguous_format)
    var_mean_15 = torch.ops.aten.var_mean.correction(clone_110, [2], correction = 0, keepdim = True)
    getitem_30: "f32[8, 196, 1]" = var_mean_15[0]
    getitem_31: "f32[8, 196, 1]" = var_mean_15[1];  var_mean_15 = None
    sub_47: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_110, getitem_31);  clone_110 = getitem_31 = None
    add_77: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-06);  getitem_30 = None
    rsqrt_15: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
    mul_75: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_15);  sub_47 = rsqrt_15 = None
    mul_76: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_75, arg40_1);  mul_75 = arg40_1 = None
    add_78: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_76, arg41_1);  mul_76 = arg41_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_205: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_78, [1568, 768]);  add_78 = None
    permute_87: "f32[768, 3072]" = torch.ops.aten.permute.default(arg140_1, [1, 0]);  arg140_1 = None
    addmm_22: "f32[1568, 3072]" = torch.ops.aten.addmm.default(arg141_1, view_205, permute_87);  arg141_1 = view_205 = permute_87 = None
    view_206: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(addmm_22, [8, 196, 3072]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_77: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_206, 0.5)
    mul_78: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_206, 0.7071067811865476);  view_206 = None
    erf_7: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_78);  mul_78 = None
    add_79: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_79: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_77, add_79);  mul_77 = add_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_207: "f32[1568, 3072]" = torch.ops.aten.reshape.default(mul_79, [1568, 3072]);  mul_79 = None
    permute_88: "f32[3072, 768]" = torch.ops.aten.permute.default(arg142_1, [1, 0]);  arg142_1 = None
    addmm_23: "f32[1568, 768]" = torch.ops.aten.addmm.default(arg143_1, view_207, permute_88);  arg143_1 = view_207 = permute_88 = None
    view_208: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(addmm_23, [8, 196, 768]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:236, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_80: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_76, view_208);  add_76 = view_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_113: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_80, memory_format = torch.contiguous_format)
    var_mean_16 = torch.ops.aten.var_mean.correction(clone_113, [2], correction = 0, keepdim = True)
    getitem_32: "f32[8, 196, 1]" = var_mean_16[0]
    getitem_33: "f32[8, 196, 1]" = var_mean_16[1];  var_mean_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_221: "f32[1, 16, 1, 1]" = torch.ops.aten.reshape.default(arg44_1, [1, -1, 1, 1]);  arg44_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    sigmoid_16: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_221)
    sub_52: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1.0, sigmoid_16);  sigmoid_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_48: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_113, getitem_33);  clone_113 = getitem_33 = None
    add_81: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-06);  getitem_32 = None
    rsqrt_16: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
    mul_80: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_16);  sub_48 = rsqrt_16 = None
    mul_81: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_80, arg42_1);  mul_80 = arg42_1 = None
    add_82: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_81, arg43_1);  mul_81 = arg43_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_213: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_82, [1568, 768])
    permute_89: "f32[768, 1536]" = torch.ops.aten.permute.default(arg144_1, [1, 0]);  arg144_1 = None
    mm_24: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_213, permute_89);  view_213 = permute_89 = None
    view_214: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(mm_24, [8, 196, 1536]);  mm_24 = None
    view_215: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.reshape.default(view_214, [8, 196, 2, 16, 48]);  view_214 = None
    permute_90: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.permute.default(view_215, [2, 0, 3, 1, 4]);  view_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    select_88: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_90, 0, 0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    expand_69: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(select_88, [8, 16, 196, 48]);  select_88 = None
    clone_117: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_69, memory_format = torch.contiguous_format);  expand_69 = None
    view_218: "f32[128, 196, 48]" = torch.ops.aten.reshape.default(clone_117, [128, 196, 48]);  clone_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    select_89: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_90, 0, 1);  permute_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    permute_93: "f32[8, 16, 48, 196]" = torch.ops.aten.permute.default(select_89, [0, 1, 3, 2]);  select_89 = None
    expand_70: "f32[8, 16, 48, 196]" = torch.ops.aten.expand.default(permute_93, [8, 16, 48, 196]);  permute_93 = None
    clone_118: "f32[8, 16, 48, 196]" = torch.ops.aten.clone.default(expand_70, memory_format = torch.contiguous_format);  expand_70 = None
    view_219: "f32[128, 48, 196]" = torch.ops.aten.reshape.default(clone_118, [128, 48, 196]);  clone_118 = None
    bmm_16: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_218, view_219);  view_218 = view_219 = None
    view_220: "f32[8, 16, 196, 196]" = torch.ops.aten.reshape.default(bmm_16, [8, 16, 196, 196]);  bmm_16 = None
    mul_82: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(view_220, 0.14433756729740643);  view_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    amax_16: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(mul_82, [-1], True)
    sub_50: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_82, amax_16);  mul_82 = amax_16 = None
    exp_16: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_50);  sub_50 = None
    sum_25: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
    div_24: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_16, sum_25);  exp_16 = sum_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    mul_83: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_52, div_24);  sub_52 = div_24 = None
    sigmoid_17: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_221);  view_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:121, code: rel_indices = torch.zeros(1, num_patches, num_patches, 3)
    full_8: "f32[1, 196, 196, 3]" = torch.ops.aten.full.default([1, 196, 196, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:126, code: rel_indices[:, :, :, 2] = indd.unsqueeze(0)
    select_80: "f32[1, 196, 196]" = torch.ops.aten.select.int(full_8, 3, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:122, code: ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
    iota_16: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_209: "i64[1, 14]" = torch.ops.aten.reshape.default(iota_16, [1, -1]);  iota_16 = None
    iota_17: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_210: "i64[14, 1]" = torch.ops.aten.reshape.default(iota_17, [-1, 1]);  iota_17 = None
    sub_49: "i64[14, 14]" = torch.ops.aten.sub.Tensor(view_209, view_210);  view_209 = view_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:123, code: indx = ind.repeat(img_size, img_size)
    repeat_8: "i64[196, 196]" = torch.ops.aten.repeat.default(sub_49, [14, 14])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:125, code: indd = indx ** 2 + indy ** 2
    pow_17: "i64[196, 196]" = torch.ops.aten.pow.Tensor_Scalar(repeat_8, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:124, code: indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
    unsqueeze_48: "i64[14, 1, 14]" = torch.ops.aten.unsqueeze.default(sub_49, 1);  sub_49 = None
    expand_65: "i64[14, 14, 14]" = torch.ops.aten.expand.default(unsqueeze_48, [14, 14, 14]);  unsqueeze_48 = None
    clone_114: "i64[14, 14, 14]" = torch.ops.aten.clone.default(expand_65, memory_format = torch.contiguous_format);  expand_65 = None
    view_211: "i64[196, 14]" = torch.ops.aten.reshape.default(clone_114, [196, 14]);  clone_114 = None
    unsqueeze_49: "i64[196, 14, 1]" = torch.ops.aten.unsqueeze.default(view_211, 2);  view_211 = None
    expand_66: "i64[196, 14, 14]" = torch.ops.aten.expand.default(unsqueeze_49, [196, 14, 14]);  unsqueeze_49 = None
    clone_115: "i64[196, 14, 14]" = torch.ops.aten.clone.default(expand_66, memory_format = torch.contiguous_format);  expand_66 = None
    view_212: "i64[196, 196]" = torch.ops.aten.reshape.default(clone_115, [196, 196]);  clone_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:125, code: indd = indx ** 2 + indy ** 2
    pow_18: "i64[196, 196]" = torch.ops.aten.pow.Tensor_Scalar(view_212, 2)
    add_83: "i64[196, 196]" = torch.ops.aten.add.Tensor(pow_17, pow_18);  pow_17 = pow_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:126, code: rel_indices[:, :, :, 2] = indd.unsqueeze(0)
    unsqueeze_50: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(add_83, 0);  add_83 = None
    copy_24: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_80, unsqueeze_50);  select_80 = unsqueeze_50 = None
    select_scatter_24: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(full_8, copy_24, 3, 2);  full_8 = copy_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:127, code: rel_indices[:, :, :, 1] = indy.unsqueeze(0)
    select_83: "f32[1, 196, 196]" = torch.ops.aten.select.int(select_scatter_24, 3, 1)
    unsqueeze_51: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(view_212, 0);  view_212 = None
    copy_25: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_83, unsqueeze_51);  select_83 = unsqueeze_51 = None
    select_scatter_25: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(select_scatter_24, copy_25, 3, 1);  select_scatter_24 = copy_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:128, code: rel_indices[:, :, :, 0] = indx.unsqueeze(0)
    select_86: "f32[1, 196, 196]" = torch.ops.aten.select.int(select_scatter_25, 3, 0)
    unsqueeze_52: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(repeat_8, 0);  repeat_8 = None
    copy_26: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_86, unsqueeze_52);  select_86 = unsqueeze_52 = None
    select_scatter_26: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(select_scatter_25, copy_26, 3, 0);  select_scatter_25 = copy_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    expand_68: "f32[8, 196, 196, 3]" = torch.ops.aten.expand.default(select_scatter_26, [8, -1, -1, -1])
    clone_116: "f32[8, 196, 196, 3]" = torch.ops.aten.clone.default(expand_68, memory_format = torch.contiguous_format);  expand_68 = None
    view_216: "f32[307328, 3]" = torch.ops.aten.reshape.default(clone_116, [307328, 3]);  clone_116 = None
    permute_91: "f32[3, 16]" = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
    mm_25: "f32[307328, 16]" = torch.ops.aten.mm.default(view_216, permute_91);  view_216 = permute_91 = None
    view_217: "f32[8, 196, 196, 16]" = torch.ops.aten.reshape.default(mm_25, [8, 196, 196, 16]);  mm_25 = None
    add_84: "f32[8, 196, 196, 16]" = torch.ops.aten.add.Tensor(view_217, arg146_1);  view_217 = arg146_1 = None
    permute_92: "f32[8, 16, 196, 196]" = torch.ops.aten.permute.default(add_84, [0, 3, 1, 2]);  add_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    clone_119: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(permute_92, memory_format = torch.contiguous_format);  permute_92 = None
    amax_17: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(clone_119, [-1], True)
    sub_51: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(clone_119, amax_17);  clone_119 = amax_17 = None
    exp_17: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_51);  sub_51 = None
    sum_26: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
    div_25: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_17, sum_26);  exp_17 = sum_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    mul_84: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sigmoid_17, div_25);  sigmoid_17 = div_25 = None
    add_85: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(mul_83, mul_84);  mul_83 = mul_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    sum_27: "f32[8, 16, 196]" = torch.ops.aten.sum.dim_IntList(add_85, [-1])
    unsqueeze_53: "f32[8, 16, 196, 1]" = torch.ops.aten.unsqueeze.default(sum_27, -1);  sum_27 = None
    div_26: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(add_85, unsqueeze_53);  add_85 = unsqueeze_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_71: "f32[8, 16, 196, 196]" = torch.ops.aten.expand.default(div_26, [8, 16, 196, 196]);  div_26 = None
    view_225: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(expand_71, [128, 196, 196]);  expand_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_222: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_82, [1568, 768]);  add_82 = None
    permute_94: "f32[768, 768]" = torch.ops.aten.permute.default(arg147_1, [1, 0]);  arg147_1 = None
    mm_26: "f32[1568, 768]" = torch.ops.aten.mm.default(view_222, permute_94);  view_222 = permute_94 = None
    view_223: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_26, [8, 196, 768]);  mm_26 = None
    view_224: "f32[8, 196, 16, 48]" = torch.ops.aten.reshape.default(view_223, [8, 196, 16, 48]);  view_223 = None
    permute_95: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_224, [0, 2, 1, 3]);  view_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_72: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(permute_95, [8, 16, 196, 48]);  permute_95 = None
    clone_121: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_72, memory_format = torch.contiguous_format);  expand_72 = None
    view_226: "f32[128, 196, 48]" = torch.ops.aten.reshape.default(clone_121, [128, 196, 48]);  clone_121 = None
    bmm_17: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(view_225, view_226);  view_225 = view_226 = None
    view_227: "f32[8, 16, 196, 48]" = torch.ops.aten.reshape.default(bmm_17, [8, 16, 196, 48]);  bmm_17 = None
    permute_96: "f32[8, 196, 16, 48]" = torch.ops.aten.permute.default(view_227, [0, 2, 1, 3]);  view_227 = None
    clone_122: "f32[8, 196, 16, 48]" = torch.ops.aten.clone.default(permute_96, memory_format = torch.contiguous_format);  permute_96 = None
    view_228: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(clone_122, [8, 196, 768]);  clone_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_229: "f32[1568, 768]" = torch.ops.aten.reshape.default(view_228, [1568, 768]);  view_228 = None
    permute_97: "f32[768, 768]" = torch.ops.aten.permute.default(arg148_1, [1, 0]);  arg148_1 = None
    addmm_24: "f32[1568, 768]" = torch.ops.aten.addmm.default(arg149_1, view_229, permute_97);  arg149_1 = view_229 = permute_97 = None
    view_230: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(addmm_24, [8, 196, 768]);  addmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:235, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_86: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_80, view_230);  add_80 = view_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_124: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_86, memory_format = torch.contiguous_format)
    var_mean_17 = torch.ops.aten.var_mean.correction(clone_124, [2], correction = 0, keepdim = True)
    getitem_34: "f32[8, 196, 1]" = var_mean_17[0]
    getitem_35: "f32[8, 196, 1]" = var_mean_17[1];  var_mean_17 = None
    sub_53: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_124, getitem_35);  clone_124 = getitem_35 = None
    add_87: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-06);  getitem_34 = None
    rsqrt_17: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
    mul_85: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_17);  sub_53 = rsqrt_17 = None
    mul_86: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_85, arg45_1);  mul_85 = arg45_1 = None
    add_88: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_86, arg46_1);  mul_86 = arg46_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_231: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_88, [1568, 768]);  add_88 = None
    permute_98: "f32[768, 3072]" = torch.ops.aten.permute.default(arg150_1, [1, 0]);  arg150_1 = None
    addmm_25: "f32[1568, 3072]" = torch.ops.aten.addmm.default(arg151_1, view_231, permute_98);  arg151_1 = view_231 = permute_98 = None
    view_232: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(addmm_25, [8, 196, 3072]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_87: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_232, 0.5)
    mul_88: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_232, 0.7071067811865476);  view_232 = None
    erf_8: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_88);  mul_88 = None
    add_89: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_89: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_87, add_89);  mul_87 = add_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_233: "f32[1568, 3072]" = torch.ops.aten.reshape.default(mul_89, [1568, 3072]);  mul_89 = None
    permute_99: "f32[3072, 768]" = torch.ops.aten.permute.default(arg152_1, [1, 0]);  arg152_1 = None
    addmm_26: "f32[1568, 768]" = torch.ops.aten.addmm.default(arg153_1, view_233, permute_99);  arg153_1 = view_233 = permute_99 = None
    view_234: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(addmm_26, [8, 196, 768]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:236, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_90: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_86, view_234);  add_86 = view_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_127: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_90, memory_format = torch.contiguous_format)
    var_mean_18 = torch.ops.aten.var_mean.correction(clone_127, [2], correction = 0, keepdim = True)
    getitem_36: "f32[8, 196, 1]" = var_mean_18[0]
    getitem_37: "f32[8, 196, 1]" = var_mean_18[1];  var_mean_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_247: "f32[1, 16, 1, 1]" = torch.ops.aten.reshape.default(arg49_1, [1, -1, 1, 1]);  arg49_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    sigmoid_18: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_247)
    sub_58: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1.0, sigmoid_18);  sigmoid_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_54: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_127, getitem_37);  clone_127 = getitem_37 = None
    add_91: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-06);  getitem_36 = None
    rsqrt_18: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_91);  add_91 = None
    mul_90: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_18);  sub_54 = rsqrt_18 = None
    mul_91: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_90, arg47_1);  mul_90 = arg47_1 = None
    add_92: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_91, arg48_1);  mul_91 = arg48_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_239: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_92, [1568, 768])
    permute_100: "f32[768, 1536]" = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
    mm_27: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_239, permute_100);  view_239 = permute_100 = None
    view_240: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(mm_27, [8, 196, 1536]);  mm_27 = None
    view_241: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.reshape.default(view_240, [8, 196, 2, 16, 48]);  view_240 = None
    permute_101: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.permute.default(view_241, [2, 0, 3, 1, 4]);  view_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    select_98: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_101, 0, 0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    expand_77: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(select_98, [8, 16, 196, 48]);  select_98 = None
    clone_131: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_77, memory_format = torch.contiguous_format);  expand_77 = None
    view_244: "f32[128, 196, 48]" = torch.ops.aten.reshape.default(clone_131, [128, 196, 48]);  clone_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    select_99: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_101, 0, 1);  permute_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    permute_104: "f32[8, 16, 48, 196]" = torch.ops.aten.permute.default(select_99, [0, 1, 3, 2]);  select_99 = None
    expand_78: "f32[8, 16, 48, 196]" = torch.ops.aten.expand.default(permute_104, [8, 16, 48, 196]);  permute_104 = None
    clone_132: "f32[8, 16, 48, 196]" = torch.ops.aten.clone.default(expand_78, memory_format = torch.contiguous_format);  expand_78 = None
    view_245: "f32[128, 48, 196]" = torch.ops.aten.reshape.default(clone_132, [128, 48, 196]);  clone_132 = None
    bmm_18: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_244, view_245);  view_244 = view_245 = None
    view_246: "f32[8, 16, 196, 196]" = torch.ops.aten.reshape.default(bmm_18, [8, 16, 196, 196]);  bmm_18 = None
    mul_92: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(view_246, 0.14433756729740643);  view_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    amax_18: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(mul_92, [-1], True)
    sub_56: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_92, amax_18);  mul_92 = amax_18 = None
    exp_18: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_56);  sub_56 = None
    sum_28: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
    div_27: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_18, sum_28);  exp_18 = sum_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    mul_93: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_58, div_27);  sub_58 = div_27 = None
    sigmoid_19: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_247);  view_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:121, code: rel_indices = torch.zeros(1, num_patches, num_patches, 3)
    full_9: "f32[1, 196, 196, 3]" = torch.ops.aten.full.default([1, 196, 196, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:126, code: rel_indices[:, :, :, 2] = indd.unsqueeze(0)
    select_90: "f32[1, 196, 196]" = torch.ops.aten.select.int(full_9, 3, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:122, code: ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
    iota_18: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_235: "i64[1, 14]" = torch.ops.aten.reshape.default(iota_18, [1, -1]);  iota_18 = None
    iota_19: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_236: "i64[14, 1]" = torch.ops.aten.reshape.default(iota_19, [-1, 1]);  iota_19 = None
    sub_55: "i64[14, 14]" = torch.ops.aten.sub.Tensor(view_235, view_236);  view_235 = view_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:123, code: indx = ind.repeat(img_size, img_size)
    repeat_9: "i64[196, 196]" = torch.ops.aten.repeat.default(sub_55, [14, 14])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:125, code: indd = indx ** 2 + indy ** 2
    pow_19: "i64[196, 196]" = torch.ops.aten.pow.Tensor_Scalar(repeat_9, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:124, code: indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
    unsqueeze_54: "i64[14, 1, 14]" = torch.ops.aten.unsqueeze.default(sub_55, 1);  sub_55 = None
    expand_73: "i64[14, 14, 14]" = torch.ops.aten.expand.default(unsqueeze_54, [14, 14, 14]);  unsqueeze_54 = None
    clone_128: "i64[14, 14, 14]" = torch.ops.aten.clone.default(expand_73, memory_format = torch.contiguous_format);  expand_73 = None
    view_237: "i64[196, 14]" = torch.ops.aten.reshape.default(clone_128, [196, 14]);  clone_128 = None
    unsqueeze_55: "i64[196, 14, 1]" = torch.ops.aten.unsqueeze.default(view_237, 2);  view_237 = None
    expand_74: "i64[196, 14, 14]" = torch.ops.aten.expand.default(unsqueeze_55, [196, 14, 14]);  unsqueeze_55 = None
    clone_129: "i64[196, 14, 14]" = torch.ops.aten.clone.default(expand_74, memory_format = torch.contiguous_format);  expand_74 = None
    view_238: "i64[196, 196]" = torch.ops.aten.reshape.default(clone_129, [196, 196]);  clone_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:125, code: indd = indx ** 2 + indy ** 2
    pow_20: "i64[196, 196]" = torch.ops.aten.pow.Tensor_Scalar(view_238, 2)
    add_93: "i64[196, 196]" = torch.ops.aten.add.Tensor(pow_19, pow_20);  pow_19 = pow_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:126, code: rel_indices[:, :, :, 2] = indd.unsqueeze(0)
    unsqueeze_56: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(add_93, 0);  add_93 = None
    copy_27: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_90, unsqueeze_56);  select_90 = unsqueeze_56 = None
    select_scatter_27: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(full_9, copy_27, 3, 2);  full_9 = copy_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:127, code: rel_indices[:, :, :, 1] = indy.unsqueeze(0)
    select_93: "f32[1, 196, 196]" = torch.ops.aten.select.int(select_scatter_27, 3, 1)
    unsqueeze_57: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(view_238, 0);  view_238 = None
    copy_28: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_93, unsqueeze_57);  select_93 = unsqueeze_57 = None
    select_scatter_28: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(select_scatter_27, copy_28, 3, 1);  select_scatter_27 = copy_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:128, code: rel_indices[:, :, :, 0] = indx.unsqueeze(0)
    select_96: "f32[1, 196, 196]" = torch.ops.aten.select.int(select_scatter_28, 3, 0)
    unsqueeze_58: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(repeat_9, 0);  repeat_9 = None
    copy_29: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_96, unsqueeze_58);  select_96 = unsqueeze_58 = None
    select_scatter_29: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(select_scatter_28, copy_29, 3, 0);  select_scatter_28 = copy_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    expand_76: "f32[8, 196, 196, 3]" = torch.ops.aten.expand.default(select_scatter_29, [8, -1, -1, -1])
    clone_130: "f32[8, 196, 196, 3]" = torch.ops.aten.clone.default(expand_76, memory_format = torch.contiguous_format);  expand_76 = None
    view_242: "f32[307328, 3]" = torch.ops.aten.reshape.default(clone_130, [307328, 3]);  clone_130 = None
    permute_102: "f32[3, 16]" = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
    mm_28: "f32[307328, 16]" = torch.ops.aten.mm.default(view_242, permute_102);  view_242 = permute_102 = None
    view_243: "f32[8, 196, 196, 16]" = torch.ops.aten.reshape.default(mm_28, [8, 196, 196, 16]);  mm_28 = None
    add_94: "f32[8, 196, 196, 16]" = torch.ops.aten.add.Tensor(view_243, arg156_1);  view_243 = arg156_1 = None
    permute_103: "f32[8, 16, 196, 196]" = torch.ops.aten.permute.default(add_94, [0, 3, 1, 2]);  add_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    clone_133: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(permute_103, memory_format = torch.contiguous_format);  permute_103 = None
    amax_19: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(clone_133, [-1], True)
    sub_57: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(clone_133, amax_19);  clone_133 = amax_19 = None
    exp_19: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_57);  sub_57 = None
    sum_29: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
    div_28: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_19, sum_29);  exp_19 = sum_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    mul_94: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sigmoid_19, div_28);  sigmoid_19 = div_28 = None
    add_95: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(mul_93, mul_94);  mul_93 = mul_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    sum_30: "f32[8, 16, 196]" = torch.ops.aten.sum.dim_IntList(add_95, [-1])
    unsqueeze_59: "f32[8, 16, 196, 1]" = torch.ops.aten.unsqueeze.default(sum_30, -1);  sum_30 = None
    div_29: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(add_95, unsqueeze_59);  add_95 = unsqueeze_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_79: "f32[8, 16, 196, 196]" = torch.ops.aten.expand.default(div_29, [8, 16, 196, 196]);  div_29 = None
    view_251: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(expand_79, [128, 196, 196]);  expand_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_248: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_92, [1568, 768]);  add_92 = None
    permute_105: "f32[768, 768]" = torch.ops.aten.permute.default(arg157_1, [1, 0]);  arg157_1 = None
    mm_29: "f32[1568, 768]" = torch.ops.aten.mm.default(view_248, permute_105);  view_248 = permute_105 = None
    view_249: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_29, [8, 196, 768]);  mm_29 = None
    view_250: "f32[8, 196, 16, 48]" = torch.ops.aten.reshape.default(view_249, [8, 196, 16, 48]);  view_249 = None
    permute_106: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_250, [0, 2, 1, 3]);  view_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_80: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(permute_106, [8, 16, 196, 48]);  permute_106 = None
    clone_135: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_80, memory_format = torch.contiguous_format);  expand_80 = None
    view_252: "f32[128, 196, 48]" = torch.ops.aten.reshape.default(clone_135, [128, 196, 48]);  clone_135 = None
    bmm_19: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(view_251, view_252);  view_251 = view_252 = None
    view_253: "f32[8, 16, 196, 48]" = torch.ops.aten.reshape.default(bmm_19, [8, 16, 196, 48]);  bmm_19 = None
    permute_107: "f32[8, 196, 16, 48]" = torch.ops.aten.permute.default(view_253, [0, 2, 1, 3]);  view_253 = None
    clone_136: "f32[8, 196, 16, 48]" = torch.ops.aten.clone.default(permute_107, memory_format = torch.contiguous_format);  permute_107 = None
    view_254: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(clone_136, [8, 196, 768]);  clone_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_255: "f32[1568, 768]" = torch.ops.aten.reshape.default(view_254, [1568, 768]);  view_254 = None
    permute_108: "f32[768, 768]" = torch.ops.aten.permute.default(arg158_1, [1, 0]);  arg158_1 = None
    addmm_27: "f32[1568, 768]" = torch.ops.aten.addmm.default(arg159_1, view_255, permute_108);  arg159_1 = view_255 = permute_108 = None
    view_256: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(addmm_27, [8, 196, 768]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:235, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_96: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_90, view_256);  add_90 = view_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_138: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_96, memory_format = torch.contiguous_format)
    var_mean_19 = torch.ops.aten.var_mean.correction(clone_138, [2], correction = 0, keepdim = True)
    getitem_38: "f32[8, 196, 1]" = var_mean_19[0]
    getitem_39: "f32[8, 196, 1]" = var_mean_19[1];  var_mean_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:364, code: cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
    expand: "f32[8, 1, 768]" = torch.ops.aten.expand.default(arg1_1, [8, -1, -1]);  arg1_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_59: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_138, getitem_39);  clone_138 = getitem_39 = None
    add_97: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-06);  getitem_38 = None
    rsqrt_19: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_97);  add_97 = None
    mul_95: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_19);  sub_59 = rsqrt_19 = None
    mul_96: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_95, arg50_1);  mul_95 = arg50_1 = None
    add_98: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_96, arg51_1);  mul_96 = arg51_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_257: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_98, [1568, 768]);  add_98 = None
    permute_109: "f32[768, 3072]" = torch.ops.aten.permute.default(arg160_1, [1, 0]);  arg160_1 = None
    addmm_28: "f32[1568, 3072]" = torch.ops.aten.addmm.default(arg161_1, view_257, permute_109);  arg161_1 = view_257 = permute_109 = None
    view_258: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(addmm_28, [8, 196, 3072]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_97: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_258, 0.5)
    mul_98: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_258, 0.7071067811865476);  view_258 = None
    erf_9: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_98);  mul_98 = None
    add_99: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_99: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_97, add_99);  mul_97 = add_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_259: "f32[1568, 3072]" = torch.ops.aten.reshape.default(mul_99, [1568, 3072]);  mul_99 = None
    permute_110: "f32[3072, 768]" = torch.ops.aten.permute.default(arg162_1, [1, 0]);  arg162_1 = None
    addmm_29: "f32[1568, 768]" = torch.ops.aten.addmm.default(arg163_1, view_259, permute_110);  arg163_1 = view_259 = permute_110 = None
    view_260: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(addmm_29, [8, 196, 768]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:236, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_100: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_96, view_260);  add_96 = view_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:367, code: x = torch.cat((cls_tokens, x), dim=1)
    cat: "f32[8, 197, 768]" = torch.ops.aten.cat.default([expand, add_100], 1);  expand = add_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_20 = torch.ops.aten.var_mean.correction(cat, [2], correction = 0, keepdim = True)
    getitem_40: "f32[8, 197, 1]" = var_mean_20[0]
    getitem_41: "f32[8, 197, 1]" = var_mean_20[1];  var_mean_20 = None
    sub_60: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(cat, getitem_41);  getitem_41 = None
    add_101: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-06);  getitem_40 = None
    rsqrt_20: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_101);  add_101 = None
    mul_100: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_20);  sub_60 = rsqrt_20 = None
    mul_101: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_100, arg52_1);  mul_100 = arg52_1 = None
    add_102: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_101, arg53_1);  mul_101 = arg53_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:175, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_261: "f32[1576, 768]" = torch.ops.aten.reshape.default(add_102, [1576, 768]);  add_102 = None
    permute_111: "f32[768, 2304]" = torch.ops.aten.permute.default(arg164_1, [1, 0]);  arg164_1 = None
    mm_30: "f32[1576, 2304]" = torch.ops.aten.mm.default(view_261, permute_111);  view_261 = permute_111 = None
    view_262: "f32[8, 197, 2304]" = torch.ops.aten.reshape.default(mm_30, [8, 197, 2304]);  mm_30 = None
    view_263: "f32[8, 197, 3, 16, 48]" = torch.ops.aten.reshape.default(view_262, [8, 197, 3, 16, 48]);  view_262 = None
    permute_112: "f32[3, 8, 16, 197, 48]" = torch.ops.aten.permute.default(view_263, [2, 0, 3, 1, 4]);  view_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:176, code: q, k, v = qkv.unbind(0)
    unbind = torch.ops.aten.unbind.int(permute_112);  permute_112 = None
    getitem_42: "f32[8, 16, 197, 48]" = unbind[0]
    getitem_43: "f32[8, 16, 197, 48]" = unbind[1]
    getitem_44: "f32[8, 16, 197, 48]" = unbind[2];  unbind = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:178, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    expand_81: "f32[8, 16, 197, 48]" = torch.ops.aten.expand.default(getitem_42, [8, 16, 197, 48]);  getitem_42 = None
    clone_141: "f32[8, 16, 197, 48]" = torch.ops.aten.clone.default(expand_81, memory_format = torch.contiguous_format);  expand_81 = None
    view_264: "f32[128, 197, 48]" = torch.ops.aten.reshape.default(clone_141, [128, 197, 48]);  clone_141 = None
    permute_113: "f32[8, 16, 48, 197]" = torch.ops.aten.permute.default(getitem_43, [0, 1, 3, 2]);  getitem_43 = None
    expand_82: "f32[8, 16, 48, 197]" = torch.ops.aten.expand.default(permute_113, [8, 16, 48, 197]);  permute_113 = None
    clone_142: "f32[8, 16, 48, 197]" = torch.ops.aten.clone.default(expand_82, memory_format = torch.contiguous_format);  expand_82 = None
    view_265: "f32[128, 48, 197]" = torch.ops.aten.reshape.default(clone_142, [128, 48, 197]);  clone_142 = None
    bmm_20: "f32[128, 197, 197]" = torch.ops.aten.bmm.default(view_264, view_265);  view_264 = view_265 = None
    view_266: "f32[8, 16, 197, 197]" = torch.ops.aten.reshape.default(bmm_20, [8, 16, 197, 197]);  bmm_20 = None
    mul_102: "f32[8, 16, 197, 197]" = torch.ops.aten.mul.Tensor(view_266, 0.14433756729740643);  view_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:179, code: attn = attn.softmax(dim=-1)
    amax_20: "f32[8, 16, 197, 1]" = torch.ops.aten.amax.default(mul_102, [-1], True)
    sub_61: "f32[8, 16, 197, 197]" = torch.ops.aten.sub.Tensor(mul_102, amax_20);  mul_102 = amax_20 = None
    exp_20: "f32[8, 16, 197, 197]" = torch.ops.aten.exp.default(sub_61);  sub_61 = None
    sum_31: "f32[8, 16, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
    div_30: "f32[8, 16, 197, 197]" = torch.ops.aten.div.Tensor(exp_20, sum_31);  exp_20 = sum_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:182, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_83: "f32[8, 16, 197, 197]" = torch.ops.aten.expand.default(div_30, [8, 16, 197, 197]);  div_30 = None
    view_267: "f32[128, 197, 197]" = torch.ops.aten.reshape.default(expand_83, [128, 197, 197]);  expand_83 = None
    expand_84: "f32[8, 16, 197, 48]" = torch.ops.aten.expand.default(getitem_44, [8, 16, 197, 48]);  getitem_44 = None
    clone_144: "f32[8, 16, 197, 48]" = torch.ops.aten.clone.default(expand_84, memory_format = torch.contiguous_format);  expand_84 = None
    view_268: "f32[128, 197, 48]" = torch.ops.aten.reshape.default(clone_144, [128, 197, 48]);  clone_144 = None
    bmm_21: "f32[128, 197, 48]" = torch.ops.aten.bmm.default(view_267, view_268);  view_267 = view_268 = None
    view_269: "f32[8, 16, 197, 48]" = torch.ops.aten.reshape.default(bmm_21, [8, 16, 197, 48]);  bmm_21 = None
    permute_114: "f32[8, 197, 16, 48]" = torch.ops.aten.permute.default(view_269, [0, 2, 1, 3]);  view_269 = None
    clone_145: "f32[8, 197, 16, 48]" = torch.ops.aten.clone.default(permute_114, memory_format = torch.contiguous_format);  permute_114 = None
    view_270: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(clone_145, [8, 197, 768]);  clone_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:183, code: x = self.proj(x)
    view_271: "f32[1576, 768]" = torch.ops.aten.reshape.default(view_270, [1576, 768]);  view_270 = None
    permute_115: "f32[768, 768]" = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
    addmm_30: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg166_1, view_271, permute_115);  arg166_1 = view_271 = permute_115 = None
    view_272: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_30, [8, 197, 768]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:235, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_103: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(cat, view_272);  cat = view_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_21 = torch.ops.aten.var_mean.correction(add_103, [2], correction = 0, keepdim = True)
    getitem_45: "f32[8, 197, 1]" = var_mean_21[0]
    getitem_46: "f32[8, 197, 1]" = var_mean_21[1];  var_mean_21 = None
    sub_62: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_103, getitem_46);  getitem_46 = None
    add_104: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_45, 1e-06);  getitem_45 = None
    rsqrt_21: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_104);  add_104 = None
    mul_103: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_21);  sub_62 = rsqrt_21 = None
    mul_104: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_103, arg54_1);  mul_103 = arg54_1 = None
    add_105: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_104, arg55_1);  mul_104 = arg55_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_273: "f32[1576, 768]" = torch.ops.aten.reshape.default(add_105, [1576, 768]);  add_105 = None
    permute_116: "f32[768, 3072]" = torch.ops.aten.permute.default(arg167_1, [1, 0]);  arg167_1 = None
    addmm_31: "f32[1576, 3072]" = torch.ops.aten.addmm.default(arg168_1, view_273, permute_116);  arg168_1 = view_273 = permute_116 = None
    view_274: "f32[8, 197, 3072]" = torch.ops.aten.reshape.default(addmm_31, [8, 197, 3072]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_105: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_274, 0.5)
    mul_106: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_274, 0.7071067811865476);  view_274 = None
    erf_10: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_106);  mul_106 = None
    add_106: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_107: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_105, add_106);  mul_105 = add_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_275: "f32[1576, 3072]" = torch.ops.aten.reshape.default(mul_107, [1576, 3072]);  mul_107 = None
    permute_117: "f32[3072, 768]" = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
    addmm_32: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg170_1, view_275, permute_117);  arg170_1 = view_275 = permute_117 = None
    view_276: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_32, [8, 197, 768]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:236, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_107: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_103, view_276);  add_103 = view_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_22 = torch.ops.aten.var_mean.correction(add_107, [2], correction = 0, keepdim = True)
    getitem_47: "f32[8, 197, 1]" = var_mean_22[0]
    getitem_48: "f32[8, 197, 1]" = var_mean_22[1];  var_mean_22 = None
    sub_63: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_107, getitem_48);  getitem_48 = None
    add_108: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_47, 1e-06);  getitem_47 = None
    rsqrt_22: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
    mul_108: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_22);  sub_63 = rsqrt_22 = None
    mul_109: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_108, arg56_1);  mul_108 = arg56_1 = None
    add_109: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_109, arg57_1);  mul_109 = arg57_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:175, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_277: "f32[1576, 768]" = torch.ops.aten.reshape.default(add_109, [1576, 768]);  add_109 = None
    permute_118: "f32[768, 2304]" = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
    mm_31: "f32[1576, 2304]" = torch.ops.aten.mm.default(view_277, permute_118);  view_277 = permute_118 = None
    view_278: "f32[8, 197, 2304]" = torch.ops.aten.reshape.default(mm_31, [8, 197, 2304]);  mm_31 = None
    view_279: "f32[8, 197, 3, 16, 48]" = torch.ops.aten.reshape.default(view_278, [8, 197, 3, 16, 48]);  view_278 = None
    permute_119: "f32[3, 8, 16, 197, 48]" = torch.ops.aten.permute.default(view_279, [2, 0, 3, 1, 4]);  view_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:176, code: q, k, v = qkv.unbind(0)
    unbind_1 = torch.ops.aten.unbind.int(permute_119);  permute_119 = None
    getitem_49: "f32[8, 16, 197, 48]" = unbind_1[0]
    getitem_50: "f32[8, 16, 197, 48]" = unbind_1[1]
    getitem_51: "f32[8, 16, 197, 48]" = unbind_1[2];  unbind_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:178, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    expand_85: "f32[8, 16, 197, 48]" = torch.ops.aten.expand.default(getitem_49, [8, 16, 197, 48]);  getitem_49 = None
    clone_149: "f32[8, 16, 197, 48]" = torch.ops.aten.clone.default(expand_85, memory_format = torch.contiguous_format);  expand_85 = None
    view_280: "f32[128, 197, 48]" = torch.ops.aten.reshape.default(clone_149, [128, 197, 48]);  clone_149 = None
    permute_120: "f32[8, 16, 48, 197]" = torch.ops.aten.permute.default(getitem_50, [0, 1, 3, 2]);  getitem_50 = None
    expand_86: "f32[8, 16, 48, 197]" = torch.ops.aten.expand.default(permute_120, [8, 16, 48, 197]);  permute_120 = None
    clone_150: "f32[8, 16, 48, 197]" = torch.ops.aten.clone.default(expand_86, memory_format = torch.contiguous_format);  expand_86 = None
    view_281: "f32[128, 48, 197]" = torch.ops.aten.reshape.default(clone_150, [128, 48, 197]);  clone_150 = None
    bmm_22: "f32[128, 197, 197]" = torch.ops.aten.bmm.default(view_280, view_281);  view_280 = view_281 = None
    view_282: "f32[8, 16, 197, 197]" = torch.ops.aten.reshape.default(bmm_22, [8, 16, 197, 197]);  bmm_22 = None
    mul_110: "f32[8, 16, 197, 197]" = torch.ops.aten.mul.Tensor(view_282, 0.14433756729740643);  view_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:179, code: attn = attn.softmax(dim=-1)
    amax_21: "f32[8, 16, 197, 1]" = torch.ops.aten.amax.default(mul_110, [-1], True)
    sub_64: "f32[8, 16, 197, 197]" = torch.ops.aten.sub.Tensor(mul_110, amax_21);  mul_110 = amax_21 = None
    exp_21: "f32[8, 16, 197, 197]" = torch.ops.aten.exp.default(sub_64);  sub_64 = None
    sum_32: "f32[8, 16, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
    div_31: "f32[8, 16, 197, 197]" = torch.ops.aten.div.Tensor(exp_21, sum_32);  exp_21 = sum_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:182, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_87: "f32[8, 16, 197, 197]" = torch.ops.aten.expand.default(div_31, [8, 16, 197, 197]);  div_31 = None
    view_283: "f32[128, 197, 197]" = torch.ops.aten.reshape.default(expand_87, [128, 197, 197]);  expand_87 = None
    expand_88: "f32[8, 16, 197, 48]" = torch.ops.aten.expand.default(getitem_51, [8, 16, 197, 48]);  getitem_51 = None
    clone_152: "f32[8, 16, 197, 48]" = torch.ops.aten.clone.default(expand_88, memory_format = torch.contiguous_format);  expand_88 = None
    view_284: "f32[128, 197, 48]" = torch.ops.aten.reshape.default(clone_152, [128, 197, 48]);  clone_152 = None
    bmm_23: "f32[128, 197, 48]" = torch.ops.aten.bmm.default(view_283, view_284);  view_283 = view_284 = None
    view_285: "f32[8, 16, 197, 48]" = torch.ops.aten.reshape.default(bmm_23, [8, 16, 197, 48]);  bmm_23 = None
    permute_121: "f32[8, 197, 16, 48]" = torch.ops.aten.permute.default(view_285, [0, 2, 1, 3]);  view_285 = None
    clone_153: "f32[8, 197, 16, 48]" = torch.ops.aten.clone.default(permute_121, memory_format = torch.contiguous_format);  permute_121 = None
    view_286: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(clone_153, [8, 197, 768]);  clone_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:183, code: x = self.proj(x)
    view_287: "f32[1576, 768]" = torch.ops.aten.reshape.default(view_286, [1576, 768]);  view_286 = None
    permute_122: "f32[768, 768]" = torch.ops.aten.permute.default(arg172_1, [1, 0]);  arg172_1 = None
    addmm_33: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg173_1, view_287, permute_122);  arg173_1 = view_287 = permute_122 = None
    view_288: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_33, [8, 197, 768]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:235, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_110: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_107, view_288);  add_107 = view_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_23 = torch.ops.aten.var_mean.correction(add_110, [2], correction = 0, keepdim = True)
    getitem_52: "f32[8, 197, 1]" = var_mean_23[0]
    getitem_53: "f32[8, 197, 1]" = var_mean_23[1];  var_mean_23 = None
    sub_65: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_110, getitem_53);  getitem_53 = None
    add_111: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-06);  getitem_52 = None
    rsqrt_23: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
    mul_111: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_23);  sub_65 = rsqrt_23 = None
    mul_112: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_111, arg58_1);  mul_111 = arg58_1 = None
    add_112: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_112, arg59_1);  mul_112 = arg59_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_289: "f32[1576, 768]" = torch.ops.aten.reshape.default(add_112, [1576, 768]);  add_112 = None
    permute_123: "f32[768, 3072]" = torch.ops.aten.permute.default(arg174_1, [1, 0]);  arg174_1 = None
    addmm_34: "f32[1576, 3072]" = torch.ops.aten.addmm.default(arg175_1, view_289, permute_123);  arg175_1 = view_289 = permute_123 = None
    view_290: "f32[8, 197, 3072]" = torch.ops.aten.reshape.default(addmm_34, [8, 197, 3072]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_113: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_290, 0.5)
    mul_114: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_290, 0.7071067811865476);  view_290 = None
    erf_11: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_114);  mul_114 = None
    add_113: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_115: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_113, add_113);  mul_113 = add_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_291: "f32[1576, 3072]" = torch.ops.aten.reshape.default(mul_115, [1576, 3072]);  mul_115 = None
    permute_124: "f32[3072, 768]" = torch.ops.aten.permute.default(arg176_1, [1, 0]);  arg176_1 = None
    addmm_35: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg177_1, view_291, permute_124);  arg177_1 = view_291 = permute_124 = None
    view_292: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_35, [8, 197, 768]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:236, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_114: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_110, view_292);  add_110 = view_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_24 = torch.ops.aten.var_mean.correction(add_114, [2], correction = 0, keepdim = True)
    getitem_54: "f32[8, 197, 1]" = var_mean_24[0]
    getitem_55: "f32[8, 197, 1]" = var_mean_24[1];  var_mean_24 = None
    sub_66: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_114, getitem_55);  add_114 = getitem_55 = None
    add_115: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-06);  getitem_54 = None
    rsqrt_24: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
    mul_116: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_24);  sub_66 = rsqrt_24 = None
    mul_117: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_116, arg60_1);  mul_116 = arg60_1 = None
    add_116: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_117, arg61_1);  mul_117 = arg61_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:374, code: x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
    select_100: "f32[8, 768]" = torch.ops.aten.select.int(add_116, 1, 0);  add_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:375, code: x = self.head_drop(x)
    clone_157: "f32[8, 768]" = torch.ops.aten.clone.default(select_100);  select_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:376, code: return x if pre_logits else self.head(x)
    permute_125: "f32[768, 1000]" = torch.ops.aten.permute.default(arg178_1, [1, 0]);  arg178_1 = None
    addmm_36: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg179_1, clone_157, permute_125);  arg179_1 = clone_157 = permute_125 = None
    return (addmm_36, select_scatter_2, select_scatter_5, select_scatter_8, select_scatter_11, select_scatter_14, select_scatter_17, select_scatter_20, select_scatter_23, select_scatter_26, select_scatter_29)
    