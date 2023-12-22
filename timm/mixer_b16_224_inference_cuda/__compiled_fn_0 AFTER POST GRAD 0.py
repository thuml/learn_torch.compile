from __future__ import annotations



def forward(self, arg0_1: "f32[768, 3, 16, 16]", arg1_1: "f32[768]", arg2_1: "f32[768]", arg3_1: "f32[768]", arg4_1: "f32[384, 196]", arg5_1: "f32[384]", arg6_1: "f32[196, 384]", arg7_1: "f32[196]", arg8_1: "f32[768]", arg9_1: "f32[768]", arg10_1: "f32[3072, 768]", arg11_1: "f32[3072]", arg12_1: "f32[768, 3072]", arg13_1: "f32[768]", arg14_1: "f32[768]", arg15_1: "f32[768]", arg16_1: "f32[384, 196]", arg17_1: "f32[384]", arg18_1: "f32[196, 384]", arg19_1: "f32[196]", arg20_1: "f32[768]", arg21_1: "f32[768]", arg22_1: "f32[3072, 768]", arg23_1: "f32[3072]", arg24_1: "f32[768, 3072]", arg25_1: "f32[768]", arg26_1: "f32[768]", arg27_1: "f32[768]", arg28_1: "f32[384, 196]", arg29_1: "f32[384]", arg30_1: "f32[196, 384]", arg31_1: "f32[196]", arg32_1: "f32[768]", arg33_1: "f32[768]", arg34_1: "f32[3072, 768]", arg35_1: "f32[3072]", arg36_1: "f32[768, 3072]", arg37_1: "f32[768]", arg38_1: "f32[768]", arg39_1: "f32[768]", arg40_1: "f32[384, 196]", arg41_1: "f32[384]", arg42_1: "f32[196, 384]", arg43_1: "f32[196]", arg44_1: "f32[768]", arg45_1: "f32[768]", arg46_1: "f32[3072, 768]", arg47_1: "f32[3072]", arg48_1: "f32[768, 3072]", arg49_1: "f32[768]", arg50_1: "f32[768]", arg51_1: "f32[768]", arg52_1: "f32[384, 196]", arg53_1: "f32[384]", arg54_1: "f32[196, 384]", arg55_1: "f32[196]", arg56_1: "f32[768]", arg57_1: "f32[768]", arg58_1: "f32[3072, 768]", arg59_1: "f32[3072]", arg60_1: "f32[768, 3072]", arg61_1: "f32[768]", arg62_1: "f32[768]", arg63_1: "f32[768]", arg64_1: "f32[384, 196]", arg65_1: "f32[384]", arg66_1: "f32[196, 384]", arg67_1: "f32[196]", arg68_1: "f32[768]", arg69_1: "f32[768]", arg70_1: "f32[3072, 768]", arg71_1: "f32[3072]", arg72_1: "f32[768, 3072]", arg73_1: "f32[768]", arg74_1: "f32[768]", arg75_1: "f32[768]", arg76_1: "f32[384, 196]", arg77_1: "f32[384]", arg78_1: "f32[196, 384]", arg79_1: "f32[196]", arg80_1: "f32[768]", arg81_1: "f32[768]", arg82_1: "f32[3072, 768]", arg83_1: "f32[3072]", arg84_1: "f32[768, 3072]", arg85_1: "f32[768]", arg86_1: "f32[768]", arg87_1: "f32[768]", arg88_1: "f32[384, 196]", arg89_1: "f32[384]", arg90_1: "f32[196, 384]", arg91_1: "f32[196]", arg92_1: "f32[768]", arg93_1: "f32[768]", arg94_1: "f32[3072, 768]", arg95_1: "f32[3072]", arg96_1: "f32[768, 3072]", arg97_1: "f32[768]", arg98_1: "f32[768]", arg99_1: "f32[768]", arg100_1: "f32[384, 196]", arg101_1: "f32[384]", arg102_1: "f32[196, 384]", arg103_1: "f32[196]", arg104_1: "f32[768]", arg105_1: "f32[768]", arg106_1: "f32[3072, 768]", arg107_1: "f32[3072]", arg108_1: "f32[768, 3072]", arg109_1: "f32[768]", arg110_1: "f32[768]", arg111_1: "f32[768]", arg112_1: "f32[384, 196]", arg113_1: "f32[384]", arg114_1: "f32[196, 384]", arg115_1: "f32[196]", arg116_1: "f32[768]", arg117_1: "f32[768]", arg118_1: "f32[3072, 768]", arg119_1: "f32[3072]", arg120_1: "f32[768, 3072]", arg121_1: "f32[768]", arg122_1: "f32[768]", arg123_1: "f32[768]", arg124_1: "f32[384, 196]", arg125_1: "f32[384]", arg126_1: "f32[196, 384]", arg127_1: "f32[196]", arg128_1: "f32[768]", arg129_1: "f32[768]", arg130_1: "f32[3072, 768]", arg131_1: "f32[3072]", arg132_1: "f32[768, 3072]", arg133_1: "f32[768]", arg134_1: "f32[768]", arg135_1: "f32[768]", arg136_1: "f32[384, 196]", arg137_1: "f32[384]", arg138_1: "f32[196, 384]", arg139_1: "f32[196]", arg140_1: "f32[768]", arg141_1: "f32[768]", arg142_1: "f32[3072, 768]", arg143_1: "f32[3072]", arg144_1: "f32[768, 3072]", arg145_1: "f32[768]", arg146_1: "f32[768]", arg147_1: "f32[768]", arg148_1: "f32[1000, 768]", arg149_1: "f32[1000]", arg150_1: "f32[8, 3, 224, 224]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution: "f32[8, 768, 14, 14]" = torch.ops.aten.convolution.default(arg150_1, arg0_1, arg1_1, [16, 16], [0, 0], [1, 1], False, [0, 0], 1);  arg150_1 = arg0_1 = arg1_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:89, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    view: "f32[8, 768, 196]" = torch.ops.aten.reshape.default(convolution, [8, 768, 196]);  convolution = None
    permute: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format)
    var_mean = torch.ops.aten.var_mean.correction(clone, [2], correction = 0, keepdim = True)
    getitem: "f32[8, 196, 1]" = var_mean[0]
    getitem_1: "f32[8, 196, 1]" = var_mean[1];  var_mean = None
    sub: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone, getitem_1);  clone = getitem_1 = None
    add: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
    rsqrt: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
    mul: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
    mul_1: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul, arg2_1);  mul = arg2_1 = None
    add_1: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_1, arg3_1);  mul_1 = arg3_1 = None
    permute_1: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_1, [0, 2, 1]);  add_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    clone_1: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
    view_1: "f32[6144, 196]" = torch.ops.aten.reshape.default(clone_1, [6144, 196]);  clone_1 = None
    permute_2: "f32[196, 384]" = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
    mm: "f32[6144, 384]" = torch.ops.aten.mm.default(view_1, permute_2);  view_1 = permute_2 = None
    view_2: "f32[8, 768, 384]" = torch.ops.aten.reshape.default(mm, [8, 768, 384]);  mm = None
    add_2: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(view_2, arg5_1);  view_2 = arg5_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_2: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_2, 0.5)
    mul_3: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_2, 0.7071067811865476);  add_2 = None
    erf: "f32[8, 768, 384]" = torch.ops.aten.erf.default(mul_3);  mul_3 = None
    add_3: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_4: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(mul_2, add_3);  mul_2 = add_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_3: "f32[6144, 384]" = torch.ops.aten.reshape.default(mul_4, [6144, 384]);  mul_4 = None
    permute_3: "f32[384, 196]" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
    
    # No stacktrace found for following nodes
    mm_default_35: "f32[6144, 196]" = torch.ops.aten.mm.default(view_3, permute_3);  view_3 = permute_3 = None
    add_tensor_35: "f32[6144, 196]" = torch.ops.aten.add.Tensor(mm_default_35, arg7_1);  mm_default_35 = arg7_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_4: "f32[8, 768, 196]" = torch.ops.aten.reshape.default(add_tensor_35, [8, 768, 196]);  add_tensor_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_4: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_4, [0, 2, 1]);  view_4 = None
    add_4: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(permute, permute_4);  permute = permute_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_4: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_4, memory_format = torch.contiguous_format)
    var_mean_1 = torch.ops.aten.var_mean.correction(clone_4, [2], correction = 0, keepdim = True)
    getitem_2: "f32[8, 196, 1]" = var_mean_1[0]
    getitem_3: "f32[8, 196, 1]" = var_mean_1[1];  var_mean_1 = None
    sub_1: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_4, getitem_3);  clone_4 = getitem_3 = None
    add_5: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-06);  getitem_2 = None
    rsqrt_1: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
    mul_5: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = rsqrt_1 = None
    mul_6: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_5, arg8_1);  mul_5 = arg8_1 = None
    add_6: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_6, arg9_1);  mul_6 = arg9_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_5: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_6, [1568, 768]);  add_6 = None
    permute_5: "f32[768, 3072]" = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
    
    # No stacktrace found for following nodes
    mm_default_34: "f32[1568, 3072]" = torch.ops.aten.mm.default(view_5, permute_5);  view_5 = permute_5 = None
    add_tensor_34: "f32[1568, 3072]" = torch.ops.aten.add.Tensor(mm_default_34, arg11_1);  mm_default_34 = arg11_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_6: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(add_tensor_34, [8, 196, 3072]);  add_tensor_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_7: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_6, 0.5)
    mul_8: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_6, 0.7071067811865476);  view_6 = None
    erf_1: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_8);  mul_8 = None
    add_7: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_9: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_7, add_7);  mul_7 = add_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_7: "f32[1568, 3072]" = torch.ops.aten.reshape.default(mul_9, [1568, 3072]);  mul_9 = None
    permute_6: "f32[3072, 768]" = torch.ops.aten.permute.default(arg12_1, [1, 0]);  arg12_1 = None
    
    # No stacktrace found for following nodes
    mm_default_33: "f32[1568, 768]" = torch.ops.aten.mm.default(view_7, permute_6);  view_7 = permute_6 = None
    add_tensor_33: "f32[1568, 768]" = torch.ops.aten.add.Tensor(mm_default_33, arg13_1);  mm_default_33 = arg13_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_8: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(add_tensor_33, [8, 196, 768]);  add_tensor_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_8: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_4, view_8);  add_4 = view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_7: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_8, memory_format = torch.contiguous_format)
    var_mean_2 = torch.ops.aten.var_mean.correction(clone_7, [2], correction = 0, keepdim = True)
    getitem_4: "f32[8, 196, 1]" = var_mean_2[0]
    getitem_5: "f32[8, 196, 1]" = var_mean_2[1];  var_mean_2 = None
    sub_2: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_7, getitem_5);  clone_7 = getitem_5 = None
    add_9: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-06);  getitem_4 = None
    rsqrt_2: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
    mul_10: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = rsqrt_2 = None
    mul_11: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_10, arg14_1);  mul_10 = arg14_1 = None
    add_10: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_11, arg15_1);  mul_11 = arg15_1 = None
    permute_7: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_10, [0, 2, 1]);  add_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    clone_8: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    view_9: "f32[6144, 196]" = torch.ops.aten.reshape.default(clone_8, [6144, 196]);  clone_8 = None
    permute_8: "f32[196, 384]" = torch.ops.aten.permute.default(arg16_1, [1, 0]);  arg16_1 = None
    mm_1: "f32[6144, 384]" = torch.ops.aten.mm.default(view_9, permute_8);  view_9 = permute_8 = None
    view_10: "f32[8, 768, 384]" = torch.ops.aten.reshape.default(mm_1, [8, 768, 384]);  mm_1 = None
    add_11: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(view_10, arg17_1);  view_10 = arg17_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_12: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_11, 0.5)
    mul_13: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_11, 0.7071067811865476);  add_11 = None
    erf_2: "f32[8, 768, 384]" = torch.ops.aten.erf.default(mul_13);  mul_13 = None
    add_12: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_14: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(mul_12, add_12);  mul_12 = add_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_11: "f32[6144, 384]" = torch.ops.aten.reshape.default(mul_14, [6144, 384]);  mul_14 = None
    permute_9: "f32[384, 196]" = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
    
    # No stacktrace found for following nodes
    mm_default_32: "f32[6144, 196]" = torch.ops.aten.mm.default(view_11, permute_9);  view_11 = permute_9 = None
    add_tensor_32: "f32[6144, 196]" = torch.ops.aten.add.Tensor(mm_default_32, arg19_1);  mm_default_32 = arg19_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_12: "f32[8, 768, 196]" = torch.ops.aten.reshape.default(add_tensor_32, [8, 768, 196]);  add_tensor_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_10: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None
    add_13: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_8, permute_10);  add_8 = permute_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_11: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_13, memory_format = torch.contiguous_format)
    var_mean_3 = torch.ops.aten.var_mean.correction(clone_11, [2], correction = 0, keepdim = True)
    getitem_6: "f32[8, 196, 1]" = var_mean_3[0]
    getitem_7: "f32[8, 196, 1]" = var_mean_3[1];  var_mean_3 = None
    sub_3: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_11, getitem_7);  clone_11 = getitem_7 = None
    add_14: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-06);  getitem_6 = None
    rsqrt_3: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
    mul_15: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = rsqrt_3 = None
    mul_16: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_15, arg20_1);  mul_15 = arg20_1 = None
    add_15: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_16, arg21_1);  mul_16 = arg21_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_13: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_15, [1568, 768]);  add_15 = None
    permute_11: "f32[768, 3072]" = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
    
    # No stacktrace found for following nodes
    mm_default_31: "f32[1568, 3072]" = torch.ops.aten.mm.default(view_13, permute_11);  view_13 = permute_11 = None
    add_tensor_31: "f32[1568, 3072]" = torch.ops.aten.add.Tensor(mm_default_31, arg23_1);  mm_default_31 = arg23_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_14: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(add_tensor_31, [8, 196, 3072]);  add_tensor_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_17: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_14, 0.5)
    mul_18: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_14, 0.7071067811865476);  view_14 = None
    erf_3: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_18);  mul_18 = None
    add_16: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_19: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_17, add_16);  mul_17 = add_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_15: "f32[1568, 3072]" = torch.ops.aten.reshape.default(mul_19, [1568, 3072]);  mul_19 = None
    permute_12: "f32[3072, 768]" = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
    
    # No stacktrace found for following nodes
    mm_default_30: "f32[1568, 768]" = torch.ops.aten.mm.default(view_15, permute_12);  view_15 = permute_12 = None
    add_tensor_30: "f32[1568, 768]" = torch.ops.aten.add.Tensor(mm_default_30, arg25_1);  mm_default_30 = arg25_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_16: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(add_tensor_30, [8, 196, 768]);  add_tensor_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_17: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_13, view_16);  add_13 = view_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_14: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_17, memory_format = torch.contiguous_format)
    var_mean_4 = torch.ops.aten.var_mean.correction(clone_14, [2], correction = 0, keepdim = True)
    getitem_8: "f32[8, 196, 1]" = var_mean_4[0]
    getitem_9: "f32[8, 196, 1]" = var_mean_4[1];  var_mean_4 = None
    sub_4: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_14, getitem_9);  clone_14 = getitem_9 = None
    add_18: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-06);  getitem_8 = None
    rsqrt_4: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
    mul_20: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = rsqrt_4 = None
    mul_21: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_20, arg26_1);  mul_20 = arg26_1 = None
    add_19: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_21, arg27_1);  mul_21 = arg27_1 = None
    permute_13: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_19, [0, 2, 1]);  add_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    clone_15: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_13, memory_format = torch.contiguous_format);  permute_13 = None
    view_17: "f32[6144, 196]" = torch.ops.aten.reshape.default(clone_15, [6144, 196]);  clone_15 = None
    permute_14: "f32[196, 384]" = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
    mm_2: "f32[6144, 384]" = torch.ops.aten.mm.default(view_17, permute_14);  view_17 = permute_14 = None
    view_18: "f32[8, 768, 384]" = torch.ops.aten.reshape.default(mm_2, [8, 768, 384]);  mm_2 = None
    add_20: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(view_18, arg29_1);  view_18 = arg29_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_22: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_20, 0.5)
    mul_23: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_20, 0.7071067811865476);  add_20 = None
    erf_4: "f32[8, 768, 384]" = torch.ops.aten.erf.default(mul_23);  mul_23 = None
    add_21: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_24: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(mul_22, add_21);  mul_22 = add_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_19: "f32[6144, 384]" = torch.ops.aten.reshape.default(mul_24, [6144, 384]);  mul_24 = None
    permute_15: "f32[384, 196]" = torch.ops.aten.permute.default(arg30_1, [1, 0]);  arg30_1 = None
    
    # No stacktrace found for following nodes
    mm_default_29: "f32[6144, 196]" = torch.ops.aten.mm.default(view_19, permute_15);  view_19 = permute_15 = None
    add_tensor_29: "f32[6144, 196]" = torch.ops.aten.add.Tensor(mm_default_29, arg31_1);  mm_default_29 = arg31_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_20: "f32[8, 768, 196]" = torch.ops.aten.reshape.default(add_tensor_29, [8, 768, 196]);  add_tensor_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_16: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_20, [0, 2, 1]);  view_20 = None
    add_22: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_17, permute_16);  add_17 = permute_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_18: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_22, memory_format = torch.contiguous_format)
    var_mean_5 = torch.ops.aten.var_mean.correction(clone_18, [2], correction = 0, keepdim = True)
    getitem_10: "f32[8, 196, 1]" = var_mean_5[0]
    getitem_11: "f32[8, 196, 1]" = var_mean_5[1];  var_mean_5 = None
    sub_5: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_18, getitem_11);  clone_18 = getitem_11 = None
    add_23: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-06);  getitem_10 = None
    rsqrt_5: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
    mul_25: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = rsqrt_5 = None
    mul_26: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_25, arg32_1);  mul_25 = arg32_1 = None
    add_24: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_26, arg33_1);  mul_26 = arg33_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_21: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_24, [1568, 768]);  add_24 = None
    permute_17: "f32[768, 3072]" = torch.ops.aten.permute.default(arg34_1, [1, 0]);  arg34_1 = None
    
    # No stacktrace found for following nodes
    mm_default_28: "f32[1568, 3072]" = torch.ops.aten.mm.default(view_21, permute_17);  view_21 = permute_17 = None
    add_tensor_28: "f32[1568, 3072]" = torch.ops.aten.add.Tensor(mm_default_28, arg35_1);  mm_default_28 = arg35_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_22: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(add_tensor_28, [8, 196, 3072]);  add_tensor_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_27: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_22, 0.5)
    mul_28: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_22, 0.7071067811865476);  view_22 = None
    erf_5: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_28);  mul_28 = None
    add_25: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_29: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_27, add_25);  mul_27 = add_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_23: "f32[1568, 3072]" = torch.ops.aten.reshape.default(mul_29, [1568, 3072]);  mul_29 = None
    permute_18: "f32[3072, 768]" = torch.ops.aten.permute.default(arg36_1, [1, 0]);  arg36_1 = None
    
    # No stacktrace found for following nodes
    mm_default_27: "f32[1568, 768]" = torch.ops.aten.mm.default(view_23, permute_18);  view_23 = permute_18 = None
    add_tensor_27: "f32[1568, 768]" = torch.ops.aten.add.Tensor(mm_default_27, arg37_1);  mm_default_27 = arg37_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_24: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(add_tensor_27, [8, 196, 768]);  add_tensor_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_26: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_22, view_24);  add_22 = view_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_21: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_26, memory_format = torch.contiguous_format)
    var_mean_6 = torch.ops.aten.var_mean.correction(clone_21, [2], correction = 0, keepdim = True)
    getitem_12: "f32[8, 196, 1]" = var_mean_6[0]
    getitem_13: "f32[8, 196, 1]" = var_mean_6[1];  var_mean_6 = None
    sub_6: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_21, getitem_13);  clone_21 = getitem_13 = None
    add_27: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-06);  getitem_12 = None
    rsqrt_6: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
    mul_30: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = rsqrt_6 = None
    mul_31: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_30, arg38_1);  mul_30 = arg38_1 = None
    add_28: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_31, arg39_1);  mul_31 = arg39_1 = None
    permute_19: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_28, [0, 2, 1]);  add_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    clone_22: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_19, memory_format = torch.contiguous_format);  permute_19 = None
    view_25: "f32[6144, 196]" = torch.ops.aten.reshape.default(clone_22, [6144, 196]);  clone_22 = None
    permute_20: "f32[196, 384]" = torch.ops.aten.permute.default(arg40_1, [1, 0]);  arg40_1 = None
    mm_3: "f32[6144, 384]" = torch.ops.aten.mm.default(view_25, permute_20);  view_25 = permute_20 = None
    view_26: "f32[8, 768, 384]" = torch.ops.aten.reshape.default(mm_3, [8, 768, 384]);  mm_3 = None
    add_29: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(view_26, arg41_1);  view_26 = arg41_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_32: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_29, 0.5)
    mul_33: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_29, 0.7071067811865476);  add_29 = None
    erf_6: "f32[8, 768, 384]" = torch.ops.aten.erf.default(mul_33);  mul_33 = None
    add_30: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_34: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(mul_32, add_30);  mul_32 = add_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_27: "f32[6144, 384]" = torch.ops.aten.reshape.default(mul_34, [6144, 384]);  mul_34 = None
    permute_21: "f32[384, 196]" = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
    
    # No stacktrace found for following nodes
    mm_default_26: "f32[6144, 196]" = torch.ops.aten.mm.default(view_27, permute_21);  view_27 = permute_21 = None
    add_tensor_26: "f32[6144, 196]" = torch.ops.aten.add.Tensor(mm_default_26, arg43_1);  mm_default_26 = arg43_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_28: "f32[8, 768, 196]" = torch.ops.aten.reshape.default(add_tensor_26, [8, 768, 196]);  add_tensor_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_22: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_28, [0, 2, 1]);  view_28 = None
    add_31: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_26, permute_22);  add_26 = permute_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_25: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_31, memory_format = torch.contiguous_format)
    var_mean_7 = torch.ops.aten.var_mean.correction(clone_25, [2], correction = 0, keepdim = True)
    getitem_14: "f32[8, 196, 1]" = var_mean_7[0]
    getitem_15: "f32[8, 196, 1]" = var_mean_7[1];  var_mean_7 = None
    sub_7: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_25, getitem_15);  clone_25 = getitem_15 = None
    add_32: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-06);  getitem_14 = None
    rsqrt_7: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
    mul_35: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = rsqrt_7 = None
    mul_36: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_35, arg44_1);  mul_35 = arg44_1 = None
    add_33: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_36, arg45_1);  mul_36 = arg45_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_29: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_33, [1568, 768]);  add_33 = None
    permute_23: "f32[768, 3072]" = torch.ops.aten.permute.default(arg46_1, [1, 0]);  arg46_1 = None
    
    # No stacktrace found for following nodes
    mm_default_25: "f32[1568, 3072]" = torch.ops.aten.mm.default(view_29, permute_23);  view_29 = permute_23 = None
    add_tensor_25: "f32[1568, 3072]" = torch.ops.aten.add.Tensor(mm_default_25, arg47_1);  mm_default_25 = arg47_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_30: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(add_tensor_25, [8, 196, 3072]);  add_tensor_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_37: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_30, 0.5)
    mul_38: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_30, 0.7071067811865476);  view_30 = None
    erf_7: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_38);  mul_38 = None
    add_34: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_39: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_37, add_34);  mul_37 = add_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_31: "f32[1568, 3072]" = torch.ops.aten.reshape.default(mul_39, [1568, 3072]);  mul_39 = None
    permute_24: "f32[3072, 768]" = torch.ops.aten.permute.default(arg48_1, [1, 0]);  arg48_1 = None
    
    # No stacktrace found for following nodes
    mm_default_24: "f32[1568, 768]" = torch.ops.aten.mm.default(view_31, permute_24);  view_31 = permute_24 = None
    add_tensor_24: "f32[1568, 768]" = torch.ops.aten.add.Tensor(mm_default_24, arg49_1);  mm_default_24 = arg49_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_32: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(add_tensor_24, [8, 196, 768]);  add_tensor_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_35: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_31, view_32);  add_31 = view_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_28: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_35, memory_format = torch.contiguous_format)
    var_mean_8 = torch.ops.aten.var_mean.correction(clone_28, [2], correction = 0, keepdim = True)
    getitem_16: "f32[8, 196, 1]" = var_mean_8[0]
    getitem_17: "f32[8, 196, 1]" = var_mean_8[1];  var_mean_8 = None
    sub_8: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_28, getitem_17);  clone_28 = getitem_17 = None
    add_36: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-06);  getitem_16 = None
    rsqrt_8: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
    mul_40: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = rsqrt_8 = None
    mul_41: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_40, arg50_1);  mul_40 = arg50_1 = None
    add_37: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_41, arg51_1);  mul_41 = arg51_1 = None
    permute_25: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_37, [0, 2, 1]);  add_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    clone_29: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_25, memory_format = torch.contiguous_format);  permute_25 = None
    view_33: "f32[6144, 196]" = torch.ops.aten.reshape.default(clone_29, [6144, 196]);  clone_29 = None
    permute_26: "f32[196, 384]" = torch.ops.aten.permute.default(arg52_1, [1, 0]);  arg52_1 = None
    mm_4: "f32[6144, 384]" = torch.ops.aten.mm.default(view_33, permute_26);  view_33 = permute_26 = None
    view_34: "f32[8, 768, 384]" = torch.ops.aten.reshape.default(mm_4, [8, 768, 384]);  mm_4 = None
    add_38: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(view_34, arg53_1);  view_34 = arg53_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_42: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_38, 0.5)
    mul_43: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_38, 0.7071067811865476);  add_38 = None
    erf_8: "f32[8, 768, 384]" = torch.ops.aten.erf.default(mul_43);  mul_43 = None
    add_39: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_44: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(mul_42, add_39);  mul_42 = add_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_35: "f32[6144, 384]" = torch.ops.aten.reshape.default(mul_44, [6144, 384]);  mul_44 = None
    permute_27: "f32[384, 196]" = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
    
    # No stacktrace found for following nodes
    mm_default_23: "f32[6144, 196]" = torch.ops.aten.mm.default(view_35, permute_27);  view_35 = permute_27 = None
    add_tensor_23: "f32[6144, 196]" = torch.ops.aten.add.Tensor(mm_default_23, arg55_1);  mm_default_23 = arg55_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_36: "f32[8, 768, 196]" = torch.ops.aten.reshape.default(add_tensor_23, [8, 768, 196]);  add_tensor_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_28: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_36, [0, 2, 1]);  view_36 = None
    add_40: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_35, permute_28);  add_35 = permute_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_32: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_40, memory_format = torch.contiguous_format)
    var_mean_9 = torch.ops.aten.var_mean.correction(clone_32, [2], correction = 0, keepdim = True)
    getitem_18: "f32[8, 196, 1]" = var_mean_9[0]
    getitem_19: "f32[8, 196, 1]" = var_mean_9[1];  var_mean_9 = None
    sub_9: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_32, getitem_19);  clone_32 = getitem_19 = None
    add_41: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-06);  getitem_18 = None
    rsqrt_9: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_41);  add_41 = None
    mul_45: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = rsqrt_9 = None
    mul_46: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_45, arg56_1);  mul_45 = arg56_1 = None
    add_42: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_46, arg57_1);  mul_46 = arg57_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_37: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_42, [1568, 768]);  add_42 = None
    permute_29: "f32[768, 3072]" = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
    
    # No stacktrace found for following nodes
    mm_default_22: "f32[1568, 3072]" = torch.ops.aten.mm.default(view_37, permute_29);  view_37 = permute_29 = None
    add_tensor_22: "f32[1568, 3072]" = torch.ops.aten.add.Tensor(mm_default_22, arg59_1);  mm_default_22 = arg59_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_38: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(add_tensor_22, [8, 196, 3072]);  add_tensor_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_47: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_38, 0.5)
    mul_48: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_38, 0.7071067811865476);  view_38 = None
    erf_9: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_48);  mul_48 = None
    add_43: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_49: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_47, add_43);  mul_47 = add_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_39: "f32[1568, 3072]" = torch.ops.aten.reshape.default(mul_49, [1568, 3072]);  mul_49 = None
    permute_30: "f32[3072, 768]" = torch.ops.aten.permute.default(arg60_1, [1, 0]);  arg60_1 = None
    
    # No stacktrace found for following nodes
    mm_default_21: "f32[1568, 768]" = torch.ops.aten.mm.default(view_39, permute_30);  view_39 = permute_30 = None
    add_tensor_21: "f32[1568, 768]" = torch.ops.aten.add.Tensor(mm_default_21, arg61_1);  mm_default_21 = arg61_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_40: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(add_tensor_21, [8, 196, 768]);  add_tensor_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_44: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_40, view_40);  add_40 = view_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_35: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_44, memory_format = torch.contiguous_format)
    var_mean_10 = torch.ops.aten.var_mean.correction(clone_35, [2], correction = 0, keepdim = True)
    getitem_20: "f32[8, 196, 1]" = var_mean_10[0]
    getitem_21: "f32[8, 196, 1]" = var_mean_10[1];  var_mean_10 = None
    sub_10: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_35, getitem_21);  clone_35 = getitem_21 = None
    add_45: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-06);  getitem_20 = None
    rsqrt_10: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
    mul_50: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = rsqrt_10 = None
    mul_51: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_50, arg62_1);  mul_50 = arg62_1 = None
    add_46: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_51, arg63_1);  mul_51 = arg63_1 = None
    permute_31: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_46, [0, 2, 1]);  add_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    clone_36: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_31, memory_format = torch.contiguous_format);  permute_31 = None
    view_41: "f32[6144, 196]" = torch.ops.aten.reshape.default(clone_36, [6144, 196]);  clone_36 = None
    permute_32: "f32[196, 384]" = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
    mm_5: "f32[6144, 384]" = torch.ops.aten.mm.default(view_41, permute_32);  view_41 = permute_32 = None
    view_42: "f32[8, 768, 384]" = torch.ops.aten.reshape.default(mm_5, [8, 768, 384]);  mm_5 = None
    add_47: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(view_42, arg65_1);  view_42 = arg65_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_52: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_47, 0.5)
    mul_53: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_47, 0.7071067811865476);  add_47 = None
    erf_10: "f32[8, 768, 384]" = torch.ops.aten.erf.default(mul_53);  mul_53 = None
    add_48: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_54: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(mul_52, add_48);  mul_52 = add_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_43: "f32[6144, 384]" = torch.ops.aten.reshape.default(mul_54, [6144, 384]);  mul_54 = None
    permute_33: "f32[384, 196]" = torch.ops.aten.permute.default(arg66_1, [1, 0]);  arg66_1 = None
    
    # No stacktrace found for following nodes
    mm_default_20: "f32[6144, 196]" = torch.ops.aten.mm.default(view_43, permute_33);  view_43 = permute_33 = None
    add_tensor_20: "f32[6144, 196]" = torch.ops.aten.add.Tensor(mm_default_20, arg67_1);  mm_default_20 = arg67_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_44: "f32[8, 768, 196]" = torch.ops.aten.reshape.default(add_tensor_20, [8, 768, 196]);  add_tensor_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_34: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_44, [0, 2, 1]);  view_44 = None
    add_49: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_44, permute_34);  add_44 = permute_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_39: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_49, memory_format = torch.contiguous_format)
    var_mean_11 = torch.ops.aten.var_mean.correction(clone_39, [2], correction = 0, keepdim = True)
    getitem_22: "f32[8, 196, 1]" = var_mean_11[0]
    getitem_23: "f32[8, 196, 1]" = var_mean_11[1];  var_mean_11 = None
    sub_11: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_39, getitem_23);  clone_39 = getitem_23 = None
    add_50: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-06);  getitem_22 = None
    rsqrt_11: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
    mul_55: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = rsqrt_11 = None
    mul_56: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_55, arg68_1);  mul_55 = arg68_1 = None
    add_51: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_56, arg69_1);  mul_56 = arg69_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_45: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_51, [1568, 768]);  add_51 = None
    permute_35: "f32[768, 3072]" = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
    
    # No stacktrace found for following nodes
    mm_default_19: "f32[1568, 3072]" = torch.ops.aten.mm.default(view_45, permute_35);  view_45 = permute_35 = None
    add_tensor_19: "f32[1568, 3072]" = torch.ops.aten.add.Tensor(mm_default_19, arg71_1);  mm_default_19 = arg71_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_46: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(add_tensor_19, [8, 196, 3072]);  add_tensor_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_57: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_46, 0.5)
    mul_58: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_46, 0.7071067811865476);  view_46 = None
    erf_11: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_58);  mul_58 = None
    add_52: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_59: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_57, add_52);  mul_57 = add_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_47: "f32[1568, 3072]" = torch.ops.aten.reshape.default(mul_59, [1568, 3072]);  mul_59 = None
    permute_36: "f32[3072, 768]" = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
    
    # No stacktrace found for following nodes
    mm_default_18: "f32[1568, 768]" = torch.ops.aten.mm.default(view_47, permute_36);  view_47 = permute_36 = None
    add_tensor_18: "f32[1568, 768]" = torch.ops.aten.add.Tensor(mm_default_18, arg73_1);  mm_default_18 = arg73_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_48: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(add_tensor_18, [8, 196, 768]);  add_tensor_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_53: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_49, view_48);  add_49 = view_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_42: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_53, memory_format = torch.contiguous_format)
    var_mean_12 = torch.ops.aten.var_mean.correction(clone_42, [2], correction = 0, keepdim = True)
    getitem_24: "f32[8, 196, 1]" = var_mean_12[0]
    getitem_25: "f32[8, 196, 1]" = var_mean_12[1];  var_mean_12 = None
    sub_12: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_42, getitem_25);  clone_42 = getitem_25 = None
    add_54: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-06);  getitem_24 = None
    rsqrt_12: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
    mul_60: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = rsqrt_12 = None
    mul_61: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_60, arg74_1);  mul_60 = arg74_1 = None
    add_55: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_61, arg75_1);  mul_61 = arg75_1 = None
    permute_37: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_55, [0, 2, 1]);  add_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    clone_43: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
    view_49: "f32[6144, 196]" = torch.ops.aten.reshape.default(clone_43, [6144, 196]);  clone_43 = None
    permute_38: "f32[196, 384]" = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None
    mm_6: "f32[6144, 384]" = torch.ops.aten.mm.default(view_49, permute_38);  view_49 = permute_38 = None
    view_50: "f32[8, 768, 384]" = torch.ops.aten.reshape.default(mm_6, [8, 768, 384]);  mm_6 = None
    add_56: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(view_50, arg77_1);  view_50 = arg77_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_62: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_56, 0.5)
    mul_63: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_56, 0.7071067811865476);  add_56 = None
    erf_12: "f32[8, 768, 384]" = torch.ops.aten.erf.default(mul_63);  mul_63 = None
    add_57: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_64: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(mul_62, add_57);  mul_62 = add_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_51: "f32[6144, 384]" = torch.ops.aten.reshape.default(mul_64, [6144, 384]);  mul_64 = None
    permute_39: "f32[384, 196]" = torch.ops.aten.permute.default(arg78_1, [1, 0]);  arg78_1 = None
    
    # No stacktrace found for following nodes
    mm_default_17: "f32[6144, 196]" = torch.ops.aten.mm.default(view_51, permute_39);  view_51 = permute_39 = None
    add_tensor_17: "f32[6144, 196]" = torch.ops.aten.add.Tensor(mm_default_17, arg79_1);  mm_default_17 = arg79_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_52: "f32[8, 768, 196]" = torch.ops.aten.reshape.default(add_tensor_17, [8, 768, 196]);  add_tensor_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_40: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_52, [0, 2, 1]);  view_52 = None
    add_58: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_53, permute_40);  add_53 = permute_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_46: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_58, memory_format = torch.contiguous_format)
    var_mean_13 = torch.ops.aten.var_mean.correction(clone_46, [2], correction = 0, keepdim = True)
    getitem_26: "f32[8, 196, 1]" = var_mean_13[0]
    getitem_27: "f32[8, 196, 1]" = var_mean_13[1];  var_mean_13 = None
    sub_13: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_46, getitem_27);  clone_46 = getitem_27 = None
    add_59: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-06);  getitem_26 = None
    rsqrt_13: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_59);  add_59 = None
    mul_65: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = rsqrt_13 = None
    mul_66: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_65, arg80_1);  mul_65 = arg80_1 = None
    add_60: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_66, arg81_1);  mul_66 = arg81_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_53: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_60, [1568, 768]);  add_60 = None
    permute_41: "f32[768, 3072]" = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
    
    # No stacktrace found for following nodes
    mm_default_16: "f32[1568, 3072]" = torch.ops.aten.mm.default(view_53, permute_41);  view_53 = permute_41 = None
    add_tensor_16: "f32[1568, 3072]" = torch.ops.aten.add.Tensor(mm_default_16, arg83_1);  mm_default_16 = arg83_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_54: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(add_tensor_16, [8, 196, 3072]);  add_tensor_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_67: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_54, 0.5)
    mul_68: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_54, 0.7071067811865476);  view_54 = None
    erf_13: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_68);  mul_68 = None
    add_61: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    mul_69: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_67, add_61);  mul_67 = add_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_55: "f32[1568, 3072]" = torch.ops.aten.reshape.default(mul_69, [1568, 3072]);  mul_69 = None
    permute_42: "f32[3072, 768]" = torch.ops.aten.permute.default(arg84_1, [1, 0]);  arg84_1 = None
    
    # No stacktrace found for following nodes
    mm_default_15: "f32[1568, 768]" = torch.ops.aten.mm.default(view_55, permute_42);  view_55 = permute_42 = None
    add_tensor_15: "f32[1568, 768]" = torch.ops.aten.add.Tensor(mm_default_15, arg85_1);  mm_default_15 = arg85_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_56: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(add_tensor_15, [8, 196, 768]);  add_tensor_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_62: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_58, view_56);  add_58 = view_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_49: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_62, memory_format = torch.contiguous_format)
    var_mean_14 = torch.ops.aten.var_mean.correction(clone_49, [2], correction = 0, keepdim = True)
    getitem_28: "f32[8, 196, 1]" = var_mean_14[0]
    getitem_29: "f32[8, 196, 1]" = var_mean_14[1];  var_mean_14 = None
    sub_14: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_49, getitem_29);  clone_49 = getitem_29 = None
    add_63: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-06);  getitem_28 = None
    rsqrt_14: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
    mul_70: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = rsqrt_14 = None
    mul_71: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_70, arg86_1);  mul_70 = arg86_1 = None
    add_64: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_71, arg87_1);  mul_71 = arg87_1 = None
    permute_43: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_64, [0, 2, 1]);  add_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    clone_50: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_43, memory_format = torch.contiguous_format);  permute_43 = None
    view_57: "f32[6144, 196]" = torch.ops.aten.reshape.default(clone_50, [6144, 196]);  clone_50 = None
    permute_44: "f32[196, 384]" = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
    mm_7: "f32[6144, 384]" = torch.ops.aten.mm.default(view_57, permute_44);  view_57 = permute_44 = None
    view_58: "f32[8, 768, 384]" = torch.ops.aten.reshape.default(mm_7, [8, 768, 384]);  mm_7 = None
    add_65: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(view_58, arg89_1);  view_58 = arg89_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_72: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_65, 0.5)
    mul_73: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_65, 0.7071067811865476);  add_65 = None
    erf_14: "f32[8, 768, 384]" = torch.ops.aten.erf.default(mul_73);  mul_73 = None
    add_66: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    mul_74: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(mul_72, add_66);  mul_72 = add_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_59: "f32[6144, 384]" = torch.ops.aten.reshape.default(mul_74, [6144, 384]);  mul_74 = None
    permute_45: "f32[384, 196]" = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
    
    # No stacktrace found for following nodes
    mm_default_14: "f32[6144, 196]" = torch.ops.aten.mm.default(view_59, permute_45);  view_59 = permute_45 = None
    add_tensor_14: "f32[6144, 196]" = torch.ops.aten.add.Tensor(mm_default_14, arg91_1);  mm_default_14 = arg91_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_60: "f32[8, 768, 196]" = torch.ops.aten.reshape.default(add_tensor_14, [8, 768, 196]);  add_tensor_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_46: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_60, [0, 2, 1]);  view_60 = None
    add_67: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_62, permute_46);  add_62 = permute_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_53: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_67, memory_format = torch.contiguous_format)
    var_mean_15 = torch.ops.aten.var_mean.correction(clone_53, [2], correction = 0, keepdim = True)
    getitem_30: "f32[8, 196, 1]" = var_mean_15[0]
    getitem_31: "f32[8, 196, 1]" = var_mean_15[1];  var_mean_15 = None
    sub_15: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_53, getitem_31);  clone_53 = getitem_31 = None
    add_68: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-06);  getitem_30 = None
    rsqrt_15: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
    mul_75: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = rsqrt_15 = None
    mul_76: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_75, arg92_1);  mul_75 = arg92_1 = None
    add_69: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_76, arg93_1);  mul_76 = arg93_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_61: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_69, [1568, 768]);  add_69 = None
    permute_47: "f32[768, 3072]" = torch.ops.aten.permute.default(arg94_1, [1, 0]);  arg94_1 = None
    
    # No stacktrace found for following nodes
    mm_default_13: "f32[1568, 3072]" = torch.ops.aten.mm.default(view_61, permute_47);  view_61 = permute_47 = None
    add_tensor_13: "f32[1568, 3072]" = torch.ops.aten.add.Tensor(mm_default_13, arg95_1);  mm_default_13 = arg95_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_62: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(add_tensor_13, [8, 196, 3072]);  add_tensor_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_77: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_62, 0.5)
    mul_78: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_62, 0.7071067811865476);  view_62 = None
    erf_15: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_78);  mul_78 = None
    add_70: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    mul_79: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_77, add_70);  mul_77 = add_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_63: "f32[1568, 3072]" = torch.ops.aten.reshape.default(mul_79, [1568, 3072]);  mul_79 = None
    permute_48: "f32[3072, 768]" = torch.ops.aten.permute.default(arg96_1, [1, 0]);  arg96_1 = None
    
    # No stacktrace found for following nodes
    mm_default_12: "f32[1568, 768]" = torch.ops.aten.mm.default(view_63, permute_48);  view_63 = permute_48 = None
    add_tensor_12: "f32[1568, 768]" = torch.ops.aten.add.Tensor(mm_default_12, arg97_1);  mm_default_12 = arg97_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_64: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(add_tensor_12, [8, 196, 768]);  add_tensor_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_71: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_67, view_64);  add_67 = view_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_56: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_71, memory_format = torch.contiguous_format)
    var_mean_16 = torch.ops.aten.var_mean.correction(clone_56, [2], correction = 0, keepdim = True)
    getitem_32: "f32[8, 196, 1]" = var_mean_16[0]
    getitem_33: "f32[8, 196, 1]" = var_mean_16[1];  var_mean_16 = None
    sub_16: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_56, getitem_33);  clone_56 = getitem_33 = None
    add_72: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-06);  getitem_32 = None
    rsqrt_16: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
    mul_80: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = rsqrt_16 = None
    mul_81: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_80, arg98_1);  mul_80 = arg98_1 = None
    add_73: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_81, arg99_1);  mul_81 = arg99_1 = None
    permute_49: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_73, [0, 2, 1]);  add_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    clone_57: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
    view_65: "f32[6144, 196]" = torch.ops.aten.reshape.default(clone_57, [6144, 196]);  clone_57 = None
    permute_50: "f32[196, 384]" = torch.ops.aten.permute.default(arg100_1, [1, 0]);  arg100_1 = None
    mm_8: "f32[6144, 384]" = torch.ops.aten.mm.default(view_65, permute_50);  view_65 = permute_50 = None
    view_66: "f32[8, 768, 384]" = torch.ops.aten.reshape.default(mm_8, [8, 768, 384]);  mm_8 = None
    add_74: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(view_66, arg101_1);  view_66 = arg101_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_82: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_74, 0.5)
    mul_83: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_74, 0.7071067811865476);  add_74 = None
    erf_16: "f32[8, 768, 384]" = torch.ops.aten.erf.default(mul_83);  mul_83 = None
    add_75: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    mul_84: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(mul_82, add_75);  mul_82 = add_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_67: "f32[6144, 384]" = torch.ops.aten.reshape.default(mul_84, [6144, 384]);  mul_84 = None
    permute_51: "f32[384, 196]" = torch.ops.aten.permute.default(arg102_1, [1, 0]);  arg102_1 = None
    
    # No stacktrace found for following nodes
    mm_default_11: "f32[6144, 196]" = torch.ops.aten.mm.default(view_67, permute_51);  view_67 = permute_51 = None
    add_tensor_11: "f32[6144, 196]" = torch.ops.aten.add.Tensor(mm_default_11, arg103_1);  mm_default_11 = arg103_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_68: "f32[8, 768, 196]" = torch.ops.aten.reshape.default(add_tensor_11, [8, 768, 196]);  add_tensor_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_52: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_68, [0, 2, 1]);  view_68 = None
    add_76: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_71, permute_52);  add_71 = permute_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_60: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_76, memory_format = torch.contiguous_format)
    var_mean_17 = torch.ops.aten.var_mean.correction(clone_60, [2], correction = 0, keepdim = True)
    getitem_34: "f32[8, 196, 1]" = var_mean_17[0]
    getitem_35: "f32[8, 196, 1]" = var_mean_17[1];  var_mean_17 = None
    sub_17: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_60, getitem_35);  clone_60 = getitem_35 = None
    add_77: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-06);  getitem_34 = None
    rsqrt_17: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
    mul_85: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = rsqrt_17 = None
    mul_86: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_85, arg104_1);  mul_85 = arg104_1 = None
    add_78: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_86, arg105_1);  mul_86 = arg105_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_69: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_78, [1568, 768]);  add_78 = None
    permute_53: "f32[768, 3072]" = torch.ops.aten.permute.default(arg106_1, [1, 0]);  arg106_1 = None
    
    # No stacktrace found for following nodes
    mm_default_10: "f32[1568, 3072]" = torch.ops.aten.mm.default(view_69, permute_53);  view_69 = permute_53 = None
    add_tensor_10: "f32[1568, 3072]" = torch.ops.aten.add.Tensor(mm_default_10, arg107_1);  mm_default_10 = arg107_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_70: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(add_tensor_10, [8, 196, 3072]);  add_tensor_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_87: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_70, 0.5)
    mul_88: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_70, 0.7071067811865476);  view_70 = None
    erf_17: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_88);  mul_88 = None
    add_79: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    mul_89: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_87, add_79);  mul_87 = add_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_71: "f32[1568, 3072]" = torch.ops.aten.reshape.default(mul_89, [1568, 3072]);  mul_89 = None
    permute_54: "f32[3072, 768]" = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
    
    # No stacktrace found for following nodes
    mm_default_9: "f32[1568, 768]" = torch.ops.aten.mm.default(view_71, permute_54);  view_71 = permute_54 = None
    add_tensor_9: "f32[1568, 768]" = torch.ops.aten.add.Tensor(mm_default_9, arg109_1);  mm_default_9 = arg109_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_72: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(add_tensor_9, [8, 196, 768]);  add_tensor_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_80: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_76, view_72);  add_76 = view_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_63: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_80, memory_format = torch.contiguous_format)
    var_mean_18 = torch.ops.aten.var_mean.correction(clone_63, [2], correction = 0, keepdim = True)
    getitem_36: "f32[8, 196, 1]" = var_mean_18[0]
    getitem_37: "f32[8, 196, 1]" = var_mean_18[1];  var_mean_18 = None
    sub_18: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_63, getitem_37);  clone_63 = getitem_37 = None
    add_81: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-06);  getitem_36 = None
    rsqrt_18: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
    mul_90: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = rsqrt_18 = None
    mul_91: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_90, arg110_1);  mul_90 = arg110_1 = None
    add_82: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_91, arg111_1);  mul_91 = arg111_1 = None
    permute_55: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_82, [0, 2, 1]);  add_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    clone_64: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_55, memory_format = torch.contiguous_format);  permute_55 = None
    view_73: "f32[6144, 196]" = torch.ops.aten.reshape.default(clone_64, [6144, 196]);  clone_64 = None
    permute_56: "f32[196, 384]" = torch.ops.aten.permute.default(arg112_1, [1, 0]);  arg112_1 = None
    mm_9: "f32[6144, 384]" = torch.ops.aten.mm.default(view_73, permute_56);  view_73 = permute_56 = None
    view_74: "f32[8, 768, 384]" = torch.ops.aten.reshape.default(mm_9, [8, 768, 384]);  mm_9 = None
    add_83: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(view_74, arg113_1);  view_74 = arg113_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_92: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_83, 0.5)
    mul_93: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_83, 0.7071067811865476);  add_83 = None
    erf_18: "f32[8, 768, 384]" = torch.ops.aten.erf.default(mul_93);  mul_93 = None
    add_84: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    mul_94: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(mul_92, add_84);  mul_92 = add_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_75: "f32[6144, 384]" = torch.ops.aten.reshape.default(mul_94, [6144, 384]);  mul_94 = None
    permute_57: "f32[384, 196]" = torch.ops.aten.permute.default(arg114_1, [1, 0]);  arg114_1 = None
    
    # No stacktrace found for following nodes
    mm_default_8: "f32[6144, 196]" = torch.ops.aten.mm.default(view_75, permute_57);  view_75 = permute_57 = None
    add_tensor_8: "f32[6144, 196]" = torch.ops.aten.add.Tensor(mm_default_8, arg115_1);  mm_default_8 = arg115_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_76: "f32[8, 768, 196]" = torch.ops.aten.reshape.default(add_tensor_8, [8, 768, 196]);  add_tensor_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_58: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_76, [0, 2, 1]);  view_76 = None
    add_85: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_80, permute_58);  add_80 = permute_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_67: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_85, memory_format = torch.contiguous_format)
    var_mean_19 = torch.ops.aten.var_mean.correction(clone_67, [2], correction = 0, keepdim = True)
    getitem_38: "f32[8, 196, 1]" = var_mean_19[0]
    getitem_39: "f32[8, 196, 1]" = var_mean_19[1];  var_mean_19 = None
    sub_19: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_67, getitem_39);  clone_67 = getitem_39 = None
    add_86: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-06);  getitem_38 = None
    rsqrt_19: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
    mul_95: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = rsqrt_19 = None
    mul_96: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_95, arg116_1);  mul_95 = arg116_1 = None
    add_87: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_96, arg117_1);  mul_96 = arg117_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_77: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_87, [1568, 768]);  add_87 = None
    permute_59: "f32[768, 3072]" = torch.ops.aten.permute.default(arg118_1, [1, 0]);  arg118_1 = None
    
    # No stacktrace found for following nodes
    mm_default_7: "f32[1568, 3072]" = torch.ops.aten.mm.default(view_77, permute_59);  view_77 = permute_59 = None
    add_tensor_7: "f32[1568, 3072]" = torch.ops.aten.add.Tensor(mm_default_7, arg119_1);  mm_default_7 = arg119_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_78: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(add_tensor_7, [8, 196, 3072]);  add_tensor_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_97: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_78, 0.5)
    mul_98: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_78, 0.7071067811865476);  view_78 = None
    erf_19: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_98);  mul_98 = None
    add_88: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    mul_99: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_97, add_88);  mul_97 = add_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_79: "f32[1568, 3072]" = torch.ops.aten.reshape.default(mul_99, [1568, 3072]);  mul_99 = None
    permute_60: "f32[3072, 768]" = torch.ops.aten.permute.default(arg120_1, [1, 0]);  arg120_1 = None
    
    # No stacktrace found for following nodes
    mm_default_6: "f32[1568, 768]" = torch.ops.aten.mm.default(view_79, permute_60);  view_79 = permute_60 = None
    add_tensor_6: "f32[1568, 768]" = torch.ops.aten.add.Tensor(mm_default_6, arg121_1);  mm_default_6 = arg121_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_80: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(add_tensor_6, [8, 196, 768]);  add_tensor_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_89: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_85, view_80);  add_85 = view_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_70: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_89, memory_format = torch.contiguous_format)
    var_mean_20 = torch.ops.aten.var_mean.correction(clone_70, [2], correction = 0, keepdim = True)
    getitem_40: "f32[8, 196, 1]" = var_mean_20[0]
    getitem_41: "f32[8, 196, 1]" = var_mean_20[1];  var_mean_20 = None
    sub_20: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_70, getitem_41);  clone_70 = getitem_41 = None
    add_90: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-06);  getitem_40 = None
    rsqrt_20: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
    mul_100: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = rsqrt_20 = None
    mul_101: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_100, arg122_1);  mul_100 = arg122_1 = None
    add_91: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_101, arg123_1);  mul_101 = arg123_1 = None
    permute_61: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_91, [0, 2, 1]);  add_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    clone_71: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_61, memory_format = torch.contiguous_format);  permute_61 = None
    view_81: "f32[6144, 196]" = torch.ops.aten.reshape.default(clone_71, [6144, 196]);  clone_71 = None
    permute_62: "f32[196, 384]" = torch.ops.aten.permute.default(arg124_1, [1, 0]);  arg124_1 = None
    mm_10: "f32[6144, 384]" = torch.ops.aten.mm.default(view_81, permute_62);  view_81 = permute_62 = None
    view_82: "f32[8, 768, 384]" = torch.ops.aten.reshape.default(mm_10, [8, 768, 384]);  mm_10 = None
    add_92: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(view_82, arg125_1);  view_82 = arg125_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_102: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_92, 0.5)
    mul_103: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_92, 0.7071067811865476);  add_92 = None
    erf_20: "f32[8, 768, 384]" = torch.ops.aten.erf.default(mul_103);  mul_103 = None
    add_93: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    mul_104: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(mul_102, add_93);  mul_102 = add_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_83: "f32[6144, 384]" = torch.ops.aten.reshape.default(mul_104, [6144, 384]);  mul_104 = None
    permute_63: "f32[384, 196]" = torch.ops.aten.permute.default(arg126_1, [1, 0]);  arg126_1 = None
    
    # No stacktrace found for following nodes
    mm_default_5: "f32[6144, 196]" = torch.ops.aten.mm.default(view_83, permute_63);  view_83 = permute_63 = None
    add_tensor_5: "f32[6144, 196]" = torch.ops.aten.add.Tensor(mm_default_5, arg127_1);  mm_default_5 = arg127_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_84: "f32[8, 768, 196]" = torch.ops.aten.reshape.default(add_tensor_5, [8, 768, 196]);  add_tensor_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_64: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_84, [0, 2, 1]);  view_84 = None
    add_94: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_89, permute_64);  add_89 = permute_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_74: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_94, memory_format = torch.contiguous_format)
    var_mean_21 = torch.ops.aten.var_mean.correction(clone_74, [2], correction = 0, keepdim = True)
    getitem_42: "f32[8, 196, 1]" = var_mean_21[0]
    getitem_43: "f32[8, 196, 1]" = var_mean_21[1];  var_mean_21 = None
    sub_21: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_74, getitem_43);  clone_74 = getitem_43 = None
    add_95: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-06);  getitem_42 = None
    rsqrt_21: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_95);  add_95 = None
    mul_105: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = rsqrt_21 = None
    mul_106: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_105, arg128_1);  mul_105 = arg128_1 = None
    add_96: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_106, arg129_1);  mul_106 = arg129_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_85: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_96, [1568, 768]);  add_96 = None
    permute_65: "f32[768, 3072]" = torch.ops.aten.permute.default(arg130_1, [1, 0]);  arg130_1 = None
    
    # No stacktrace found for following nodes
    mm_default_4: "f32[1568, 3072]" = torch.ops.aten.mm.default(view_85, permute_65);  view_85 = permute_65 = None
    add_tensor_4: "f32[1568, 3072]" = torch.ops.aten.add.Tensor(mm_default_4, arg131_1);  mm_default_4 = arg131_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_86: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(add_tensor_4, [8, 196, 3072]);  add_tensor_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_107: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_86, 0.5)
    mul_108: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_86, 0.7071067811865476);  view_86 = None
    erf_21: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_108);  mul_108 = None
    add_97: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    mul_109: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_107, add_97);  mul_107 = add_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_87: "f32[1568, 3072]" = torch.ops.aten.reshape.default(mul_109, [1568, 3072]);  mul_109 = None
    permute_66: "f32[3072, 768]" = torch.ops.aten.permute.default(arg132_1, [1, 0]);  arg132_1 = None
    
    # No stacktrace found for following nodes
    mm_default_3: "f32[1568, 768]" = torch.ops.aten.mm.default(view_87, permute_66);  view_87 = permute_66 = None
    add_tensor_3: "f32[1568, 768]" = torch.ops.aten.add.Tensor(mm_default_3, arg133_1);  mm_default_3 = arg133_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_88: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(add_tensor_3, [8, 196, 768]);  add_tensor_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_98: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_94, view_88);  add_94 = view_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_77: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_98, memory_format = torch.contiguous_format)
    var_mean_22 = torch.ops.aten.var_mean.correction(clone_77, [2], correction = 0, keepdim = True)
    getitem_44: "f32[8, 196, 1]" = var_mean_22[0]
    getitem_45: "f32[8, 196, 1]" = var_mean_22[1];  var_mean_22 = None
    sub_22: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_77, getitem_45);  clone_77 = getitem_45 = None
    add_99: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-06);  getitem_44 = None
    rsqrt_22: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_99);  add_99 = None
    mul_110: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = rsqrt_22 = None
    mul_111: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_110, arg134_1);  mul_110 = arg134_1 = None
    add_100: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_111, arg135_1);  mul_111 = arg135_1 = None
    permute_67: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_100, [0, 2, 1]);  add_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    clone_78: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_67, memory_format = torch.contiguous_format);  permute_67 = None
    view_89: "f32[6144, 196]" = torch.ops.aten.reshape.default(clone_78, [6144, 196]);  clone_78 = None
    permute_68: "f32[196, 384]" = torch.ops.aten.permute.default(arg136_1, [1, 0]);  arg136_1 = None
    mm_11: "f32[6144, 384]" = torch.ops.aten.mm.default(view_89, permute_68);  view_89 = permute_68 = None
    view_90: "f32[8, 768, 384]" = torch.ops.aten.reshape.default(mm_11, [8, 768, 384]);  mm_11 = None
    add_101: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(view_90, arg137_1);  view_90 = arg137_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_112: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_101, 0.5)
    mul_113: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_101, 0.7071067811865476);  add_101 = None
    erf_22: "f32[8, 768, 384]" = torch.ops.aten.erf.default(mul_113);  mul_113 = None
    add_102: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    mul_114: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(mul_112, add_102);  mul_112 = add_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_91: "f32[6144, 384]" = torch.ops.aten.reshape.default(mul_114, [6144, 384]);  mul_114 = None
    permute_69: "f32[384, 196]" = torch.ops.aten.permute.default(arg138_1, [1, 0]);  arg138_1 = None
    
    # No stacktrace found for following nodes
    mm_default_2: "f32[6144, 196]" = torch.ops.aten.mm.default(view_91, permute_69);  view_91 = permute_69 = None
    add_tensor_2: "f32[6144, 196]" = torch.ops.aten.add.Tensor(mm_default_2, arg139_1);  mm_default_2 = arg139_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_92: "f32[8, 768, 196]" = torch.ops.aten.reshape.default(add_tensor_2, [8, 768, 196]);  add_tensor_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_70: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_92, [0, 2, 1]);  view_92 = None
    add_103: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_98, permute_70);  add_98 = permute_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_81: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_103, memory_format = torch.contiguous_format)
    var_mean_23 = torch.ops.aten.var_mean.correction(clone_81, [2], correction = 0, keepdim = True)
    getitem_46: "f32[8, 196, 1]" = var_mean_23[0]
    getitem_47: "f32[8, 196, 1]" = var_mean_23[1];  var_mean_23 = None
    sub_23: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_81, getitem_47);  clone_81 = getitem_47 = None
    add_104: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-06);  getitem_46 = None
    rsqrt_23: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_104);  add_104 = None
    mul_115: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = rsqrt_23 = None
    mul_116: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_115, arg140_1);  mul_115 = arg140_1 = None
    add_105: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_116, arg141_1);  mul_116 = arg141_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_93: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_105, [1568, 768]);  add_105 = None
    permute_71: "f32[768, 3072]" = torch.ops.aten.permute.default(arg142_1, [1, 0]);  arg142_1 = None
    
    # No stacktrace found for following nodes
    mm_default_1: "f32[1568, 3072]" = torch.ops.aten.mm.default(view_93, permute_71);  view_93 = permute_71 = None
    add_tensor_1: "f32[1568, 3072]" = torch.ops.aten.add.Tensor(mm_default_1, arg143_1);  mm_default_1 = arg143_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_94: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(add_tensor_1, [8, 196, 3072]);  add_tensor_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_117: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_94, 0.5)
    mul_118: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_94, 0.7071067811865476);  view_94 = None
    erf_23: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_118);  mul_118 = None
    add_106: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    mul_119: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_117, add_106);  mul_117 = add_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_95: "f32[1568, 3072]" = torch.ops.aten.reshape.default(mul_119, [1568, 3072]);  mul_119 = None
    permute_72: "f32[3072, 768]" = torch.ops.aten.permute.default(arg144_1, [1, 0]);  arg144_1 = None
    
    # No stacktrace found for following nodes
    mm_default: "f32[1568, 768]" = torch.ops.aten.mm.default(view_95, permute_72);  view_95 = permute_72 = None
    add_tensor: "f32[1568, 768]" = torch.ops.aten.add.Tensor(mm_default, arg145_1);  mm_default = arg145_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_96: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(add_tensor, [8, 196, 768]);  add_tensor = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_107: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_103, view_96);  add_103 = view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:266, code: x = self.norm(x)
    clone_84: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_107, memory_format = torch.contiguous_format);  add_107 = None
    var_mean_24 = torch.ops.aten.var_mean.correction(clone_84, [2], correction = 0, keepdim = True)
    getitem_48: "f32[8, 196, 1]" = var_mean_24[0]
    getitem_49: "f32[8, 196, 1]" = var_mean_24[1];  var_mean_24 = None
    sub_24: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_84, getitem_49);  clone_84 = getitem_49 = None
    add_108: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-06);  getitem_48 = None
    rsqrt_24: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
    mul_120: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = rsqrt_24 = None
    mul_121: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_120, arg146_1);  mul_120 = arg146_1 = None
    add_109: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_121, arg147_1);  mul_121 = arg147_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:271, code: x = x.mean(dim=1)
    mean: "f32[8, 768]" = torch.ops.aten.mean.dim(add_109, [1]);  add_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:273, code: return x if pre_logits else self.head(x)
    permute_73: "f32[768, 1000]" = torch.ops.aten.permute.default(arg148_1, [1, 0]);  arg148_1 = None
    addmm_36: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg149_1, mean, permute_73);  arg149_1 = mean = permute_73 = None
    return (addmm_36,)
    