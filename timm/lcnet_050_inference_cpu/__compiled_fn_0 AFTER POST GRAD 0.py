from __future__ import annotations



def forward(self, arg0_1: "f32[8]", arg1_1: "f32[8]", arg2_1: "f32[8]", arg3_1: "f32[8]", arg4_1: "f32[16]", arg5_1: "f32[16]", arg6_1: "f32[16]", arg7_1: "f32[16]", arg8_1: "f32[32]", arg9_1: "f32[32]", arg10_1: "f32[32]", arg11_1: "f32[32]", arg12_1: "f32[32]", arg13_1: "f32[32]", arg14_1: "f32[32]", arg15_1: "f32[32]", arg16_1: "f32[64]", arg17_1: "f32[64]", arg18_1: "f32[64]", arg19_1: "f32[64]", arg20_1: "f32[64]", arg21_1: "f32[64]", arg22_1: "f32[64]", arg23_1: "f32[64]", arg24_1: "f32[128]", arg25_1: "f32[128]", arg26_1: "f32[128]", arg27_1: "f32[128]", arg28_1: "f32[128]", arg29_1: "f32[128]", arg30_1: "f32[128]", arg31_1: "f32[128]", arg32_1: "f32[128]", arg33_1: "f32[128]", arg34_1: "f32[128]", arg35_1: "f32[128]", arg36_1: "f32[128]", arg37_1: "f32[128]", arg38_1: "f32[128]", arg39_1: "f32[128]", arg40_1: "f32[128]", arg41_1: "f32[128]", arg42_1: "f32[128]", arg43_1: "f32[128]", arg44_1: "f32[128]", arg45_1: "f32[128]", arg46_1: "f32[128]", arg47_1: "f32[128]", arg48_1: "f32[256]", arg49_1: "f32[256]", arg50_1: "f32[256]", arg51_1: "f32[256]", arg52_1: "f32[256]", arg53_1: "f32[256]", arg54_1: "f32[1000, 1280]", arg55_1: "f32[1000]", arg56_1: "f32[8, 3, 3, 3]", arg57_1: "f32[8, 1, 3, 3]", arg58_1: "f32[16, 8, 1, 1]", arg59_1: "f32[16, 1, 3, 3]", arg60_1: "f32[32, 16, 1, 1]", arg61_1: "f32[32, 1, 3, 3]", arg62_1: "f32[32, 32, 1, 1]", arg63_1: "f32[32, 1, 3, 3]", arg64_1: "f32[64, 32, 1, 1]", arg65_1: "f32[64, 1, 3, 3]", arg66_1: "f32[64, 64, 1, 1]", arg67_1: "f32[64, 1, 3, 3]", arg68_1: "f32[128, 64, 1, 1]", arg69_1: "f32[128, 1, 5, 5]", arg70_1: "f32[128, 128, 1, 1]", arg71_1: "f32[128, 1, 5, 5]", arg72_1: "f32[128, 128, 1, 1]", arg73_1: "f32[128, 1, 5, 5]", arg74_1: "f32[128, 128, 1, 1]", arg75_1: "f32[128, 1, 5, 5]", arg76_1: "f32[128, 128, 1, 1]", arg77_1: "f32[128, 1, 5, 5]", arg78_1: "f32[128, 128, 1, 1]", arg79_1: "f32[128, 1, 5, 5]", arg80_1: "f32[32, 128, 1, 1]", arg81_1: "f32[32]", arg82_1: "f32[128, 32, 1, 1]", arg83_1: "f32[128]", arg84_1: "f32[256, 128, 1, 1]", arg85_1: "f32[256, 1, 5, 5]", arg86_1: "f32[64, 256, 1, 1]", arg87_1: "f32[64]", arg88_1: "f32[256, 64, 1, 1]", arg89_1: "f32[256]", arg90_1: "f32[256, 256, 1, 1]", arg91_1: "f32[1280, 256, 1, 1]", arg92_1: "f32[1280]", arg93_1: "f32[8]", arg94_1: "f32[8]", arg95_1: "f32[8]", arg96_1: "f32[8]", arg97_1: "f32[16]", arg98_1: "f32[16]", arg99_1: "f32[16]", arg100_1: "f32[16]", arg101_1: "f32[32]", arg102_1: "f32[32]", arg103_1: "f32[32]", arg104_1: "f32[32]", arg105_1: "f32[32]", arg106_1: "f32[32]", arg107_1: "f32[32]", arg108_1: "f32[32]", arg109_1: "f32[64]", arg110_1: "f32[64]", arg111_1: "f32[64]", arg112_1: "f32[64]", arg113_1: "f32[64]", arg114_1: "f32[64]", arg115_1: "f32[64]", arg116_1: "f32[64]", arg117_1: "f32[128]", arg118_1: "f32[128]", arg119_1: "f32[128]", arg120_1: "f32[128]", arg121_1: "f32[128]", arg122_1: "f32[128]", arg123_1: "f32[128]", arg124_1: "f32[128]", arg125_1: "f32[128]", arg126_1: "f32[128]", arg127_1: "f32[128]", arg128_1: "f32[128]", arg129_1: "f32[128]", arg130_1: "f32[128]", arg131_1: "f32[128]", arg132_1: "f32[128]", arg133_1: "f32[128]", arg134_1: "f32[128]", arg135_1: "f32[128]", arg136_1: "f32[128]", arg137_1: "f32[128]", arg138_1: "f32[128]", arg139_1: "f32[128]", arg140_1: "f32[128]", arg141_1: "f32[256]", arg142_1: "f32[256]", arg143_1: "f32[256]", arg144_1: "f32[256]", arg145_1: "f32[256]", arg146_1: "f32[256]", arg147_1: "f32[8, 3, 224, 224]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilenetv3.py:135, code: x = self.conv_stem(x)
    convolution: "f32[8, 8, 112, 112]" = torch.ops.aten.convolution.default(arg147_1, arg56_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg147_1 = arg56_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(arg93_1, -1);  arg93_1 = None
    unsqueeze_1: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    sub: "f32[8, 8, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1);  convolution = unsqueeze_1 = None
    add: "f32[8]" = torch.ops.aten.add.Tensor(arg94_1, 1e-05);  arg94_1 = None
    sqrt: "f32[8]" = torch.ops.aten.sqrt.default(add);  add = None
    reciprocal: "f32[8]" = torch.ops.aten.reciprocal.default(sqrt);  sqrt = None
    mul: "f32[8]" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
    unsqueeze_2: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(mul, -1);  mul = None
    unsqueeze_3: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    mul_1: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(sub, unsqueeze_3);  sub = unsqueeze_3 = None
    unsqueeze_4: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(arg0_1, -1);  arg0_1 = None
    unsqueeze_5: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_2: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None
    unsqueeze_6: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(arg1_1, -1);  arg1_1 = None
    unsqueeze_7: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_1: "f32[8, 8, 112, 112]" = torch.ops.aten.add.Tensor(mul_2, unsqueeze_7);  mul_2 = unsqueeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_2: "f32[8, 8, 112, 112]" = torch.ops.aten.add.Tensor(add_1, 3)
    clamp_min: "f32[8, 8, 112, 112]" = torch.ops.aten.clamp_min.default(add_2, 0);  add_2 = None
    clamp_max: "f32[8, 8, 112, 112]" = torch.ops.aten.clamp_max.default(clamp_min, 6);  clamp_min = None
    mul_3: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(add_1, clamp_max);  add_1 = clamp_max = None
    div: "f32[8, 8, 112, 112]" = torch.ops.aten.div.Tensor(mul_3, 6);  mul_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_1: "f32[8, 8, 112, 112]" = torch.ops.aten.convolution.default(div, arg57_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  div = arg57_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_8: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(arg95_1, -1);  arg95_1 = None
    unsqueeze_9: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    sub_1: "f32[8, 8, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_9);  convolution_1 = unsqueeze_9 = None
    add_3: "f32[8]" = torch.ops.aten.add.Tensor(arg96_1, 1e-05);  arg96_1 = None
    sqrt_1: "f32[8]" = torch.ops.aten.sqrt.default(add_3);  add_3 = None
    reciprocal_1: "f32[8]" = torch.ops.aten.reciprocal.default(sqrt_1);  sqrt_1 = None
    mul_4: "f32[8]" = torch.ops.aten.mul.Tensor(reciprocal_1, 1);  reciprocal_1 = None
    unsqueeze_10: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(mul_4, -1);  mul_4 = None
    unsqueeze_11: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    mul_5: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(sub_1, unsqueeze_11);  sub_1 = unsqueeze_11 = None
    unsqueeze_12: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
    unsqueeze_13: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_6: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(mul_5, unsqueeze_13);  mul_5 = unsqueeze_13 = None
    unsqueeze_14: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, -1);  arg3_1 = None
    unsqueeze_15: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_4: "f32[8, 8, 112, 112]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_15);  mul_6 = unsqueeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_5: "f32[8, 8, 112, 112]" = torch.ops.aten.add.Tensor(add_4, 3)
    clamp_min_1: "f32[8, 8, 112, 112]" = torch.ops.aten.clamp_min.default(add_5, 0);  add_5 = None
    clamp_max_1: "f32[8, 8, 112, 112]" = torch.ops.aten.clamp_max.default(clamp_min_1, 6);  clamp_min_1 = None
    mul_7: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(add_4, clamp_max_1);  add_4 = clamp_max_1 = None
    div_1: "f32[8, 8, 112, 112]" = torch.ops.aten.div.Tensor(mul_7, 6);  mul_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_2: "f32[8, 16, 112, 112]" = torch.ops.aten.convolution.default(div_1, arg58_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_1 = arg58_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_16: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg97_1, -1);  arg97_1 = None
    unsqueeze_17: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    sub_2: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_17);  convolution_2 = unsqueeze_17 = None
    add_6: "f32[16]" = torch.ops.aten.add.Tensor(arg98_1, 1e-05);  arg98_1 = None
    sqrt_2: "f32[16]" = torch.ops.aten.sqrt.default(add_6);  add_6 = None
    reciprocal_2: "f32[16]" = torch.ops.aten.reciprocal.default(sqrt_2);  sqrt_2 = None
    mul_8: "f32[16]" = torch.ops.aten.mul.Tensor(reciprocal_2, 1);  reciprocal_2 = None
    unsqueeze_18: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(mul_8, -1);  mul_8 = None
    unsqueeze_19: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    mul_9: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_2, unsqueeze_19);  sub_2 = unsqueeze_19 = None
    unsqueeze_20: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
    unsqueeze_21: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_10: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(mul_9, unsqueeze_21);  mul_9 = unsqueeze_21 = None
    unsqueeze_22: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
    unsqueeze_23: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_7: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(mul_10, unsqueeze_23);  mul_10 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_8: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(add_7, 3)
    clamp_min_2: "f32[8, 16, 112, 112]" = torch.ops.aten.clamp_min.default(add_8, 0);  add_8 = None
    clamp_max_2: "f32[8, 16, 112, 112]" = torch.ops.aten.clamp_max.default(clamp_min_2, 6);  clamp_min_2 = None
    mul_11: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(add_7, clamp_max_2);  add_7 = clamp_max_2 = None
    div_2: "f32[8, 16, 112, 112]" = torch.ops.aten.div.Tensor(mul_11, 6);  mul_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_3: "f32[8, 16, 56, 56]" = torch.ops.aten.convolution.default(div_2, arg59_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 16);  div_2 = arg59_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_24: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg99_1, -1);  arg99_1 = None
    unsqueeze_25: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
    sub_3: "f32[8, 16, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_25);  convolution_3 = unsqueeze_25 = None
    add_9: "f32[16]" = torch.ops.aten.add.Tensor(arg100_1, 1e-05);  arg100_1 = None
    sqrt_3: "f32[16]" = torch.ops.aten.sqrt.default(add_9);  add_9 = None
    reciprocal_3: "f32[16]" = torch.ops.aten.reciprocal.default(sqrt_3);  sqrt_3 = None
    mul_12: "f32[16]" = torch.ops.aten.mul.Tensor(reciprocal_3, 1);  reciprocal_3 = None
    unsqueeze_26: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(mul_12, -1);  mul_12 = None
    unsqueeze_27: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
    mul_13: "f32[8, 16, 56, 56]" = torch.ops.aten.mul.Tensor(sub_3, unsqueeze_27);  sub_3 = unsqueeze_27 = None
    unsqueeze_28: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg6_1, -1);  arg6_1 = None
    unsqueeze_29: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    mul_14: "f32[8, 16, 56, 56]" = torch.ops.aten.mul.Tensor(mul_13, unsqueeze_29);  mul_13 = unsqueeze_29 = None
    unsqueeze_30: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
    unsqueeze_31: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
    add_10: "f32[8, 16, 56, 56]" = torch.ops.aten.add.Tensor(mul_14, unsqueeze_31);  mul_14 = unsqueeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_11: "f32[8, 16, 56, 56]" = torch.ops.aten.add.Tensor(add_10, 3)
    clamp_min_3: "f32[8, 16, 56, 56]" = torch.ops.aten.clamp_min.default(add_11, 0);  add_11 = None
    clamp_max_3: "f32[8, 16, 56, 56]" = torch.ops.aten.clamp_max.default(clamp_min_3, 6);  clamp_min_3 = None
    mul_15: "f32[8, 16, 56, 56]" = torch.ops.aten.mul.Tensor(add_10, clamp_max_3);  add_10 = clamp_max_3 = None
    div_3: "f32[8, 16, 56, 56]" = torch.ops.aten.div.Tensor(mul_15, 6);  mul_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_4: "f32[8, 32, 56, 56]" = torch.ops.aten.convolution.default(div_3, arg60_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_3 = arg60_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_32: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg101_1, -1);  arg101_1 = None
    unsqueeze_33: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    sub_4: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_33);  convolution_4 = unsqueeze_33 = None
    add_12: "f32[32]" = torch.ops.aten.add.Tensor(arg102_1, 1e-05);  arg102_1 = None
    sqrt_4: "f32[32]" = torch.ops.aten.sqrt.default(add_12);  add_12 = None
    reciprocal_4: "f32[32]" = torch.ops.aten.reciprocal.default(sqrt_4);  sqrt_4 = None
    mul_16: "f32[32]" = torch.ops.aten.mul.Tensor(reciprocal_4, 1);  reciprocal_4 = None
    unsqueeze_34: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(mul_16, -1);  mul_16 = None
    unsqueeze_35: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
    mul_17: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_4, unsqueeze_35);  sub_4 = unsqueeze_35 = None
    unsqueeze_36: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg8_1, -1);  arg8_1 = None
    unsqueeze_37: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
    mul_18: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(mul_17, unsqueeze_37);  mul_17 = unsqueeze_37 = None
    unsqueeze_38: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
    unsqueeze_39: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
    add_13: "f32[8, 32, 56, 56]" = torch.ops.aten.add.Tensor(mul_18, unsqueeze_39);  mul_18 = unsqueeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_14: "f32[8, 32, 56, 56]" = torch.ops.aten.add.Tensor(add_13, 3)
    clamp_min_4: "f32[8, 32, 56, 56]" = torch.ops.aten.clamp_min.default(add_14, 0);  add_14 = None
    clamp_max_4: "f32[8, 32, 56, 56]" = torch.ops.aten.clamp_max.default(clamp_min_4, 6);  clamp_min_4 = None
    mul_19: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(add_13, clamp_max_4);  add_13 = clamp_max_4 = None
    div_4: "f32[8, 32, 56, 56]" = torch.ops.aten.div.Tensor(mul_19, 6);  mul_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_5: "f32[8, 32, 56, 56]" = torch.ops.aten.convolution.default(div_4, arg61_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32);  div_4 = arg61_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_40: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg103_1, -1);  arg103_1 = None
    unsqueeze_41: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    sub_5: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_41);  convolution_5 = unsqueeze_41 = None
    add_15: "f32[32]" = torch.ops.aten.add.Tensor(arg104_1, 1e-05);  arg104_1 = None
    sqrt_5: "f32[32]" = torch.ops.aten.sqrt.default(add_15);  add_15 = None
    reciprocal_5: "f32[32]" = torch.ops.aten.reciprocal.default(sqrt_5);  sqrt_5 = None
    mul_20: "f32[32]" = torch.ops.aten.mul.Tensor(reciprocal_5, 1);  reciprocal_5 = None
    unsqueeze_42: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(mul_20, -1);  mul_20 = None
    unsqueeze_43: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    mul_21: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_5, unsqueeze_43);  sub_5 = unsqueeze_43 = None
    unsqueeze_44: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
    unsqueeze_45: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    mul_22: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(mul_21, unsqueeze_45);  mul_21 = unsqueeze_45 = None
    unsqueeze_46: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg11_1, -1);  arg11_1 = None
    unsqueeze_47: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
    add_16: "f32[8, 32, 56, 56]" = torch.ops.aten.add.Tensor(mul_22, unsqueeze_47);  mul_22 = unsqueeze_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_17: "f32[8, 32, 56, 56]" = torch.ops.aten.add.Tensor(add_16, 3)
    clamp_min_5: "f32[8, 32, 56, 56]" = torch.ops.aten.clamp_min.default(add_17, 0);  add_17 = None
    clamp_max_5: "f32[8, 32, 56, 56]" = torch.ops.aten.clamp_max.default(clamp_min_5, 6);  clamp_min_5 = None
    mul_23: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(add_16, clamp_max_5);  add_16 = clamp_max_5 = None
    div_5: "f32[8, 32, 56, 56]" = torch.ops.aten.div.Tensor(mul_23, 6);  mul_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_6: "f32[8, 32, 56, 56]" = torch.ops.aten.convolution.default(div_5, arg62_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_5 = arg62_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_48: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg105_1, -1);  arg105_1 = None
    unsqueeze_49: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
    sub_6: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_49);  convolution_6 = unsqueeze_49 = None
    add_18: "f32[32]" = torch.ops.aten.add.Tensor(arg106_1, 1e-05);  arg106_1 = None
    sqrt_6: "f32[32]" = torch.ops.aten.sqrt.default(add_18);  add_18 = None
    reciprocal_6: "f32[32]" = torch.ops.aten.reciprocal.default(sqrt_6);  sqrt_6 = None
    mul_24: "f32[32]" = torch.ops.aten.mul.Tensor(reciprocal_6, 1);  reciprocal_6 = None
    unsqueeze_50: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(mul_24, -1);  mul_24 = None
    unsqueeze_51: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
    mul_25: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_6, unsqueeze_51);  sub_6 = unsqueeze_51 = None
    unsqueeze_52: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg12_1, -1);  arg12_1 = None
    unsqueeze_53: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    mul_26: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(mul_25, unsqueeze_53);  mul_25 = unsqueeze_53 = None
    unsqueeze_54: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg13_1, -1);  arg13_1 = None
    unsqueeze_55: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
    add_19: "f32[8, 32, 56, 56]" = torch.ops.aten.add.Tensor(mul_26, unsqueeze_55);  mul_26 = unsqueeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_20: "f32[8, 32, 56, 56]" = torch.ops.aten.add.Tensor(add_19, 3)
    clamp_min_6: "f32[8, 32, 56, 56]" = torch.ops.aten.clamp_min.default(add_20, 0);  add_20 = None
    clamp_max_6: "f32[8, 32, 56, 56]" = torch.ops.aten.clamp_max.default(clamp_min_6, 6);  clamp_min_6 = None
    mul_27: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(add_19, clamp_max_6);  add_19 = clamp_max_6 = None
    div_6: "f32[8, 32, 56, 56]" = torch.ops.aten.div.Tensor(mul_27, 6);  mul_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_7: "f32[8, 32, 28, 28]" = torch.ops.aten.convolution.default(div_6, arg63_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 32);  div_6 = arg63_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_56: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg107_1, -1);  arg107_1 = None
    unsqueeze_57: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    sub_7: "f32[8, 32, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_57);  convolution_7 = unsqueeze_57 = None
    add_21: "f32[32]" = torch.ops.aten.add.Tensor(arg108_1, 1e-05);  arg108_1 = None
    sqrt_7: "f32[32]" = torch.ops.aten.sqrt.default(add_21);  add_21 = None
    reciprocal_7: "f32[32]" = torch.ops.aten.reciprocal.default(sqrt_7);  sqrt_7 = None
    mul_28: "f32[32]" = torch.ops.aten.mul.Tensor(reciprocal_7, 1);  reciprocal_7 = None
    unsqueeze_58: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(mul_28, -1);  mul_28 = None
    unsqueeze_59: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
    mul_29: "f32[8, 32, 28, 28]" = torch.ops.aten.mul.Tensor(sub_7, unsqueeze_59);  sub_7 = unsqueeze_59 = None
    unsqueeze_60: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
    unsqueeze_61: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
    mul_30: "f32[8, 32, 28, 28]" = torch.ops.aten.mul.Tensor(mul_29, unsqueeze_61);  mul_29 = unsqueeze_61 = None
    unsqueeze_62: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
    unsqueeze_63: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
    add_22: "f32[8, 32, 28, 28]" = torch.ops.aten.add.Tensor(mul_30, unsqueeze_63);  mul_30 = unsqueeze_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_23: "f32[8, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_22, 3)
    clamp_min_7: "f32[8, 32, 28, 28]" = torch.ops.aten.clamp_min.default(add_23, 0);  add_23 = None
    clamp_max_7: "f32[8, 32, 28, 28]" = torch.ops.aten.clamp_max.default(clamp_min_7, 6);  clamp_min_7 = None
    mul_31: "f32[8, 32, 28, 28]" = torch.ops.aten.mul.Tensor(add_22, clamp_max_7);  add_22 = clamp_max_7 = None
    div_7: "f32[8, 32, 28, 28]" = torch.ops.aten.div.Tensor(mul_31, 6);  mul_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_8: "f32[8, 64, 28, 28]" = torch.ops.aten.convolution.default(div_7, arg64_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_7 = arg64_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_64: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg109_1, -1);  arg109_1 = None
    unsqueeze_65: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
    sub_8: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_65);  convolution_8 = unsqueeze_65 = None
    add_24: "f32[64]" = torch.ops.aten.add.Tensor(arg110_1, 1e-05);  arg110_1 = None
    sqrt_8: "f32[64]" = torch.ops.aten.sqrt.default(add_24);  add_24 = None
    reciprocal_8: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_8);  sqrt_8 = None
    mul_32: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_8, 1);  reciprocal_8 = None
    unsqueeze_66: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_32, -1);  mul_32 = None
    unsqueeze_67: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
    mul_33: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_8, unsqueeze_67);  sub_8 = unsqueeze_67 = None
    unsqueeze_68: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg16_1, -1);  arg16_1 = None
    unsqueeze_69: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    mul_34: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(mul_33, unsqueeze_69);  mul_33 = unsqueeze_69 = None
    unsqueeze_70: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg17_1, -1);  arg17_1 = None
    unsqueeze_71: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    add_25: "f32[8, 64, 28, 28]" = torch.ops.aten.add.Tensor(mul_34, unsqueeze_71);  mul_34 = unsqueeze_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_26: "f32[8, 64, 28, 28]" = torch.ops.aten.add.Tensor(add_25, 3)
    clamp_min_8: "f32[8, 64, 28, 28]" = torch.ops.aten.clamp_min.default(add_26, 0);  add_26 = None
    clamp_max_8: "f32[8, 64, 28, 28]" = torch.ops.aten.clamp_max.default(clamp_min_8, 6);  clamp_min_8 = None
    mul_35: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(add_25, clamp_max_8);  add_25 = clamp_max_8 = None
    div_8: "f32[8, 64, 28, 28]" = torch.ops.aten.div.Tensor(mul_35, 6);  mul_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_9: "f32[8, 64, 28, 28]" = torch.ops.aten.convolution.default(div_8, arg65_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 64);  div_8 = arg65_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_72: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg111_1, -1);  arg111_1 = None
    unsqueeze_73: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
    sub_9: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_73);  convolution_9 = unsqueeze_73 = None
    add_27: "f32[64]" = torch.ops.aten.add.Tensor(arg112_1, 1e-05);  arg112_1 = None
    sqrt_9: "f32[64]" = torch.ops.aten.sqrt.default(add_27);  add_27 = None
    reciprocal_9: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_9);  sqrt_9 = None
    mul_36: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_9, 1);  reciprocal_9 = None
    unsqueeze_74: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_36, -1);  mul_36 = None
    unsqueeze_75: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
    mul_37: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_9, unsqueeze_75);  sub_9 = unsqueeze_75 = None
    unsqueeze_76: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg18_1, -1);  arg18_1 = None
    unsqueeze_77: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
    mul_38: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(mul_37, unsqueeze_77);  mul_37 = unsqueeze_77 = None
    unsqueeze_78: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg19_1, -1);  arg19_1 = None
    unsqueeze_79: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    add_28: "f32[8, 64, 28, 28]" = torch.ops.aten.add.Tensor(mul_38, unsqueeze_79);  mul_38 = unsqueeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_29: "f32[8, 64, 28, 28]" = torch.ops.aten.add.Tensor(add_28, 3)
    clamp_min_9: "f32[8, 64, 28, 28]" = torch.ops.aten.clamp_min.default(add_29, 0);  add_29 = None
    clamp_max_9: "f32[8, 64, 28, 28]" = torch.ops.aten.clamp_max.default(clamp_min_9, 6);  clamp_min_9 = None
    mul_39: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(add_28, clamp_max_9);  add_28 = clamp_max_9 = None
    div_9: "f32[8, 64, 28, 28]" = torch.ops.aten.div.Tensor(mul_39, 6);  mul_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_10: "f32[8, 64, 28, 28]" = torch.ops.aten.convolution.default(div_9, arg66_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_9 = arg66_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_80: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg113_1, -1);  arg113_1 = None
    unsqueeze_81: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
    sub_10: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_81);  convolution_10 = unsqueeze_81 = None
    add_30: "f32[64]" = torch.ops.aten.add.Tensor(arg114_1, 1e-05);  arg114_1 = None
    sqrt_10: "f32[64]" = torch.ops.aten.sqrt.default(add_30);  add_30 = None
    reciprocal_10: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_10);  sqrt_10 = None
    mul_40: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_10, 1);  reciprocal_10 = None
    unsqueeze_82: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_40, -1);  mul_40 = None
    unsqueeze_83: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    mul_41: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_10, unsqueeze_83);  sub_10 = unsqueeze_83 = None
    unsqueeze_84: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg20_1, -1);  arg20_1 = None
    unsqueeze_85: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    mul_42: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(mul_41, unsqueeze_85);  mul_41 = unsqueeze_85 = None
    unsqueeze_86: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg21_1, -1);  arg21_1 = None
    unsqueeze_87: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
    add_31: "f32[8, 64, 28, 28]" = torch.ops.aten.add.Tensor(mul_42, unsqueeze_87);  mul_42 = unsqueeze_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_32: "f32[8, 64, 28, 28]" = torch.ops.aten.add.Tensor(add_31, 3)
    clamp_min_10: "f32[8, 64, 28, 28]" = torch.ops.aten.clamp_min.default(add_32, 0);  add_32 = None
    clamp_max_10: "f32[8, 64, 28, 28]" = torch.ops.aten.clamp_max.default(clamp_min_10, 6);  clamp_min_10 = None
    mul_43: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(add_31, clamp_max_10);  add_31 = clamp_max_10 = None
    div_10: "f32[8, 64, 28, 28]" = torch.ops.aten.div.Tensor(mul_43, 6);  mul_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_11: "f32[8, 64, 14, 14]" = torch.ops.aten.convolution.default(div_10, arg67_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 64);  div_10 = arg67_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_88: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg115_1, -1);  arg115_1 = None
    unsqueeze_89: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
    sub_11: "f32[8, 64, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_89);  convolution_11 = unsqueeze_89 = None
    add_33: "f32[64]" = torch.ops.aten.add.Tensor(arg116_1, 1e-05);  arg116_1 = None
    sqrt_11: "f32[64]" = torch.ops.aten.sqrt.default(add_33);  add_33 = None
    reciprocal_11: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_11);  sqrt_11 = None
    mul_44: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_11, 1);  reciprocal_11 = None
    unsqueeze_90: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_44, -1);  mul_44 = None
    unsqueeze_91: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    mul_45: "f32[8, 64, 14, 14]" = torch.ops.aten.mul.Tensor(sub_11, unsqueeze_91);  sub_11 = unsqueeze_91 = None
    unsqueeze_92: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg22_1, -1);  arg22_1 = None
    unsqueeze_93: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
    mul_46: "f32[8, 64, 14, 14]" = torch.ops.aten.mul.Tensor(mul_45, unsqueeze_93);  mul_45 = unsqueeze_93 = None
    unsqueeze_94: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg23_1, -1);  arg23_1 = None
    unsqueeze_95: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
    add_34: "f32[8, 64, 14, 14]" = torch.ops.aten.add.Tensor(mul_46, unsqueeze_95);  mul_46 = unsqueeze_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_35: "f32[8, 64, 14, 14]" = torch.ops.aten.add.Tensor(add_34, 3)
    clamp_min_11: "f32[8, 64, 14, 14]" = torch.ops.aten.clamp_min.default(add_35, 0);  add_35 = None
    clamp_max_11: "f32[8, 64, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_11, 6);  clamp_min_11 = None
    mul_47: "f32[8, 64, 14, 14]" = torch.ops.aten.mul.Tensor(add_34, clamp_max_11);  add_34 = clamp_max_11 = None
    div_11: "f32[8, 64, 14, 14]" = torch.ops.aten.div.Tensor(mul_47, 6);  mul_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_12: "f32[8, 128, 14, 14]" = torch.ops.aten.convolution.default(div_11, arg68_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_11 = arg68_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_96: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg117_1, -1);  arg117_1 = None
    unsqueeze_97: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
    sub_12: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_97);  convolution_12 = unsqueeze_97 = None
    add_36: "f32[128]" = torch.ops.aten.add.Tensor(arg118_1, 1e-05);  arg118_1 = None
    sqrt_12: "f32[128]" = torch.ops.aten.sqrt.default(add_36);  add_36 = None
    reciprocal_12: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_12);  sqrt_12 = None
    mul_48: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_12, 1);  reciprocal_12 = None
    unsqueeze_98: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_48, -1);  mul_48 = None
    unsqueeze_99: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
    mul_49: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_12, unsqueeze_99);  sub_12 = unsqueeze_99 = None
    unsqueeze_100: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg24_1, -1);  arg24_1 = None
    unsqueeze_101: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
    mul_50: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_101);  mul_49 = unsqueeze_101 = None
    unsqueeze_102: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg25_1, -1);  arg25_1 = None
    unsqueeze_103: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    add_37: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_50, unsqueeze_103);  mul_50 = unsqueeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_38: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(add_37, 3)
    clamp_min_12: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_min.default(add_38, 0);  add_38 = None
    clamp_max_12: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_12, 6);  clamp_min_12 = None
    mul_51: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(add_37, clamp_max_12);  add_37 = clamp_max_12 = None
    div_12: "f32[8, 128, 14, 14]" = torch.ops.aten.div.Tensor(mul_51, 6);  mul_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_13: "f32[8, 128, 14, 14]" = torch.ops.aten.convolution.default(div_12, arg69_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 128);  div_12 = arg69_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_104: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg119_1, -1);  arg119_1 = None
    unsqueeze_105: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
    sub_13: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_105);  convolution_13 = unsqueeze_105 = None
    add_39: "f32[128]" = torch.ops.aten.add.Tensor(arg120_1, 1e-05);  arg120_1 = None
    sqrt_13: "f32[128]" = torch.ops.aten.sqrt.default(add_39);  add_39 = None
    reciprocal_13: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_13);  sqrt_13 = None
    mul_52: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_13, 1);  reciprocal_13 = None
    unsqueeze_106: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_52, -1);  mul_52 = None
    unsqueeze_107: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
    mul_53: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_13, unsqueeze_107);  sub_13 = unsqueeze_107 = None
    unsqueeze_108: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg26_1, -1);  arg26_1 = None
    unsqueeze_109: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
    mul_54: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_53, unsqueeze_109);  mul_53 = unsqueeze_109 = None
    unsqueeze_110: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg27_1, -1);  arg27_1 = None
    unsqueeze_111: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
    add_40: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_54, unsqueeze_111);  mul_54 = unsqueeze_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_41: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(add_40, 3)
    clamp_min_13: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_min.default(add_41, 0);  add_41 = None
    clamp_max_13: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_13, 6);  clamp_min_13 = None
    mul_55: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(add_40, clamp_max_13);  add_40 = clamp_max_13 = None
    div_13: "f32[8, 128, 14, 14]" = torch.ops.aten.div.Tensor(mul_55, 6);  mul_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_14: "f32[8, 128, 14, 14]" = torch.ops.aten.convolution.default(div_13, arg70_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_13 = arg70_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_112: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg121_1, -1);  arg121_1 = None
    unsqueeze_113: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
    sub_14: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_113);  convolution_14 = unsqueeze_113 = None
    add_42: "f32[128]" = torch.ops.aten.add.Tensor(arg122_1, 1e-05);  arg122_1 = None
    sqrt_14: "f32[128]" = torch.ops.aten.sqrt.default(add_42);  add_42 = None
    reciprocal_14: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_14);  sqrt_14 = None
    mul_56: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_14, 1);  reciprocal_14 = None
    unsqueeze_114: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_56, -1);  mul_56 = None
    unsqueeze_115: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
    mul_57: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_14, unsqueeze_115);  sub_14 = unsqueeze_115 = None
    unsqueeze_116: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg28_1, -1);  arg28_1 = None
    unsqueeze_117: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
    mul_58: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_57, unsqueeze_117);  mul_57 = unsqueeze_117 = None
    unsqueeze_118: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg29_1, -1);  arg29_1 = None
    unsqueeze_119: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
    add_43: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_58, unsqueeze_119);  mul_58 = unsqueeze_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_44: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(add_43, 3)
    clamp_min_14: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_min.default(add_44, 0);  add_44 = None
    clamp_max_14: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_14, 6);  clamp_min_14 = None
    mul_59: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(add_43, clamp_max_14);  add_43 = clamp_max_14 = None
    div_14: "f32[8, 128, 14, 14]" = torch.ops.aten.div.Tensor(mul_59, 6);  mul_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_15: "f32[8, 128, 14, 14]" = torch.ops.aten.convolution.default(div_14, arg71_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 128);  div_14 = arg71_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_120: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg123_1, -1);  arg123_1 = None
    unsqueeze_121: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
    sub_15: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_121);  convolution_15 = unsqueeze_121 = None
    add_45: "f32[128]" = torch.ops.aten.add.Tensor(arg124_1, 1e-05);  arg124_1 = None
    sqrt_15: "f32[128]" = torch.ops.aten.sqrt.default(add_45);  add_45 = None
    reciprocal_15: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_15);  sqrt_15 = None
    mul_60: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_15, 1);  reciprocal_15 = None
    unsqueeze_122: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_60, -1);  mul_60 = None
    unsqueeze_123: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
    mul_61: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_15, unsqueeze_123);  sub_15 = unsqueeze_123 = None
    unsqueeze_124: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg30_1, -1);  arg30_1 = None
    unsqueeze_125: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
    mul_62: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_61, unsqueeze_125);  mul_61 = unsqueeze_125 = None
    unsqueeze_126: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg31_1, -1);  arg31_1 = None
    unsqueeze_127: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
    add_46: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_62, unsqueeze_127);  mul_62 = unsqueeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_47: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(add_46, 3)
    clamp_min_15: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_min.default(add_47, 0);  add_47 = None
    clamp_max_15: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_15, 6);  clamp_min_15 = None
    mul_63: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(add_46, clamp_max_15);  add_46 = clamp_max_15 = None
    div_15: "f32[8, 128, 14, 14]" = torch.ops.aten.div.Tensor(mul_63, 6);  mul_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_16: "f32[8, 128, 14, 14]" = torch.ops.aten.convolution.default(div_15, arg72_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_15 = arg72_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_128: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg125_1, -1);  arg125_1 = None
    unsqueeze_129: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
    sub_16: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_129);  convolution_16 = unsqueeze_129 = None
    add_48: "f32[128]" = torch.ops.aten.add.Tensor(arg126_1, 1e-05);  arg126_1 = None
    sqrt_16: "f32[128]" = torch.ops.aten.sqrt.default(add_48);  add_48 = None
    reciprocal_16: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_16);  sqrt_16 = None
    mul_64: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_16, 1);  reciprocal_16 = None
    unsqueeze_130: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_64, -1);  mul_64 = None
    unsqueeze_131: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
    mul_65: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_16, unsqueeze_131);  sub_16 = unsqueeze_131 = None
    unsqueeze_132: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg32_1, -1);  arg32_1 = None
    unsqueeze_133: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
    mul_66: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_65, unsqueeze_133);  mul_65 = unsqueeze_133 = None
    unsqueeze_134: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg33_1, -1);  arg33_1 = None
    unsqueeze_135: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
    add_49: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_66, unsqueeze_135);  mul_66 = unsqueeze_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_50: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(add_49, 3)
    clamp_min_16: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_min.default(add_50, 0);  add_50 = None
    clamp_max_16: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_16, 6);  clamp_min_16 = None
    mul_67: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(add_49, clamp_max_16);  add_49 = clamp_max_16 = None
    div_16: "f32[8, 128, 14, 14]" = torch.ops.aten.div.Tensor(mul_67, 6);  mul_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_17: "f32[8, 128, 14, 14]" = torch.ops.aten.convolution.default(div_16, arg73_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 128);  div_16 = arg73_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_136: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg127_1, -1);  arg127_1 = None
    unsqueeze_137: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
    sub_17: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_137);  convolution_17 = unsqueeze_137 = None
    add_51: "f32[128]" = torch.ops.aten.add.Tensor(arg128_1, 1e-05);  arg128_1 = None
    sqrt_17: "f32[128]" = torch.ops.aten.sqrt.default(add_51);  add_51 = None
    reciprocal_17: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_17);  sqrt_17 = None
    mul_68: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_17, 1);  reciprocal_17 = None
    unsqueeze_138: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_68, -1);  mul_68 = None
    unsqueeze_139: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
    mul_69: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_17, unsqueeze_139);  sub_17 = unsqueeze_139 = None
    unsqueeze_140: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg34_1, -1);  arg34_1 = None
    unsqueeze_141: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
    mul_70: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_69, unsqueeze_141);  mul_69 = unsqueeze_141 = None
    unsqueeze_142: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg35_1, -1);  arg35_1 = None
    unsqueeze_143: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
    add_52: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_70, unsqueeze_143);  mul_70 = unsqueeze_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_53: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(add_52, 3)
    clamp_min_17: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_min.default(add_53, 0);  add_53 = None
    clamp_max_17: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_17, 6);  clamp_min_17 = None
    mul_71: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(add_52, clamp_max_17);  add_52 = clamp_max_17 = None
    div_17: "f32[8, 128, 14, 14]" = torch.ops.aten.div.Tensor(mul_71, 6);  mul_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_18: "f32[8, 128, 14, 14]" = torch.ops.aten.convolution.default(div_17, arg74_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_17 = arg74_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_144: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg129_1, -1);  arg129_1 = None
    unsqueeze_145: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
    sub_18: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_145);  convolution_18 = unsqueeze_145 = None
    add_54: "f32[128]" = torch.ops.aten.add.Tensor(arg130_1, 1e-05);  arg130_1 = None
    sqrt_18: "f32[128]" = torch.ops.aten.sqrt.default(add_54);  add_54 = None
    reciprocal_18: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_18);  sqrt_18 = None
    mul_72: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_18, 1);  reciprocal_18 = None
    unsqueeze_146: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_72, -1);  mul_72 = None
    unsqueeze_147: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
    mul_73: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_18, unsqueeze_147);  sub_18 = unsqueeze_147 = None
    unsqueeze_148: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg36_1, -1);  arg36_1 = None
    unsqueeze_149: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
    mul_74: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_73, unsqueeze_149);  mul_73 = unsqueeze_149 = None
    unsqueeze_150: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg37_1, -1);  arg37_1 = None
    unsqueeze_151: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
    add_55: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_74, unsqueeze_151);  mul_74 = unsqueeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_56: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(add_55, 3)
    clamp_min_18: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_min.default(add_56, 0);  add_56 = None
    clamp_max_18: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_18, 6);  clamp_min_18 = None
    mul_75: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(add_55, clamp_max_18);  add_55 = clamp_max_18 = None
    div_18: "f32[8, 128, 14, 14]" = torch.ops.aten.div.Tensor(mul_75, 6);  mul_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_19: "f32[8, 128, 14, 14]" = torch.ops.aten.convolution.default(div_18, arg75_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 128);  div_18 = arg75_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_152: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg131_1, -1);  arg131_1 = None
    unsqueeze_153: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
    sub_19: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_153);  convolution_19 = unsqueeze_153 = None
    add_57: "f32[128]" = torch.ops.aten.add.Tensor(arg132_1, 1e-05);  arg132_1 = None
    sqrt_19: "f32[128]" = torch.ops.aten.sqrt.default(add_57);  add_57 = None
    reciprocal_19: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_19);  sqrt_19 = None
    mul_76: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_19, 1);  reciprocal_19 = None
    unsqueeze_154: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_76, -1);  mul_76 = None
    unsqueeze_155: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
    mul_77: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_19, unsqueeze_155);  sub_19 = unsqueeze_155 = None
    unsqueeze_156: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg38_1, -1);  arg38_1 = None
    unsqueeze_157: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
    mul_78: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_77, unsqueeze_157);  mul_77 = unsqueeze_157 = None
    unsqueeze_158: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg39_1, -1);  arg39_1 = None
    unsqueeze_159: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
    add_58: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_78, unsqueeze_159);  mul_78 = unsqueeze_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_59: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(add_58, 3)
    clamp_min_19: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_min.default(add_59, 0);  add_59 = None
    clamp_max_19: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_19, 6);  clamp_min_19 = None
    mul_79: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(add_58, clamp_max_19);  add_58 = clamp_max_19 = None
    div_19: "f32[8, 128, 14, 14]" = torch.ops.aten.div.Tensor(mul_79, 6);  mul_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_20: "f32[8, 128, 14, 14]" = torch.ops.aten.convolution.default(div_19, arg76_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_19 = arg76_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_160: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg133_1, -1);  arg133_1 = None
    unsqueeze_161: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
    sub_20: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_161);  convolution_20 = unsqueeze_161 = None
    add_60: "f32[128]" = torch.ops.aten.add.Tensor(arg134_1, 1e-05);  arg134_1 = None
    sqrt_20: "f32[128]" = torch.ops.aten.sqrt.default(add_60);  add_60 = None
    reciprocal_20: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_20);  sqrt_20 = None
    mul_80: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_20, 1);  reciprocal_20 = None
    unsqueeze_162: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_80, -1);  mul_80 = None
    unsqueeze_163: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
    mul_81: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_20, unsqueeze_163);  sub_20 = unsqueeze_163 = None
    unsqueeze_164: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg40_1, -1);  arg40_1 = None
    unsqueeze_165: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
    mul_82: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_81, unsqueeze_165);  mul_81 = unsqueeze_165 = None
    unsqueeze_166: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg41_1, -1);  arg41_1 = None
    unsqueeze_167: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
    add_61: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_82, unsqueeze_167);  mul_82 = unsqueeze_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_62: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(add_61, 3)
    clamp_min_20: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_min.default(add_62, 0);  add_62 = None
    clamp_max_20: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_20, 6);  clamp_min_20 = None
    mul_83: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(add_61, clamp_max_20);  add_61 = clamp_max_20 = None
    div_20: "f32[8, 128, 14, 14]" = torch.ops.aten.div.Tensor(mul_83, 6);  mul_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_21: "f32[8, 128, 14, 14]" = torch.ops.aten.convolution.default(div_20, arg77_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 128);  div_20 = arg77_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_168: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg135_1, -1);  arg135_1 = None
    unsqueeze_169: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
    sub_21: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_169);  convolution_21 = unsqueeze_169 = None
    add_63: "f32[128]" = torch.ops.aten.add.Tensor(arg136_1, 1e-05);  arg136_1 = None
    sqrt_21: "f32[128]" = torch.ops.aten.sqrt.default(add_63);  add_63 = None
    reciprocal_21: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_21);  sqrt_21 = None
    mul_84: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_21, 1);  reciprocal_21 = None
    unsqueeze_170: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_84, -1);  mul_84 = None
    unsqueeze_171: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
    mul_85: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_21, unsqueeze_171);  sub_21 = unsqueeze_171 = None
    unsqueeze_172: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg42_1, -1);  arg42_1 = None
    unsqueeze_173: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
    mul_86: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_85, unsqueeze_173);  mul_85 = unsqueeze_173 = None
    unsqueeze_174: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg43_1, -1);  arg43_1 = None
    unsqueeze_175: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
    add_64: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_86, unsqueeze_175);  mul_86 = unsqueeze_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_65: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(add_64, 3)
    clamp_min_21: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_min.default(add_65, 0);  add_65 = None
    clamp_max_21: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_21, 6);  clamp_min_21 = None
    mul_87: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(add_64, clamp_max_21);  add_64 = clamp_max_21 = None
    div_21: "f32[8, 128, 14, 14]" = torch.ops.aten.div.Tensor(mul_87, 6);  mul_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_22: "f32[8, 128, 14, 14]" = torch.ops.aten.convolution.default(div_21, arg78_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_21 = arg78_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_176: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg137_1, -1);  arg137_1 = None
    unsqueeze_177: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
    sub_22: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_177);  convolution_22 = unsqueeze_177 = None
    add_66: "f32[128]" = torch.ops.aten.add.Tensor(arg138_1, 1e-05);  arg138_1 = None
    sqrt_22: "f32[128]" = torch.ops.aten.sqrt.default(add_66);  add_66 = None
    reciprocal_22: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_22);  sqrt_22 = None
    mul_88: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_22, 1);  reciprocal_22 = None
    unsqueeze_178: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_88, -1);  mul_88 = None
    unsqueeze_179: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
    mul_89: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_22, unsqueeze_179);  sub_22 = unsqueeze_179 = None
    unsqueeze_180: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg44_1, -1);  arg44_1 = None
    unsqueeze_181: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
    mul_90: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_89, unsqueeze_181);  mul_89 = unsqueeze_181 = None
    unsqueeze_182: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg45_1, -1);  arg45_1 = None
    unsqueeze_183: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
    add_67: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_90, unsqueeze_183);  mul_90 = unsqueeze_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_68: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(add_67, 3)
    clamp_min_22: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_min.default(add_68, 0);  add_68 = None
    clamp_max_22: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_22, 6);  clamp_min_22 = None
    mul_91: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(add_67, clamp_max_22);  add_67 = clamp_max_22 = None
    div_22: "f32[8, 128, 14, 14]" = torch.ops.aten.div.Tensor(mul_91, 6);  mul_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_23: "f32[8, 128, 7, 7]" = torch.ops.aten.convolution.default(div_22, arg79_1, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 128);  div_22 = arg79_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_184: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg139_1, -1);  arg139_1 = None
    unsqueeze_185: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
    sub_23: "f32[8, 128, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_185);  convolution_23 = unsqueeze_185 = None
    add_69: "f32[128]" = torch.ops.aten.add.Tensor(arg140_1, 1e-05);  arg140_1 = None
    sqrt_23: "f32[128]" = torch.ops.aten.sqrt.default(add_69);  add_69 = None
    reciprocal_23: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_23);  sqrt_23 = None
    mul_92: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_23, 1);  reciprocal_23 = None
    unsqueeze_186: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_92, -1);  mul_92 = None
    unsqueeze_187: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
    mul_93: "f32[8, 128, 7, 7]" = torch.ops.aten.mul.Tensor(sub_23, unsqueeze_187);  sub_23 = unsqueeze_187 = None
    unsqueeze_188: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg46_1, -1);  arg46_1 = None
    unsqueeze_189: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
    mul_94: "f32[8, 128, 7, 7]" = torch.ops.aten.mul.Tensor(mul_93, unsqueeze_189);  mul_93 = unsqueeze_189 = None
    unsqueeze_190: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg47_1, -1);  arg47_1 = None
    unsqueeze_191: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
    add_70: "f32[8, 128, 7, 7]" = torch.ops.aten.add.Tensor(mul_94, unsqueeze_191);  mul_94 = unsqueeze_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_71: "f32[8, 128, 7, 7]" = torch.ops.aten.add.Tensor(add_70, 3)
    clamp_min_23: "f32[8, 128, 7, 7]" = torch.ops.aten.clamp_min.default(add_71, 0);  add_71 = None
    clamp_max_23: "f32[8, 128, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_23, 6);  clamp_min_23 = None
    mul_95: "f32[8, 128, 7, 7]" = torch.ops.aten.mul.Tensor(add_70, clamp_max_23);  add_70 = clamp_max_23 = None
    div_23: "f32[8, 128, 7, 7]" = torch.ops.aten.div.Tensor(mul_95, 6);  mul_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean: "f32[8, 128, 1, 1]" = torch.ops.aten.mean.dim(div_23, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_24: "f32[8, 32, 1, 1]" = torch.ops.aten.convolution.default(mean, arg80_1, arg81_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean = arg80_1 = arg81_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    relu: "f32[8, 32, 1, 1]" = torch.ops.aten.relu.default(convolution_24);  convolution_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_25: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(relu, arg82_1, arg83_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu = arg82_1 = arg83_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    add_72: "f32[8, 128, 1, 1]" = torch.ops.aten.add.Tensor(convolution_25, 3);  convolution_25 = None
    clamp_min_24: "f32[8, 128, 1, 1]" = torch.ops.aten.clamp_min.default(add_72, 0);  add_72 = None
    clamp_max_24: "f32[8, 128, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_24, 6);  clamp_min_24 = None
    div_24: "f32[8, 128, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_24, 6);  clamp_max_24 = None
    mul_96: "f32[8, 128, 7, 7]" = torch.ops.aten.mul.Tensor(div_23, div_24);  div_23 = div_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_26: "f32[8, 256, 7, 7]" = torch.ops.aten.convolution.default(mul_96, arg84_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_96 = arg84_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_192: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg141_1, -1);  arg141_1 = None
    unsqueeze_193: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
    sub_24: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_193);  convolution_26 = unsqueeze_193 = None
    add_73: "f32[256]" = torch.ops.aten.add.Tensor(arg142_1, 1e-05);  arg142_1 = None
    sqrt_24: "f32[256]" = torch.ops.aten.sqrt.default(add_73);  add_73 = None
    reciprocal_24: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_24);  sqrt_24 = None
    mul_97: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_24, 1);  reciprocal_24 = None
    unsqueeze_194: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_97, -1);  mul_97 = None
    unsqueeze_195: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
    mul_98: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(sub_24, unsqueeze_195);  sub_24 = unsqueeze_195 = None
    unsqueeze_196: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg48_1, -1);  arg48_1 = None
    unsqueeze_197: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
    mul_99: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(mul_98, unsqueeze_197);  mul_98 = unsqueeze_197 = None
    unsqueeze_198: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg49_1, -1);  arg49_1 = None
    unsqueeze_199: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
    add_74: "f32[8, 256, 7, 7]" = torch.ops.aten.add.Tensor(mul_99, unsqueeze_199);  mul_99 = unsqueeze_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_75: "f32[8, 256, 7, 7]" = torch.ops.aten.add.Tensor(add_74, 3)
    clamp_min_25: "f32[8, 256, 7, 7]" = torch.ops.aten.clamp_min.default(add_75, 0);  add_75 = None
    clamp_max_25: "f32[8, 256, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_25, 6);  clamp_min_25 = None
    mul_100: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(add_74, clamp_max_25);  add_74 = clamp_max_25 = None
    div_25: "f32[8, 256, 7, 7]" = torch.ops.aten.div.Tensor(mul_100, 6);  mul_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_27: "f32[8, 256, 7, 7]" = torch.ops.aten.convolution.default(div_25, arg85_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 256);  div_25 = arg85_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_200: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg143_1, -1);  arg143_1 = None
    unsqueeze_201: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
    sub_25: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_201);  convolution_27 = unsqueeze_201 = None
    add_76: "f32[256]" = torch.ops.aten.add.Tensor(arg144_1, 1e-05);  arg144_1 = None
    sqrt_25: "f32[256]" = torch.ops.aten.sqrt.default(add_76);  add_76 = None
    reciprocal_25: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_25);  sqrt_25 = None
    mul_101: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_25, 1);  reciprocal_25 = None
    unsqueeze_202: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_101, -1);  mul_101 = None
    unsqueeze_203: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
    mul_102: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(sub_25, unsqueeze_203);  sub_25 = unsqueeze_203 = None
    unsqueeze_204: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg50_1, -1);  arg50_1 = None
    unsqueeze_205: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
    mul_103: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(mul_102, unsqueeze_205);  mul_102 = unsqueeze_205 = None
    unsqueeze_206: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg51_1, -1);  arg51_1 = None
    unsqueeze_207: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
    add_77: "f32[8, 256, 7, 7]" = torch.ops.aten.add.Tensor(mul_103, unsqueeze_207);  mul_103 = unsqueeze_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_78: "f32[8, 256, 7, 7]" = torch.ops.aten.add.Tensor(add_77, 3)
    clamp_min_26: "f32[8, 256, 7, 7]" = torch.ops.aten.clamp_min.default(add_78, 0);  add_78 = None
    clamp_max_26: "f32[8, 256, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_26, 6);  clamp_min_26 = None
    mul_104: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(add_77, clamp_max_26);  add_77 = clamp_max_26 = None
    div_26: "f32[8, 256, 7, 7]" = torch.ops.aten.div.Tensor(mul_104, 6);  mul_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_1: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(div_26, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_28: "f32[8, 64, 1, 1]" = torch.ops.aten.convolution.default(mean_1, arg86_1, arg87_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_1 = arg86_1 = arg87_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    relu_1: "f32[8, 64, 1, 1]" = torch.ops.aten.relu.default(convolution_28);  convolution_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_29: "f32[8, 256, 1, 1]" = torch.ops.aten.convolution.default(relu_1, arg88_1, arg89_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_1 = arg88_1 = arg89_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    add_79: "f32[8, 256, 1, 1]" = torch.ops.aten.add.Tensor(convolution_29, 3);  convolution_29 = None
    clamp_min_27: "f32[8, 256, 1, 1]" = torch.ops.aten.clamp_min.default(add_79, 0);  add_79 = None
    clamp_max_27: "f32[8, 256, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_27, 6);  clamp_min_27 = None
    div_27: "f32[8, 256, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_27, 6);  clamp_max_27 = None
    mul_105: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(div_26, div_27);  div_26 = div_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_30: "f32[8, 256, 7, 7]" = torch.ops.aten.convolution.default(mul_105, arg90_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_105 = arg90_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_208: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg145_1, -1);  arg145_1 = None
    unsqueeze_209: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
    sub_26: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_209);  convolution_30 = unsqueeze_209 = None
    add_80: "f32[256]" = torch.ops.aten.add.Tensor(arg146_1, 1e-05);  arg146_1 = None
    sqrt_26: "f32[256]" = torch.ops.aten.sqrt.default(add_80);  add_80 = None
    reciprocal_26: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_26);  sqrt_26 = None
    mul_106: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_26, 1);  reciprocal_26 = None
    unsqueeze_210: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_106, -1);  mul_106 = None
    unsqueeze_211: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
    mul_107: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(sub_26, unsqueeze_211);  sub_26 = unsqueeze_211 = None
    unsqueeze_212: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg52_1, -1);  arg52_1 = None
    unsqueeze_213: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
    mul_108: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(mul_107, unsqueeze_213);  mul_107 = unsqueeze_213 = None
    unsqueeze_214: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg53_1, -1);  arg53_1 = None
    unsqueeze_215: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
    add_81: "f32[8, 256, 7, 7]" = torch.ops.aten.add.Tensor(mul_108, unsqueeze_215);  mul_108 = unsqueeze_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_82: "f32[8, 256, 7, 7]" = torch.ops.aten.add.Tensor(add_81, 3)
    clamp_min_28: "f32[8, 256, 7, 7]" = torch.ops.aten.clamp_min.default(add_82, 0);  add_82 = None
    clamp_max_28: "f32[8, 256, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_28, 6);  clamp_min_28 = None
    mul_109: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(add_81, clamp_max_28);  add_81 = clamp_max_28 = None
    div_28: "f32[8, 256, 7, 7]" = torch.ops.aten.div.Tensor(mul_109, 6);  mul_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean_2: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(div_28, [-1, -2], True);  div_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilenetv3.py:145, code: x = self.conv_head(x)
    convolution_31: "f32[8, 1280, 1, 1]" = torch.ops.aten.convolution.default(mean_2, arg91_1, arg92_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_2 = arg91_1 = arg92_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilenetv3.py:146, code: x = self.act2(x)
    add_83: "f32[8, 1280, 1, 1]" = torch.ops.aten.add.Tensor(convolution_31, 3)
    clamp_min_29: "f32[8, 1280, 1, 1]" = torch.ops.aten.clamp_min.default(add_83, 0);  add_83 = None
    clamp_max_29: "f32[8, 1280, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_29, 6);  clamp_min_29 = None
    mul_110: "f32[8, 1280, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_31, clamp_max_29);  convolution_31 = clamp_max_29 = None
    div_29: "f32[8, 1280, 1, 1]" = torch.ops.aten.div.Tensor(mul_110, 6);  mul_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/linear.py:19, code: return F.linear(input, self.weight, self.bias)
    view_1: "f32[8, 1280]" = torch.ops.aten.reshape.default(div_29, [8, 1280]);  div_29 = None
    permute: "f32[1280, 1000]" = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
    addmm: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg55_1, view_1, permute);  arg55_1 = view_1 = permute = None
    return (addmm,)
    