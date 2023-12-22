from __future__ import annotations



def forward(self, arg0_1: "f32[64]", arg1_1: "f32[64]", arg2_1: "f32[64]", arg3_1: "f32[64]", arg4_1: "f32[64]", arg5_1: "f32[64]", arg6_1: "f32[128]", arg7_1: "f32[128]", arg8_1: "f32[128]", arg9_1: "f32[128]", arg10_1: "f32[128]", arg11_1: "f32[128]", arg12_1: "f32[128]", arg13_1: "f32[128]", arg14_1: "f32[256]", arg15_1: "f32[256]", arg16_1: "f32[160]", arg17_1: "f32[160]", arg18_1: "f32[160]", arg19_1: "f32[160]", arg20_1: "f32[160]", arg21_1: "f32[160]", arg22_1: "f32[160]", arg23_1: "f32[160]", arg24_1: "f32[512]", arg25_1: "f32[512]", arg26_1: "f32[192]", arg27_1: "f32[192]", arg28_1: "f32[192]", arg29_1: "f32[192]", arg30_1: "f32[192]", arg31_1: "f32[192]", arg32_1: "f32[192]", arg33_1: "f32[192]", arg34_1: "f32[768]", arg35_1: "f32[768]", arg36_1: "f32[224]", arg37_1: "f32[224]", arg38_1: "f32[224]", arg39_1: "f32[224]", arg40_1: "f32[224]", arg41_1: "f32[224]", arg42_1: "f32[224]", arg43_1: "f32[224]", arg44_1: "f32[1024]", arg45_1: "f32[1024]", arg46_1: "f32[64, 3, 3, 3]", arg47_1: "f32[64, 1, 3, 3]", arg48_1: "f32[64, 64, 1, 1]", arg49_1: "f32[64, 1, 3, 3]", arg50_1: "f32[64, 64, 1, 1]", arg51_1: "f32[128, 64, 1, 1]", arg52_1: "f32[128, 1, 3, 3]", arg53_1: "f32[128, 128, 1, 1]", arg54_1: "f32[128, 1, 3, 3]", arg55_1: "f32[128, 128, 1, 1]", arg56_1: "f32[128, 1, 3, 3]", arg57_1: "f32[128, 128, 1, 1]", arg58_1: "f32[256, 448, 1, 1]", arg59_1: "f32[256, 256, 1, 1]", arg60_1: "f32[256]", arg61_1: "f32[160, 256, 1, 1]", arg62_1: "f32[160, 1, 3, 3]", arg63_1: "f32[160, 160, 1, 1]", arg64_1: "f32[160, 1, 3, 3]", arg65_1: "f32[160, 160, 1, 1]", arg66_1: "f32[160, 1, 3, 3]", arg67_1: "f32[160, 160, 1, 1]", arg68_1: "f32[512, 736, 1, 1]", arg69_1: "f32[512, 512, 1, 1]", arg70_1: "f32[512]", arg71_1: "f32[192, 512, 1, 1]", arg72_1: "f32[192, 1, 3, 3]", arg73_1: "f32[192, 192, 1, 1]", arg74_1: "f32[192, 1, 3, 3]", arg75_1: "f32[192, 192, 1, 1]", arg76_1: "f32[192, 1, 3, 3]", arg77_1: "f32[192, 192, 1, 1]", arg78_1: "f32[768, 1088, 1, 1]", arg79_1: "f32[768, 768, 1, 1]", arg80_1: "f32[768]", arg81_1: "f32[224, 768, 1, 1]", arg82_1: "f32[224, 1, 3, 3]", arg83_1: "f32[224, 224, 1, 1]", arg84_1: "f32[224, 1, 3, 3]", arg85_1: "f32[224, 224, 1, 1]", arg86_1: "f32[224, 1, 3, 3]", arg87_1: "f32[224, 224, 1, 1]", arg88_1: "f32[1024, 1440, 1, 1]", arg89_1: "f32[1024, 1024, 1, 1]", arg90_1: "f32[1024]", arg91_1: "f32[1000, 1024]", arg92_1: "f32[1000]", arg93_1: "f32[64]", arg94_1: "f32[64]", arg95_1: "f32[64]", arg96_1: "f32[64]", arg97_1: "f32[64]", arg98_1: "f32[64]", arg99_1: "f32[128]", arg100_1: "f32[128]", arg101_1: "f32[128]", arg102_1: "f32[128]", arg103_1: "f32[128]", arg104_1: "f32[128]", arg105_1: "f32[128]", arg106_1: "f32[128]", arg107_1: "f32[256]", arg108_1: "f32[256]", arg109_1: "f32[160]", arg110_1: "f32[160]", arg111_1: "f32[160]", arg112_1: "f32[160]", arg113_1: "f32[160]", arg114_1: "f32[160]", arg115_1: "f32[160]", arg116_1: "f32[160]", arg117_1: "f32[512]", arg118_1: "f32[512]", arg119_1: "f32[192]", arg120_1: "f32[192]", arg121_1: "f32[192]", arg122_1: "f32[192]", arg123_1: "f32[192]", arg124_1: "f32[192]", arg125_1: "f32[192]", arg126_1: "f32[192]", arg127_1: "f32[768]", arg128_1: "f32[768]", arg129_1: "f32[224]", arg130_1: "f32[224]", arg131_1: "f32[224]", arg132_1: "f32[224]", arg133_1: "f32[224]", arg134_1: "f32[224]", arg135_1: "f32[224]", arg136_1: "f32[224]", arg137_1: "f32[1024]", arg138_1: "f32[1024]", arg139_1: "f32[8, 3, 288, 288]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution: "f32[8, 64, 144, 144]" = torch.ops.aten.convolution.default(arg139_1, arg46_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg139_1 = arg46_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg93_1, -1);  arg93_1 = None
    unsqueeze_1: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    sub: "f32[8, 64, 144, 144]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1);  convolution = unsqueeze_1 = None
    add: "f32[64]" = torch.ops.aten.add.Tensor(arg94_1, 1e-05);  arg94_1 = None
    sqrt: "f32[64]" = torch.ops.aten.sqrt.default(add);  add = None
    reciprocal: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt);  sqrt = None
    mul: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
    unsqueeze_2: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul, -1);  mul = None
    unsqueeze_3: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    mul_1: "f32[8, 64, 144, 144]" = torch.ops.aten.mul.Tensor(sub, unsqueeze_3);  sub = unsqueeze_3 = None
    unsqueeze_4: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg0_1, -1);  arg0_1 = None
    unsqueeze_5: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_2: "f32[8, 64, 144, 144]" = torch.ops.aten.mul.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None
    unsqueeze_6: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg1_1, -1);  arg1_1 = None
    unsqueeze_7: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_1: "f32[8, 64, 144, 144]" = torch.ops.aten.add.Tensor(mul_2, unsqueeze_7);  mul_2 = unsqueeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu: "f32[8, 64, 144, 144]" = torch.ops.aten.relu.default(add_1);  add_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    convolution_1: "f32[8, 64, 144, 144]" = torch.ops.aten.convolution.default(relu, arg47_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 64);  relu = arg47_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    convolution_2: "f32[8, 64, 144, 144]" = torch.ops.aten.convolution.default(convolution_1, arg48_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_1 = arg48_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_8: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg95_1, -1);  arg95_1 = None
    unsqueeze_9: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    sub_1: "f32[8, 64, 144, 144]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_9);  convolution_2 = unsqueeze_9 = None
    add_2: "f32[64]" = torch.ops.aten.add.Tensor(arg96_1, 1e-05);  arg96_1 = None
    sqrt_1: "f32[64]" = torch.ops.aten.sqrt.default(add_2);  add_2 = None
    reciprocal_1: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_1);  sqrt_1 = None
    mul_3: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_1, 1);  reciprocal_1 = None
    unsqueeze_10: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_3, -1);  mul_3 = None
    unsqueeze_11: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    mul_4: "f32[8, 64, 144, 144]" = torch.ops.aten.mul.Tensor(sub_1, unsqueeze_11);  sub_1 = unsqueeze_11 = None
    unsqueeze_12: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
    unsqueeze_13: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_5: "f32[8, 64, 144, 144]" = torch.ops.aten.mul.Tensor(mul_4, unsqueeze_13);  mul_4 = unsqueeze_13 = None
    unsqueeze_14: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, -1);  arg3_1 = None
    unsqueeze_15: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_3: "f32[8, 64, 144, 144]" = torch.ops.aten.add.Tensor(mul_5, unsqueeze_15);  mul_5 = unsqueeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_1: "f32[8, 64, 144, 144]" = torch.ops.aten.relu.default(add_3);  add_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    convolution_3: "f32[8, 64, 72, 72]" = torch.ops.aten.convolution.default(relu_1, arg49_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 64);  relu_1 = arg49_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    convolution_4: "f32[8, 64, 72, 72]" = torch.ops.aten.convolution.default(convolution_3, arg50_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_3 = arg50_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_16: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg97_1, -1);  arg97_1 = None
    unsqueeze_17: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    sub_2: "f32[8, 64, 72, 72]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_17);  convolution_4 = unsqueeze_17 = None
    add_4: "f32[64]" = torch.ops.aten.add.Tensor(arg98_1, 1e-05);  arg98_1 = None
    sqrt_2: "f32[64]" = torch.ops.aten.sqrt.default(add_4);  add_4 = None
    reciprocal_2: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_2);  sqrt_2 = None
    mul_6: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_2, 1);  reciprocal_2 = None
    unsqueeze_18: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_6, -1);  mul_6 = None
    unsqueeze_19: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    mul_7: "f32[8, 64, 72, 72]" = torch.ops.aten.mul.Tensor(sub_2, unsqueeze_19);  sub_2 = unsqueeze_19 = None
    unsqueeze_20: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
    unsqueeze_21: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_8: "f32[8, 64, 72, 72]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_21);  mul_7 = unsqueeze_21 = None
    unsqueeze_22: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
    unsqueeze_23: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_5: "f32[8, 64, 72, 72]" = torch.ops.aten.add.Tensor(mul_8, unsqueeze_23);  mul_8 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_2: "f32[8, 64, 72, 72]" = torch.ops.aten.relu.default(add_5);  add_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_5: "f32[8, 128, 72, 72]" = torch.ops.aten.convolution.default(relu_2, arg51_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg51_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_24: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg99_1, -1);  arg99_1 = None
    unsqueeze_25: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
    sub_3: "f32[8, 128, 72, 72]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_25);  convolution_5 = unsqueeze_25 = None
    add_6: "f32[128]" = torch.ops.aten.add.Tensor(arg100_1, 1e-05);  arg100_1 = None
    sqrt_3: "f32[128]" = torch.ops.aten.sqrt.default(add_6);  add_6 = None
    reciprocal_3: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_3);  sqrt_3 = None
    mul_9: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_3, 1);  reciprocal_3 = None
    unsqueeze_26: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_9, -1);  mul_9 = None
    unsqueeze_27: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
    mul_10: "f32[8, 128, 72, 72]" = torch.ops.aten.mul.Tensor(sub_3, unsqueeze_27);  sub_3 = unsqueeze_27 = None
    unsqueeze_28: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg6_1, -1);  arg6_1 = None
    unsqueeze_29: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    mul_11: "f32[8, 128, 72, 72]" = torch.ops.aten.mul.Tensor(mul_10, unsqueeze_29);  mul_10 = unsqueeze_29 = None
    unsqueeze_30: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
    unsqueeze_31: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
    add_7: "f32[8, 128, 72, 72]" = torch.ops.aten.add.Tensor(mul_11, unsqueeze_31);  mul_11 = unsqueeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_3: "f32[8, 128, 72, 72]" = torch.ops.aten.relu.default(add_7);  add_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    convolution_6: "f32[8, 128, 72, 72]" = torch.ops.aten.convolution.default(relu_3, arg52_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 128);  relu_3 = arg52_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    convolution_7: "f32[8, 128, 72, 72]" = torch.ops.aten.convolution.default(convolution_6, arg53_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_6 = arg53_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_32: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg101_1, -1);  arg101_1 = None
    unsqueeze_33: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    sub_4: "f32[8, 128, 72, 72]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_33);  convolution_7 = unsqueeze_33 = None
    add_8: "f32[128]" = torch.ops.aten.add.Tensor(arg102_1, 1e-05);  arg102_1 = None
    sqrt_4: "f32[128]" = torch.ops.aten.sqrt.default(add_8);  add_8 = None
    reciprocal_4: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_4);  sqrt_4 = None
    mul_12: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_4, 1);  reciprocal_4 = None
    unsqueeze_34: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_12, -1);  mul_12 = None
    unsqueeze_35: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
    mul_13: "f32[8, 128, 72, 72]" = torch.ops.aten.mul.Tensor(sub_4, unsqueeze_35);  sub_4 = unsqueeze_35 = None
    unsqueeze_36: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg8_1, -1);  arg8_1 = None
    unsqueeze_37: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
    mul_14: "f32[8, 128, 72, 72]" = torch.ops.aten.mul.Tensor(mul_13, unsqueeze_37);  mul_13 = unsqueeze_37 = None
    unsqueeze_38: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
    unsqueeze_39: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
    add_9: "f32[8, 128, 72, 72]" = torch.ops.aten.add.Tensor(mul_14, unsqueeze_39);  mul_14 = unsqueeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_4: "f32[8, 128, 72, 72]" = torch.ops.aten.relu.default(add_9);  add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    convolution_8: "f32[8, 128, 72, 72]" = torch.ops.aten.convolution.default(relu_4, arg54_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 128);  arg54_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    convolution_9: "f32[8, 128, 72, 72]" = torch.ops.aten.convolution.default(convolution_8, arg55_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_8 = arg55_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_40: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg103_1, -1);  arg103_1 = None
    unsqueeze_41: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    sub_5: "f32[8, 128, 72, 72]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_41);  convolution_9 = unsqueeze_41 = None
    add_10: "f32[128]" = torch.ops.aten.add.Tensor(arg104_1, 1e-05);  arg104_1 = None
    sqrt_5: "f32[128]" = torch.ops.aten.sqrt.default(add_10);  add_10 = None
    reciprocal_5: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_5);  sqrt_5 = None
    mul_15: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_5, 1);  reciprocal_5 = None
    unsqueeze_42: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_15, -1);  mul_15 = None
    unsqueeze_43: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    mul_16: "f32[8, 128, 72, 72]" = torch.ops.aten.mul.Tensor(sub_5, unsqueeze_43);  sub_5 = unsqueeze_43 = None
    unsqueeze_44: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
    unsqueeze_45: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    mul_17: "f32[8, 128, 72, 72]" = torch.ops.aten.mul.Tensor(mul_16, unsqueeze_45);  mul_16 = unsqueeze_45 = None
    unsqueeze_46: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg11_1, -1);  arg11_1 = None
    unsqueeze_47: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
    add_11: "f32[8, 128, 72, 72]" = torch.ops.aten.add.Tensor(mul_17, unsqueeze_47);  mul_17 = unsqueeze_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_5: "f32[8, 128, 72, 72]" = torch.ops.aten.relu.default(add_11);  add_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    convolution_10: "f32[8, 128, 72, 72]" = torch.ops.aten.convolution.default(relu_5, arg56_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 128);  arg56_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    convolution_11: "f32[8, 128, 72, 72]" = torch.ops.aten.convolution.default(convolution_10, arg57_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_10 = arg57_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_48: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg105_1, -1);  arg105_1 = None
    unsqueeze_49: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
    sub_6: "f32[8, 128, 72, 72]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_49);  convolution_11 = unsqueeze_49 = None
    add_12: "f32[128]" = torch.ops.aten.add.Tensor(arg106_1, 1e-05);  arg106_1 = None
    sqrt_6: "f32[128]" = torch.ops.aten.sqrt.default(add_12);  add_12 = None
    reciprocal_6: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_6);  sqrt_6 = None
    mul_18: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_6, 1);  reciprocal_6 = None
    unsqueeze_50: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_18, -1);  mul_18 = None
    unsqueeze_51: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
    mul_19: "f32[8, 128, 72, 72]" = torch.ops.aten.mul.Tensor(sub_6, unsqueeze_51);  sub_6 = unsqueeze_51 = None
    unsqueeze_52: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg12_1, -1);  arg12_1 = None
    unsqueeze_53: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    mul_20: "f32[8, 128, 72, 72]" = torch.ops.aten.mul.Tensor(mul_19, unsqueeze_53);  mul_19 = unsqueeze_53 = None
    unsqueeze_54: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg13_1, -1);  arg13_1 = None
    unsqueeze_55: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
    add_13: "f32[8, 128, 72, 72]" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_55);  mul_20 = unsqueeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_6: "f32[8, 128, 72, 72]" = torch.ops.aten.relu.default(add_13);  add_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vovnet.py:39, code: x = torch.cat(concat_list, dim=1)
    cat: "f32[8, 448, 72, 72]" = torch.ops.aten.cat.default([relu_2, relu_4, relu_5, relu_6], 1);  relu_2 = relu_4 = relu_5 = relu_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_12: "f32[8, 256, 72, 72]" = torch.ops.aten.convolution.default(cat, arg58_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat = arg58_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_56: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg107_1, -1);  arg107_1 = None
    unsqueeze_57: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    sub_7: "f32[8, 256, 72, 72]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_57);  convolution_12 = unsqueeze_57 = None
    add_14: "f32[256]" = torch.ops.aten.add.Tensor(arg108_1, 1e-05);  arg108_1 = None
    sqrt_7: "f32[256]" = torch.ops.aten.sqrt.default(add_14);  add_14 = None
    reciprocal_7: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_7);  sqrt_7 = None
    mul_21: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_7, 1);  reciprocal_7 = None
    unsqueeze_58: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_21, -1);  mul_21 = None
    unsqueeze_59: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
    mul_22: "f32[8, 256, 72, 72]" = torch.ops.aten.mul.Tensor(sub_7, unsqueeze_59);  sub_7 = unsqueeze_59 = None
    unsqueeze_60: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
    unsqueeze_61: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
    mul_23: "f32[8, 256, 72, 72]" = torch.ops.aten.mul.Tensor(mul_22, unsqueeze_61);  mul_22 = unsqueeze_61 = None
    unsqueeze_62: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
    unsqueeze_63: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
    add_15: "f32[8, 256, 72, 72]" = torch.ops.aten.add.Tensor(mul_23, unsqueeze_63);  mul_23 = unsqueeze_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_7: "f32[8, 256, 72, 72]" = torch.ops.aten.relu.default(add_15);  add_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:66, code: x_se = x.mean((2, 3), keepdim=True)
    mean: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(relu_7, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:70, code: x_se = self.fc(x_se)
    convolution_13: "f32[8, 256, 1, 1]" = torch.ops.aten.convolution.default(mean, arg59_1, arg60_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean = arg59_1 = arg60_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:71, code: return x * self.gate(x_se)
    add_16: "f32[8, 256, 1, 1]" = torch.ops.aten.add.Tensor(convolution_13, 3);  convolution_13 = None
    clamp_min: "f32[8, 256, 1, 1]" = torch.ops.aten.clamp_min.default(add_16, 0);  add_16 = None
    clamp_max: "f32[8, 256, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min, 6);  clamp_min = None
    div: "f32[8, 256, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max, 6);  clamp_max = None
    mul_24: "f32[8, 256, 72, 72]" = torch.ops.aten.mul.Tensor(relu_7, div);  relu_7 = div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vovnet.py:145, code: x = self.pool(x)
    max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices.default(mul_24, [3, 3], [2, 2], [0, 0], [1, 1], True);  mul_24 = None
    getitem: "f32[8, 256, 36, 36]" = max_pool2d_with_indices[0];  max_pool2d_with_indices = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_14: "f32[8, 160, 36, 36]" = torch.ops.aten.convolution.default(getitem, arg61_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg61_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_64: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg109_1, -1);  arg109_1 = None
    unsqueeze_65: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
    sub_8: "f32[8, 160, 36, 36]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_65);  convolution_14 = unsqueeze_65 = None
    add_17: "f32[160]" = torch.ops.aten.add.Tensor(arg110_1, 1e-05);  arg110_1 = None
    sqrt_8: "f32[160]" = torch.ops.aten.sqrt.default(add_17);  add_17 = None
    reciprocal_8: "f32[160]" = torch.ops.aten.reciprocal.default(sqrt_8);  sqrt_8 = None
    mul_25: "f32[160]" = torch.ops.aten.mul.Tensor(reciprocal_8, 1);  reciprocal_8 = None
    unsqueeze_66: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(mul_25, -1);  mul_25 = None
    unsqueeze_67: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
    mul_26: "f32[8, 160, 36, 36]" = torch.ops.aten.mul.Tensor(sub_8, unsqueeze_67);  sub_8 = unsqueeze_67 = None
    unsqueeze_68: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg16_1, -1);  arg16_1 = None
    unsqueeze_69: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    mul_27: "f32[8, 160, 36, 36]" = torch.ops.aten.mul.Tensor(mul_26, unsqueeze_69);  mul_26 = unsqueeze_69 = None
    unsqueeze_70: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg17_1, -1);  arg17_1 = None
    unsqueeze_71: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    add_18: "f32[8, 160, 36, 36]" = torch.ops.aten.add.Tensor(mul_27, unsqueeze_71);  mul_27 = unsqueeze_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_8: "f32[8, 160, 36, 36]" = torch.ops.aten.relu.default(add_18);  add_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    convolution_15: "f32[8, 160, 36, 36]" = torch.ops.aten.convolution.default(relu_8, arg62_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 160);  relu_8 = arg62_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    convolution_16: "f32[8, 160, 36, 36]" = torch.ops.aten.convolution.default(convolution_15, arg63_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_15 = arg63_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_72: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg111_1, -1);  arg111_1 = None
    unsqueeze_73: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
    sub_9: "f32[8, 160, 36, 36]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_73);  convolution_16 = unsqueeze_73 = None
    add_19: "f32[160]" = torch.ops.aten.add.Tensor(arg112_1, 1e-05);  arg112_1 = None
    sqrt_9: "f32[160]" = torch.ops.aten.sqrt.default(add_19);  add_19 = None
    reciprocal_9: "f32[160]" = torch.ops.aten.reciprocal.default(sqrt_9);  sqrt_9 = None
    mul_28: "f32[160]" = torch.ops.aten.mul.Tensor(reciprocal_9, 1);  reciprocal_9 = None
    unsqueeze_74: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(mul_28, -1);  mul_28 = None
    unsqueeze_75: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
    mul_29: "f32[8, 160, 36, 36]" = torch.ops.aten.mul.Tensor(sub_9, unsqueeze_75);  sub_9 = unsqueeze_75 = None
    unsqueeze_76: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg18_1, -1);  arg18_1 = None
    unsqueeze_77: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
    mul_30: "f32[8, 160, 36, 36]" = torch.ops.aten.mul.Tensor(mul_29, unsqueeze_77);  mul_29 = unsqueeze_77 = None
    unsqueeze_78: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg19_1, -1);  arg19_1 = None
    unsqueeze_79: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    add_20: "f32[8, 160, 36, 36]" = torch.ops.aten.add.Tensor(mul_30, unsqueeze_79);  mul_30 = unsqueeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_9: "f32[8, 160, 36, 36]" = torch.ops.aten.relu.default(add_20);  add_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    convolution_17: "f32[8, 160, 36, 36]" = torch.ops.aten.convolution.default(relu_9, arg64_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 160);  arg64_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    convolution_18: "f32[8, 160, 36, 36]" = torch.ops.aten.convolution.default(convolution_17, arg65_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_17 = arg65_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_80: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg113_1, -1);  arg113_1 = None
    unsqueeze_81: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
    sub_10: "f32[8, 160, 36, 36]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_81);  convolution_18 = unsqueeze_81 = None
    add_21: "f32[160]" = torch.ops.aten.add.Tensor(arg114_1, 1e-05);  arg114_1 = None
    sqrt_10: "f32[160]" = torch.ops.aten.sqrt.default(add_21);  add_21 = None
    reciprocal_10: "f32[160]" = torch.ops.aten.reciprocal.default(sqrt_10);  sqrt_10 = None
    mul_31: "f32[160]" = torch.ops.aten.mul.Tensor(reciprocal_10, 1);  reciprocal_10 = None
    unsqueeze_82: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(mul_31, -1);  mul_31 = None
    unsqueeze_83: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    mul_32: "f32[8, 160, 36, 36]" = torch.ops.aten.mul.Tensor(sub_10, unsqueeze_83);  sub_10 = unsqueeze_83 = None
    unsqueeze_84: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg20_1, -1);  arg20_1 = None
    unsqueeze_85: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    mul_33: "f32[8, 160, 36, 36]" = torch.ops.aten.mul.Tensor(mul_32, unsqueeze_85);  mul_32 = unsqueeze_85 = None
    unsqueeze_86: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg21_1, -1);  arg21_1 = None
    unsqueeze_87: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
    add_22: "f32[8, 160, 36, 36]" = torch.ops.aten.add.Tensor(mul_33, unsqueeze_87);  mul_33 = unsqueeze_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_10: "f32[8, 160, 36, 36]" = torch.ops.aten.relu.default(add_22);  add_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    convolution_19: "f32[8, 160, 36, 36]" = torch.ops.aten.convolution.default(relu_10, arg66_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 160);  arg66_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    convolution_20: "f32[8, 160, 36, 36]" = torch.ops.aten.convolution.default(convolution_19, arg67_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_19 = arg67_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_88: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg115_1, -1);  arg115_1 = None
    unsqueeze_89: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
    sub_11: "f32[8, 160, 36, 36]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_89);  convolution_20 = unsqueeze_89 = None
    add_23: "f32[160]" = torch.ops.aten.add.Tensor(arg116_1, 1e-05);  arg116_1 = None
    sqrt_11: "f32[160]" = torch.ops.aten.sqrt.default(add_23);  add_23 = None
    reciprocal_11: "f32[160]" = torch.ops.aten.reciprocal.default(sqrt_11);  sqrt_11 = None
    mul_34: "f32[160]" = torch.ops.aten.mul.Tensor(reciprocal_11, 1);  reciprocal_11 = None
    unsqueeze_90: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(mul_34, -1);  mul_34 = None
    unsqueeze_91: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    mul_35: "f32[8, 160, 36, 36]" = torch.ops.aten.mul.Tensor(sub_11, unsqueeze_91);  sub_11 = unsqueeze_91 = None
    unsqueeze_92: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg22_1, -1);  arg22_1 = None
    unsqueeze_93: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
    mul_36: "f32[8, 160, 36, 36]" = torch.ops.aten.mul.Tensor(mul_35, unsqueeze_93);  mul_35 = unsqueeze_93 = None
    unsqueeze_94: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg23_1, -1);  arg23_1 = None
    unsqueeze_95: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
    add_24: "f32[8, 160, 36, 36]" = torch.ops.aten.add.Tensor(mul_36, unsqueeze_95);  mul_36 = unsqueeze_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_11: "f32[8, 160, 36, 36]" = torch.ops.aten.relu.default(add_24);  add_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vovnet.py:39, code: x = torch.cat(concat_list, dim=1)
    cat_1: "f32[8, 736, 36, 36]" = torch.ops.aten.cat.default([getitem, relu_9, relu_10, relu_11], 1);  getitem = relu_9 = relu_10 = relu_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_21: "f32[8, 512, 36, 36]" = torch.ops.aten.convolution.default(cat_1, arg68_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_1 = arg68_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_96: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg117_1, -1);  arg117_1 = None
    unsqueeze_97: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
    sub_12: "f32[8, 512, 36, 36]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_97);  convolution_21 = unsqueeze_97 = None
    add_25: "f32[512]" = torch.ops.aten.add.Tensor(arg118_1, 1e-05);  arg118_1 = None
    sqrt_12: "f32[512]" = torch.ops.aten.sqrt.default(add_25);  add_25 = None
    reciprocal_12: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_12);  sqrt_12 = None
    mul_37: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_12, 1);  reciprocal_12 = None
    unsqueeze_98: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_37, -1);  mul_37 = None
    unsqueeze_99: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
    mul_38: "f32[8, 512, 36, 36]" = torch.ops.aten.mul.Tensor(sub_12, unsqueeze_99);  sub_12 = unsqueeze_99 = None
    unsqueeze_100: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg24_1, -1);  arg24_1 = None
    unsqueeze_101: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
    mul_39: "f32[8, 512, 36, 36]" = torch.ops.aten.mul.Tensor(mul_38, unsqueeze_101);  mul_38 = unsqueeze_101 = None
    unsqueeze_102: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg25_1, -1);  arg25_1 = None
    unsqueeze_103: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    add_26: "f32[8, 512, 36, 36]" = torch.ops.aten.add.Tensor(mul_39, unsqueeze_103);  mul_39 = unsqueeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_12: "f32[8, 512, 36, 36]" = torch.ops.aten.relu.default(add_26);  add_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:66, code: x_se = x.mean((2, 3), keepdim=True)
    mean_1: "f32[8, 512, 1, 1]" = torch.ops.aten.mean.dim(relu_12, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:70, code: x_se = self.fc(x_se)
    convolution_22: "f32[8, 512, 1, 1]" = torch.ops.aten.convolution.default(mean_1, arg69_1, arg70_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_1 = arg69_1 = arg70_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:71, code: return x * self.gate(x_se)
    add_27: "f32[8, 512, 1, 1]" = torch.ops.aten.add.Tensor(convolution_22, 3);  convolution_22 = None
    clamp_min_1: "f32[8, 512, 1, 1]" = torch.ops.aten.clamp_min.default(add_27, 0);  add_27 = None
    clamp_max_1: "f32[8, 512, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_1, 6);  clamp_min_1 = None
    div_1: "f32[8, 512, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_1, 6);  clamp_max_1 = None
    mul_40: "f32[8, 512, 36, 36]" = torch.ops.aten.mul.Tensor(relu_12, div_1);  relu_12 = div_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vovnet.py:145, code: x = self.pool(x)
    max_pool2d_with_indices_1 = torch.ops.aten.max_pool2d_with_indices.default(mul_40, [3, 3], [2, 2], [0, 0], [1, 1], True);  mul_40 = None
    getitem_2: "f32[8, 512, 18, 18]" = max_pool2d_with_indices_1[0];  max_pool2d_with_indices_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_23: "f32[8, 192, 18, 18]" = torch.ops.aten.convolution.default(getitem_2, arg71_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg71_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_104: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg119_1, -1);  arg119_1 = None
    unsqueeze_105: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
    sub_13: "f32[8, 192, 18, 18]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_105);  convolution_23 = unsqueeze_105 = None
    add_28: "f32[192]" = torch.ops.aten.add.Tensor(arg120_1, 1e-05);  arg120_1 = None
    sqrt_13: "f32[192]" = torch.ops.aten.sqrt.default(add_28);  add_28 = None
    reciprocal_13: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_13);  sqrt_13 = None
    mul_41: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_13, 1);  reciprocal_13 = None
    unsqueeze_106: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_41, -1);  mul_41 = None
    unsqueeze_107: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
    mul_42: "f32[8, 192, 18, 18]" = torch.ops.aten.mul.Tensor(sub_13, unsqueeze_107);  sub_13 = unsqueeze_107 = None
    unsqueeze_108: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg26_1, -1);  arg26_1 = None
    unsqueeze_109: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
    mul_43: "f32[8, 192, 18, 18]" = torch.ops.aten.mul.Tensor(mul_42, unsqueeze_109);  mul_42 = unsqueeze_109 = None
    unsqueeze_110: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg27_1, -1);  arg27_1 = None
    unsqueeze_111: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
    add_29: "f32[8, 192, 18, 18]" = torch.ops.aten.add.Tensor(mul_43, unsqueeze_111);  mul_43 = unsqueeze_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_13: "f32[8, 192, 18, 18]" = torch.ops.aten.relu.default(add_29);  add_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    convolution_24: "f32[8, 192, 18, 18]" = torch.ops.aten.convolution.default(relu_13, arg72_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 192);  relu_13 = arg72_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    convolution_25: "f32[8, 192, 18, 18]" = torch.ops.aten.convolution.default(convolution_24, arg73_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_24 = arg73_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_112: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg121_1, -1);  arg121_1 = None
    unsqueeze_113: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
    sub_14: "f32[8, 192, 18, 18]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_113);  convolution_25 = unsqueeze_113 = None
    add_30: "f32[192]" = torch.ops.aten.add.Tensor(arg122_1, 1e-05);  arg122_1 = None
    sqrt_14: "f32[192]" = torch.ops.aten.sqrt.default(add_30);  add_30 = None
    reciprocal_14: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_14);  sqrt_14 = None
    mul_44: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_14, 1);  reciprocal_14 = None
    unsqueeze_114: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_44, -1);  mul_44 = None
    unsqueeze_115: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
    mul_45: "f32[8, 192, 18, 18]" = torch.ops.aten.mul.Tensor(sub_14, unsqueeze_115);  sub_14 = unsqueeze_115 = None
    unsqueeze_116: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg28_1, -1);  arg28_1 = None
    unsqueeze_117: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
    mul_46: "f32[8, 192, 18, 18]" = torch.ops.aten.mul.Tensor(mul_45, unsqueeze_117);  mul_45 = unsqueeze_117 = None
    unsqueeze_118: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg29_1, -1);  arg29_1 = None
    unsqueeze_119: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
    add_31: "f32[8, 192, 18, 18]" = torch.ops.aten.add.Tensor(mul_46, unsqueeze_119);  mul_46 = unsqueeze_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_14: "f32[8, 192, 18, 18]" = torch.ops.aten.relu.default(add_31);  add_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    convolution_26: "f32[8, 192, 18, 18]" = torch.ops.aten.convolution.default(relu_14, arg74_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 192);  arg74_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    convolution_27: "f32[8, 192, 18, 18]" = torch.ops.aten.convolution.default(convolution_26, arg75_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_26 = arg75_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_120: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg123_1, -1);  arg123_1 = None
    unsqueeze_121: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
    sub_15: "f32[8, 192, 18, 18]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_121);  convolution_27 = unsqueeze_121 = None
    add_32: "f32[192]" = torch.ops.aten.add.Tensor(arg124_1, 1e-05);  arg124_1 = None
    sqrt_15: "f32[192]" = torch.ops.aten.sqrt.default(add_32);  add_32 = None
    reciprocal_15: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_15);  sqrt_15 = None
    mul_47: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_15, 1);  reciprocal_15 = None
    unsqueeze_122: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_47, -1);  mul_47 = None
    unsqueeze_123: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
    mul_48: "f32[8, 192, 18, 18]" = torch.ops.aten.mul.Tensor(sub_15, unsqueeze_123);  sub_15 = unsqueeze_123 = None
    unsqueeze_124: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg30_1, -1);  arg30_1 = None
    unsqueeze_125: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
    mul_49: "f32[8, 192, 18, 18]" = torch.ops.aten.mul.Tensor(mul_48, unsqueeze_125);  mul_48 = unsqueeze_125 = None
    unsqueeze_126: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg31_1, -1);  arg31_1 = None
    unsqueeze_127: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
    add_33: "f32[8, 192, 18, 18]" = torch.ops.aten.add.Tensor(mul_49, unsqueeze_127);  mul_49 = unsqueeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_15: "f32[8, 192, 18, 18]" = torch.ops.aten.relu.default(add_33);  add_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    convolution_28: "f32[8, 192, 18, 18]" = torch.ops.aten.convolution.default(relu_15, arg76_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 192);  arg76_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    convolution_29: "f32[8, 192, 18, 18]" = torch.ops.aten.convolution.default(convolution_28, arg77_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_28 = arg77_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_128: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg125_1, -1);  arg125_1 = None
    unsqueeze_129: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
    sub_16: "f32[8, 192, 18, 18]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_129);  convolution_29 = unsqueeze_129 = None
    add_34: "f32[192]" = torch.ops.aten.add.Tensor(arg126_1, 1e-05);  arg126_1 = None
    sqrt_16: "f32[192]" = torch.ops.aten.sqrt.default(add_34);  add_34 = None
    reciprocal_16: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_16);  sqrt_16 = None
    mul_50: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_16, 1);  reciprocal_16 = None
    unsqueeze_130: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_50, -1);  mul_50 = None
    unsqueeze_131: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
    mul_51: "f32[8, 192, 18, 18]" = torch.ops.aten.mul.Tensor(sub_16, unsqueeze_131);  sub_16 = unsqueeze_131 = None
    unsqueeze_132: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg32_1, -1);  arg32_1 = None
    unsqueeze_133: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
    mul_52: "f32[8, 192, 18, 18]" = torch.ops.aten.mul.Tensor(mul_51, unsqueeze_133);  mul_51 = unsqueeze_133 = None
    unsqueeze_134: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg33_1, -1);  arg33_1 = None
    unsqueeze_135: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
    add_35: "f32[8, 192, 18, 18]" = torch.ops.aten.add.Tensor(mul_52, unsqueeze_135);  mul_52 = unsqueeze_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_16: "f32[8, 192, 18, 18]" = torch.ops.aten.relu.default(add_35);  add_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vovnet.py:39, code: x = torch.cat(concat_list, dim=1)
    cat_2: "f32[8, 1088, 18, 18]" = torch.ops.aten.cat.default([getitem_2, relu_14, relu_15, relu_16], 1);  getitem_2 = relu_14 = relu_15 = relu_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_30: "f32[8, 768, 18, 18]" = torch.ops.aten.convolution.default(cat_2, arg78_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_2 = arg78_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_136: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg127_1, -1);  arg127_1 = None
    unsqueeze_137: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
    sub_17: "f32[8, 768, 18, 18]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_137);  convolution_30 = unsqueeze_137 = None
    add_36: "f32[768]" = torch.ops.aten.add.Tensor(arg128_1, 1e-05);  arg128_1 = None
    sqrt_17: "f32[768]" = torch.ops.aten.sqrt.default(add_36);  add_36 = None
    reciprocal_17: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_17);  sqrt_17 = None
    mul_53: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_17, 1);  reciprocal_17 = None
    unsqueeze_138: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_53, -1);  mul_53 = None
    unsqueeze_139: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
    mul_54: "f32[8, 768, 18, 18]" = torch.ops.aten.mul.Tensor(sub_17, unsqueeze_139);  sub_17 = unsqueeze_139 = None
    unsqueeze_140: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg34_1, -1);  arg34_1 = None
    unsqueeze_141: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
    mul_55: "f32[8, 768, 18, 18]" = torch.ops.aten.mul.Tensor(mul_54, unsqueeze_141);  mul_54 = unsqueeze_141 = None
    unsqueeze_142: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg35_1, -1);  arg35_1 = None
    unsqueeze_143: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
    add_37: "f32[8, 768, 18, 18]" = torch.ops.aten.add.Tensor(mul_55, unsqueeze_143);  mul_55 = unsqueeze_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_17: "f32[8, 768, 18, 18]" = torch.ops.aten.relu.default(add_37);  add_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:66, code: x_se = x.mean((2, 3), keepdim=True)
    mean_2: "f32[8, 768, 1, 1]" = torch.ops.aten.mean.dim(relu_17, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:70, code: x_se = self.fc(x_se)
    convolution_31: "f32[8, 768, 1, 1]" = torch.ops.aten.convolution.default(mean_2, arg79_1, arg80_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_2 = arg79_1 = arg80_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:71, code: return x * self.gate(x_se)
    add_38: "f32[8, 768, 1, 1]" = torch.ops.aten.add.Tensor(convolution_31, 3);  convolution_31 = None
    clamp_min_2: "f32[8, 768, 1, 1]" = torch.ops.aten.clamp_min.default(add_38, 0);  add_38 = None
    clamp_max_2: "f32[8, 768, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_2, 6);  clamp_min_2 = None
    div_2: "f32[8, 768, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_2, 6);  clamp_max_2 = None
    mul_56: "f32[8, 768, 18, 18]" = torch.ops.aten.mul.Tensor(relu_17, div_2);  relu_17 = div_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vovnet.py:145, code: x = self.pool(x)
    max_pool2d_with_indices_2 = torch.ops.aten.max_pool2d_with_indices.default(mul_56, [3, 3], [2, 2], [0, 0], [1, 1], True);  mul_56 = None
    getitem_4: "f32[8, 768, 9, 9]" = max_pool2d_with_indices_2[0];  max_pool2d_with_indices_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_32: "f32[8, 224, 9, 9]" = torch.ops.aten.convolution.default(getitem_4, arg81_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg81_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_144: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(arg129_1, -1);  arg129_1 = None
    unsqueeze_145: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
    sub_18: "f32[8, 224, 9, 9]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_145);  convolution_32 = unsqueeze_145 = None
    add_39: "f32[224]" = torch.ops.aten.add.Tensor(arg130_1, 1e-05);  arg130_1 = None
    sqrt_18: "f32[224]" = torch.ops.aten.sqrt.default(add_39);  add_39 = None
    reciprocal_18: "f32[224]" = torch.ops.aten.reciprocal.default(sqrt_18);  sqrt_18 = None
    mul_57: "f32[224]" = torch.ops.aten.mul.Tensor(reciprocal_18, 1);  reciprocal_18 = None
    unsqueeze_146: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(mul_57, -1);  mul_57 = None
    unsqueeze_147: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
    mul_58: "f32[8, 224, 9, 9]" = torch.ops.aten.mul.Tensor(sub_18, unsqueeze_147);  sub_18 = unsqueeze_147 = None
    unsqueeze_148: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(arg36_1, -1);  arg36_1 = None
    unsqueeze_149: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
    mul_59: "f32[8, 224, 9, 9]" = torch.ops.aten.mul.Tensor(mul_58, unsqueeze_149);  mul_58 = unsqueeze_149 = None
    unsqueeze_150: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(arg37_1, -1);  arg37_1 = None
    unsqueeze_151: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
    add_40: "f32[8, 224, 9, 9]" = torch.ops.aten.add.Tensor(mul_59, unsqueeze_151);  mul_59 = unsqueeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_18: "f32[8, 224, 9, 9]" = torch.ops.aten.relu.default(add_40);  add_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    convolution_33: "f32[8, 224, 9, 9]" = torch.ops.aten.convolution.default(relu_18, arg82_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 224);  relu_18 = arg82_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    convolution_34: "f32[8, 224, 9, 9]" = torch.ops.aten.convolution.default(convolution_33, arg83_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_33 = arg83_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_152: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(arg131_1, -1);  arg131_1 = None
    unsqueeze_153: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
    sub_19: "f32[8, 224, 9, 9]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_153);  convolution_34 = unsqueeze_153 = None
    add_41: "f32[224]" = torch.ops.aten.add.Tensor(arg132_1, 1e-05);  arg132_1 = None
    sqrt_19: "f32[224]" = torch.ops.aten.sqrt.default(add_41);  add_41 = None
    reciprocal_19: "f32[224]" = torch.ops.aten.reciprocal.default(sqrt_19);  sqrt_19 = None
    mul_60: "f32[224]" = torch.ops.aten.mul.Tensor(reciprocal_19, 1);  reciprocal_19 = None
    unsqueeze_154: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(mul_60, -1);  mul_60 = None
    unsqueeze_155: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
    mul_61: "f32[8, 224, 9, 9]" = torch.ops.aten.mul.Tensor(sub_19, unsqueeze_155);  sub_19 = unsqueeze_155 = None
    unsqueeze_156: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(arg38_1, -1);  arg38_1 = None
    unsqueeze_157: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
    mul_62: "f32[8, 224, 9, 9]" = torch.ops.aten.mul.Tensor(mul_61, unsqueeze_157);  mul_61 = unsqueeze_157 = None
    unsqueeze_158: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(arg39_1, -1);  arg39_1 = None
    unsqueeze_159: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
    add_42: "f32[8, 224, 9, 9]" = torch.ops.aten.add.Tensor(mul_62, unsqueeze_159);  mul_62 = unsqueeze_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_19: "f32[8, 224, 9, 9]" = torch.ops.aten.relu.default(add_42);  add_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    convolution_35: "f32[8, 224, 9, 9]" = torch.ops.aten.convolution.default(relu_19, arg84_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 224);  arg84_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    convolution_36: "f32[8, 224, 9, 9]" = torch.ops.aten.convolution.default(convolution_35, arg85_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_35 = arg85_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_160: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(arg133_1, -1);  arg133_1 = None
    unsqueeze_161: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
    sub_20: "f32[8, 224, 9, 9]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_161);  convolution_36 = unsqueeze_161 = None
    add_43: "f32[224]" = torch.ops.aten.add.Tensor(arg134_1, 1e-05);  arg134_1 = None
    sqrt_20: "f32[224]" = torch.ops.aten.sqrt.default(add_43);  add_43 = None
    reciprocal_20: "f32[224]" = torch.ops.aten.reciprocal.default(sqrt_20);  sqrt_20 = None
    mul_63: "f32[224]" = torch.ops.aten.mul.Tensor(reciprocal_20, 1);  reciprocal_20 = None
    unsqueeze_162: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(mul_63, -1);  mul_63 = None
    unsqueeze_163: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
    mul_64: "f32[8, 224, 9, 9]" = torch.ops.aten.mul.Tensor(sub_20, unsqueeze_163);  sub_20 = unsqueeze_163 = None
    unsqueeze_164: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(arg40_1, -1);  arg40_1 = None
    unsqueeze_165: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
    mul_65: "f32[8, 224, 9, 9]" = torch.ops.aten.mul.Tensor(mul_64, unsqueeze_165);  mul_64 = unsqueeze_165 = None
    unsqueeze_166: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(arg41_1, -1);  arg41_1 = None
    unsqueeze_167: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
    add_44: "f32[8, 224, 9, 9]" = torch.ops.aten.add.Tensor(mul_65, unsqueeze_167);  mul_65 = unsqueeze_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_20: "f32[8, 224, 9, 9]" = torch.ops.aten.relu.default(add_44);  add_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    convolution_37: "f32[8, 224, 9, 9]" = torch.ops.aten.convolution.default(relu_20, arg86_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 224);  arg86_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    convolution_38: "f32[8, 224, 9, 9]" = torch.ops.aten.convolution.default(convolution_37, arg87_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_37 = arg87_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_168: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(arg135_1, -1);  arg135_1 = None
    unsqueeze_169: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
    sub_21: "f32[8, 224, 9, 9]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_169);  convolution_38 = unsqueeze_169 = None
    add_45: "f32[224]" = torch.ops.aten.add.Tensor(arg136_1, 1e-05);  arg136_1 = None
    sqrt_21: "f32[224]" = torch.ops.aten.sqrt.default(add_45);  add_45 = None
    reciprocal_21: "f32[224]" = torch.ops.aten.reciprocal.default(sqrt_21);  sqrt_21 = None
    mul_66: "f32[224]" = torch.ops.aten.mul.Tensor(reciprocal_21, 1);  reciprocal_21 = None
    unsqueeze_170: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(mul_66, -1);  mul_66 = None
    unsqueeze_171: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
    mul_67: "f32[8, 224, 9, 9]" = torch.ops.aten.mul.Tensor(sub_21, unsqueeze_171);  sub_21 = unsqueeze_171 = None
    unsqueeze_172: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(arg42_1, -1);  arg42_1 = None
    unsqueeze_173: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
    mul_68: "f32[8, 224, 9, 9]" = torch.ops.aten.mul.Tensor(mul_67, unsqueeze_173);  mul_67 = unsqueeze_173 = None
    unsqueeze_174: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(arg43_1, -1);  arg43_1 = None
    unsqueeze_175: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
    add_46: "f32[8, 224, 9, 9]" = torch.ops.aten.add.Tensor(mul_68, unsqueeze_175);  mul_68 = unsqueeze_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_21: "f32[8, 224, 9, 9]" = torch.ops.aten.relu.default(add_46);  add_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vovnet.py:39, code: x = torch.cat(concat_list, dim=1)
    cat_3: "f32[8, 1440, 9, 9]" = torch.ops.aten.cat.default([getitem_4, relu_19, relu_20, relu_21], 1);  getitem_4 = relu_19 = relu_20 = relu_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_39: "f32[8, 1024, 9, 9]" = torch.ops.aten.convolution.default(cat_3, arg88_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_3 = arg88_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_176: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg137_1, -1);  arg137_1 = None
    unsqueeze_177: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
    sub_22: "f32[8, 1024, 9, 9]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_177);  convolution_39 = unsqueeze_177 = None
    add_47: "f32[1024]" = torch.ops.aten.add.Tensor(arg138_1, 1e-05);  arg138_1 = None
    sqrt_22: "f32[1024]" = torch.ops.aten.sqrt.default(add_47);  add_47 = None
    reciprocal_22: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_22);  sqrt_22 = None
    mul_69: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_22, 1);  reciprocal_22 = None
    unsqueeze_178: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_69, -1);  mul_69 = None
    unsqueeze_179: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
    mul_70: "f32[8, 1024, 9, 9]" = torch.ops.aten.mul.Tensor(sub_22, unsqueeze_179);  sub_22 = unsqueeze_179 = None
    unsqueeze_180: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg44_1, -1);  arg44_1 = None
    unsqueeze_181: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
    mul_71: "f32[8, 1024, 9, 9]" = torch.ops.aten.mul.Tensor(mul_70, unsqueeze_181);  mul_70 = unsqueeze_181 = None
    unsqueeze_182: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg45_1, -1);  arg45_1 = None
    unsqueeze_183: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
    add_48: "f32[8, 1024, 9, 9]" = torch.ops.aten.add.Tensor(mul_71, unsqueeze_183);  mul_71 = unsqueeze_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_22: "f32[8, 1024, 9, 9]" = torch.ops.aten.relu.default(add_48);  add_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:66, code: x_se = x.mean((2, 3), keepdim=True)
    mean_3: "f32[8, 1024, 1, 1]" = torch.ops.aten.mean.dim(relu_22, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:70, code: x_se = self.fc(x_se)
    convolution_40: "f32[8, 1024, 1, 1]" = torch.ops.aten.convolution.default(mean_3, arg89_1, arg90_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_3 = arg89_1 = arg90_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:71, code: return x * self.gate(x_se)
    add_49: "f32[8, 1024, 1, 1]" = torch.ops.aten.add.Tensor(convolution_40, 3);  convolution_40 = None
    clamp_min_3: "f32[8, 1024, 1, 1]" = torch.ops.aten.clamp_min.default(add_49, 0);  add_49 = None
    clamp_max_3: "f32[8, 1024, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_3, 6);  clamp_min_3 = None
    div_3: "f32[8, 1024, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_3, 6);  clamp_max_3 = None
    mul_72: "f32[8, 1024, 9, 9]" = torch.ops.aten.mul.Tensor(relu_22, div_3);  relu_22 = div_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean_4: "f32[8, 1024, 1, 1]" = torch.ops.aten.mean.dim(mul_72, [-1, -2], True);  mul_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view: "f32[8, 1024]" = torch.ops.aten.reshape.default(mean_4, [8, 1024]);  mean_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:134, code: x = self.fc(x)
    permute: "f32[1024, 1000]" = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
    addmm: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg92_1, view, permute);  arg92_1 = view = permute = None
    return (addmm,)
    