from __future__ import annotations



def forward(self, arg0_1: "f32[24]", arg1_1: "f32[24]", arg2_1: "f32[32]", arg3_1: "f32[32]", arg4_1: "f32[64]", arg5_1: "f32[64]", arg6_1: "f32[64]", arg7_1: "f32[64]", arg8_1: "f32[64]", arg9_1: "f32[64]", arg10_1: "f32[256]", arg11_1: "f32[256]", arg12_1: "f32[256]", arg13_1: "f32[256]", arg14_1: "f32[64]", arg15_1: "f32[64]", arg16_1: "f32[64]", arg17_1: "f32[64]", arg18_1: "f32[256]", arg19_1: "f32[256]", arg20_1: "f32[128]", arg21_1: "f32[128]", arg22_1: "f32[128]", arg23_1: "f32[128]", arg24_1: "f32[512]", arg25_1: "f32[512]", arg26_1: "f32[512]", arg27_1: "f32[512]", arg28_1: "f32[128]", arg29_1: "f32[128]", arg30_1: "f32[128]", arg31_1: "f32[128]", arg32_1: "f32[512]", arg33_1: "f32[512]", arg34_1: "f32[256]", arg35_1: "f32[256]", arg36_1: "f32[256]", arg37_1: "f32[256]", arg38_1: "f32[1024]", arg39_1: "f32[1024]", arg40_1: "f32[1024]", arg41_1: "f32[1024]", arg42_1: "f32[256]", arg43_1: "f32[256]", arg44_1: "f32[23, 16]", arg45_1: "f32[23, 16]", arg46_1: "f32[256]", arg47_1: "f32[256]", arg48_1: "f32[1024]", arg49_1: "f32[1024]", arg50_1: "f32[512]", arg51_1: "f32[512]", arg52_1: "f32[23, 16]", arg53_1: "f32[23, 16]", arg54_1: "f32[512]", arg55_1: "f32[512]", arg56_1: "f32[2048]", arg57_1: "f32[2048]", arg58_1: "f32[2048]", arg59_1: "f32[2048]", arg60_1: "f32[512]", arg61_1: "f32[512]", arg62_1: "f32[23, 16]", arg63_1: "f32[23, 16]", arg64_1: "f32[512]", arg65_1: "f32[512]", arg66_1: "f32[2048]", arg67_1: "f32[2048]", arg68_1: "f32[24, 3, 3, 3]", arg69_1: "f32[32, 24, 3, 3]", arg70_1: "f32[64, 32, 3, 3]", arg71_1: "f32[64, 64, 1, 1]", arg72_1: "f32[64, 16, 3, 3]", arg73_1: "f32[1, 1, 3]", arg74_1: "f32[256, 64, 1, 1]", arg75_1: "f32[256, 64, 1, 1]", arg76_1: "f32[64, 256, 1, 1]", arg77_1: "f32[64, 16, 3, 3]", arg78_1: "f32[1, 1, 3]", arg79_1: "f32[256, 64, 1, 1]", arg80_1: "f32[128, 256, 1, 1]", arg81_1: "f32[128, 16, 3, 3]", arg82_1: "f32[1, 1, 5]", arg83_1: "f32[512, 128, 1, 1]", arg84_1: "f32[512, 256, 1, 1]", arg85_1: "f32[128, 512, 1, 1]", arg86_1: "f32[128, 16, 3, 3]", arg87_1: "f32[1, 1, 5]", arg88_1: "f32[512, 128, 1, 1]", arg89_1: "f32[256, 512, 1, 1]", arg90_1: "f32[256, 16, 3, 3]", arg91_1: "f32[1, 1, 5]", arg92_1: "f32[1024, 256, 1, 1]", arg93_1: "f32[1024, 512, 1, 1]", arg94_1: "f32[256, 1024, 1, 1]", arg95_1: "f32[128, 256, 1, 1]", arg96_1: "f32[384, 256, 1, 1]", arg97_1: "f32[1024, 256, 1, 1]", arg98_1: "f32[512, 1024, 1, 1]", arg99_1: "f32[128, 512, 1, 1]", arg100_1: "f32[640, 512, 1, 1]", arg101_1: "f32[2048, 512, 1, 1]", arg102_1: "f32[2048, 1024, 1, 1]", arg103_1: "f32[512, 2048, 1, 1]", arg104_1: "f32[128, 512, 1, 1]", arg105_1: "f32[640, 512, 1, 1]", arg106_1: "f32[2048, 512, 1, 1]", arg107_1: "f32[1000, 2048]", arg108_1: "f32[1000]", arg109_1: "f32[24]", arg110_1: "f32[24]", arg111_1: "f32[32]", arg112_1: "f32[32]", arg113_1: "f32[64]", arg114_1: "f32[64]", arg115_1: "f32[64]", arg116_1: "f32[64]", arg117_1: "f32[64]", arg118_1: "f32[64]", arg119_1: "f32[256]", arg120_1: "f32[256]", arg121_1: "f32[256]", arg122_1: "f32[256]", arg123_1: "f32[64]", arg124_1: "f32[64]", arg125_1: "f32[64]", arg126_1: "f32[64]", arg127_1: "f32[256]", arg128_1: "f32[256]", arg129_1: "f32[128]", arg130_1: "f32[128]", arg131_1: "f32[128]", arg132_1: "f32[128]", arg133_1: "f32[512]", arg134_1: "f32[512]", arg135_1: "f32[512]", arg136_1: "f32[512]", arg137_1: "f32[128]", arg138_1: "f32[128]", arg139_1: "f32[128]", arg140_1: "f32[128]", arg141_1: "f32[512]", arg142_1: "f32[512]", arg143_1: "f32[256]", arg144_1: "f32[256]", arg145_1: "f32[256]", arg146_1: "f32[256]", arg147_1: "f32[1024]", arg148_1: "f32[1024]", arg149_1: "f32[1024]", arg150_1: "f32[1024]", arg151_1: "f32[256]", arg152_1: "f32[256]", arg153_1: "f32[256]", arg154_1: "f32[256]", arg155_1: "f32[1024]", arg156_1: "f32[1024]", arg157_1: "f32[512]", arg158_1: "f32[512]", arg159_1: "f32[512]", arg160_1: "f32[512]", arg161_1: "f32[2048]", arg162_1: "f32[2048]", arg163_1: "f32[2048]", arg164_1: "f32[2048]", arg165_1: "f32[512]", arg166_1: "f32[512]", arg167_1: "f32[512]", arg168_1: "f32[512]", arg169_1: "f32[2048]", arg170_1: "f32[2048]", arg171_1: "f32[8, 3, 256, 256]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution: "f32[8, 24, 128, 128]" = torch.ops.aten.convolution.default(arg171_1, arg68_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg171_1 = arg68_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type: "f32[24]" = torch.ops.prims.convert_element_type.default(arg109_1, torch.float32);  arg109_1 = None
    convert_element_type_1: "f32[24]" = torch.ops.prims.convert_element_type.default(arg110_1, torch.float32);  arg110_1 = None
    add: "f32[24]" = torch.ops.aten.add.Tensor(convert_element_type_1, 1e-05);  convert_element_type_1 = None
    sqrt: "f32[24]" = torch.ops.aten.sqrt.default(add);  add = None
    reciprocal: "f32[24]" = torch.ops.aten.reciprocal.default(sqrt);  sqrt = None
    mul: "f32[24]" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
    unsqueeze: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type, -1);  convert_element_type = None
    unsqueeze_1: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    unsqueeze_2: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(mul, -1);  mul = None
    unsqueeze_3: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    sub: "f32[8, 24, 128, 128]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1);  convolution = unsqueeze_1 = None
    mul_1: "f32[8, 24, 128, 128]" = torch.ops.aten.mul.Tensor(sub, unsqueeze_3);  sub = unsqueeze_3 = None
    unsqueeze_4: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg0_1, -1);  arg0_1 = None
    unsqueeze_5: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_2: "f32[8, 24, 128, 128]" = torch.ops.aten.mul.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None
    unsqueeze_6: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg1_1, -1);  arg1_1 = None
    unsqueeze_7: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_1: "f32[8, 24, 128, 128]" = torch.ops.aten.add.Tensor(mul_2, unsqueeze_7);  mul_2 = unsqueeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid: "f32[8, 24, 128, 128]" = torch.ops.aten.sigmoid.default(add_1)
    mul_3: "f32[8, 24, 128, 128]" = torch.ops.aten.mul.Tensor(add_1, sigmoid);  add_1 = sigmoid = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_1: "f32[8, 32, 128, 128]" = torch.ops.aten.convolution.default(mul_3, arg69_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  mul_3 = arg69_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_2: "f32[32]" = torch.ops.prims.convert_element_type.default(arg111_1, torch.float32);  arg111_1 = None
    convert_element_type_3: "f32[32]" = torch.ops.prims.convert_element_type.default(arg112_1, torch.float32);  arg112_1 = None
    add_2: "f32[32]" = torch.ops.aten.add.Tensor(convert_element_type_3, 1e-05);  convert_element_type_3 = None
    sqrt_1: "f32[32]" = torch.ops.aten.sqrt.default(add_2);  add_2 = None
    reciprocal_1: "f32[32]" = torch.ops.aten.reciprocal.default(sqrt_1);  sqrt_1 = None
    mul_4: "f32[32]" = torch.ops.aten.mul.Tensor(reciprocal_1, 1);  reciprocal_1 = None
    unsqueeze_8: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, -1);  convert_element_type_2 = None
    unsqueeze_9: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    unsqueeze_10: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(mul_4, -1);  mul_4 = None
    unsqueeze_11: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    sub_1: "f32[8, 32, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_9);  convolution_1 = unsqueeze_9 = None
    mul_5: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(sub_1, unsqueeze_11);  sub_1 = unsqueeze_11 = None
    unsqueeze_12: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
    unsqueeze_13: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_6: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(mul_5, unsqueeze_13);  mul_5 = unsqueeze_13 = None
    unsqueeze_14: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, -1);  arg3_1 = None
    unsqueeze_15: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_3: "f32[8, 32, 128, 128]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_15);  mul_6 = unsqueeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_1: "f32[8, 32, 128, 128]" = torch.ops.aten.sigmoid.default(add_3)
    mul_7: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(add_3, sigmoid_1);  add_3 = sigmoid_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_2: "f32[8, 64, 128, 128]" = torch.ops.aten.convolution.default(mul_7, arg70_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  mul_7 = arg70_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_4: "f32[64]" = torch.ops.prims.convert_element_type.default(arg113_1, torch.float32);  arg113_1 = None
    convert_element_type_5: "f32[64]" = torch.ops.prims.convert_element_type.default(arg114_1, torch.float32);  arg114_1 = None
    add_4: "f32[64]" = torch.ops.aten.add.Tensor(convert_element_type_5, 1e-05);  convert_element_type_5 = None
    sqrt_2: "f32[64]" = torch.ops.aten.sqrt.default(add_4);  add_4 = None
    reciprocal_2: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_2);  sqrt_2 = None
    mul_8: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_2, 1);  reciprocal_2 = None
    unsqueeze_16: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_4, -1);  convert_element_type_4 = None
    unsqueeze_17: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    unsqueeze_18: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_8, -1);  mul_8 = None
    unsqueeze_19: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    sub_2: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_17);  convolution_2 = unsqueeze_17 = None
    mul_9: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_2, unsqueeze_19);  sub_2 = unsqueeze_19 = None
    unsqueeze_20: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
    unsqueeze_21: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_10: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(mul_9, unsqueeze_21);  mul_9 = unsqueeze_21 = None
    unsqueeze_22: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
    unsqueeze_23: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_5: "f32[8, 64, 128, 128]" = torch.ops.aten.add.Tensor(mul_10, unsqueeze_23);  mul_10 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_2: "f32[8, 64, 128, 128]" = torch.ops.aten.sigmoid.default(add_5)
    mul_11: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(add_5, sigmoid_2);  add_5 = sigmoid_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:1245, code: x = self.stem(x)
    max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices.default(mul_11, [3, 3], [2, 2], [1, 1]);  mul_11 = None
    getitem: "f32[8, 64, 64, 64]" = max_pool2d_with_indices[0];  max_pool2d_with_indices = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_3: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(getitem, arg71_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg71_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_6: "f32[64]" = torch.ops.prims.convert_element_type.default(arg115_1, torch.float32);  arg115_1 = None
    convert_element_type_7: "f32[64]" = torch.ops.prims.convert_element_type.default(arg116_1, torch.float32);  arg116_1 = None
    add_6: "f32[64]" = torch.ops.aten.add.Tensor(convert_element_type_7, 1e-05);  convert_element_type_7 = None
    sqrt_3: "f32[64]" = torch.ops.aten.sqrt.default(add_6);  add_6 = None
    reciprocal_3: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_3);  sqrt_3 = None
    mul_12: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_3, 1);  reciprocal_3 = None
    unsqueeze_24: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_6, -1);  convert_element_type_6 = None
    unsqueeze_25: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
    unsqueeze_26: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_12, -1);  mul_12 = None
    unsqueeze_27: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
    sub_3: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_25);  convolution_3 = unsqueeze_25 = None
    mul_13: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_3, unsqueeze_27);  sub_3 = unsqueeze_27 = None
    unsqueeze_28: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg6_1, -1);  arg6_1 = None
    unsqueeze_29: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    mul_14: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_13, unsqueeze_29);  mul_13 = unsqueeze_29 = None
    unsqueeze_30: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
    unsqueeze_31: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
    add_7: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_14, unsqueeze_31);  mul_14 = unsqueeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_3: "f32[8, 64, 64, 64]" = torch.ops.aten.sigmoid.default(add_7)
    mul_15: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_7, sigmoid_3);  add_7 = sigmoid_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_4: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(mul_15, arg72_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 4);  mul_15 = arg72_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_8: "f32[64]" = torch.ops.prims.convert_element_type.default(arg117_1, torch.float32);  arg117_1 = None
    convert_element_type_9: "f32[64]" = torch.ops.prims.convert_element_type.default(arg118_1, torch.float32);  arg118_1 = None
    add_8: "f32[64]" = torch.ops.aten.add.Tensor(convert_element_type_9, 1e-05);  convert_element_type_9 = None
    sqrt_4: "f32[64]" = torch.ops.aten.sqrt.default(add_8);  add_8 = None
    reciprocal_4: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_4);  sqrt_4 = None
    mul_16: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_4, 1);  reciprocal_4 = None
    unsqueeze_32: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_8, -1);  convert_element_type_8 = None
    unsqueeze_33: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    unsqueeze_34: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_16, -1);  mul_16 = None
    unsqueeze_35: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
    sub_4: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_33);  convolution_4 = unsqueeze_33 = None
    mul_17: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_4, unsqueeze_35);  sub_4 = unsqueeze_35 = None
    unsqueeze_36: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg8_1, -1);  arg8_1 = None
    unsqueeze_37: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
    mul_18: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_17, unsqueeze_37);  mul_17 = unsqueeze_37 = None
    unsqueeze_38: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
    unsqueeze_39: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
    add_9: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_18, unsqueeze_39);  mul_18 = unsqueeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_4: "f32[8, 64, 64, 64]" = torch.ops.aten.sigmoid.default(add_9)
    mul_19: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_9, sigmoid_4);  add_9 = sigmoid_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:85, code: y = x.mean((2, 3)).view(x.shape[0], 1, -1)  # view for 1d conv
    mean: "f32[8, 64]" = torch.ops.aten.mean.dim(mul_19, [2, 3])
    view: "f32[8, 1, 64]" = torch.ops.aten.view.default(mean, [8, 1, -1]);  mean = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:86, code: y = self.conv(y)
    convolution_5: "f32[8, 1, 64]" = torch.ops.aten.convolution.default(view, arg73_1, None, [1], [1], [1], False, [0], 1);  view = arg73_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_5: "f32[8, 1, 64]" = torch.ops.aten.sigmoid.default(convolution_5);  convolution_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:90, code: y = self.gate(y).view(x.shape[0], -1, 1, 1)
    view_1: "f32[8, 64, 1, 1]" = torch.ops.aten.view.default(sigmoid_5, [8, -1, 1, 1]);  sigmoid_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:91, code: return x * y.expand_as(x)
    expand: "f32[8, 64, 64, 64]" = torch.ops.aten.expand.default(view_1, [8, 64, 64, 64]);  view_1 = None
    mul_20: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_19, expand);  mul_19 = expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_6: "f32[8, 256, 64, 64]" = torch.ops.aten.convolution.default(mul_20, arg74_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_20 = arg74_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_10: "f32[256]" = torch.ops.prims.convert_element_type.default(arg119_1, torch.float32);  arg119_1 = None
    convert_element_type_11: "f32[256]" = torch.ops.prims.convert_element_type.default(arg120_1, torch.float32);  arg120_1 = None
    add_10: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_11, 1e-05);  convert_element_type_11 = None
    sqrt_5: "f32[256]" = torch.ops.aten.sqrt.default(add_10);  add_10 = None
    reciprocal_5: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_5);  sqrt_5 = None
    mul_21: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_5, 1);  reciprocal_5 = None
    unsqueeze_40: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_10, -1);  convert_element_type_10 = None
    unsqueeze_41: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    unsqueeze_42: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_21, -1);  mul_21 = None
    unsqueeze_43: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    sub_5: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_41);  convolution_6 = unsqueeze_41 = None
    mul_22: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_5, unsqueeze_43);  sub_5 = unsqueeze_43 = None
    unsqueeze_44: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
    unsqueeze_45: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    mul_23: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_22, unsqueeze_45);  mul_22 = unsqueeze_45 = None
    unsqueeze_46: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg11_1, -1);  arg11_1 = None
    unsqueeze_47: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
    add_11: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_23, unsqueeze_47);  mul_23 = unsqueeze_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_7: "f32[8, 256, 64, 64]" = torch.ops.aten.convolution.default(getitem, arg75_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem = arg75_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_12: "f32[256]" = torch.ops.prims.convert_element_type.default(arg121_1, torch.float32);  arg121_1 = None
    convert_element_type_13: "f32[256]" = torch.ops.prims.convert_element_type.default(arg122_1, torch.float32);  arg122_1 = None
    add_12: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_13, 1e-05);  convert_element_type_13 = None
    sqrt_6: "f32[256]" = torch.ops.aten.sqrt.default(add_12);  add_12 = None
    reciprocal_6: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_6);  sqrt_6 = None
    mul_24: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_6, 1);  reciprocal_6 = None
    unsqueeze_48: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_12, -1);  convert_element_type_12 = None
    unsqueeze_49: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
    unsqueeze_50: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_24, -1);  mul_24 = None
    unsqueeze_51: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
    sub_6: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_49);  convolution_7 = unsqueeze_49 = None
    mul_25: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_6, unsqueeze_51);  sub_6 = unsqueeze_51 = None
    unsqueeze_52: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg12_1, -1);  arg12_1 = None
    unsqueeze_53: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    mul_26: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_25, unsqueeze_53);  mul_25 = unsqueeze_53 = None
    unsqueeze_54: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg13_1, -1);  arg13_1 = None
    unsqueeze_55: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
    add_13: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_26, unsqueeze_55);  mul_26 = unsqueeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    add_14: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(add_11, add_13);  add_11 = add_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    sigmoid_6: "f32[8, 256, 64, 64]" = torch.ops.aten.sigmoid.default(add_14)
    mul_27: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(add_14, sigmoid_6);  add_14 = sigmoid_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_8: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(mul_27, arg76_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg76_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_14: "f32[64]" = torch.ops.prims.convert_element_type.default(arg123_1, torch.float32);  arg123_1 = None
    convert_element_type_15: "f32[64]" = torch.ops.prims.convert_element_type.default(arg124_1, torch.float32);  arg124_1 = None
    add_15: "f32[64]" = torch.ops.aten.add.Tensor(convert_element_type_15, 1e-05);  convert_element_type_15 = None
    sqrt_7: "f32[64]" = torch.ops.aten.sqrt.default(add_15);  add_15 = None
    reciprocal_7: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_7);  sqrt_7 = None
    mul_28: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_7, 1);  reciprocal_7 = None
    unsqueeze_56: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_14, -1);  convert_element_type_14 = None
    unsqueeze_57: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    unsqueeze_58: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_28, -1);  mul_28 = None
    unsqueeze_59: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
    sub_7: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_57);  convolution_8 = unsqueeze_57 = None
    mul_29: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_7, unsqueeze_59);  sub_7 = unsqueeze_59 = None
    unsqueeze_60: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
    unsqueeze_61: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
    mul_30: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_29, unsqueeze_61);  mul_29 = unsqueeze_61 = None
    unsqueeze_62: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
    unsqueeze_63: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
    add_16: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_30, unsqueeze_63);  mul_30 = unsqueeze_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_7: "f32[8, 64, 64, 64]" = torch.ops.aten.sigmoid.default(add_16)
    mul_31: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_16, sigmoid_7);  add_16 = sigmoid_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_9: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(mul_31, arg77_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 4);  mul_31 = arg77_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_16: "f32[64]" = torch.ops.prims.convert_element_type.default(arg125_1, torch.float32);  arg125_1 = None
    convert_element_type_17: "f32[64]" = torch.ops.prims.convert_element_type.default(arg126_1, torch.float32);  arg126_1 = None
    add_17: "f32[64]" = torch.ops.aten.add.Tensor(convert_element_type_17, 1e-05);  convert_element_type_17 = None
    sqrt_8: "f32[64]" = torch.ops.aten.sqrt.default(add_17);  add_17 = None
    reciprocal_8: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_8);  sqrt_8 = None
    mul_32: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_8, 1);  reciprocal_8 = None
    unsqueeze_64: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_16, -1);  convert_element_type_16 = None
    unsqueeze_65: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
    unsqueeze_66: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_32, -1);  mul_32 = None
    unsqueeze_67: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
    sub_8: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_65);  convolution_9 = unsqueeze_65 = None
    mul_33: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_8, unsqueeze_67);  sub_8 = unsqueeze_67 = None
    unsqueeze_68: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg16_1, -1);  arg16_1 = None
    unsqueeze_69: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    mul_34: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_33, unsqueeze_69);  mul_33 = unsqueeze_69 = None
    unsqueeze_70: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg17_1, -1);  arg17_1 = None
    unsqueeze_71: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    add_18: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_34, unsqueeze_71);  mul_34 = unsqueeze_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_8: "f32[8, 64, 64, 64]" = torch.ops.aten.sigmoid.default(add_18)
    mul_35: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_18, sigmoid_8);  add_18 = sigmoid_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:85, code: y = x.mean((2, 3)).view(x.shape[0], 1, -1)  # view for 1d conv
    mean_1: "f32[8, 64]" = torch.ops.aten.mean.dim(mul_35, [2, 3])
    view_2: "f32[8, 1, 64]" = torch.ops.aten.view.default(mean_1, [8, 1, -1]);  mean_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:86, code: y = self.conv(y)
    convolution_10: "f32[8, 1, 64]" = torch.ops.aten.convolution.default(view_2, arg78_1, None, [1], [1], [1], False, [0], 1);  view_2 = arg78_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_9: "f32[8, 1, 64]" = torch.ops.aten.sigmoid.default(convolution_10);  convolution_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:90, code: y = self.gate(y).view(x.shape[0], -1, 1, 1)
    view_3: "f32[8, 64, 1, 1]" = torch.ops.aten.view.default(sigmoid_9, [8, -1, 1, 1]);  sigmoid_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:91, code: return x * y.expand_as(x)
    expand_1: "f32[8, 64, 64, 64]" = torch.ops.aten.expand.default(view_3, [8, 64, 64, 64]);  view_3 = None
    mul_36: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_35, expand_1);  mul_35 = expand_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_11: "f32[8, 256, 64, 64]" = torch.ops.aten.convolution.default(mul_36, arg79_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_36 = arg79_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_18: "f32[256]" = torch.ops.prims.convert_element_type.default(arg127_1, torch.float32);  arg127_1 = None
    convert_element_type_19: "f32[256]" = torch.ops.prims.convert_element_type.default(arg128_1, torch.float32);  arg128_1 = None
    add_19: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_19, 1e-05);  convert_element_type_19 = None
    sqrt_9: "f32[256]" = torch.ops.aten.sqrt.default(add_19);  add_19 = None
    reciprocal_9: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_9);  sqrt_9 = None
    mul_37: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_9, 1);  reciprocal_9 = None
    unsqueeze_72: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_18, -1);  convert_element_type_18 = None
    unsqueeze_73: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
    unsqueeze_74: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_37, -1);  mul_37 = None
    unsqueeze_75: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
    sub_9: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_73);  convolution_11 = unsqueeze_73 = None
    mul_38: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_9, unsqueeze_75);  sub_9 = unsqueeze_75 = None
    unsqueeze_76: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg18_1, -1);  arg18_1 = None
    unsqueeze_77: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
    mul_39: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_38, unsqueeze_77);  mul_38 = unsqueeze_77 = None
    unsqueeze_78: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg19_1, -1);  arg19_1 = None
    unsqueeze_79: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    add_20: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_39, unsqueeze_79);  mul_39 = unsqueeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    add_21: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(add_20, mul_27);  add_20 = mul_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    sigmoid_10: "f32[8, 256, 64, 64]" = torch.ops.aten.sigmoid.default(add_21)
    mul_40: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(add_21, sigmoid_10);  add_21 = sigmoid_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_12: "f32[8, 128, 64, 64]" = torch.ops.aten.convolution.default(mul_40, arg80_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg80_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_20: "f32[128]" = torch.ops.prims.convert_element_type.default(arg129_1, torch.float32);  arg129_1 = None
    convert_element_type_21: "f32[128]" = torch.ops.prims.convert_element_type.default(arg130_1, torch.float32);  arg130_1 = None
    add_22: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_21, 1e-05);  convert_element_type_21 = None
    sqrt_10: "f32[128]" = torch.ops.aten.sqrt.default(add_22);  add_22 = None
    reciprocal_10: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_10);  sqrt_10 = None
    mul_41: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_10, 1);  reciprocal_10 = None
    unsqueeze_80: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_20, -1);  convert_element_type_20 = None
    unsqueeze_81: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
    unsqueeze_82: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_41, -1);  mul_41 = None
    unsqueeze_83: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    sub_10: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_81);  convolution_12 = unsqueeze_81 = None
    mul_42: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_10, unsqueeze_83);  sub_10 = unsqueeze_83 = None
    unsqueeze_84: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg20_1, -1);  arg20_1 = None
    unsqueeze_85: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    mul_43: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(mul_42, unsqueeze_85);  mul_42 = unsqueeze_85 = None
    unsqueeze_86: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg21_1, -1);  arg21_1 = None
    unsqueeze_87: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
    add_23: "f32[8, 128, 64, 64]" = torch.ops.aten.add.Tensor(mul_43, unsqueeze_87);  mul_43 = unsqueeze_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_11: "f32[8, 128, 64, 64]" = torch.ops.aten.sigmoid.default(add_23)
    mul_44: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(add_23, sigmoid_11);  add_23 = sigmoid_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_13: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(mul_44, arg81_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 8);  mul_44 = arg81_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_22: "f32[128]" = torch.ops.prims.convert_element_type.default(arg131_1, torch.float32);  arg131_1 = None
    convert_element_type_23: "f32[128]" = torch.ops.prims.convert_element_type.default(arg132_1, torch.float32);  arg132_1 = None
    add_24: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_23, 1e-05);  convert_element_type_23 = None
    sqrt_11: "f32[128]" = torch.ops.aten.sqrt.default(add_24);  add_24 = None
    reciprocal_11: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_11);  sqrt_11 = None
    mul_45: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_11, 1);  reciprocal_11 = None
    unsqueeze_88: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_22, -1);  convert_element_type_22 = None
    unsqueeze_89: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
    unsqueeze_90: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_45, -1);  mul_45 = None
    unsqueeze_91: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    sub_11: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_89);  convolution_13 = unsqueeze_89 = None
    mul_46: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_11, unsqueeze_91);  sub_11 = unsqueeze_91 = None
    unsqueeze_92: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg22_1, -1);  arg22_1 = None
    unsqueeze_93: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
    mul_47: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_46, unsqueeze_93);  mul_46 = unsqueeze_93 = None
    unsqueeze_94: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg23_1, -1);  arg23_1 = None
    unsqueeze_95: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
    add_25: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_47, unsqueeze_95);  mul_47 = unsqueeze_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_12: "f32[8, 128, 32, 32]" = torch.ops.aten.sigmoid.default(add_25)
    mul_48: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_25, sigmoid_12);  add_25 = sigmoid_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:85, code: y = x.mean((2, 3)).view(x.shape[0], 1, -1)  # view for 1d conv
    mean_2: "f32[8, 128]" = torch.ops.aten.mean.dim(mul_48, [2, 3])
    view_4: "f32[8, 1, 128]" = torch.ops.aten.view.default(mean_2, [8, 1, -1]);  mean_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:86, code: y = self.conv(y)
    convolution_14: "f32[8, 1, 128]" = torch.ops.aten.convolution.default(view_4, arg82_1, None, [1], [2], [1], False, [0], 1);  view_4 = arg82_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_13: "f32[8, 1, 128]" = torch.ops.aten.sigmoid.default(convolution_14);  convolution_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:90, code: y = self.gate(y).view(x.shape[0], -1, 1, 1)
    view_5: "f32[8, 128, 1, 1]" = torch.ops.aten.view.default(sigmoid_13, [8, -1, 1, 1]);  sigmoid_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:91, code: return x * y.expand_as(x)
    expand_2: "f32[8, 128, 32, 32]" = torch.ops.aten.expand.default(view_5, [8, 128, 32, 32]);  view_5 = None
    mul_49: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_48, expand_2);  mul_48 = expand_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_15: "f32[8, 512, 32, 32]" = torch.ops.aten.convolution.default(mul_49, arg83_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_49 = arg83_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_24: "f32[512]" = torch.ops.prims.convert_element_type.default(arg133_1, torch.float32);  arg133_1 = None
    convert_element_type_25: "f32[512]" = torch.ops.prims.convert_element_type.default(arg134_1, torch.float32);  arg134_1 = None
    add_26: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_25, 1e-05);  convert_element_type_25 = None
    sqrt_12: "f32[512]" = torch.ops.aten.sqrt.default(add_26);  add_26 = None
    reciprocal_12: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_12);  sqrt_12 = None
    mul_50: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_12, 1);  reciprocal_12 = None
    unsqueeze_96: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_24, -1);  convert_element_type_24 = None
    unsqueeze_97: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
    unsqueeze_98: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_50, -1);  mul_50 = None
    unsqueeze_99: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
    sub_12: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_97);  convolution_15 = unsqueeze_97 = None
    mul_51: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_12, unsqueeze_99);  sub_12 = unsqueeze_99 = None
    unsqueeze_100: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg24_1, -1);  arg24_1 = None
    unsqueeze_101: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
    mul_52: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(mul_51, unsqueeze_101);  mul_51 = unsqueeze_101 = None
    unsqueeze_102: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg25_1, -1);  arg25_1 = None
    unsqueeze_103: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    add_27: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_52, unsqueeze_103);  mul_52 = unsqueeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_16: "f32[8, 512, 32, 32]" = torch.ops.aten.convolution.default(mul_40, arg84_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  mul_40 = arg84_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_26: "f32[512]" = torch.ops.prims.convert_element_type.default(arg135_1, torch.float32);  arg135_1 = None
    convert_element_type_27: "f32[512]" = torch.ops.prims.convert_element_type.default(arg136_1, torch.float32);  arg136_1 = None
    add_28: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_27, 1e-05);  convert_element_type_27 = None
    sqrt_13: "f32[512]" = torch.ops.aten.sqrt.default(add_28);  add_28 = None
    reciprocal_13: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_13);  sqrt_13 = None
    mul_53: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_13, 1);  reciprocal_13 = None
    unsqueeze_104: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_26, -1);  convert_element_type_26 = None
    unsqueeze_105: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
    unsqueeze_106: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_53, -1);  mul_53 = None
    unsqueeze_107: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
    sub_13: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_105);  convolution_16 = unsqueeze_105 = None
    mul_54: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_13, unsqueeze_107);  sub_13 = unsqueeze_107 = None
    unsqueeze_108: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg26_1, -1);  arg26_1 = None
    unsqueeze_109: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
    mul_55: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(mul_54, unsqueeze_109);  mul_54 = unsqueeze_109 = None
    unsqueeze_110: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg27_1, -1);  arg27_1 = None
    unsqueeze_111: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
    add_29: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_55, unsqueeze_111);  mul_55 = unsqueeze_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    add_30: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(add_27, add_29);  add_27 = add_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    sigmoid_14: "f32[8, 512, 32, 32]" = torch.ops.aten.sigmoid.default(add_30)
    mul_56: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(add_30, sigmoid_14);  add_30 = sigmoid_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_17: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(mul_56, arg85_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg85_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_28: "f32[128]" = torch.ops.prims.convert_element_type.default(arg137_1, torch.float32);  arg137_1 = None
    convert_element_type_29: "f32[128]" = torch.ops.prims.convert_element_type.default(arg138_1, torch.float32);  arg138_1 = None
    add_31: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_29, 1e-05);  convert_element_type_29 = None
    sqrt_14: "f32[128]" = torch.ops.aten.sqrt.default(add_31);  add_31 = None
    reciprocal_14: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_14);  sqrt_14 = None
    mul_57: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_14, 1);  reciprocal_14 = None
    unsqueeze_112: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_28, -1);  convert_element_type_28 = None
    unsqueeze_113: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
    unsqueeze_114: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_57, -1);  mul_57 = None
    unsqueeze_115: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
    sub_14: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_113);  convolution_17 = unsqueeze_113 = None
    mul_58: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_14, unsqueeze_115);  sub_14 = unsqueeze_115 = None
    unsqueeze_116: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg28_1, -1);  arg28_1 = None
    unsqueeze_117: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
    mul_59: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_58, unsqueeze_117);  mul_58 = unsqueeze_117 = None
    unsqueeze_118: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg29_1, -1);  arg29_1 = None
    unsqueeze_119: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
    add_32: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_59, unsqueeze_119);  mul_59 = unsqueeze_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_15: "f32[8, 128, 32, 32]" = torch.ops.aten.sigmoid.default(add_32)
    mul_60: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_32, sigmoid_15);  add_32 = sigmoid_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_18: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(mul_60, arg86_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  mul_60 = arg86_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_30: "f32[128]" = torch.ops.prims.convert_element_type.default(arg139_1, torch.float32);  arg139_1 = None
    convert_element_type_31: "f32[128]" = torch.ops.prims.convert_element_type.default(arg140_1, torch.float32);  arg140_1 = None
    add_33: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_31, 1e-05);  convert_element_type_31 = None
    sqrt_15: "f32[128]" = torch.ops.aten.sqrt.default(add_33);  add_33 = None
    reciprocal_15: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_15);  sqrt_15 = None
    mul_61: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_15, 1);  reciprocal_15 = None
    unsqueeze_120: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_30, -1);  convert_element_type_30 = None
    unsqueeze_121: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
    unsqueeze_122: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_61, -1);  mul_61 = None
    unsqueeze_123: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
    sub_15: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_121);  convolution_18 = unsqueeze_121 = None
    mul_62: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_15, unsqueeze_123);  sub_15 = unsqueeze_123 = None
    unsqueeze_124: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg30_1, -1);  arg30_1 = None
    unsqueeze_125: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
    mul_63: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_62, unsqueeze_125);  mul_62 = unsqueeze_125 = None
    unsqueeze_126: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg31_1, -1);  arg31_1 = None
    unsqueeze_127: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
    add_34: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_63, unsqueeze_127);  mul_63 = unsqueeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_16: "f32[8, 128, 32, 32]" = torch.ops.aten.sigmoid.default(add_34)
    mul_64: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_34, sigmoid_16);  add_34 = sigmoid_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:85, code: y = x.mean((2, 3)).view(x.shape[0], 1, -1)  # view for 1d conv
    mean_3: "f32[8, 128]" = torch.ops.aten.mean.dim(mul_64, [2, 3])
    view_6: "f32[8, 1, 128]" = torch.ops.aten.view.default(mean_3, [8, 1, -1]);  mean_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:86, code: y = self.conv(y)
    convolution_19: "f32[8, 1, 128]" = torch.ops.aten.convolution.default(view_6, arg87_1, None, [1], [2], [1], False, [0], 1);  view_6 = arg87_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_17: "f32[8, 1, 128]" = torch.ops.aten.sigmoid.default(convolution_19);  convolution_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:90, code: y = self.gate(y).view(x.shape[0], -1, 1, 1)
    view_7: "f32[8, 128, 1, 1]" = torch.ops.aten.view.default(sigmoid_17, [8, -1, 1, 1]);  sigmoid_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:91, code: return x * y.expand_as(x)
    expand_3: "f32[8, 128, 32, 32]" = torch.ops.aten.expand.default(view_7, [8, 128, 32, 32]);  view_7 = None
    mul_65: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_64, expand_3);  mul_64 = expand_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_20: "f32[8, 512, 32, 32]" = torch.ops.aten.convolution.default(mul_65, arg88_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_65 = arg88_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_32: "f32[512]" = torch.ops.prims.convert_element_type.default(arg141_1, torch.float32);  arg141_1 = None
    convert_element_type_33: "f32[512]" = torch.ops.prims.convert_element_type.default(arg142_1, torch.float32);  arg142_1 = None
    add_35: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_33, 1e-05);  convert_element_type_33 = None
    sqrt_16: "f32[512]" = torch.ops.aten.sqrt.default(add_35);  add_35 = None
    reciprocal_16: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_16);  sqrt_16 = None
    mul_66: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_16, 1);  reciprocal_16 = None
    unsqueeze_128: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_32, -1);  convert_element_type_32 = None
    unsqueeze_129: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
    unsqueeze_130: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_66, -1);  mul_66 = None
    unsqueeze_131: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
    sub_16: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_129);  convolution_20 = unsqueeze_129 = None
    mul_67: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_16, unsqueeze_131);  sub_16 = unsqueeze_131 = None
    unsqueeze_132: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg32_1, -1);  arg32_1 = None
    unsqueeze_133: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
    mul_68: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(mul_67, unsqueeze_133);  mul_67 = unsqueeze_133 = None
    unsqueeze_134: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg33_1, -1);  arg33_1 = None
    unsqueeze_135: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
    add_36: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_68, unsqueeze_135);  mul_68 = unsqueeze_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    add_37: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(add_36, mul_56);  add_36 = mul_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    sigmoid_18: "f32[8, 512, 32, 32]" = torch.ops.aten.sigmoid.default(add_37)
    mul_69: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(add_37, sigmoid_18);  add_37 = sigmoid_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_21: "f32[8, 256, 32, 32]" = torch.ops.aten.convolution.default(mul_69, arg89_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg89_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_34: "f32[256]" = torch.ops.prims.convert_element_type.default(arg143_1, torch.float32);  arg143_1 = None
    convert_element_type_35: "f32[256]" = torch.ops.prims.convert_element_type.default(arg144_1, torch.float32);  arg144_1 = None
    add_38: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_35, 1e-05);  convert_element_type_35 = None
    sqrt_17: "f32[256]" = torch.ops.aten.sqrt.default(add_38);  add_38 = None
    reciprocal_17: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_17);  sqrt_17 = None
    mul_70: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_17, 1);  reciprocal_17 = None
    unsqueeze_136: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_34, -1);  convert_element_type_34 = None
    unsqueeze_137: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
    unsqueeze_138: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_70, -1);  mul_70 = None
    unsqueeze_139: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
    sub_17: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_137);  convolution_21 = unsqueeze_137 = None
    mul_71: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_17, unsqueeze_139);  sub_17 = unsqueeze_139 = None
    unsqueeze_140: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg34_1, -1);  arg34_1 = None
    unsqueeze_141: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
    mul_72: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(mul_71, unsqueeze_141);  mul_71 = unsqueeze_141 = None
    unsqueeze_142: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg35_1, -1);  arg35_1 = None
    unsqueeze_143: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
    add_39: "f32[8, 256, 32, 32]" = torch.ops.aten.add.Tensor(mul_72, unsqueeze_143);  mul_72 = unsqueeze_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_19: "f32[8, 256, 32, 32]" = torch.ops.aten.sigmoid.default(add_39)
    mul_73: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(add_39, sigmoid_19);  add_39 = sigmoid_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_22: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(mul_73, arg90_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 16);  mul_73 = arg90_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_36: "f32[256]" = torch.ops.prims.convert_element_type.default(arg145_1, torch.float32);  arg145_1 = None
    convert_element_type_37: "f32[256]" = torch.ops.prims.convert_element_type.default(arg146_1, torch.float32);  arg146_1 = None
    add_40: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_37, 1e-05);  convert_element_type_37 = None
    sqrt_18: "f32[256]" = torch.ops.aten.sqrt.default(add_40);  add_40 = None
    reciprocal_18: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_18);  sqrt_18 = None
    mul_74: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_18, 1);  reciprocal_18 = None
    unsqueeze_144: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_36, -1);  convert_element_type_36 = None
    unsqueeze_145: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
    unsqueeze_146: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_74, -1);  mul_74 = None
    unsqueeze_147: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
    sub_18: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_145);  convolution_22 = unsqueeze_145 = None
    mul_75: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_18, unsqueeze_147);  sub_18 = unsqueeze_147 = None
    unsqueeze_148: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg36_1, -1);  arg36_1 = None
    unsqueeze_149: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
    mul_76: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_75, unsqueeze_149);  mul_75 = unsqueeze_149 = None
    unsqueeze_150: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg37_1, -1);  arg37_1 = None
    unsqueeze_151: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
    add_41: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_76, unsqueeze_151);  mul_76 = unsqueeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_20: "f32[8, 256, 16, 16]" = torch.ops.aten.sigmoid.default(add_41)
    mul_77: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_41, sigmoid_20);  add_41 = sigmoid_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:85, code: y = x.mean((2, 3)).view(x.shape[0], 1, -1)  # view for 1d conv
    mean_4: "f32[8, 256]" = torch.ops.aten.mean.dim(mul_77, [2, 3])
    view_8: "f32[8, 1, 256]" = torch.ops.aten.view.default(mean_4, [8, 1, -1]);  mean_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:86, code: y = self.conv(y)
    convolution_23: "f32[8, 1, 256]" = torch.ops.aten.convolution.default(view_8, arg91_1, None, [1], [2], [1], False, [0], 1);  view_8 = arg91_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_21: "f32[8, 1, 256]" = torch.ops.aten.sigmoid.default(convolution_23);  convolution_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:90, code: y = self.gate(y).view(x.shape[0], -1, 1, 1)
    view_9: "f32[8, 256, 1, 1]" = torch.ops.aten.view.default(sigmoid_21, [8, -1, 1, 1]);  sigmoid_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:91, code: return x * y.expand_as(x)
    expand_4: "f32[8, 256, 16, 16]" = torch.ops.aten.expand.default(view_9, [8, 256, 16, 16]);  view_9 = None
    mul_78: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_77, expand_4);  mul_77 = expand_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_24: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(mul_78, arg92_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_78 = arg92_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_38: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg147_1, torch.float32);  arg147_1 = None
    convert_element_type_39: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg148_1, torch.float32);  arg148_1 = None
    add_42: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_39, 1e-05);  convert_element_type_39 = None
    sqrt_19: "f32[1024]" = torch.ops.aten.sqrt.default(add_42);  add_42 = None
    reciprocal_19: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_19);  sqrt_19 = None
    mul_79: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_19, 1);  reciprocal_19 = None
    unsqueeze_152: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_38, -1);  convert_element_type_38 = None
    unsqueeze_153: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
    unsqueeze_154: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_79, -1);  mul_79 = None
    unsqueeze_155: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
    sub_19: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_153);  convolution_24 = unsqueeze_153 = None
    mul_80: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_19, unsqueeze_155);  sub_19 = unsqueeze_155 = None
    unsqueeze_156: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg38_1, -1);  arg38_1 = None
    unsqueeze_157: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
    mul_81: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_80, unsqueeze_157);  mul_80 = unsqueeze_157 = None
    unsqueeze_158: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg39_1, -1);  arg39_1 = None
    unsqueeze_159: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
    add_43: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_81, unsqueeze_159);  mul_81 = unsqueeze_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_25: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(mul_69, arg93_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  mul_69 = arg93_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_40: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg149_1, torch.float32);  arg149_1 = None
    convert_element_type_41: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg150_1, torch.float32);  arg150_1 = None
    add_44: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_41, 1e-05);  convert_element_type_41 = None
    sqrt_20: "f32[1024]" = torch.ops.aten.sqrt.default(add_44);  add_44 = None
    reciprocal_20: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_20);  sqrt_20 = None
    mul_82: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_20, 1);  reciprocal_20 = None
    unsqueeze_160: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_40, -1);  convert_element_type_40 = None
    unsqueeze_161: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
    unsqueeze_162: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_82, -1);  mul_82 = None
    unsqueeze_163: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
    sub_20: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_161);  convolution_25 = unsqueeze_161 = None
    mul_83: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_20, unsqueeze_163);  sub_20 = unsqueeze_163 = None
    unsqueeze_164: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg40_1, -1);  arg40_1 = None
    unsqueeze_165: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
    mul_84: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_83, unsqueeze_165);  mul_83 = unsqueeze_165 = None
    unsqueeze_166: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg41_1, -1);  arg41_1 = None
    unsqueeze_167: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
    add_45: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_84, unsqueeze_167);  mul_84 = unsqueeze_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    add_46: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_43, add_45);  add_43 = add_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    sigmoid_22: "f32[8, 1024, 16, 16]" = torch.ops.aten.sigmoid.default(add_46)
    mul_85: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(add_46, sigmoid_22);  add_46 = sigmoid_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_26: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(mul_85, arg94_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg94_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_42: "f32[256]" = torch.ops.prims.convert_element_type.default(arg151_1, torch.float32);  arg151_1 = None
    convert_element_type_43: "f32[256]" = torch.ops.prims.convert_element_type.default(arg152_1, torch.float32);  arg152_1 = None
    add_47: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_43, 1e-05);  convert_element_type_43 = None
    sqrt_21: "f32[256]" = torch.ops.aten.sqrt.default(add_47);  add_47 = None
    reciprocal_21: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_21);  sqrt_21 = None
    mul_86: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_21, 1);  reciprocal_21 = None
    unsqueeze_168: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_42, -1);  convert_element_type_42 = None
    unsqueeze_169: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
    unsqueeze_170: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_86, -1);  mul_86 = None
    unsqueeze_171: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
    sub_21: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_169);  convolution_26 = unsqueeze_169 = None
    mul_87: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_21, unsqueeze_171);  sub_21 = unsqueeze_171 = None
    unsqueeze_172: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg42_1, -1);  arg42_1 = None
    unsqueeze_173: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
    mul_88: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_87, unsqueeze_173);  mul_87 = unsqueeze_173 = None
    unsqueeze_174: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg43_1, -1);  arg43_1 = None
    unsqueeze_175: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
    add_48: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_88, unsqueeze_175);  mul_88 = unsqueeze_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_23: "f32[8, 256, 16, 16]" = torch.ops.aten.sigmoid.default(add_48)
    mul_89: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_48, sigmoid_23);  add_48 = sigmoid_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:177, code: q = self.q(x)
    convolution_27: "f32[8, 128, 16, 16]" = torch.ops.aten.convolution.default(mul_89, arg95_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg95_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:179, code: q = q.reshape(
    view_10: "f32[64, 16, 2, 8, 2, 8]" = torch.ops.aten.view.default(convolution_27, [-1, 16, 2, 8, 2, 8]);  convolution_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:181, code: num_h_blocks, self.block_size_ds, num_w_blocks, self.block_size_ds).permute(0, 1, 3, 5, 2, 4)
    permute: "f32[64, 16, 8, 8, 2, 2]" = torch.ops.aten.permute.default(view_10, [0, 1, 3, 5, 2, 4]);  view_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:183, code: q = q.reshape(B * self.num_heads, self.dim_head_qk, -1, num_blocks).transpose(1, 3)
    clone: "f32[64, 16, 8, 8, 2, 2]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
    view_11: "f32[64, 16, 64, 4]" = torch.ops.aten.view.default(clone, [64, 16, 64, 4]);  clone = None
    permute_1: "f32[64, 4, 64, 16]" = torch.ops.aten.permute.default(view_11, [0, 3, 2, 1]);  view_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:186, code: kv = self.kv(x)
    convolution_28: "f32[8, 384, 16, 16]" = torch.ops.aten.convolution.default(mul_89, arg96_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_89 = arg96_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:190, code: kv = F.pad(kv, [self.halo_size, self.halo_size, self.halo_size, self.halo_size])
    constant_pad_nd: "f32[8, 384, 20, 20]" = torch.ops.aten.constant_pad_nd.default(convolution_28, [2, 2, 2, 2], 0.0);  convolution_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:191, code: kv = kv.unfold(2, self.win_size, self.block_size).unfold(3, self.win_size, self.block_size).reshape(
    unfold: "f32[8, 384, 2, 20, 12]" = torch.ops.aten.unfold.default(constant_pad_nd, 2, 12, 8);  constant_pad_nd = None
    unfold_1: "f32[8, 384, 2, 2, 12, 12]" = torch.ops.aten.unfold.default(unfold, 3, 12, 8);  unfold = None
    clone_1: "f32[8, 384, 2, 2, 12, 12]" = torch.ops.aten.clone.default(unfold_1, memory_format = torch.contiguous_format);  unfold_1 = None
    view_12: "f32[64, 48, 4, 144]" = torch.ops.aten.view.default(clone_1, [64, 48, 4, 144]);  clone_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:192, code: B * self.num_heads, self.dim_head_qk + self.dim_head_v, num_blocks, -1).permute(0, 2, 3, 1)
    permute_2: "f32[64, 4, 144, 48]" = torch.ops.aten.permute.default(view_12, [0, 2, 3, 1]);  view_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:193, code: k, v = torch.split(kv, [self.dim_head_qk, self.dim_head_v], dim=-1)
    split_with_sizes = torch.ops.aten.split_with_sizes.default(permute_2, [16, 32], 3);  permute_2 = None
    getitem_2: "f32[64, 4, 144, 16]" = split_with_sizes[0]
    getitem_3: "f32[64, 4, 144, 32]" = split_with_sizes[1];  split_with_sizes = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:199, code: attn = (q @ k.transpose(-1, -2)) * self.scale + self.pos_embed(q)
    permute_3: "f32[64, 4, 16, 144]" = torch.ops.aten.permute.default(getitem_2, [0, 1, 3, 2]);  getitem_2 = None
    expand_5: "f32[64, 4, 64, 16]" = torch.ops.aten.expand.default(permute_1, [64, 4, 64, 16])
    clone_2: "f32[64, 4, 64, 16]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
    view_13: "f32[256, 64, 16]" = torch.ops.aten.view.default(clone_2, [256, 64, 16]);  clone_2 = None
    expand_6: "f32[64, 4, 16, 144]" = torch.ops.aten.expand.default(permute_3, [64, 4, 16, 144]);  permute_3 = None
    clone_3: "f32[64, 4, 16, 144]" = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format);  expand_6 = None
    view_14: "f32[256, 16, 144]" = torch.ops.aten.view.default(clone_3, [256, 16, 144]);  clone_3 = None
    bmm: "f32[256, 64, 144]" = torch.ops.aten.bmm.default(view_13, view_14);  view_13 = view_14 = None
    view_15: "f32[64, 4, 64, 144]" = torch.ops.aten.view.default(bmm, [64, 4, 64, 144]);  bmm = None
    mul_90: "f32[64, 4, 64, 144]" = torch.ops.aten.mul.Tensor(view_15, 0.25);  view_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:85, code: q = q.reshape(-1, self.block_size, self.block_size, self.dim_head)
    clone_4: "f32[64, 4, 64, 16]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
    view_16: "f32[256, 8, 8, 16]" = torch.ops.aten.view.default(clone_4, [256, 8, 8, 16]);  clone_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:45, code: x = (q @ rel_k.transpose(-1, -2))
    permute_4: "f32[16, 23]" = torch.ops.aten.permute.default(arg44_1, [1, 0]);  arg44_1 = None
    view_17: "f32[16384, 16]" = torch.ops.aten.view.default(view_16, [16384, 16])
    mm: "f32[16384, 23]" = torch.ops.aten.mm.default(view_17, permute_4);  view_17 = permute_4 = None
    view_18: "f32[256, 8, 8, 23]" = torch.ops.aten.view.default(mm, [256, 8, 8, 23]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:46, code: x = x.reshape(-1, W, rel_size)
    view_19: "f32[2048, 8, 23]" = torch.ops.aten.view.default(view_18, [-1, 8, 23]);  view_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:49, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    constant_pad_nd_1: "f32[2048, 8, 24]" = torch.ops.aten.constant_pad_nd.default(view_19, [0, 1], 0.0);  view_19 = None
    view_20: "f32[2048, 192]" = torch.ops.aten.view.default(constant_pad_nd_1, [2048, 192]);  constant_pad_nd_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:50, code: x_pad = F.pad(x_pad, [0, rel_size - W])
    constant_pad_nd_2: "f32[2048, 207]" = torch.ops.aten.constant_pad_nd.default(view_20, [0, 15], 0.0);  view_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:53, code: x_pad = x_pad.reshape(-1, W + 1, rel_size)
    view_21: "f32[2048, 9, 23]" = torch.ops.aten.view.default(constant_pad_nd_2, [-1, 9, 23]);  constant_pad_nd_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:54, code: x = x_pad[:, :W, win_size - 1:]
    slice_1: "f32[2048, 9, 23]" = torch.ops.aten.slice.Tensor(view_21, 0, 0, 9223372036854775807);  view_21 = None
    slice_2: "f32[2048, 8, 23]" = torch.ops.aten.slice.Tensor(slice_1, 1, 0, 8);  slice_1 = None
    slice_3: "f32[2048, 8, 12]" = torch.ops.aten.slice.Tensor(slice_2, 2, 11, 9223372036854775807);  slice_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:57, code: x = x.reshape(B, H, 1, W, win_size).expand(-1, -1, win_size, -1, -1)
    view_22: "f32[256, 8, 1, 8, 12]" = torch.ops.aten.view.default(slice_3, [256, 8, 1, 8, 12]);  slice_3 = None
    expand_7: "f32[256, 8, 12, 8, 12]" = torch.ops.aten.expand.default(view_22, [-1, -1, 12, -1, -1]);  view_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:58, code: return x.permute(permute_mask)
    permute_5: "f32[256, 8, 8, 12, 12]" = torch.ops.aten.permute.default(expand_7, [0, 1, 3, 2, 4]);  expand_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:89, code: q = q.transpose(1, 2)
    permute_6: "f32[256, 8, 8, 16]" = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:45, code: x = (q @ rel_k.transpose(-1, -2))
    permute_7: "f32[16, 23]" = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
    clone_5: "f32[256, 8, 8, 16]" = torch.ops.aten.clone.default(permute_6, memory_format = torch.contiguous_format);  permute_6 = None
    view_23: "f32[16384, 16]" = torch.ops.aten.view.default(clone_5, [16384, 16]);  clone_5 = None
    mm_1: "f32[16384, 23]" = torch.ops.aten.mm.default(view_23, permute_7);  view_23 = permute_7 = None
    view_24: "f32[256, 8, 8, 23]" = torch.ops.aten.view.default(mm_1, [256, 8, 8, 23]);  mm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:46, code: x = x.reshape(-1, W, rel_size)
    view_25: "f32[2048, 8, 23]" = torch.ops.aten.view.default(view_24, [-1, 8, 23]);  view_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:49, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    constant_pad_nd_3: "f32[2048, 8, 24]" = torch.ops.aten.constant_pad_nd.default(view_25, [0, 1], 0.0);  view_25 = None
    view_26: "f32[2048, 192]" = torch.ops.aten.view.default(constant_pad_nd_3, [2048, 192]);  constant_pad_nd_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:50, code: x_pad = F.pad(x_pad, [0, rel_size - W])
    constant_pad_nd_4: "f32[2048, 207]" = torch.ops.aten.constant_pad_nd.default(view_26, [0, 15], 0.0);  view_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:53, code: x_pad = x_pad.reshape(-1, W + 1, rel_size)
    view_27: "f32[2048, 9, 23]" = torch.ops.aten.view.default(constant_pad_nd_4, [-1, 9, 23]);  constant_pad_nd_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:54, code: x = x_pad[:, :W, win_size - 1:]
    slice_4: "f32[2048, 9, 23]" = torch.ops.aten.slice.Tensor(view_27, 0, 0, 9223372036854775807);  view_27 = None
    slice_5: "f32[2048, 8, 23]" = torch.ops.aten.slice.Tensor(slice_4, 1, 0, 8);  slice_4 = None
    slice_6: "f32[2048, 8, 12]" = torch.ops.aten.slice.Tensor(slice_5, 2, 11, 9223372036854775807);  slice_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:57, code: x = x.reshape(B, H, 1, W, win_size).expand(-1, -1, win_size, -1, -1)
    view_28: "f32[256, 8, 1, 8, 12]" = torch.ops.aten.view.default(slice_6, [256, 8, 1, 8, 12]);  slice_6 = None
    expand_8: "f32[256, 8, 12, 8, 12]" = torch.ops.aten.expand.default(view_28, [-1, -1, 12, -1, -1]);  view_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:58, code: return x.permute(permute_mask)
    permute_8: "f32[256, 8, 8, 12, 12]" = torch.ops.aten.permute.default(expand_8, [0, 3, 1, 4, 2]);  expand_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:92, code: rel_logits = rel_logits_h + rel_logits_w
    add_49: "f32[256, 8, 8, 12, 12]" = torch.ops.aten.add.Tensor(permute_8, permute_5);  permute_8 = permute_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:93, code: rel_logits = rel_logits.reshape(B, BB, HW, -1)
    clone_6: "f32[256, 8, 8, 12, 12]" = torch.ops.aten.clone.default(add_49, memory_format = torch.contiguous_format);  add_49 = None
    view_29: "f32[64, 4, 64, 144]" = torch.ops.aten.view.default(clone_6, [64, 4, 64, 144]);  clone_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:199, code: attn = (q @ k.transpose(-1, -2)) * self.scale + self.pos_embed(q)
    add_50: "f32[64, 4, 64, 144]" = torch.ops.aten.add.Tensor(mul_90, view_29);  mul_90 = view_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:201, code: attn = attn.softmax(dim=-1)
    amax: "f32[64, 4, 64, 1]" = torch.ops.aten.amax.default(add_50, [-1], True)
    sub_22: "f32[64, 4, 64, 144]" = torch.ops.aten.sub.Tensor(add_50, amax);  add_50 = amax = None
    exp: "f32[64, 4, 64, 144]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
    sum_1: "f32[64, 4, 64, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div: "f32[64, 4, 64, 144]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:203, code: out = (attn @ v).transpose(1, 3)  # B * num_heads, dim_head_v, block_size ** 2, num_blocks
    expand_9: "f32[64, 4, 64, 144]" = torch.ops.aten.expand.default(div, [64, 4, 64, 144]);  div = None
    view_30: "f32[256, 64, 144]" = torch.ops.aten.view.default(expand_9, [256, 64, 144]);  expand_9 = None
    expand_10: "f32[64, 4, 144, 32]" = torch.ops.aten.expand.default(getitem_3, [64, 4, 144, 32]);  getitem_3 = None
    clone_7: "f32[64, 4, 144, 32]" = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
    view_31: "f32[256, 144, 32]" = torch.ops.aten.view.default(clone_7, [256, 144, 32]);  clone_7 = None
    bmm_1: "f32[256, 64, 32]" = torch.ops.aten.bmm.default(view_30, view_31);  view_30 = view_31 = None
    view_32: "f32[64, 4, 64, 32]" = torch.ops.aten.view.default(bmm_1, [64, 4, 64, 32]);  bmm_1 = None
    permute_9: "f32[64, 32, 64, 4]" = torch.ops.aten.permute.default(view_32, [0, 3, 2, 1]);  view_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:205, code: out = out.reshape(-1, self.block_size_ds, self.block_size_ds, num_h_blocks, num_w_blocks)
    clone_8: "f32[64, 32, 64, 4]" = torch.ops.aten.clone.default(permute_9, memory_format = torch.contiguous_format);  permute_9 = None
    view_33: "f32[2048, 8, 8, 2, 2]" = torch.ops.aten.view.default(clone_8, [2048, 8, 8, 2, 2]);  clone_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:206, code: out = out.permute(0, 3, 1, 4, 2).contiguous().view(
    permute_10: "f32[2048, 2, 8, 2, 8]" = torch.ops.aten.permute.default(view_33, [0, 3, 1, 4, 2]);  view_33 = None
    clone_9: "f32[2048, 2, 8, 2, 8]" = torch.ops.aten.clone.default(permute_10, memory_format = torch.contiguous_format);  permute_10 = None
    view_34: "f32[8, 256, 16, 16]" = torch.ops.aten.view.default(clone_9, [8, 256, 16, 16]);  clone_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_44: "f32[256]" = torch.ops.prims.convert_element_type.default(arg153_1, torch.float32);  arg153_1 = None
    convert_element_type_45: "f32[256]" = torch.ops.prims.convert_element_type.default(arg154_1, torch.float32);  arg154_1 = None
    add_51: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_45, 1e-05);  convert_element_type_45 = None
    sqrt_22: "f32[256]" = torch.ops.aten.sqrt.default(add_51);  add_51 = None
    reciprocal_22: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_22);  sqrt_22 = None
    mul_91: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_22, 1);  reciprocal_22 = None
    unsqueeze_176: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_44, -1);  convert_element_type_44 = None
    unsqueeze_177: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
    unsqueeze_178: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_91, -1);  mul_91 = None
    unsqueeze_179: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
    sub_23: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(view_34, unsqueeze_177);  view_34 = unsqueeze_177 = None
    mul_92: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_23, unsqueeze_179);  sub_23 = unsqueeze_179 = None
    unsqueeze_180: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg46_1, -1);  arg46_1 = None
    unsqueeze_181: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
    mul_93: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_92, unsqueeze_181);  mul_92 = unsqueeze_181 = None
    unsqueeze_182: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg47_1, -1);  arg47_1 = None
    unsqueeze_183: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
    add_52: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_93, unsqueeze_183);  mul_93 = unsqueeze_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_24: "f32[8, 256, 16, 16]" = torch.ops.aten.sigmoid.default(add_52)
    mul_94: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_52, sigmoid_24);  add_52 = sigmoid_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_29: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(mul_94, arg97_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_94 = arg97_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_46: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg155_1, torch.float32);  arg155_1 = None
    convert_element_type_47: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg156_1, torch.float32);  arg156_1 = None
    add_53: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_47, 1e-05);  convert_element_type_47 = None
    sqrt_23: "f32[1024]" = torch.ops.aten.sqrt.default(add_53);  add_53 = None
    reciprocal_23: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_23);  sqrt_23 = None
    mul_95: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_23, 1);  reciprocal_23 = None
    unsqueeze_184: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_46, -1);  convert_element_type_46 = None
    unsqueeze_185: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
    unsqueeze_186: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_95, -1);  mul_95 = None
    unsqueeze_187: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
    sub_24: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_185);  convolution_29 = unsqueeze_185 = None
    mul_96: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_24, unsqueeze_187);  sub_24 = unsqueeze_187 = None
    unsqueeze_188: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg48_1, -1);  arg48_1 = None
    unsqueeze_189: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
    mul_97: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_96, unsqueeze_189);  mul_96 = unsqueeze_189 = None
    unsqueeze_190: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg49_1, -1);  arg49_1 = None
    unsqueeze_191: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
    add_54: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_97, unsqueeze_191);  mul_97 = unsqueeze_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:888, code: x = x + self.shortcut(shortcut)
    add_55: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_54, mul_85);  add_54 = mul_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:889, code: return self.act(x)
    sigmoid_25: "f32[8, 1024, 16, 16]" = torch.ops.aten.sigmoid.default(add_55)
    mul_98: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(add_55, sigmoid_25);  add_55 = sigmoid_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_30: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(mul_98, arg98_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg98_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_48: "f32[512]" = torch.ops.prims.convert_element_type.default(arg157_1, torch.float32);  arg157_1 = None
    convert_element_type_49: "f32[512]" = torch.ops.prims.convert_element_type.default(arg158_1, torch.float32);  arg158_1 = None
    add_56: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_49, 1e-05);  convert_element_type_49 = None
    sqrt_24: "f32[512]" = torch.ops.aten.sqrt.default(add_56);  add_56 = None
    reciprocal_24: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_24);  sqrt_24 = None
    mul_99: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_24, 1);  reciprocal_24 = None
    unsqueeze_192: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_48, -1);  convert_element_type_48 = None
    unsqueeze_193: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
    unsqueeze_194: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_99, -1);  mul_99 = None
    unsqueeze_195: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
    sub_25: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_193);  convolution_30 = unsqueeze_193 = None
    mul_100: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_25, unsqueeze_195);  sub_25 = unsqueeze_195 = None
    unsqueeze_196: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg50_1, -1);  arg50_1 = None
    unsqueeze_197: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
    mul_101: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_100, unsqueeze_197);  mul_100 = unsqueeze_197 = None
    unsqueeze_198: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg51_1, -1);  arg51_1 = None
    unsqueeze_199: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
    add_57: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_101, unsqueeze_199);  mul_101 = unsqueeze_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_26: "f32[8, 512, 16, 16]" = torch.ops.aten.sigmoid.default(add_57)
    mul_102: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(add_57, sigmoid_26);  add_57 = sigmoid_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:177, code: q = self.q(x)
    convolution_31: "f32[8, 128, 8, 8]" = torch.ops.aten.convolution.default(mul_102, arg99_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  arg99_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:179, code: q = q.reshape(
    view_35: "f32[64, 16, 2, 4, 2, 4]" = torch.ops.aten.view.default(convolution_31, [-1, 16, 2, 4, 2, 4]);  convolution_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:181, code: num_h_blocks, self.block_size_ds, num_w_blocks, self.block_size_ds).permute(0, 1, 3, 5, 2, 4)
    permute_11: "f32[64, 16, 4, 4, 2, 2]" = torch.ops.aten.permute.default(view_35, [0, 1, 3, 5, 2, 4]);  view_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:183, code: q = q.reshape(B * self.num_heads, self.dim_head_qk, -1, num_blocks).transpose(1, 3)
    clone_10: "f32[64, 16, 4, 4, 2, 2]" = torch.ops.aten.clone.default(permute_11, memory_format = torch.contiguous_format);  permute_11 = None
    view_36: "f32[64, 16, 16, 4]" = torch.ops.aten.view.default(clone_10, [64, 16, 16, 4]);  clone_10 = None
    permute_12: "f32[64, 4, 16, 16]" = torch.ops.aten.permute.default(view_36, [0, 3, 2, 1]);  view_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:186, code: kv = self.kv(x)
    convolution_32: "f32[8, 640, 16, 16]" = torch.ops.aten.convolution.default(mul_102, arg100_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_102 = arg100_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:190, code: kv = F.pad(kv, [self.halo_size, self.halo_size, self.halo_size, self.halo_size])
    constant_pad_nd_5: "f32[8, 640, 20, 20]" = torch.ops.aten.constant_pad_nd.default(convolution_32, [2, 2, 2, 2], 0.0);  convolution_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:191, code: kv = kv.unfold(2, self.win_size, self.block_size).unfold(3, self.win_size, self.block_size).reshape(
    unfold_2: "f32[8, 640, 2, 20, 12]" = torch.ops.aten.unfold.default(constant_pad_nd_5, 2, 12, 8);  constant_pad_nd_5 = None
    unfold_3: "f32[8, 640, 2, 2, 12, 12]" = torch.ops.aten.unfold.default(unfold_2, 3, 12, 8);  unfold_2 = None
    clone_11: "f32[8, 640, 2, 2, 12, 12]" = torch.ops.aten.clone.default(unfold_3, memory_format = torch.contiguous_format);  unfold_3 = None
    view_37: "f32[64, 80, 4, 144]" = torch.ops.aten.view.default(clone_11, [64, 80, 4, 144]);  clone_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:192, code: B * self.num_heads, self.dim_head_qk + self.dim_head_v, num_blocks, -1).permute(0, 2, 3, 1)
    permute_13: "f32[64, 4, 144, 80]" = torch.ops.aten.permute.default(view_37, [0, 2, 3, 1]);  view_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:193, code: k, v = torch.split(kv, [self.dim_head_qk, self.dim_head_v], dim=-1)
    split_with_sizes_1 = torch.ops.aten.split_with_sizes.default(permute_13, [16, 64], 3);  permute_13 = None
    getitem_4: "f32[64, 4, 144, 16]" = split_with_sizes_1[0]
    getitem_5: "f32[64, 4, 144, 64]" = split_with_sizes_1[1];  split_with_sizes_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:199, code: attn = (q @ k.transpose(-1, -2)) * self.scale + self.pos_embed(q)
    permute_14: "f32[64, 4, 16, 144]" = torch.ops.aten.permute.default(getitem_4, [0, 1, 3, 2]);  getitem_4 = None
    expand_11: "f32[64, 4, 16, 16]" = torch.ops.aten.expand.default(permute_12, [64, 4, 16, 16])
    clone_12: "f32[64, 4, 16, 16]" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
    view_38: "f32[256, 16, 16]" = torch.ops.aten.view.default(clone_12, [256, 16, 16]);  clone_12 = None
    expand_12: "f32[64, 4, 16, 144]" = torch.ops.aten.expand.default(permute_14, [64, 4, 16, 144]);  permute_14 = None
    clone_13: "f32[64, 4, 16, 144]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
    view_39: "f32[256, 16, 144]" = torch.ops.aten.view.default(clone_13, [256, 16, 144]);  clone_13 = None
    bmm_2: "f32[256, 16, 144]" = torch.ops.aten.bmm.default(view_38, view_39);  view_38 = view_39 = None
    view_40: "f32[64, 4, 16, 144]" = torch.ops.aten.view.default(bmm_2, [64, 4, 16, 144]);  bmm_2 = None
    mul_103: "f32[64, 4, 16, 144]" = torch.ops.aten.mul.Tensor(view_40, 0.25);  view_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:85, code: q = q.reshape(-1, self.block_size, self.block_size, self.dim_head)
    clone_14: "f32[64, 4, 16, 16]" = torch.ops.aten.clone.default(permute_12, memory_format = torch.contiguous_format);  permute_12 = None
    view_41: "f32[256, 4, 4, 16]" = torch.ops.aten.view.default(clone_14, [256, 4, 4, 16]);  clone_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:45, code: x = (q @ rel_k.transpose(-1, -2))
    permute_15: "f32[16, 23]" = torch.ops.aten.permute.default(arg52_1, [1, 0]);  arg52_1 = None
    view_42: "f32[4096, 16]" = torch.ops.aten.view.default(view_41, [4096, 16])
    mm_2: "f32[4096, 23]" = torch.ops.aten.mm.default(view_42, permute_15);  view_42 = permute_15 = None
    view_43: "f32[256, 4, 4, 23]" = torch.ops.aten.view.default(mm_2, [256, 4, 4, 23]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:46, code: x = x.reshape(-1, W, rel_size)
    view_44: "f32[1024, 4, 23]" = torch.ops.aten.view.default(view_43, [-1, 4, 23]);  view_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:49, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    constant_pad_nd_6: "f32[1024, 4, 24]" = torch.ops.aten.constant_pad_nd.default(view_44, [0, 1], 0.0);  view_44 = None
    view_45: "f32[1024, 96]" = torch.ops.aten.view.default(constant_pad_nd_6, [1024, 96]);  constant_pad_nd_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:50, code: x_pad = F.pad(x_pad, [0, rel_size - W])
    constant_pad_nd_7: "f32[1024, 115]" = torch.ops.aten.constant_pad_nd.default(view_45, [0, 19], 0.0);  view_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:53, code: x_pad = x_pad.reshape(-1, W + 1, rel_size)
    view_46: "f32[1024, 5, 23]" = torch.ops.aten.view.default(constant_pad_nd_7, [-1, 5, 23]);  constant_pad_nd_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:54, code: x = x_pad[:, :W, win_size - 1:]
    slice_7: "f32[1024, 5, 23]" = torch.ops.aten.slice.Tensor(view_46, 0, 0, 9223372036854775807);  view_46 = None
    slice_8: "f32[1024, 4, 23]" = torch.ops.aten.slice.Tensor(slice_7, 1, 0, 4);  slice_7 = None
    slice_9: "f32[1024, 4, 12]" = torch.ops.aten.slice.Tensor(slice_8, 2, 11, 9223372036854775807);  slice_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:57, code: x = x.reshape(B, H, 1, W, win_size).expand(-1, -1, win_size, -1, -1)
    view_47: "f32[256, 4, 1, 4, 12]" = torch.ops.aten.view.default(slice_9, [256, 4, 1, 4, 12]);  slice_9 = None
    expand_13: "f32[256, 4, 12, 4, 12]" = torch.ops.aten.expand.default(view_47, [-1, -1, 12, -1, -1]);  view_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:58, code: return x.permute(permute_mask)
    permute_16: "f32[256, 4, 4, 12, 12]" = torch.ops.aten.permute.default(expand_13, [0, 1, 3, 2, 4]);  expand_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:89, code: q = q.transpose(1, 2)
    permute_17: "f32[256, 4, 4, 16]" = torch.ops.aten.permute.default(view_41, [0, 2, 1, 3]);  view_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:45, code: x = (q @ rel_k.transpose(-1, -2))
    permute_18: "f32[16, 23]" = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
    clone_15: "f32[256, 4, 4, 16]" = torch.ops.aten.clone.default(permute_17, memory_format = torch.contiguous_format);  permute_17 = None
    view_48: "f32[4096, 16]" = torch.ops.aten.view.default(clone_15, [4096, 16]);  clone_15 = None
    mm_3: "f32[4096, 23]" = torch.ops.aten.mm.default(view_48, permute_18);  view_48 = permute_18 = None
    view_49: "f32[256, 4, 4, 23]" = torch.ops.aten.view.default(mm_3, [256, 4, 4, 23]);  mm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:46, code: x = x.reshape(-1, W, rel_size)
    view_50: "f32[1024, 4, 23]" = torch.ops.aten.view.default(view_49, [-1, 4, 23]);  view_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:49, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    constant_pad_nd_8: "f32[1024, 4, 24]" = torch.ops.aten.constant_pad_nd.default(view_50, [0, 1], 0.0);  view_50 = None
    view_51: "f32[1024, 96]" = torch.ops.aten.view.default(constant_pad_nd_8, [1024, 96]);  constant_pad_nd_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:50, code: x_pad = F.pad(x_pad, [0, rel_size - W])
    constant_pad_nd_9: "f32[1024, 115]" = torch.ops.aten.constant_pad_nd.default(view_51, [0, 19], 0.0);  view_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:53, code: x_pad = x_pad.reshape(-1, W + 1, rel_size)
    view_52: "f32[1024, 5, 23]" = torch.ops.aten.view.default(constant_pad_nd_9, [-1, 5, 23]);  constant_pad_nd_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:54, code: x = x_pad[:, :W, win_size - 1:]
    slice_10: "f32[1024, 5, 23]" = torch.ops.aten.slice.Tensor(view_52, 0, 0, 9223372036854775807);  view_52 = None
    slice_11: "f32[1024, 4, 23]" = torch.ops.aten.slice.Tensor(slice_10, 1, 0, 4);  slice_10 = None
    slice_12: "f32[1024, 4, 12]" = torch.ops.aten.slice.Tensor(slice_11, 2, 11, 9223372036854775807);  slice_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:57, code: x = x.reshape(B, H, 1, W, win_size).expand(-1, -1, win_size, -1, -1)
    view_53: "f32[256, 4, 1, 4, 12]" = torch.ops.aten.view.default(slice_12, [256, 4, 1, 4, 12]);  slice_12 = None
    expand_14: "f32[256, 4, 12, 4, 12]" = torch.ops.aten.expand.default(view_53, [-1, -1, 12, -1, -1]);  view_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:58, code: return x.permute(permute_mask)
    permute_19: "f32[256, 4, 4, 12, 12]" = torch.ops.aten.permute.default(expand_14, [0, 3, 1, 4, 2]);  expand_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:92, code: rel_logits = rel_logits_h + rel_logits_w
    add_58: "f32[256, 4, 4, 12, 12]" = torch.ops.aten.add.Tensor(permute_19, permute_16);  permute_19 = permute_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:93, code: rel_logits = rel_logits.reshape(B, BB, HW, -1)
    clone_16: "f32[256, 4, 4, 12, 12]" = torch.ops.aten.clone.default(add_58, memory_format = torch.contiguous_format);  add_58 = None
    view_54: "f32[64, 4, 16, 144]" = torch.ops.aten.view.default(clone_16, [64, 4, 16, 144]);  clone_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:199, code: attn = (q @ k.transpose(-1, -2)) * self.scale + self.pos_embed(q)
    add_59: "f32[64, 4, 16, 144]" = torch.ops.aten.add.Tensor(mul_103, view_54);  mul_103 = view_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:201, code: attn = attn.softmax(dim=-1)
    amax_1: "f32[64, 4, 16, 1]" = torch.ops.aten.amax.default(add_59, [-1], True)
    sub_26: "f32[64, 4, 16, 144]" = torch.ops.aten.sub.Tensor(add_59, amax_1);  add_59 = amax_1 = None
    exp_1: "f32[64, 4, 16, 144]" = torch.ops.aten.exp.default(sub_26);  sub_26 = None
    sum_2: "f32[64, 4, 16, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_1: "f32[64, 4, 16, 144]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:203, code: out = (attn @ v).transpose(1, 3)  # B * num_heads, dim_head_v, block_size ** 2, num_blocks
    expand_15: "f32[64, 4, 16, 144]" = torch.ops.aten.expand.default(div_1, [64, 4, 16, 144]);  div_1 = None
    view_55: "f32[256, 16, 144]" = torch.ops.aten.view.default(expand_15, [256, 16, 144]);  expand_15 = None
    expand_16: "f32[64, 4, 144, 64]" = torch.ops.aten.expand.default(getitem_5, [64, 4, 144, 64]);  getitem_5 = None
    clone_17: "f32[64, 4, 144, 64]" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
    view_56: "f32[256, 144, 64]" = torch.ops.aten.view.default(clone_17, [256, 144, 64]);  clone_17 = None
    bmm_3: "f32[256, 16, 64]" = torch.ops.aten.bmm.default(view_55, view_56);  view_55 = view_56 = None
    view_57: "f32[64, 4, 16, 64]" = torch.ops.aten.view.default(bmm_3, [64, 4, 16, 64]);  bmm_3 = None
    permute_20: "f32[64, 64, 16, 4]" = torch.ops.aten.permute.default(view_57, [0, 3, 2, 1]);  view_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:205, code: out = out.reshape(-1, self.block_size_ds, self.block_size_ds, num_h_blocks, num_w_blocks)
    clone_18: "f32[64, 64, 16, 4]" = torch.ops.aten.clone.default(permute_20, memory_format = torch.contiguous_format);  permute_20 = None
    view_58: "f32[4096, 4, 4, 2, 2]" = torch.ops.aten.view.default(clone_18, [4096, 4, 4, 2, 2]);  clone_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:206, code: out = out.permute(0, 3, 1, 4, 2).contiguous().view(
    permute_21: "f32[4096, 2, 4, 2, 4]" = torch.ops.aten.permute.default(view_58, [0, 3, 1, 4, 2]);  view_58 = None
    clone_19: "f32[4096, 2, 4, 2, 4]" = torch.ops.aten.clone.default(permute_21, memory_format = torch.contiguous_format);  permute_21 = None
    view_59: "f32[8, 512, 8, 8]" = torch.ops.aten.view.default(clone_19, [8, 512, 8, 8]);  clone_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_50: "f32[512]" = torch.ops.prims.convert_element_type.default(arg159_1, torch.float32);  arg159_1 = None
    convert_element_type_51: "f32[512]" = torch.ops.prims.convert_element_type.default(arg160_1, torch.float32);  arg160_1 = None
    add_60: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_51, 1e-05);  convert_element_type_51 = None
    sqrt_25: "f32[512]" = torch.ops.aten.sqrt.default(add_60);  add_60 = None
    reciprocal_25: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_25);  sqrt_25 = None
    mul_104: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_25, 1);  reciprocal_25 = None
    unsqueeze_200: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_50, -1);  convert_element_type_50 = None
    unsqueeze_201: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
    unsqueeze_202: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_104, -1);  mul_104 = None
    unsqueeze_203: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
    sub_27: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(view_59, unsqueeze_201);  view_59 = unsqueeze_201 = None
    mul_105: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_27, unsqueeze_203);  sub_27 = unsqueeze_203 = None
    unsqueeze_204: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg54_1, -1);  arg54_1 = None
    unsqueeze_205: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
    mul_106: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_105, unsqueeze_205);  mul_105 = unsqueeze_205 = None
    unsqueeze_206: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg55_1, -1);  arg55_1 = None
    unsqueeze_207: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
    add_61: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_106, unsqueeze_207);  mul_106 = unsqueeze_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_27: "f32[8, 512, 8, 8]" = torch.ops.aten.sigmoid.default(add_61)
    mul_107: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_61, sigmoid_27);  add_61 = sigmoid_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_33: "f32[8, 2048, 8, 8]" = torch.ops.aten.convolution.default(mul_107, arg101_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_107 = arg101_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_52: "f32[2048]" = torch.ops.prims.convert_element_type.default(arg161_1, torch.float32);  arg161_1 = None
    convert_element_type_53: "f32[2048]" = torch.ops.prims.convert_element_type.default(arg162_1, torch.float32);  arg162_1 = None
    add_62: "f32[2048]" = torch.ops.aten.add.Tensor(convert_element_type_53, 1e-05);  convert_element_type_53 = None
    sqrt_26: "f32[2048]" = torch.ops.aten.sqrt.default(add_62);  add_62 = None
    reciprocal_26: "f32[2048]" = torch.ops.aten.reciprocal.default(sqrt_26);  sqrt_26 = None
    mul_108: "f32[2048]" = torch.ops.aten.mul.Tensor(reciprocal_26, 1);  reciprocal_26 = None
    unsqueeze_208: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_52, -1);  convert_element_type_52 = None
    unsqueeze_209: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
    unsqueeze_210: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(mul_108, -1);  mul_108 = None
    unsqueeze_211: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
    sub_28: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_209);  convolution_33 = unsqueeze_209 = None
    mul_109: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(sub_28, unsqueeze_211);  sub_28 = unsqueeze_211 = None
    unsqueeze_212: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg56_1, -1);  arg56_1 = None
    unsqueeze_213: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
    mul_110: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(mul_109, unsqueeze_213);  mul_109 = unsqueeze_213 = None
    unsqueeze_214: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg57_1, -1);  arg57_1 = None
    unsqueeze_215: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
    add_63: "f32[8, 2048, 8, 8]" = torch.ops.aten.add.Tensor(mul_110, unsqueeze_215);  mul_110 = unsqueeze_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_34: "f32[8, 2048, 8, 8]" = torch.ops.aten.convolution.default(mul_98, arg102_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  mul_98 = arg102_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_54: "f32[2048]" = torch.ops.prims.convert_element_type.default(arg163_1, torch.float32);  arg163_1 = None
    convert_element_type_55: "f32[2048]" = torch.ops.prims.convert_element_type.default(arg164_1, torch.float32);  arg164_1 = None
    add_64: "f32[2048]" = torch.ops.aten.add.Tensor(convert_element_type_55, 1e-05);  convert_element_type_55 = None
    sqrt_27: "f32[2048]" = torch.ops.aten.sqrt.default(add_64);  add_64 = None
    reciprocal_27: "f32[2048]" = torch.ops.aten.reciprocal.default(sqrt_27);  sqrt_27 = None
    mul_111: "f32[2048]" = torch.ops.aten.mul.Tensor(reciprocal_27, 1);  reciprocal_27 = None
    unsqueeze_216: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_54, -1);  convert_element_type_54 = None
    unsqueeze_217: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
    unsqueeze_218: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(mul_111, -1);  mul_111 = None
    unsqueeze_219: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, -1);  unsqueeze_218 = None
    sub_29: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_217);  convolution_34 = unsqueeze_217 = None
    mul_112: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(sub_29, unsqueeze_219);  sub_29 = unsqueeze_219 = None
    unsqueeze_220: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg58_1, -1);  arg58_1 = None
    unsqueeze_221: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, -1);  unsqueeze_220 = None
    mul_113: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(mul_112, unsqueeze_221);  mul_112 = unsqueeze_221 = None
    unsqueeze_222: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg59_1, -1);  arg59_1 = None
    unsqueeze_223: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, -1);  unsqueeze_222 = None
    add_65: "f32[8, 2048, 8, 8]" = torch.ops.aten.add.Tensor(mul_113, unsqueeze_223);  mul_113 = unsqueeze_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:888, code: x = x + self.shortcut(shortcut)
    add_66: "f32[8, 2048, 8, 8]" = torch.ops.aten.add.Tensor(add_63, add_65);  add_63 = add_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:889, code: return self.act(x)
    sigmoid_28: "f32[8, 2048, 8, 8]" = torch.ops.aten.sigmoid.default(add_66)
    mul_114: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(add_66, sigmoid_28);  add_66 = sigmoid_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_35: "f32[8, 512, 8, 8]" = torch.ops.aten.convolution.default(mul_114, arg103_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg103_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_56: "f32[512]" = torch.ops.prims.convert_element_type.default(arg165_1, torch.float32);  arg165_1 = None
    convert_element_type_57: "f32[512]" = torch.ops.prims.convert_element_type.default(arg166_1, torch.float32);  arg166_1 = None
    add_67: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_57, 1e-05);  convert_element_type_57 = None
    sqrt_28: "f32[512]" = torch.ops.aten.sqrt.default(add_67);  add_67 = None
    reciprocal_28: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_28);  sqrt_28 = None
    mul_115: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_28, 1);  reciprocal_28 = None
    unsqueeze_224: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_56, -1);  convert_element_type_56 = None
    unsqueeze_225: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, -1);  unsqueeze_224 = None
    unsqueeze_226: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_115, -1);  mul_115 = None
    unsqueeze_227: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, -1);  unsqueeze_226 = None
    sub_30: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_225);  convolution_35 = unsqueeze_225 = None
    mul_116: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_30, unsqueeze_227);  sub_30 = unsqueeze_227 = None
    unsqueeze_228: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg60_1, -1);  arg60_1 = None
    unsqueeze_229: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, -1);  unsqueeze_228 = None
    mul_117: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_116, unsqueeze_229);  mul_116 = unsqueeze_229 = None
    unsqueeze_230: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg61_1, -1);  arg61_1 = None
    unsqueeze_231: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, -1);  unsqueeze_230 = None
    add_68: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_117, unsqueeze_231);  mul_117 = unsqueeze_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_29: "f32[8, 512, 8, 8]" = torch.ops.aten.sigmoid.default(add_68)
    mul_118: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_68, sigmoid_29);  add_68 = sigmoid_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:177, code: q = self.q(x)
    convolution_36: "f32[8, 128, 8, 8]" = torch.ops.aten.convolution.default(mul_118, arg104_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg104_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:179, code: q = q.reshape(
    view_60: "f32[64, 16, 1, 8, 1, 8]" = torch.ops.aten.view.default(convolution_36, [-1, 16, 1, 8, 1, 8]);  convolution_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:181, code: num_h_blocks, self.block_size_ds, num_w_blocks, self.block_size_ds).permute(0, 1, 3, 5, 2, 4)
    permute_22: "f32[64, 16, 8, 8, 1, 1]" = torch.ops.aten.permute.default(view_60, [0, 1, 3, 5, 2, 4]);  view_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:183, code: q = q.reshape(B * self.num_heads, self.dim_head_qk, -1, num_blocks).transpose(1, 3)
    view_61: "f32[64, 16, 64, 1]" = torch.ops.aten.view.default(permute_22, [64, 16, -1, 1]);  permute_22 = None
    permute_23: "f32[64, 1, 64, 16]" = torch.ops.aten.permute.default(view_61, [0, 3, 2, 1]);  view_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:186, code: kv = self.kv(x)
    convolution_37: "f32[8, 640, 8, 8]" = torch.ops.aten.convolution.default(mul_118, arg105_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_118 = arg105_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:190, code: kv = F.pad(kv, [self.halo_size, self.halo_size, self.halo_size, self.halo_size])
    constant_pad_nd_10: "f32[8, 640, 12, 12]" = torch.ops.aten.constant_pad_nd.default(convolution_37, [2, 2, 2, 2], 0.0);  convolution_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:191, code: kv = kv.unfold(2, self.win_size, self.block_size).unfold(3, self.win_size, self.block_size).reshape(
    unfold_4: "f32[8, 640, 1, 12, 12]" = torch.ops.aten.unfold.default(constant_pad_nd_10, 2, 12, 8);  constant_pad_nd_10 = None
    unfold_5: "f32[8, 640, 1, 1, 12, 12]" = torch.ops.aten.unfold.default(unfold_4, 3, 12, 8);  unfold_4 = None
    view_62: "f32[64, 80, 1, 144]" = torch.ops.aten.view.default(unfold_5, [64, 80, 1, -1]);  unfold_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:192, code: B * self.num_heads, self.dim_head_qk + self.dim_head_v, num_blocks, -1).permute(0, 2, 3, 1)
    permute_24: "f32[64, 1, 144, 80]" = torch.ops.aten.permute.default(view_62, [0, 2, 3, 1]);  view_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:193, code: k, v = torch.split(kv, [self.dim_head_qk, self.dim_head_v], dim=-1)
    split_with_sizes_2 = torch.ops.aten.split_with_sizes.default(permute_24, [16, 64], 3);  permute_24 = None
    getitem_6: "f32[64, 1, 144, 16]" = split_with_sizes_2[0]
    getitem_7: "f32[64, 1, 144, 64]" = split_with_sizes_2[1];  split_with_sizes_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:199, code: attn = (q @ k.transpose(-1, -2)) * self.scale + self.pos_embed(q)
    permute_25: "f32[64, 1, 16, 144]" = torch.ops.aten.permute.default(getitem_6, [0, 1, 3, 2]);  getitem_6 = None
    expand_17: "f32[64, 1, 64, 16]" = torch.ops.aten.expand.default(permute_23, [64, 1, 64, 16])
    view_63: "f32[64, 64, 16]" = torch.ops.aten.view.default(expand_17, [64, 64, 16]);  expand_17 = None
    expand_18: "f32[64, 1, 16, 144]" = torch.ops.aten.expand.default(permute_25, [64, 1, 16, 144]);  permute_25 = None
    view_64: "f32[64, 16, 144]" = torch.ops.aten.view.default(expand_18, [64, 16, 144]);  expand_18 = None
    bmm_4: "f32[64, 64, 144]" = torch.ops.aten.bmm.default(view_63, view_64);  view_63 = view_64 = None
    view_65: "f32[64, 1, 64, 144]" = torch.ops.aten.view.default(bmm_4, [64, 1, 64, 144]);  bmm_4 = None
    mul_119: "f32[64, 1, 64, 144]" = torch.ops.aten.mul.Tensor(view_65, 0.25);  view_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:85, code: q = q.reshape(-1, self.block_size, self.block_size, self.dim_head)
    view_66: "f32[64, 8, 8, 16]" = torch.ops.aten.view.default(permute_23, [64, 8, 8, 16]);  permute_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:45, code: x = (q @ rel_k.transpose(-1, -2))
    permute_26: "f32[16, 23]" = torch.ops.aten.permute.default(arg62_1, [1, 0]);  arg62_1 = None
    clone_20: "f32[64, 8, 8, 16]" = torch.ops.aten.clone.default(view_66, memory_format = torch.contiguous_format)
    view_67: "f32[4096, 16]" = torch.ops.aten.view.default(clone_20, [4096, 16]);  clone_20 = None
    mm_4: "f32[4096, 23]" = torch.ops.aten.mm.default(view_67, permute_26);  view_67 = permute_26 = None
    view_68: "f32[64, 8, 8, 23]" = torch.ops.aten.view.default(mm_4, [64, 8, 8, 23]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:46, code: x = x.reshape(-1, W, rel_size)
    view_69: "f32[512, 8, 23]" = torch.ops.aten.view.default(view_68, [-1, 8, 23]);  view_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:49, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    constant_pad_nd_11: "f32[512, 8, 24]" = torch.ops.aten.constant_pad_nd.default(view_69, [0, 1], 0.0);  view_69 = None
    view_70: "f32[512, 192]" = torch.ops.aten.view.default(constant_pad_nd_11, [512, 192]);  constant_pad_nd_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:50, code: x_pad = F.pad(x_pad, [0, rel_size - W])
    constant_pad_nd_12: "f32[512, 207]" = torch.ops.aten.constant_pad_nd.default(view_70, [0, 15], 0.0);  view_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:53, code: x_pad = x_pad.reshape(-1, W + 1, rel_size)
    view_71: "f32[512, 9, 23]" = torch.ops.aten.view.default(constant_pad_nd_12, [-1, 9, 23]);  constant_pad_nd_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:54, code: x = x_pad[:, :W, win_size - 1:]
    slice_13: "f32[512, 9, 23]" = torch.ops.aten.slice.Tensor(view_71, 0, 0, 9223372036854775807);  view_71 = None
    slice_14: "f32[512, 8, 23]" = torch.ops.aten.slice.Tensor(slice_13, 1, 0, 8);  slice_13 = None
    slice_15: "f32[512, 8, 12]" = torch.ops.aten.slice.Tensor(slice_14, 2, 11, 9223372036854775807);  slice_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:57, code: x = x.reshape(B, H, 1, W, win_size).expand(-1, -1, win_size, -1, -1)
    view_72: "f32[64, 8, 1, 8, 12]" = torch.ops.aten.view.default(slice_15, [64, 8, 1, 8, 12]);  slice_15 = None
    expand_19: "f32[64, 8, 12, 8, 12]" = torch.ops.aten.expand.default(view_72, [-1, -1, 12, -1, -1]);  view_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:58, code: return x.permute(permute_mask)
    permute_27: "f32[64, 8, 8, 12, 12]" = torch.ops.aten.permute.default(expand_19, [0, 1, 3, 2, 4]);  expand_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:89, code: q = q.transpose(1, 2)
    permute_28: "f32[64, 8, 8, 16]" = torch.ops.aten.permute.default(view_66, [0, 2, 1, 3]);  view_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:45, code: x = (q @ rel_k.transpose(-1, -2))
    permute_29: "f32[16, 23]" = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
    clone_21: "f32[64, 8, 8, 16]" = torch.ops.aten.clone.default(permute_28, memory_format = torch.contiguous_format);  permute_28 = None
    view_73: "f32[4096, 16]" = torch.ops.aten.view.default(clone_21, [4096, 16]);  clone_21 = None
    mm_5: "f32[4096, 23]" = torch.ops.aten.mm.default(view_73, permute_29);  view_73 = permute_29 = None
    view_74: "f32[64, 8, 8, 23]" = torch.ops.aten.view.default(mm_5, [64, 8, 8, 23]);  mm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:46, code: x = x.reshape(-1, W, rel_size)
    view_75: "f32[512, 8, 23]" = torch.ops.aten.view.default(view_74, [-1, 8, 23]);  view_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:49, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    constant_pad_nd_13: "f32[512, 8, 24]" = torch.ops.aten.constant_pad_nd.default(view_75, [0, 1], 0.0);  view_75 = None
    view_76: "f32[512, 192]" = torch.ops.aten.view.default(constant_pad_nd_13, [512, 192]);  constant_pad_nd_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:50, code: x_pad = F.pad(x_pad, [0, rel_size - W])
    constant_pad_nd_14: "f32[512, 207]" = torch.ops.aten.constant_pad_nd.default(view_76, [0, 15], 0.0);  view_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:53, code: x_pad = x_pad.reshape(-1, W + 1, rel_size)
    view_77: "f32[512, 9, 23]" = torch.ops.aten.view.default(constant_pad_nd_14, [-1, 9, 23]);  constant_pad_nd_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:54, code: x = x_pad[:, :W, win_size - 1:]
    slice_16: "f32[512, 9, 23]" = torch.ops.aten.slice.Tensor(view_77, 0, 0, 9223372036854775807);  view_77 = None
    slice_17: "f32[512, 8, 23]" = torch.ops.aten.slice.Tensor(slice_16, 1, 0, 8);  slice_16 = None
    slice_18: "f32[512, 8, 12]" = torch.ops.aten.slice.Tensor(slice_17, 2, 11, 9223372036854775807);  slice_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:57, code: x = x.reshape(B, H, 1, W, win_size).expand(-1, -1, win_size, -1, -1)
    view_78: "f32[64, 8, 1, 8, 12]" = torch.ops.aten.view.default(slice_18, [64, 8, 1, 8, 12]);  slice_18 = None
    expand_20: "f32[64, 8, 12, 8, 12]" = torch.ops.aten.expand.default(view_78, [-1, -1, 12, -1, -1]);  view_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:58, code: return x.permute(permute_mask)
    permute_30: "f32[64, 8, 8, 12, 12]" = torch.ops.aten.permute.default(expand_20, [0, 3, 1, 4, 2]);  expand_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:92, code: rel_logits = rel_logits_h + rel_logits_w
    add_69: "f32[64, 8, 8, 12, 12]" = torch.ops.aten.add.Tensor(permute_30, permute_27);  permute_30 = permute_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:93, code: rel_logits = rel_logits.reshape(B, BB, HW, -1)
    clone_22: "f32[64, 8, 8, 12, 12]" = torch.ops.aten.clone.default(add_69, memory_format = torch.contiguous_format);  add_69 = None
    view_79: "f32[64, 1, 64, 144]" = torch.ops.aten.view.default(clone_22, [64, 1, 64, 144]);  clone_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:199, code: attn = (q @ k.transpose(-1, -2)) * self.scale + self.pos_embed(q)
    add_70: "f32[64, 1, 64, 144]" = torch.ops.aten.add.Tensor(mul_119, view_79);  mul_119 = view_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:201, code: attn = attn.softmax(dim=-1)
    amax_2: "f32[64, 1, 64, 1]" = torch.ops.aten.amax.default(add_70, [-1], True)
    sub_31: "f32[64, 1, 64, 144]" = torch.ops.aten.sub.Tensor(add_70, amax_2);  add_70 = amax_2 = None
    exp_2: "f32[64, 1, 64, 144]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
    sum_3: "f32[64, 1, 64, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_2: "f32[64, 1, 64, 144]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:203, code: out = (attn @ v).transpose(1, 3)  # B * num_heads, dim_head_v, block_size ** 2, num_blocks
    expand_21: "f32[64, 1, 64, 144]" = torch.ops.aten.expand.default(div_2, [64, 1, 64, 144]);  div_2 = None
    view_80: "f32[64, 64, 144]" = torch.ops.aten.view.default(expand_21, [64, 64, 144]);  expand_21 = None
    expand_22: "f32[64, 1, 144, 64]" = torch.ops.aten.expand.default(getitem_7, [64, 1, 144, 64]);  getitem_7 = None
    view_81: "f32[64, 144, 64]" = torch.ops.aten.view.default(expand_22, [64, 144, 64]);  expand_22 = None
    bmm_5: "f32[64, 64, 64]" = torch.ops.aten.bmm.default(view_80, view_81);  view_80 = view_81 = None
    view_82: "f32[64, 1, 64, 64]" = torch.ops.aten.view.default(bmm_5, [64, 1, 64, 64]);  bmm_5 = None
    permute_31: "f32[64, 64, 64, 1]" = torch.ops.aten.permute.default(view_82, [0, 3, 2, 1]);  view_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:205, code: out = out.reshape(-1, self.block_size_ds, self.block_size_ds, num_h_blocks, num_w_blocks)
    clone_23: "f32[64, 64, 64, 1]" = torch.ops.aten.clone.default(permute_31, memory_format = torch.contiguous_format);  permute_31 = None
    view_83: "f32[4096, 8, 8, 1, 1]" = torch.ops.aten.view.default(clone_23, [4096, 8, 8, 1, 1]);  clone_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:206, code: out = out.permute(0, 3, 1, 4, 2).contiguous().view(
    permute_32: "f32[4096, 1, 8, 1, 8]" = torch.ops.aten.permute.default(view_83, [0, 3, 1, 4, 2]);  view_83 = None
    view_84: "f32[8, 512, 8, 8]" = torch.ops.aten.view.default(permute_32, [8, 512, 8, 8]);  permute_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_58: "f32[512]" = torch.ops.prims.convert_element_type.default(arg167_1, torch.float32);  arg167_1 = None
    convert_element_type_59: "f32[512]" = torch.ops.prims.convert_element_type.default(arg168_1, torch.float32);  arg168_1 = None
    add_71: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_59, 1e-05);  convert_element_type_59 = None
    sqrt_29: "f32[512]" = torch.ops.aten.sqrt.default(add_71);  add_71 = None
    reciprocal_29: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_29);  sqrt_29 = None
    mul_120: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_29, 1);  reciprocal_29 = None
    unsqueeze_232: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_58, -1);  convert_element_type_58 = None
    unsqueeze_233: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, -1);  unsqueeze_232 = None
    unsqueeze_234: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_120, -1);  mul_120 = None
    unsqueeze_235: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, -1);  unsqueeze_234 = None
    sub_32: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(view_84, unsqueeze_233);  view_84 = unsqueeze_233 = None
    mul_121: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_32, unsqueeze_235);  sub_32 = unsqueeze_235 = None
    unsqueeze_236: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg64_1, -1);  arg64_1 = None
    unsqueeze_237: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, -1);  unsqueeze_236 = None
    mul_122: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_121, unsqueeze_237);  mul_121 = unsqueeze_237 = None
    unsqueeze_238: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg65_1, -1);  arg65_1 = None
    unsqueeze_239: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, -1);  unsqueeze_238 = None
    add_72: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_122, unsqueeze_239);  mul_122 = unsqueeze_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_30: "f32[8, 512, 8, 8]" = torch.ops.aten.sigmoid.default(add_72)
    mul_123: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_72, sigmoid_30);  add_72 = sigmoid_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_38: "f32[8, 2048, 8, 8]" = torch.ops.aten.convolution.default(mul_123, arg106_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_123 = arg106_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_60: "f32[2048]" = torch.ops.prims.convert_element_type.default(arg169_1, torch.float32);  arg169_1 = None
    convert_element_type_61: "f32[2048]" = torch.ops.prims.convert_element_type.default(arg170_1, torch.float32);  arg170_1 = None
    add_73: "f32[2048]" = torch.ops.aten.add.Tensor(convert_element_type_61, 1e-05);  convert_element_type_61 = None
    sqrt_30: "f32[2048]" = torch.ops.aten.sqrt.default(add_73);  add_73 = None
    reciprocal_30: "f32[2048]" = torch.ops.aten.reciprocal.default(sqrt_30);  sqrt_30 = None
    mul_124: "f32[2048]" = torch.ops.aten.mul.Tensor(reciprocal_30, 1);  reciprocal_30 = None
    unsqueeze_240: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_60, -1);  convert_element_type_60 = None
    unsqueeze_241: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, -1);  unsqueeze_240 = None
    unsqueeze_242: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(mul_124, -1);  mul_124 = None
    unsqueeze_243: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, -1);  unsqueeze_242 = None
    sub_33: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_241);  convolution_38 = unsqueeze_241 = None
    mul_125: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(sub_33, unsqueeze_243);  sub_33 = unsqueeze_243 = None
    unsqueeze_244: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg66_1, -1);  arg66_1 = None
    unsqueeze_245: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, -1);  unsqueeze_244 = None
    mul_126: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(mul_125, unsqueeze_245);  mul_125 = unsqueeze_245 = None
    unsqueeze_246: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg67_1, -1);  arg67_1 = None
    unsqueeze_247: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, -1);  unsqueeze_246 = None
    add_74: "f32[8, 2048, 8, 8]" = torch.ops.aten.add.Tensor(mul_126, unsqueeze_247);  mul_126 = unsqueeze_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:888, code: x = x + self.shortcut(shortcut)
    add_75: "f32[8, 2048, 8, 8]" = torch.ops.aten.add.Tensor(add_74, mul_114);  add_74 = mul_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:889, code: return self.act(x)
    sigmoid_31: "f32[8, 2048, 8, 8]" = torch.ops.aten.sigmoid.default(add_75)
    mul_127: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(add_75, sigmoid_31);  add_75 = sigmoid_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean_5: "f32[8, 2048, 1, 1]" = torch.ops.aten.mean.dim(mul_127, [-1, -2], True);  mul_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_85: "f32[8, 2048]" = torch.ops.aten.view.default(mean_5, [8, 2048]);  mean_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:131, code: x = self.drop(x)
    clone_24: "f32[8, 2048]" = torch.ops.aten.clone.default(view_85);  view_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:134, code: x = self.fc(x)
    permute_33: "f32[2048, 1000]" = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
    addmm: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg108_1, clone_24, permute_33);  arg108_1 = clone_24 = permute_33 = None
    return (addmm,)
    