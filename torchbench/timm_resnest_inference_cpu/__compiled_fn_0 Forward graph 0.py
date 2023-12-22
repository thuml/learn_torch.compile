from __future__ import annotations



def forward(self, arg0_1: "f32[32, 3, 3, 3]", arg1_1: "f32[32]", arg2_1: "f32[32]", arg3_1: "f32[32, 32, 3, 3]", arg4_1: "f32[32]", arg5_1: "f32[32]", arg6_1: "f32[64, 32, 3, 3]", arg7_1: "f32[64]", arg8_1: "f32[64]", arg9_1: "f32[64, 64, 1, 1]", arg10_1: "f32[64]", arg11_1: "f32[64]", arg12_1: "f32[128, 32, 3, 3]", arg13_1: "f32[128]", arg14_1: "f32[128]", arg15_1: "f32[32, 64, 1, 1]", arg16_1: "f32[32]", arg17_1: "f32[32]", arg18_1: "f32[32]", arg19_1: "f32[128, 32, 1, 1]", arg20_1: "f32[128]", arg21_1: "f32[256, 64, 1, 1]", arg22_1: "f32[256]", arg23_1: "f32[256]", arg24_1: "f32[256, 64, 1, 1]", arg25_1: "f32[256]", arg26_1: "f32[256]", arg27_1: "f32[128, 256, 1, 1]", arg28_1: "f32[128]", arg29_1: "f32[128]", arg30_1: "f32[256, 64, 3, 3]", arg31_1: "f32[256]", arg32_1: "f32[256]", arg33_1: "f32[64, 128, 1, 1]", arg34_1: "f32[64]", arg35_1: "f32[64]", arg36_1: "f32[64]", arg37_1: "f32[256, 64, 1, 1]", arg38_1: "f32[256]", arg39_1: "f32[512, 128, 1, 1]", arg40_1: "f32[512]", arg41_1: "f32[512]", arg42_1: "f32[512, 256, 1, 1]", arg43_1: "f32[512]", arg44_1: "f32[512]", arg45_1: "f32[256, 512, 1, 1]", arg46_1: "f32[256]", arg47_1: "f32[256]", arg48_1: "f32[512, 128, 3, 3]", arg49_1: "f32[512]", arg50_1: "f32[512]", arg51_1: "f32[128, 256, 1, 1]", arg52_1: "f32[128]", arg53_1: "f32[128]", arg54_1: "f32[128]", arg55_1: "f32[512, 128, 1, 1]", arg56_1: "f32[512]", arg57_1: "f32[1024, 256, 1, 1]", arg58_1: "f32[1024]", arg59_1: "f32[1024]", arg60_1: "f32[1024, 512, 1, 1]", arg61_1: "f32[1024]", arg62_1: "f32[1024]", arg63_1: "f32[512, 1024, 1, 1]", arg64_1: "f32[512]", arg65_1: "f32[512]", arg66_1: "f32[1024, 256, 3, 3]", arg67_1: "f32[1024]", arg68_1: "f32[1024]", arg69_1: "f32[256, 512, 1, 1]", arg70_1: "f32[256]", arg71_1: "f32[256]", arg72_1: "f32[256]", arg73_1: "f32[1024, 256, 1, 1]", arg74_1: "f32[1024]", arg75_1: "f32[2048, 512, 1, 1]", arg76_1: "f32[2048]", arg77_1: "f32[2048]", arg78_1: "f32[2048, 1024, 1, 1]", arg79_1: "f32[2048]", arg80_1: "f32[2048]", arg81_1: "f32[1000, 2048]", arg82_1: "f32[1000]", arg83_1: "f32[32]", arg84_1: "f32[32]", arg85_1: "i64[]", arg86_1: "f32[32]", arg87_1: "f32[32]", arg88_1: "i64[]", arg89_1: "f32[64]", arg90_1: "f32[64]", arg91_1: "i64[]", arg92_1: "f32[64]", arg93_1: "f32[64]", arg94_1: "i64[]", arg95_1: "f32[128]", arg96_1: "f32[128]", arg97_1: "i64[]", arg98_1: "f32[32]", arg99_1: "f32[32]", arg100_1: "i64[]", arg101_1: "f32[256]", arg102_1: "f32[256]", arg103_1: "i64[]", arg104_1: "f32[256]", arg105_1: "f32[256]", arg106_1: "i64[]", arg107_1: "f32[128]", arg108_1: "f32[128]", arg109_1: "i64[]", arg110_1: "f32[256]", arg111_1: "f32[256]", arg112_1: "i64[]", arg113_1: "f32[64]", arg114_1: "f32[64]", arg115_1: "i64[]", arg116_1: "f32[512]", arg117_1: "f32[512]", arg118_1: "i64[]", arg119_1: "f32[512]", arg120_1: "f32[512]", arg121_1: "i64[]", arg122_1: "f32[256]", arg123_1: "f32[256]", arg124_1: "i64[]", arg125_1: "f32[512]", arg126_1: "f32[512]", arg127_1: "i64[]", arg128_1: "f32[128]", arg129_1: "f32[128]", arg130_1: "i64[]", arg131_1: "f32[1024]", arg132_1: "f32[1024]", arg133_1: "i64[]", arg134_1: "f32[1024]", arg135_1: "f32[1024]", arg136_1: "i64[]", arg137_1: "f32[512]", arg138_1: "f32[512]", arg139_1: "i64[]", arg140_1: "f32[1024]", arg141_1: "f32[1024]", arg142_1: "i64[]", arg143_1: "f32[256]", arg144_1: "f32[256]", arg145_1: "i64[]", arg146_1: "f32[2048]", arg147_1: "f32[2048]", arg148_1: "i64[]", arg149_1: "f32[2048]", arg150_1: "f32[2048]", arg151_1: "i64[]", arg152_1: "f32[4, 3, 224, 224]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:520, code: x = self.conv1(x)
    convolution: "f32[4, 32, 112, 112]" = torch.ops.aten.convolution.default(arg152_1, arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg152_1 = arg0_1 = None
    convert_element_type: "f32[32]" = torch.ops.prims.convert_element_type.default(arg83_1, torch.float32);  arg83_1 = None
    convert_element_type_1: "f32[32]" = torch.ops.prims.convert_element_type.default(arg84_1, torch.float32);  arg84_1 = None
    add: "f32[32]" = torch.ops.aten.add.Tensor(convert_element_type_1, 1e-05);  convert_element_type_1 = None
    sqrt: "f32[32]" = torch.ops.aten.sqrt.default(add);  add = None
    reciprocal: "f32[32]" = torch.ops.aten.reciprocal.default(sqrt);  sqrt = None
    mul: "f32[32]" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
    unsqueeze: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type, -1);  convert_element_type = None
    unsqueeze_1: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    unsqueeze_2: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(mul, -1);  mul = None
    unsqueeze_3: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    sub: "f32[4, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1);  convolution = unsqueeze_1 = None
    mul_1: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub, unsqueeze_3);  sub = unsqueeze_3 = None
    unsqueeze_4: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg1_1, -1);  arg1_1 = None
    unsqueeze_5: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_2: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None
    unsqueeze_6: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
    unsqueeze_7: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_1: "f32[4, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_2, unsqueeze_7);  mul_2 = unsqueeze_7 = None
    relu: "f32[4, 32, 112, 112]" = torch.ops.aten.relu.default(add_1);  add_1 = None
    convolution_1: "f32[4, 32, 112, 112]" = torch.ops.aten.convolution.default(relu, arg3_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu = arg3_1 = None
    convert_element_type_2: "f32[32]" = torch.ops.prims.convert_element_type.default(arg86_1, torch.float32);  arg86_1 = None
    convert_element_type_3: "f32[32]" = torch.ops.prims.convert_element_type.default(arg87_1, torch.float32);  arg87_1 = None
    add_2: "f32[32]" = torch.ops.aten.add.Tensor(convert_element_type_3, 1e-05);  convert_element_type_3 = None
    sqrt_1: "f32[32]" = torch.ops.aten.sqrt.default(add_2);  add_2 = None
    reciprocal_1: "f32[32]" = torch.ops.aten.reciprocal.default(sqrt_1);  sqrt_1 = None
    mul_3: "f32[32]" = torch.ops.aten.mul.Tensor(reciprocal_1, 1);  reciprocal_1 = None
    unsqueeze_8: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, -1);  convert_element_type_2 = None
    unsqueeze_9: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    unsqueeze_10: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(mul_3, -1);  mul_3 = None
    unsqueeze_11: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    sub_1: "f32[4, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_9);  convolution_1 = unsqueeze_9 = None
    mul_4: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_1, unsqueeze_11);  sub_1 = unsqueeze_11 = None
    unsqueeze_12: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
    unsqueeze_13: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_5: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_4, unsqueeze_13);  mul_4 = unsqueeze_13 = None
    unsqueeze_14: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
    unsqueeze_15: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_3: "f32[4, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_5, unsqueeze_15);  mul_5 = unsqueeze_15 = None
    relu_1: "f32[4, 32, 112, 112]" = torch.ops.aten.relu.default(add_3);  add_3 = None
    convolution_2: "f32[4, 64, 112, 112]" = torch.ops.aten.convolution.default(relu_1, arg6_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_1 = arg6_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:521, code: x = self.bn1(x)
    convert_element_type_4: "f32[64]" = torch.ops.prims.convert_element_type.default(arg89_1, torch.float32);  arg89_1 = None
    convert_element_type_5: "f32[64]" = torch.ops.prims.convert_element_type.default(arg90_1, torch.float32);  arg90_1 = None
    add_4: "f32[64]" = torch.ops.aten.add.Tensor(convert_element_type_5, 1e-05);  convert_element_type_5 = None
    sqrt_2: "f32[64]" = torch.ops.aten.sqrt.default(add_4);  add_4 = None
    reciprocal_2: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_2);  sqrt_2 = None
    mul_6: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_2, 1);  reciprocal_2 = None
    unsqueeze_16: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_4, -1);  convert_element_type_4 = None
    unsqueeze_17: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    unsqueeze_18: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_6, -1);  mul_6 = None
    unsqueeze_19: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    sub_2: "f32[4, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_17);  convolution_2 = unsqueeze_17 = None
    mul_7: "f32[4, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_2, unsqueeze_19);  sub_2 = unsqueeze_19 = None
    unsqueeze_20: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
    unsqueeze_21: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_8: "f32[4, 64, 112, 112]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_21);  mul_7 = unsqueeze_21 = None
    unsqueeze_22: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg8_1, -1);  arg8_1 = None
    unsqueeze_23: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_5: "f32[4, 64, 112, 112]" = torch.ops.aten.add.Tensor(mul_8, unsqueeze_23);  mul_8 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:522, code: x = self.act1(x)
    relu_2: "f32[4, 64, 112, 112]" = torch.ops.aten.relu.default(add_5);  add_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:523, code: x = self.maxpool(x)
    max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices.default(relu_2, [3, 3], [2, 2], [1, 1]);  relu_2 = None
    getitem: "f32[4, 64, 56, 56]" = max_pool2d_with_indices[0];  max_pool2d_with_indices = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_3: "f32[4, 64, 56, 56]" = torch.ops.aten.convolution.default(getitem, arg9_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg9_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    convert_element_type_6: "f32[64]" = torch.ops.prims.convert_element_type.default(arg92_1, torch.float32);  arg92_1 = None
    convert_element_type_7: "f32[64]" = torch.ops.prims.convert_element_type.default(arg93_1, torch.float32);  arg93_1 = None
    add_6: "f32[64]" = torch.ops.aten.add.Tensor(convert_element_type_7, 1e-05);  convert_element_type_7 = None
    sqrt_3: "f32[64]" = torch.ops.aten.sqrt.default(add_6);  add_6 = None
    reciprocal_3: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_3);  sqrt_3 = None
    mul_9: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_3, 1);  reciprocal_3 = None
    unsqueeze_24: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_6, -1);  convert_element_type_6 = None
    unsqueeze_25: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
    unsqueeze_26: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_9, -1);  mul_9 = None
    unsqueeze_27: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
    sub_3: "f32[4, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_25);  convolution_3 = unsqueeze_25 = None
    mul_10: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_3, unsqueeze_27);  sub_3 = unsqueeze_27 = None
    unsqueeze_28: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
    unsqueeze_29: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    mul_11: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_10, unsqueeze_29);  mul_10 = unsqueeze_29 = None
    unsqueeze_30: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg11_1, -1);  arg11_1 = None
    unsqueeze_31: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
    add_7: "f32[4, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_11, unsqueeze_31);  mul_11 = unsqueeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    relu_3: "f32[4, 64, 56, 56]" = torch.ops.aten.relu.default(add_7);  add_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_4: "f32[4, 128, 56, 56]" = torch.ops.aten.convolution.default(relu_3, arg12_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_3 = arg12_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    convert_element_type_8: "f32[128]" = torch.ops.prims.convert_element_type.default(arg95_1, torch.float32);  arg95_1 = None
    convert_element_type_9: "f32[128]" = torch.ops.prims.convert_element_type.default(arg96_1, torch.float32);  arg96_1 = None
    add_8: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_9, 1e-05);  convert_element_type_9 = None
    sqrt_4: "f32[128]" = torch.ops.aten.sqrt.default(add_8);  add_8 = None
    reciprocal_4: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_4);  sqrt_4 = None
    mul_12: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_4, 1);  reciprocal_4 = None
    unsqueeze_32: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_8, -1);  convert_element_type_8 = None
    unsqueeze_33: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    unsqueeze_34: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_12, -1);  mul_12 = None
    unsqueeze_35: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
    sub_4: "f32[4, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_33);  convolution_4 = unsqueeze_33 = None
    mul_13: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_4, unsqueeze_35);  sub_4 = unsqueeze_35 = None
    unsqueeze_36: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg13_1, -1);  arg13_1 = None
    unsqueeze_37: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
    mul_14: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(mul_13, unsqueeze_37);  mul_13 = unsqueeze_37 = None
    unsqueeze_38: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
    unsqueeze_39: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
    add_9: "f32[4, 128, 56, 56]" = torch.ops.aten.add.Tensor(mul_14, unsqueeze_39);  mul_14 = unsqueeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    relu_4: "f32[4, 128, 56, 56]" = torch.ops.aten.relu.default(add_9);  add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_1: "f32[4, 2, 64, 56, 56]" = torch.ops.aten.view.default(relu_4, [4, 2, 64, 56, 56]);  relu_4 = None
    sum_1: "f32[4, 64, 56, 56]" = torch.ops.aten.sum.dim_IntList(view_1, [1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    mean: "f32[4, 64, 1, 1]" = torch.ops.aten.mean.dim(sum_1, [2, 3], True);  sum_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_5: "f32[4, 32, 1, 1]" = torch.ops.aten.convolution.default(mean, arg15_1, arg16_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean = arg15_1 = arg16_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    convert_element_type_10: "f32[32]" = torch.ops.prims.convert_element_type.default(arg98_1, torch.float32);  arg98_1 = None
    convert_element_type_11: "f32[32]" = torch.ops.prims.convert_element_type.default(arg99_1, torch.float32);  arg99_1 = None
    add_10: "f32[32]" = torch.ops.aten.add.Tensor(convert_element_type_11, 1e-05);  convert_element_type_11 = None
    sqrt_5: "f32[32]" = torch.ops.aten.sqrt.default(add_10);  add_10 = None
    reciprocal_5: "f32[32]" = torch.ops.aten.reciprocal.default(sqrt_5);  sqrt_5 = None
    mul_15: "f32[32]" = torch.ops.aten.mul.Tensor(reciprocal_5, 1);  reciprocal_5 = None
    unsqueeze_40: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_10, -1);  convert_element_type_10 = None
    unsqueeze_41: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    unsqueeze_42: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(mul_15, -1);  mul_15 = None
    unsqueeze_43: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    sub_5: "f32[4, 32, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_41);  convolution_5 = unsqueeze_41 = None
    mul_16: "f32[4, 32, 1, 1]" = torch.ops.aten.mul.Tensor(sub_5, unsqueeze_43);  sub_5 = unsqueeze_43 = None
    unsqueeze_44: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg17_1, -1);  arg17_1 = None
    unsqueeze_45: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    mul_17: "f32[4, 32, 1, 1]" = torch.ops.aten.mul.Tensor(mul_16, unsqueeze_45);  mul_16 = unsqueeze_45 = None
    unsqueeze_46: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg18_1, -1);  arg18_1 = None
    unsqueeze_47: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
    add_11: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(mul_17, unsqueeze_47);  mul_17 = unsqueeze_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    relu_5: "f32[4, 32, 1, 1]" = torch.ops.aten.relu.default(add_11);  add_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_6: "f32[4, 128, 1, 1]" = torch.ops.aten.convolution.default(relu_5, arg19_1, arg20_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_5 = arg19_1 = arg20_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_2: "f32[4, 1, 2, 64]" = torch.ops.aten.view.default(convolution_6, [4, 1, 2, -1]);  convolution_6 = None
    permute: "f32[4, 2, 1, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    amax: "f32[4, 1, 1, 64]" = torch.ops.aten.amax.default(permute, [1], True)
    sub_6: "f32[4, 2, 1, 64]" = torch.ops.aten.sub.Tensor(permute, amax);  permute = amax = None
    exp: "f32[4, 2, 1, 64]" = torch.ops.aten.exp.default(sub_6);  sub_6 = None
    sum_2: "f32[4, 1, 1, 64]" = torch.ops.aten.sum.dim_IntList(exp, [1], True)
    div: "f32[4, 2, 1, 64]" = torch.ops.aten.div.Tensor(exp, sum_2);  exp = sum_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_3: "f32[4, 128]" = torch.ops.aten.view.default(div, [4, -1]);  div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_4: "f32[4, 128, 1, 1]" = torch.ops.aten.view.default(view_3, [4, -1, 1, 1]);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_5: "f32[4, 2, 64, 1, 1]" = torch.ops.aten.view.default(view_4, [4, 2, 64, 1, 1]);  view_4 = None
    mul_18: "f32[4, 2, 64, 56, 56]" = torch.ops.aten.mul.Tensor(view_1, view_5);  view_1 = view_5 = None
    sum_3: "f32[4, 64, 56, 56]" = torch.ops.aten.sum.dim_IntList(mul_18, [1]);  mul_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_7: "f32[4, 256, 56, 56]" = torch.ops.aten.convolution.default(sum_3, arg21_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  sum_3 = arg21_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    convert_element_type_12: "f32[256]" = torch.ops.prims.convert_element_type.default(arg101_1, torch.float32);  arg101_1 = None
    convert_element_type_13: "f32[256]" = torch.ops.prims.convert_element_type.default(arg102_1, torch.float32);  arg102_1 = None
    add_12: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_13, 1e-05);  convert_element_type_13 = None
    sqrt_6: "f32[256]" = torch.ops.aten.sqrt.default(add_12);  add_12 = None
    reciprocal_6: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_6);  sqrt_6 = None
    mul_19: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_6, 1);  reciprocal_6 = None
    unsqueeze_48: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_12, -1);  convert_element_type_12 = None
    unsqueeze_49: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
    unsqueeze_50: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_19, -1);  mul_19 = None
    unsqueeze_51: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
    sub_7: "f32[4, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_49);  convolution_7 = unsqueeze_49 = None
    mul_20: "f32[4, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_7, unsqueeze_51);  sub_7 = unsqueeze_51 = None
    unsqueeze_52: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg22_1, -1);  arg22_1 = None
    unsqueeze_53: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    mul_21: "f32[4, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_20, unsqueeze_53);  mul_20 = unsqueeze_53 = None
    unsqueeze_54: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg23_1, -1);  arg23_1 = None
    unsqueeze_55: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
    add_13: "f32[4, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_21, unsqueeze_55);  mul_21 = unsqueeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:113, code: shortcut = self.downsample(x)
    convolution_8: "f32[4, 256, 56, 56]" = torch.ops.aten.convolution.default(getitem, arg24_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem = arg24_1 = None
    convert_element_type_14: "f32[256]" = torch.ops.prims.convert_element_type.default(arg104_1, torch.float32);  arg104_1 = None
    convert_element_type_15: "f32[256]" = torch.ops.prims.convert_element_type.default(arg105_1, torch.float32);  arg105_1 = None
    add_14: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_15, 1e-05);  convert_element_type_15 = None
    sqrt_7: "f32[256]" = torch.ops.aten.sqrt.default(add_14);  add_14 = None
    reciprocal_7: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_7);  sqrt_7 = None
    mul_22: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_7, 1);  reciprocal_7 = None
    unsqueeze_56: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_14, -1);  convert_element_type_14 = None
    unsqueeze_57: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    unsqueeze_58: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_22, -1);  mul_22 = None
    unsqueeze_59: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
    sub_8: "f32[4, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_57);  convolution_8 = unsqueeze_57 = None
    mul_23: "f32[4, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_8, unsqueeze_59);  sub_8 = unsqueeze_59 = None
    unsqueeze_60: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg25_1, -1);  arg25_1 = None
    unsqueeze_61: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
    mul_24: "f32[4, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_23, unsqueeze_61);  mul_23 = unsqueeze_61 = None
    unsqueeze_62: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg26_1, -1);  arg26_1 = None
    unsqueeze_63: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
    add_15: "f32[4, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_24, unsqueeze_63);  mul_24 = unsqueeze_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    add_16: "f32[4, 256, 56, 56]" = torch.ops.aten.add.Tensor(add_13, add_15);  add_13 = add_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    relu_6: "f32[4, 256, 56, 56]" = torch.ops.aten.relu.default(add_16);  add_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_9: "f32[4, 128, 56, 56]" = torch.ops.aten.convolution.default(relu_6, arg27_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg27_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    convert_element_type_16: "f32[128]" = torch.ops.prims.convert_element_type.default(arg107_1, torch.float32);  arg107_1 = None
    convert_element_type_17: "f32[128]" = torch.ops.prims.convert_element_type.default(arg108_1, torch.float32);  arg108_1 = None
    add_17: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_17, 1e-05);  convert_element_type_17 = None
    sqrt_8: "f32[128]" = torch.ops.aten.sqrt.default(add_17);  add_17 = None
    reciprocal_8: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_8);  sqrt_8 = None
    mul_25: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_8, 1);  reciprocal_8 = None
    unsqueeze_64: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_16, -1);  convert_element_type_16 = None
    unsqueeze_65: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
    unsqueeze_66: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_25, -1);  mul_25 = None
    unsqueeze_67: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
    sub_9: "f32[4, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_65);  convolution_9 = unsqueeze_65 = None
    mul_26: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_9, unsqueeze_67);  sub_9 = unsqueeze_67 = None
    unsqueeze_68: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg28_1, -1);  arg28_1 = None
    unsqueeze_69: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    mul_27: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(mul_26, unsqueeze_69);  mul_26 = unsqueeze_69 = None
    unsqueeze_70: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg29_1, -1);  arg29_1 = None
    unsqueeze_71: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    add_18: "f32[4, 128, 56, 56]" = torch.ops.aten.add.Tensor(mul_27, unsqueeze_71);  mul_27 = unsqueeze_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    relu_7: "f32[4, 128, 56, 56]" = torch.ops.aten.relu.default(add_18);  add_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_10: "f32[4, 256, 56, 56]" = torch.ops.aten.convolution.default(relu_7, arg30_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_7 = arg30_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    convert_element_type_18: "f32[256]" = torch.ops.prims.convert_element_type.default(arg110_1, torch.float32);  arg110_1 = None
    convert_element_type_19: "f32[256]" = torch.ops.prims.convert_element_type.default(arg111_1, torch.float32);  arg111_1 = None
    add_19: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_19, 1e-05);  convert_element_type_19 = None
    sqrt_9: "f32[256]" = torch.ops.aten.sqrt.default(add_19);  add_19 = None
    reciprocal_9: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_9);  sqrt_9 = None
    mul_28: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_9, 1);  reciprocal_9 = None
    unsqueeze_72: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_18, -1);  convert_element_type_18 = None
    unsqueeze_73: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
    unsqueeze_74: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_28, -1);  mul_28 = None
    unsqueeze_75: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
    sub_10: "f32[4, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_73);  convolution_10 = unsqueeze_73 = None
    mul_29: "f32[4, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_10, unsqueeze_75);  sub_10 = unsqueeze_75 = None
    unsqueeze_76: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg31_1, -1);  arg31_1 = None
    unsqueeze_77: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
    mul_30: "f32[4, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_29, unsqueeze_77);  mul_29 = unsqueeze_77 = None
    unsqueeze_78: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg32_1, -1);  arg32_1 = None
    unsqueeze_79: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    add_20: "f32[4, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_30, unsqueeze_79);  mul_30 = unsqueeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    relu_8: "f32[4, 256, 56, 56]" = torch.ops.aten.relu.default(add_20);  add_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_7: "f32[4, 2, 128, 56, 56]" = torch.ops.aten.view.default(relu_8, [4, 2, 128, 56, 56]);  relu_8 = None
    sum_4: "f32[4, 128, 56, 56]" = torch.ops.aten.sum.dim_IntList(view_7, [1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    mean_1: "f32[4, 128, 1, 1]" = torch.ops.aten.mean.dim(sum_4, [2, 3], True);  sum_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_11: "f32[4, 64, 1, 1]" = torch.ops.aten.convolution.default(mean_1, arg33_1, arg34_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_1 = arg33_1 = arg34_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    convert_element_type_20: "f32[64]" = torch.ops.prims.convert_element_type.default(arg113_1, torch.float32);  arg113_1 = None
    convert_element_type_21: "f32[64]" = torch.ops.prims.convert_element_type.default(arg114_1, torch.float32);  arg114_1 = None
    add_21: "f32[64]" = torch.ops.aten.add.Tensor(convert_element_type_21, 1e-05);  convert_element_type_21 = None
    sqrt_10: "f32[64]" = torch.ops.aten.sqrt.default(add_21);  add_21 = None
    reciprocal_10: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_10);  sqrt_10 = None
    mul_31: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_10, 1);  reciprocal_10 = None
    unsqueeze_80: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_20, -1);  convert_element_type_20 = None
    unsqueeze_81: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
    unsqueeze_82: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_31, -1);  mul_31 = None
    unsqueeze_83: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    sub_11: "f32[4, 64, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_81);  convolution_11 = unsqueeze_81 = None
    mul_32: "f32[4, 64, 1, 1]" = torch.ops.aten.mul.Tensor(sub_11, unsqueeze_83);  sub_11 = unsqueeze_83 = None
    unsqueeze_84: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg35_1, -1);  arg35_1 = None
    unsqueeze_85: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    mul_33: "f32[4, 64, 1, 1]" = torch.ops.aten.mul.Tensor(mul_32, unsqueeze_85);  mul_32 = unsqueeze_85 = None
    unsqueeze_86: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg36_1, -1);  arg36_1 = None
    unsqueeze_87: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
    add_22: "f32[4, 64, 1, 1]" = torch.ops.aten.add.Tensor(mul_33, unsqueeze_87);  mul_33 = unsqueeze_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    relu_9: "f32[4, 64, 1, 1]" = torch.ops.aten.relu.default(add_22);  add_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_12: "f32[4, 256, 1, 1]" = torch.ops.aten.convolution.default(relu_9, arg37_1, arg38_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_9 = arg37_1 = arg38_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_8: "f32[4, 1, 2, 128]" = torch.ops.aten.view.default(convolution_12, [4, 1, 2, -1]);  convolution_12 = None
    permute_1: "f32[4, 2, 1, 128]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    amax_1: "f32[4, 1, 1, 128]" = torch.ops.aten.amax.default(permute_1, [1], True)
    sub_12: "f32[4, 2, 1, 128]" = torch.ops.aten.sub.Tensor(permute_1, amax_1);  permute_1 = amax_1 = None
    exp_1: "f32[4, 2, 1, 128]" = torch.ops.aten.exp.default(sub_12);  sub_12 = None
    sum_5: "f32[4, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(exp_1, [1], True)
    div_1: "f32[4, 2, 1, 128]" = torch.ops.aten.div.Tensor(exp_1, sum_5);  exp_1 = sum_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_9: "f32[4, 256]" = torch.ops.aten.view.default(div_1, [4, -1]);  div_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_10: "f32[4, 256, 1, 1]" = torch.ops.aten.view.default(view_9, [4, -1, 1, 1]);  view_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_11: "f32[4, 2, 128, 1, 1]" = torch.ops.aten.view.default(view_10, [4, 2, 128, 1, 1]);  view_10 = None
    mul_34: "f32[4, 2, 128, 56, 56]" = torch.ops.aten.mul.Tensor(view_7, view_11);  view_7 = view_11 = None
    sum_6: "f32[4, 128, 56, 56]" = torch.ops.aten.sum.dim_IntList(mul_34, [1]);  mul_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:107, code: out = self.avd_last(out)
    avg_pool2d: "f32[4, 128, 28, 28]" = torch.ops.aten.avg_pool2d.default(sum_6, [3, 3], [2, 2], [1, 1]);  sum_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_13: "f32[4, 512, 28, 28]" = torch.ops.aten.convolution.default(avg_pool2d, arg39_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  avg_pool2d = arg39_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    convert_element_type_22: "f32[512]" = torch.ops.prims.convert_element_type.default(arg116_1, torch.float32);  arg116_1 = None
    convert_element_type_23: "f32[512]" = torch.ops.prims.convert_element_type.default(arg117_1, torch.float32);  arg117_1 = None
    add_23: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_23, 1e-05);  convert_element_type_23 = None
    sqrt_11: "f32[512]" = torch.ops.aten.sqrt.default(add_23);  add_23 = None
    reciprocal_11: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_11);  sqrt_11 = None
    mul_35: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_11, 1);  reciprocal_11 = None
    unsqueeze_88: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_22, -1);  convert_element_type_22 = None
    unsqueeze_89: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
    unsqueeze_90: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_35, -1);  mul_35 = None
    unsqueeze_91: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    sub_13: "f32[4, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_89);  convolution_13 = unsqueeze_89 = None
    mul_36: "f32[4, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_13, unsqueeze_91);  sub_13 = unsqueeze_91 = None
    unsqueeze_92: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg40_1, -1);  arg40_1 = None
    unsqueeze_93: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
    mul_37: "f32[4, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_36, unsqueeze_93);  mul_36 = unsqueeze_93 = None
    unsqueeze_94: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg41_1, -1);  arg41_1 = None
    unsqueeze_95: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
    add_24: "f32[4, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_37, unsqueeze_95);  mul_37 = unsqueeze_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:113, code: shortcut = self.downsample(x)
    avg_pool2d_1: "f32[4, 256, 28, 28]" = torch.ops.aten.avg_pool2d.default(relu_6, [2, 2], [2, 2], [0, 0], True, False);  relu_6 = None
    convolution_14: "f32[4, 512, 28, 28]" = torch.ops.aten.convolution.default(avg_pool2d_1, arg42_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  avg_pool2d_1 = arg42_1 = None
    convert_element_type_24: "f32[512]" = torch.ops.prims.convert_element_type.default(arg119_1, torch.float32);  arg119_1 = None
    convert_element_type_25: "f32[512]" = torch.ops.prims.convert_element_type.default(arg120_1, torch.float32);  arg120_1 = None
    add_25: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_25, 1e-05);  convert_element_type_25 = None
    sqrt_12: "f32[512]" = torch.ops.aten.sqrt.default(add_25);  add_25 = None
    reciprocal_12: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_12);  sqrt_12 = None
    mul_38: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_12, 1);  reciprocal_12 = None
    unsqueeze_96: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_24, -1);  convert_element_type_24 = None
    unsqueeze_97: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
    unsqueeze_98: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_38, -1);  mul_38 = None
    unsqueeze_99: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
    sub_14: "f32[4, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_97);  convolution_14 = unsqueeze_97 = None
    mul_39: "f32[4, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_14, unsqueeze_99);  sub_14 = unsqueeze_99 = None
    unsqueeze_100: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg43_1, -1);  arg43_1 = None
    unsqueeze_101: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
    mul_40: "f32[4, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_39, unsqueeze_101);  mul_39 = unsqueeze_101 = None
    unsqueeze_102: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg44_1, -1);  arg44_1 = None
    unsqueeze_103: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    add_26: "f32[4, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_40, unsqueeze_103);  mul_40 = unsqueeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    add_27: "f32[4, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_24, add_26);  add_24 = add_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    relu_10: "f32[4, 512, 28, 28]" = torch.ops.aten.relu.default(add_27);  add_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_15: "f32[4, 256, 28, 28]" = torch.ops.aten.convolution.default(relu_10, arg45_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg45_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    convert_element_type_26: "f32[256]" = torch.ops.prims.convert_element_type.default(arg122_1, torch.float32);  arg122_1 = None
    convert_element_type_27: "f32[256]" = torch.ops.prims.convert_element_type.default(arg123_1, torch.float32);  arg123_1 = None
    add_28: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_27, 1e-05);  convert_element_type_27 = None
    sqrt_13: "f32[256]" = torch.ops.aten.sqrt.default(add_28);  add_28 = None
    reciprocal_13: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_13);  sqrt_13 = None
    mul_41: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_13, 1);  reciprocal_13 = None
    unsqueeze_104: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_26, -1);  convert_element_type_26 = None
    unsqueeze_105: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
    unsqueeze_106: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_41, -1);  mul_41 = None
    unsqueeze_107: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
    sub_15: "f32[4, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_105);  convolution_15 = unsqueeze_105 = None
    mul_42: "f32[4, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_15, unsqueeze_107);  sub_15 = unsqueeze_107 = None
    unsqueeze_108: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg46_1, -1);  arg46_1 = None
    unsqueeze_109: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
    mul_43: "f32[4, 256, 28, 28]" = torch.ops.aten.mul.Tensor(mul_42, unsqueeze_109);  mul_42 = unsqueeze_109 = None
    unsqueeze_110: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg47_1, -1);  arg47_1 = None
    unsqueeze_111: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
    add_29: "f32[4, 256, 28, 28]" = torch.ops.aten.add.Tensor(mul_43, unsqueeze_111);  mul_43 = unsqueeze_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    relu_11: "f32[4, 256, 28, 28]" = torch.ops.aten.relu.default(add_29);  add_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_16: "f32[4, 512, 28, 28]" = torch.ops.aten.convolution.default(relu_11, arg48_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_11 = arg48_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    convert_element_type_28: "f32[512]" = torch.ops.prims.convert_element_type.default(arg125_1, torch.float32);  arg125_1 = None
    convert_element_type_29: "f32[512]" = torch.ops.prims.convert_element_type.default(arg126_1, torch.float32);  arg126_1 = None
    add_30: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_29, 1e-05);  convert_element_type_29 = None
    sqrt_14: "f32[512]" = torch.ops.aten.sqrt.default(add_30);  add_30 = None
    reciprocal_14: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_14);  sqrt_14 = None
    mul_44: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_14, 1);  reciprocal_14 = None
    unsqueeze_112: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_28, -1);  convert_element_type_28 = None
    unsqueeze_113: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
    unsqueeze_114: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_44, -1);  mul_44 = None
    unsqueeze_115: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
    sub_16: "f32[4, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_113);  convolution_16 = unsqueeze_113 = None
    mul_45: "f32[4, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_16, unsqueeze_115);  sub_16 = unsqueeze_115 = None
    unsqueeze_116: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg49_1, -1);  arg49_1 = None
    unsqueeze_117: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
    mul_46: "f32[4, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_45, unsqueeze_117);  mul_45 = unsqueeze_117 = None
    unsqueeze_118: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg50_1, -1);  arg50_1 = None
    unsqueeze_119: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
    add_31: "f32[4, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_46, unsqueeze_119);  mul_46 = unsqueeze_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    relu_12: "f32[4, 512, 28, 28]" = torch.ops.aten.relu.default(add_31);  add_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_13: "f32[4, 2, 256, 28, 28]" = torch.ops.aten.view.default(relu_12, [4, 2, 256, 28, 28]);  relu_12 = None
    sum_7: "f32[4, 256, 28, 28]" = torch.ops.aten.sum.dim_IntList(view_13, [1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    mean_2: "f32[4, 256, 1, 1]" = torch.ops.aten.mean.dim(sum_7, [2, 3], True);  sum_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_17: "f32[4, 128, 1, 1]" = torch.ops.aten.convolution.default(mean_2, arg51_1, arg52_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_2 = arg51_1 = arg52_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    convert_element_type_30: "f32[128]" = torch.ops.prims.convert_element_type.default(arg128_1, torch.float32);  arg128_1 = None
    convert_element_type_31: "f32[128]" = torch.ops.prims.convert_element_type.default(arg129_1, torch.float32);  arg129_1 = None
    add_32: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_31, 1e-05);  convert_element_type_31 = None
    sqrt_15: "f32[128]" = torch.ops.aten.sqrt.default(add_32);  add_32 = None
    reciprocal_15: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_15);  sqrt_15 = None
    mul_47: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_15, 1);  reciprocal_15 = None
    unsqueeze_120: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_30, -1);  convert_element_type_30 = None
    unsqueeze_121: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
    unsqueeze_122: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_47, -1);  mul_47 = None
    unsqueeze_123: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
    sub_17: "f32[4, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_121);  convolution_17 = unsqueeze_121 = None
    mul_48: "f32[4, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_17, unsqueeze_123);  sub_17 = unsqueeze_123 = None
    unsqueeze_124: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg53_1, -1);  arg53_1 = None
    unsqueeze_125: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
    mul_49: "f32[4, 128, 1, 1]" = torch.ops.aten.mul.Tensor(mul_48, unsqueeze_125);  mul_48 = unsqueeze_125 = None
    unsqueeze_126: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg54_1, -1);  arg54_1 = None
    unsqueeze_127: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
    add_33: "f32[4, 128, 1, 1]" = torch.ops.aten.add.Tensor(mul_49, unsqueeze_127);  mul_49 = unsqueeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    relu_13: "f32[4, 128, 1, 1]" = torch.ops.aten.relu.default(add_33);  add_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_18: "f32[4, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_13, arg55_1, arg56_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_13 = arg55_1 = arg56_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_14: "f32[4, 1, 2, 256]" = torch.ops.aten.view.default(convolution_18, [4, 1, 2, -1]);  convolution_18 = None
    permute_2: "f32[4, 2, 1, 256]" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    amax_2: "f32[4, 1, 1, 256]" = torch.ops.aten.amax.default(permute_2, [1], True)
    sub_18: "f32[4, 2, 1, 256]" = torch.ops.aten.sub.Tensor(permute_2, amax_2);  permute_2 = amax_2 = None
    exp_2: "f32[4, 2, 1, 256]" = torch.ops.aten.exp.default(sub_18);  sub_18 = None
    sum_8: "f32[4, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(exp_2, [1], True)
    div_2: "f32[4, 2, 1, 256]" = torch.ops.aten.div.Tensor(exp_2, sum_8);  exp_2 = sum_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_15: "f32[4, 512]" = torch.ops.aten.view.default(div_2, [4, -1]);  div_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_16: "f32[4, 512, 1, 1]" = torch.ops.aten.view.default(view_15, [4, -1, 1, 1]);  view_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_17: "f32[4, 2, 256, 1, 1]" = torch.ops.aten.view.default(view_16, [4, 2, 256, 1, 1]);  view_16 = None
    mul_50: "f32[4, 2, 256, 28, 28]" = torch.ops.aten.mul.Tensor(view_13, view_17);  view_13 = view_17 = None
    sum_9: "f32[4, 256, 28, 28]" = torch.ops.aten.sum.dim_IntList(mul_50, [1]);  mul_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:107, code: out = self.avd_last(out)
    avg_pool2d_2: "f32[4, 256, 14, 14]" = torch.ops.aten.avg_pool2d.default(sum_9, [3, 3], [2, 2], [1, 1]);  sum_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_19: "f32[4, 1024, 14, 14]" = torch.ops.aten.convolution.default(avg_pool2d_2, arg57_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  avg_pool2d_2 = arg57_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    convert_element_type_32: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg131_1, torch.float32);  arg131_1 = None
    convert_element_type_33: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg132_1, torch.float32);  arg132_1 = None
    add_34: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_33, 1e-05);  convert_element_type_33 = None
    sqrt_16: "f32[1024]" = torch.ops.aten.sqrt.default(add_34);  add_34 = None
    reciprocal_16: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_16);  sqrt_16 = None
    mul_51: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_16, 1);  reciprocal_16 = None
    unsqueeze_128: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_32, -1);  convert_element_type_32 = None
    unsqueeze_129: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
    unsqueeze_130: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_51, -1);  mul_51 = None
    unsqueeze_131: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
    sub_19: "f32[4, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_129);  convolution_19 = unsqueeze_129 = None
    mul_52: "f32[4, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_19, unsqueeze_131);  sub_19 = unsqueeze_131 = None
    unsqueeze_132: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg58_1, -1);  arg58_1 = None
    unsqueeze_133: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
    mul_53: "f32[4, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_52, unsqueeze_133);  mul_52 = unsqueeze_133 = None
    unsqueeze_134: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg59_1, -1);  arg59_1 = None
    unsqueeze_135: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
    add_35: "f32[4, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_53, unsqueeze_135);  mul_53 = unsqueeze_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:113, code: shortcut = self.downsample(x)
    avg_pool2d_3: "f32[4, 512, 14, 14]" = torch.ops.aten.avg_pool2d.default(relu_10, [2, 2], [2, 2], [0, 0], True, False);  relu_10 = None
    convolution_20: "f32[4, 1024, 14, 14]" = torch.ops.aten.convolution.default(avg_pool2d_3, arg60_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  avg_pool2d_3 = arg60_1 = None
    convert_element_type_34: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg134_1, torch.float32);  arg134_1 = None
    convert_element_type_35: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg135_1, torch.float32);  arg135_1 = None
    add_36: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_35, 1e-05);  convert_element_type_35 = None
    sqrt_17: "f32[1024]" = torch.ops.aten.sqrt.default(add_36);  add_36 = None
    reciprocal_17: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_17);  sqrt_17 = None
    mul_54: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_17, 1);  reciprocal_17 = None
    unsqueeze_136: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_34, -1);  convert_element_type_34 = None
    unsqueeze_137: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
    unsqueeze_138: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_54, -1);  mul_54 = None
    unsqueeze_139: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
    sub_20: "f32[4, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_137);  convolution_20 = unsqueeze_137 = None
    mul_55: "f32[4, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_20, unsqueeze_139);  sub_20 = unsqueeze_139 = None
    unsqueeze_140: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg61_1, -1);  arg61_1 = None
    unsqueeze_141: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
    mul_56: "f32[4, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_55, unsqueeze_141);  mul_55 = unsqueeze_141 = None
    unsqueeze_142: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg62_1, -1);  arg62_1 = None
    unsqueeze_143: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
    add_37: "f32[4, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_56, unsqueeze_143);  mul_56 = unsqueeze_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    add_38: "f32[4, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_35, add_37);  add_35 = add_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    relu_14: "f32[4, 1024, 14, 14]" = torch.ops.aten.relu.default(add_38);  add_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_21: "f32[4, 512, 14, 14]" = torch.ops.aten.convolution.default(relu_14, arg63_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg63_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    convert_element_type_36: "f32[512]" = torch.ops.prims.convert_element_type.default(arg137_1, torch.float32);  arg137_1 = None
    convert_element_type_37: "f32[512]" = torch.ops.prims.convert_element_type.default(arg138_1, torch.float32);  arg138_1 = None
    add_39: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_37, 1e-05);  convert_element_type_37 = None
    sqrt_18: "f32[512]" = torch.ops.aten.sqrt.default(add_39);  add_39 = None
    reciprocal_18: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_18);  sqrt_18 = None
    mul_57: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_18, 1);  reciprocal_18 = None
    unsqueeze_144: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_36, -1);  convert_element_type_36 = None
    unsqueeze_145: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
    unsqueeze_146: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_57, -1);  mul_57 = None
    unsqueeze_147: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
    sub_21: "f32[4, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_145);  convolution_21 = unsqueeze_145 = None
    mul_58: "f32[4, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_21, unsqueeze_147);  sub_21 = unsqueeze_147 = None
    unsqueeze_148: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg64_1, -1);  arg64_1 = None
    unsqueeze_149: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
    mul_59: "f32[4, 512, 14, 14]" = torch.ops.aten.mul.Tensor(mul_58, unsqueeze_149);  mul_58 = unsqueeze_149 = None
    unsqueeze_150: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg65_1, -1);  arg65_1 = None
    unsqueeze_151: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
    add_40: "f32[4, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_59, unsqueeze_151);  mul_59 = unsqueeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    relu_15: "f32[4, 512, 14, 14]" = torch.ops.aten.relu.default(add_40);  add_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_22: "f32[4, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_15, arg66_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  relu_15 = arg66_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    convert_element_type_38: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg140_1, torch.float32);  arg140_1 = None
    convert_element_type_39: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg141_1, torch.float32);  arg141_1 = None
    add_41: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_39, 1e-05);  convert_element_type_39 = None
    sqrt_19: "f32[1024]" = torch.ops.aten.sqrt.default(add_41);  add_41 = None
    reciprocal_19: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_19);  sqrt_19 = None
    mul_60: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_19, 1);  reciprocal_19 = None
    unsqueeze_152: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_38, -1);  convert_element_type_38 = None
    unsqueeze_153: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
    unsqueeze_154: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_60, -1);  mul_60 = None
    unsqueeze_155: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
    sub_22: "f32[4, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_153);  convolution_22 = unsqueeze_153 = None
    mul_61: "f32[4, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_22, unsqueeze_155);  sub_22 = unsqueeze_155 = None
    unsqueeze_156: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg67_1, -1);  arg67_1 = None
    unsqueeze_157: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
    mul_62: "f32[4, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_61, unsqueeze_157);  mul_61 = unsqueeze_157 = None
    unsqueeze_158: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg68_1, -1);  arg68_1 = None
    unsqueeze_159: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
    add_42: "f32[4, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_62, unsqueeze_159);  mul_62 = unsqueeze_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    relu_16: "f32[4, 1024, 14, 14]" = torch.ops.aten.relu.default(add_42);  add_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_19: "f32[4, 2, 512, 14, 14]" = torch.ops.aten.view.default(relu_16, [4, 2, 512, 14, 14]);  relu_16 = None
    sum_10: "f32[4, 512, 14, 14]" = torch.ops.aten.sum.dim_IntList(view_19, [1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    mean_3: "f32[4, 512, 1, 1]" = torch.ops.aten.mean.dim(sum_10, [2, 3], True);  sum_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_23: "f32[4, 256, 1, 1]" = torch.ops.aten.convolution.default(mean_3, arg69_1, arg70_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_3 = arg69_1 = arg70_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    convert_element_type_40: "f32[256]" = torch.ops.prims.convert_element_type.default(arg143_1, torch.float32);  arg143_1 = None
    convert_element_type_41: "f32[256]" = torch.ops.prims.convert_element_type.default(arg144_1, torch.float32);  arg144_1 = None
    add_43: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_41, 1e-05);  convert_element_type_41 = None
    sqrt_20: "f32[256]" = torch.ops.aten.sqrt.default(add_43);  add_43 = None
    reciprocal_20: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_20);  sqrt_20 = None
    mul_63: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_20, 1);  reciprocal_20 = None
    unsqueeze_160: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_40, -1);  convert_element_type_40 = None
    unsqueeze_161: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
    unsqueeze_162: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_63, -1);  mul_63 = None
    unsqueeze_163: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
    sub_23: "f32[4, 256, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_161);  convolution_23 = unsqueeze_161 = None
    mul_64: "f32[4, 256, 1, 1]" = torch.ops.aten.mul.Tensor(sub_23, unsqueeze_163);  sub_23 = unsqueeze_163 = None
    unsqueeze_164: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg71_1, -1);  arg71_1 = None
    unsqueeze_165: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
    mul_65: "f32[4, 256, 1, 1]" = torch.ops.aten.mul.Tensor(mul_64, unsqueeze_165);  mul_64 = unsqueeze_165 = None
    unsqueeze_166: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg72_1, -1);  arg72_1 = None
    unsqueeze_167: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
    add_44: "f32[4, 256, 1, 1]" = torch.ops.aten.add.Tensor(mul_65, unsqueeze_167);  mul_65 = unsqueeze_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    relu_17: "f32[4, 256, 1, 1]" = torch.ops.aten.relu.default(add_44);  add_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_24: "f32[4, 1024, 1, 1]" = torch.ops.aten.convolution.default(relu_17, arg73_1, arg74_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_17 = arg73_1 = arg74_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_20: "f32[4, 1, 2, 512]" = torch.ops.aten.view.default(convolution_24, [4, 1, 2, -1]);  convolution_24 = None
    permute_3: "f32[4, 2, 1, 512]" = torch.ops.aten.permute.default(view_20, [0, 2, 1, 3]);  view_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    amax_3: "f32[4, 1, 1, 512]" = torch.ops.aten.amax.default(permute_3, [1], True)
    sub_24: "f32[4, 2, 1, 512]" = torch.ops.aten.sub.Tensor(permute_3, amax_3);  permute_3 = amax_3 = None
    exp_3: "f32[4, 2, 1, 512]" = torch.ops.aten.exp.default(sub_24);  sub_24 = None
    sum_11: "f32[4, 1, 1, 512]" = torch.ops.aten.sum.dim_IntList(exp_3, [1], True)
    div_3: "f32[4, 2, 1, 512]" = torch.ops.aten.div.Tensor(exp_3, sum_11);  exp_3 = sum_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_21: "f32[4, 1024]" = torch.ops.aten.view.default(div_3, [4, -1]);  div_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_22: "f32[4, 1024, 1, 1]" = torch.ops.aten.view.default(view_21, [4, -1, 1, 1]);  view_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_23: "f32[4, 2, 512, 1, 1]" = torch.ops.aten.view.default(view_22, [4, 2, 512, 1, 1]);  view_22 = None
    mul_66: "f32[4, 2, 512, 14, 14]" = torch.ops.aten.mul.Tensor(view_19, view_23);  view_19 = view_23 = None
    sum_12: "f32[4, 512, 14, 14]" = torch.ops.aten.sum.dim_IntList(mul_66, [1]);  mul_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:107, code: out = self.avd_last(out)
    avg_pool2d_4: "f32[4, 512, 7, 7]" = torch.ops.aten.avg_pool2d.default(sum_12, [3, 3], [2, 2], [1, 1]);  sum_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_25: "f32[4, 2048, 7, 7]" = torch.ops.aten.convolution.default(avg_pool2d_4, arg75_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  avg_pool2d_4 = arg75_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    convert_element_type_42: "f32[2048]" = torch.ops.prims.convert_element_type.default(arg146_1, torch.float32);  arg146_1 = None
    convert_element_type_43: "f32[2048]" = torch.ops.prims.convert_element_type.default(arg147_1, torch.float32);  arg147_1 = None
    add_45: "f32[2048]" = torch.ops.aten.add.Tensor(convert_element_type_43, 1e-05);  convert_element_type_43 = None
    sqrt_21: "f32[2048]" = torch.ops.aten.sqrt.default(add_45);  add_45 = None
    reciprocal_21: "f32[2048]" = torch.ops.aten.reciprocal.default(sqrt_21);  sqrt_21 = None
    mul_67: "f32[2048]" = torch.ops.aten.mul.Tensor(reciprocal_21, 1);  reciprocal_21 = None
    unsqueeze_168: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_42, -1);  convert_element_type_42 = None
    unsqueeze_169: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
    unsqueeze_170: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(mul_67, -1);  mul_67 = None
    unsqueeze_171: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
    sub_25: "f32[4, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_169);  convolution_25 = unsqueeze_169 = None
    mul_68: "f32[4, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_25, unsqueeze_171);  sub_25 = unsqueeze_171 = None
    unsqueeze_172: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg76_1, -1);  arg76_1 = None
    unsqueeze_173: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
    mul_69: "f32[4, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(mul_68, unsqueeze_173);  mul_68 = unsqueeze_173 = None
    unsqueeze_174: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg77_1, -1);  arg77_1 = None
    unsqueeze_175: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
    add_46: "f32[4, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_69, unsqueeze_175);  mul_69 = unsqueeze_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:113, code: shortcut = self.downsample(x)
    avg_pool2d_5: "f32[4, 1024, 7, 7]" = torch.ops.aten.avg_pool2d.default(relu_14, [2, 2], [2, 2], [0, 0], True, False);  relu_14 = None
    convolution_26: "f32[4, 2048, 7, 7]" = torch.ops.aten.convolution.default(avg_pool2d_5, arg78_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  avg_pool2d_5 = arg78_1 = None
    convert_element_type_44: "f32[2048]" = torch.ops.prims.convert_element_type.default(arg149_1, torch.float32);  arg149_1 = None
    convert_element_type_45: "f32[2048]" = torch.ops.prims.convert_element_type.default(arg150_1, torch.float32);  arg150_1 = None
    add_47: "f32[2048]" = torch.ops.aten.add.Tensor(convert_element_type_45, 1e-05);  convert_element_type_45 = None
    sqrt_22: "f32[2048]" = torch.ops.aten.sqrt.default(add_47);  add_47 = None
    reciprocal_22: "f32[2048]" = torch.ops.aten.reciprocal.default(sqrt_22);  sqrt_22 = None
    mul_70: "f32[2048]" = torch.ops.aten.mul.Tensor(reciprocal_22, 1);  reciprocal_22 = None
    unsqueeze_176: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_44, -1);  convert_element_type_44 = None
    unsqueeze_177: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
    unsqueeze_178: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(mul_70, -1);  mul_70 = None
    unsqueeze_179: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
    sub_26: "f32[4, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_177);  convolution_26 = unsqueeze_177 = None
    mul_71: "f32[4, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_26, unsqueeze_179);  sub_26 = unsqueeze_179 = None
    unsqueeze_180: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg79_1, -1);  arg79_1 = None
    unsqueeze_181: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
    mul_72: "f32[4, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(mul_71, unsqueeze_181);  mul_71 = unsqueeze_181 = None
    unsqueeze_182: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg80_1, -1);  arg80_1 = None
    unsqueeze_183: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
    add_48: "f32[4, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_72, unsqueeze_183);  mul_72 = unsqueeze_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    add_49: "f32[4, 2048, 7, 7]" = torch.ops.aten.add.Tensor(add_46, add_48);  add_46 = add_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    relu_18: "f32[4, 2048, 7, 7]" = torch.ops.aten.relu.default(add_49);  add_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean_4: "f32[4, 2048, 1, 1]" = torch.ops.aten.mean.dim(relu_18, [-1, -2], True);  relu_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_24: "f32[4, 2048]" = torch.ops.aten.view.default(mean_4, [4, 2048]);  mean_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:538, code: return x if pre_logits else self.fc(x)
    permute_4: "f32[2048, 1000]" = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
    addmm: "f32[4, 1000]" = torch.ops.aten.addmm.default(arg82_1, view_24, permute_4);  arg82_1 = view_24 = permute_4 = None
    return (addmm,)
    