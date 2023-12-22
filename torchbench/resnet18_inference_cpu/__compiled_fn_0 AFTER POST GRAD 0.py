from __future__ import annotations



def forward(self, arg0_1: "f32[64, 3, 7, 7]", arg1_1: "f32[64]", arg2_1: "f32[64]", arg3_1: "f32[64, 64, 3, 3]", arg4_1: "f32[64]", arg5_1: "f32[64]", arg6_1: "f32[64, 64, 3, 3]", arg7_1: "f32[64]", arg8_1: "f32[64]", arg9_1: "f32[64, 64, 3, 3]", arg10_1: "f32[64]", arg11_1: "f32[64]", arg12_1: "f32[64, 64, 3, 3]", arg13_1: "f32[64]", arg14_1: "f32[64]", arg15_1: "f32[128, 64, 3, 3]", arg16_1: "f32[128]", arg17_1: "f32[128]", arg18_1: "f32[128, 128, 3, 3]", arg19_1: "f32[128]", arg20_1: "f32[128]", arg21_1: "f32[128, 64, 1, 1]", arg22_1: "f32[128]", arg23_1: "f32[128]", arg24_1: "f32[128, 128, 3, 3]", arg25_1: "f32[128]", arg26_1: "f32[128]", arg27_1: "f32[128, 128, 3, 3]", arg28_1: "f32[128]", arg29_1: "f32[128]", arg30_1: "f32[256, 128, 3, 3]", arg31_1: "f32[256]", arg32_1: "f32[256]", arg33_1: "f32[256, 256, 3, 3]", arg34_1: "f32[256]", arg35_1: "f32[256]", arg36_1: "f32[256, 128, 1, 1]", arg37_1: "f32[256]", arg38_1: "f32[256]", arg39_1: "f32[256, 256, 3, 3]", arg40_1: "f32[256]", arg41_1: "f32[256]", arg42_1: "f32[256, 256, 3, 3]", arg43_1: "f32[256]", arg44_1: "f32[256]", arg45_1: "f32[512, 256, 3, 3]", arg46_1: "f32[512]", arg47_1: "f32[512]", arg48_1: "f32[512, 512, 3, 3]", arg49_1: "f32[512]", arg50_1: "f32[512]", arg51_1: "f32[512, 256, 1, 1]", arg52_1: "f32[512]", arg53_1: "f32[512]", arg54_1: "f32[512, 512, 3, 3]", arg55_1: "f32[512]", arg56_1: "f32[512]", arg57_1: "f32[512, 512, 3, 3]", arg58_1: "f32[512]", arg59_1: "f32[512]", arg60_1: "f32[1000, 512]", arg61_1: "f32[1000]", arg62_1: "f32[64]", arg63_1: "f32[64]", arg64_1: "i64[]", arg65_1: "f32[64]", arg66_1: "f32[64]", arg67_1: "i64[]", arg68_1: "f32[64]", arg69_1: "f32[64]", arg70_1: "i64[]", arg71_1: "f32[64]", arg72_1: "f32[64]", arg73_1: "i64[]", arg74_1: "f32[64]", arg75_1: "f32[64]", arg76_1: "i64[]", arg77_1: "f32[128]", arg78_1: "f32[128]", arg79_1: "i64[]", arg80_1: "f32[128]", arg81_1: "f32[128]", arg82_1: "i64[]", arg83_1: "f32[128]", arg84_1: "f32[128]", arg85_1: "i64[]", arg86_1: "f32[128]", arg87_1: "f32[128]", arg88_1: "i64[]", arg89_1: "f32[128]", arg90_1: "f32[128]", arg91_1: "i64[]", arg92_1: "f32[256]", arg93_1: "f32[256]", arg94_1: "i64[]", arg95_1: "f32[256]", arg96_1: "f32[256]", arg97_1: "i64[]", arg98_1: "f32[256]", arg99_1: "f32[256]", arg100_1: "i64[]", arg101_1: "f32[256]", arg102_1: "f32[256]", arg103_1: "i64[]", arg104_1: "f32[256]", arg105_1: "f32[256]", arg106_1: "i64[]", arg107_1: "f32[512]", arg108_1: "f32[512]", arg109_1: "i64[]", arg110_1: "f32[512]", arg111_1: "f32[512]", arg112_1: "i64[]", arg113_1: "f32[512]", arg114_1: "f32[512]", arg115_1: "i64[]", arg116_1: "f32[512]", arg117_1: "f32[512]", arg118_1: "i64[]", arg119_1: "f32[512]", arg120_1: "f32[512]", arg121_1: "i64[]", arg122_1: "f32[4, 3, 224, 224]"):
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:268, code: x = self.conv1(x)
    convolution: "f32[4, 64, 112, 112]" = torch.ops.aten.convolution.default(arg122_1, arg0_1, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1);  arg122_1 = arg0_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:269, code: x = self.bn1(x)
    unsqueeze: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg62_1, -1);  arg62_1 = None
    unsqueeze_1: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    sub: "f32[4, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1);  convolution = unsqueeze_1 = None
    add: "f32[64]" = torch.ops.aten.add.Tensor(arg63_1, 1e-05);  arg63_1 = None
    sqrt: "f32[64]" = torch.ops.aten.sqrt.default(add);  add = None
    reciprocal: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt);  sqrt = None
    mul: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
    unsqueeze_2: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul, -1);  mul = None
    unsqueeze_3: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    mul_1: "f32[4, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub, unsqueeze_3);  sub = unsqueeze_3 = None
    unsqueeze_4: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg1_1, -1);  arg1_1 = None
    unsqueeze_5: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_2: "f32[4, 64, 112, 112]" = torch.ops.aten.mul.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None
    unsqueeze_6: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
    unsqueeze_7: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_1: "f32[4, 64, 112, 112]" = torch.ops.aten.add.Tensor(mul_2, unsqueeze_7);  mul_2 = unsqueeze_7 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:270, code: x = self.relu(x)
    relu: "f32[4, 64, 112, 112]" = torch.ops.aten.relu.default(add_1);  add_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:271, code: x = self.maxpool(x)
    max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices.default(relu, [3, 3], [2, 2], [1, 1]);  relu = None
    getitem: "f32[4, 64, 56, 56]" = max_pool2d_with_indices[0];  max_pool2d_with_indices = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:92, code: out = self.conv1(x)
    convolution_1: "f32[4, 64, 56, 56]" = torch.ops.aten.convolution.default(getitem, arg3_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg3_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:93, code: out = self.bn1(out)
    unsqueeze_8: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg65_1, -1);  arg65_1 = None
    unsqueeze_9: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    sub_1: "f32[4, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_9);  convolution_1 = unsqueeze_9 = None
    add_2: "f32[64]" = torch.ops.aten.add.Tensor(arg66_1, 1e-05);  arg66_1 = None
    sqrt_1: "f32[64]" = torch.ops.aten.sqrt.default(add_2);  add_2 = None
    reciprocal_1: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_1);  sqrt_1 = None
    mul_3: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_1, 1);  reciprocal_1 = None
    unsqueeze_10: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_3, -1);  mul_3 = None
    unsqueeze_11: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    mul_4: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_1, unsqueeze_11);  sub_1 = unsqueeze_11 = None
    unsqueeze_12: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
    unsqueeze_13: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_5: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_4, unsqueeze_13);  mul_4 = unsqueeze_13 = None
    unsqueeze_14: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
    unsqueeze_15: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_3: "f32[4, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_5, unsqueeze_15);  mul_5 = unsqueeze_15 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:94, code: out = self.relu(out)
    relu_1: "f32[4, 64, 56, 56]" = torch.ops.aten.relu.default(add_3);  add_3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:96, code: out = self.conv2(out)
    convolution_2: "f32[4, 64, 56, 56]" = torch.ops.aten.convolution.default(relu_1, arg6_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_1 = arg6_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:97, code: out = self.bn2(out)
    unsqueeze_16: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg68_1, -1);  arg68_1 = None
    unsqueeze_17: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    sub_2: "f32[4, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_17);  convolution_2 = unsqueeze_17 = None
    add_4: "f32[64]" = torch.ops.aten.add.Tensor(arg69_1, 1e-05);  arg69_1 = None
    sqrt_2: "f32[64]" = torch.ops.aten.sqrt.default(add_4);  add_4 = None
    reciprocal_2: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_2);  sqrt_2 = None
    mul_6: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_2, 1);  reciprocal_2 = None
    unsqueeze_18: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_6, -1);  mul_6 = None
    unsqueeze_19: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    mul_7: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_2, unsqueeze_19);  sub_2 = unsqueeze_19 = None
    unsqueeze_20: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
    unsqueeze_21: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_8: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_21);  mul_7 = unsqueeze_21 = None
    unsqueeze_22: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg8_1, -1);  arg8_1 = None
    unsqueeze_23: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_5: "f32[4, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_8, unsqueeze_23);  mul_8 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:102, code: out += identity
    add_6: "f32[4, 64, 56, 56]" = torch.ops.aten.add.Tensor(add_5, getitem);  add_5 = getitem = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:103, code: out = self.relu(out)
    relu_2: "f32[4, 64, 56, 56]" = torch.ops.aten.relu.default(add_6);  add_6 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:92, code: out = self.conv1(x)
    convolution_3: "f32[4, 64, 56, 56]" = torch.ops.aten.convolution.default(relu_2, arg9_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg9_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:93, code: out = self.bn1(out)
    unsqueeze_24: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg71_1, -1);  arg71_1 = None
    unsqueeze_25: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
    sub_3: "f32[4, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_25);  convolution_3 = unsqueeze_25 = None
    add_7: "f32[64]" = torch.ops.aten.add.Tensor(arg72_1, 1e-05);  arg72_1 = None
    sqrt_3: "f32[64]" = torch.ops.aten.sqrt.default(add_7);  add_7 = None
    reciprocal_3: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_3);  sqrt_3 = None
    mul_9: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_3, 1);  reciprocal_3 = None
    unsqueeze_26: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_9, -1);  mul_9 = None
    unsqueeze_27: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
    mul_10: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_3, unsqueeze_27);  sub_3 = unsqueeze_27 = None
    unsqueeze_28: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
    unsqueeze_29: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    mul_11: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_10, unsqueeze_29);  mul_10 = unsqueeze_29 = None
    unsqueeze_30: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg11_1, -1);  arg11_1 = None
    unsqueeze_31: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
    add_8: "f32[4, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_11, unsqueeze_31);  mul_11 = unsqueeze_31 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:94, code: out = self.relu(out)
    relu_3: "f32[4, 64, 56, 56]" = torch.ops.aten.relu.default(add_8);  add_8 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:96, code: out = self.conv2(out)
    convolution_4: "f32[4, 64, 56, 56]" = torch.ops.aten.convolution.default(relu_3, arg12_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_3 = arg12_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:97, code: out = self.bn2(out)
    unsqueeze_32: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg74_1, -1);  arg74_1 = None
    unsqueeze_33: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    sub_4: "f32[4, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_33);  convolution_4 = unsqueeze_33 = None
    add_9: "f32[64]" = torch.ops.aten.add.Tensor(arg75_1, 1e-05);  arg75_1 = None
    sqrt_4: "f32[64]" = torch.ops.aten.sqrt.default(add_9);  add_9 = None
    reciprocal_4: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_4);  sqrt_4 = None
    mul_12: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_4, 1);  reciprocal_4 = None
    unsqueeze_34: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_12, -1);  mul_12 = None
    unsqueeze_35: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
    mul_13: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_4, unsqueeze_35);  sub_4 = unsqueeze_35 = None
    unsqueeze_36: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg13_1, -1);  arg13_1 = None
    unsqueeze_37: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
    mul_14: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_13, unsqueeze_37);  mul_13 = unsqueeze_37 = None
    unsqueeze_38: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
    unsqueeze_39: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
    add_10: "f32[4, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_14, unsqueeze_39);  mul_14 = unsqueeze_39 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:102, code: out += identity
    add_11: "f32[4, 64, 56, 56]" = torch.ops.aten.add.Tensor(add_10, relu_2);  add_10 = relu_2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:103, code: out = self.relu(out)
    relu_4: "f32[4, 64, 56, 56]" = torch.ops.aten.relu.default(add_11);  add_11 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:92, code: out = self.conv1(x)
    convolution_5: "f32[4, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_4, arg15_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg15_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:93, code: out = self.bn1(out)
    unsqueeze_40: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg77_1, -1);  arg77_1 = None
    unsqueeze_41: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    sub_5: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_41);  convolution_5 = unsqueeze_41 = None
    add_12: "f32[128]" = torch.ops.aten.add.Tensor(arg78_1, 1e-05);  arg78_1 = None
    sqrt_5: "f32[128]" = torch.ops.aten.sqrt.default(add_12);  add_12 = None
    reciprocal_5: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_5);  sqrt_5 = None
    mul_15: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_5, 1);  reciprocal_5 = None
    unsqueeze_42: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_15, -1);  mul_15 = None
    unsqueeze_43: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    mul_16: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_5, unsqueeze_43);  sub_5 = unsqueeze_43 = None
    unsqueeze_44: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg16_1, -1);  arg16_1 = None
    unsqueeze_45: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    mul_17: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_16, unsqueeze_45);  mul_16 = unsqueeze_45 = None
    unsqueeze_46: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg17_1, -1);  arg17_1 = None
    unsqueeze_47: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
    add_13: "f32[4, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_17, unsqueeze_47);  mul_17 = unsqueeze_47 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:94, code: out = self.relu(out)
    relu_5: "f32[4, 128, 28, 28]" = torch.ops.aten.relu.default(add_13);  add_13 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:96, code: out = self.conv2(out)
    convolution_6: "f32[4, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_5, arg18_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_5 = arg18_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:97, code: out = self.bn2(out)
    unsqueeze_48: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg80_1, -1);  arg80_1 = None
    unsqueeze_49: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
    sub_6: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_49);  convolution_6 = unsqueeze_49 = None
    add_14: "f32[128]" = torch.ops.aten.add.Tensor(arg81_1, 1e-05);  arg81_1 = None
    sqrt_6: "f32[128]" = torch.ops.aten.sqrt.default(add_14);  add_14 = None
    reciprocal_6: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_6);  sqrt_6 = None
    mul_18: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_6, 1);  reciprocal_6 = None
    unsqueeze_50: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_18, -1);  mul_18 = None
    unsqueeze_51: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
    mul_19: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_6, unsqueeze_51);  sub_6 = unsqueeze_51 = None
    unsqueeze_52: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg19_1, -1);  arg19_1 = None
    unsqueeze_53: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    mul_20: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_19, unsqueeze_53);  mul_19 = unsqueeze_53 = None
    unsqueeze_54: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg20_1, -1);  arg20_1 = None
    unsqueeze_55: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
    add_15: "f32[4, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_55);  mul_20 = unsqueeze_55 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:100, code: identity = self.downsample(x)
    convolution_7: "f32[4, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_4, arg21_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  relu_4 = arg21_1 = None
    unsqueeze_56: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg83_1, -1);  arg83_1 = None
    unsqueeze_57: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    sub_7: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_57);  convolution_7 = unsqueeze_57 = None
    add_16: "f32[128]" = torch.ops.aten.add.Tensor(arg84_1, 1e-05);  arg84_1 = None
    sqrt_7: "f32[128]" = torch.ops.aten.sqrt.default(add_16);  add_16 = None
    reciprocal_7: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_7);  sqrt_7 = None
    mul_21: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_7, 1);  reciprocal_7 = None
    unsqueeze_58: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_21, -1);  mul_21 = None
    unsqueeze_59: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
    mul_22: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_7, unsqueeze_59);  sub_7 = unsqueeze_59 = None
    unsqueeze_60: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg22_1, -1);  arg22_1 = None
    unsqueeze_61: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
    mul_23: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_22, unsqueeze_61);  mul_22 = unsqueeze_61 = None
    unsqueeze_62: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg23_1, -1);  arg23_1 = None
    unsqueeze_63: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
    add_17: "f32[4, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_23, unsqueeze_63);  mul_23 = unsqueeze_63 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:102, code: out += identity
    add_18: "f32[4, 128, 28, 28]" = torch.ops.aten.add.Tensor(add_15, add_17);  add_15 = add_17 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:103, code: out = self.relu(out)
    relu_6: "f32[4, 128, 28, 28]" = torch.ops.aten.relu.default(add_18);  add_18 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:92, code: out = self.conv1(x)
    convolution_8: "f32[4, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_6, arg24_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg24_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:93, code: out = self.bn1(out)
    unsqueeze_64: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg86_1, -1);  arg86_1 = None
    unsqueeze_65: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
    sub_8: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_65);  convolution_8 = unsqueeze_65 = None
    add_19: "f32[128]" = torch.ops.aten.add.Tensor(arg87_1, 1e-05);  arg87_1 = None
    sqrt_8: "f32[128]" = torch.ops.aten.sqrt.default(add_19);  add_19 = None
    reciprocal_8: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_8);  sqrt_8 = None
    mul_24: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_8, 1);  reciprocal_8 = None
    unsqueeze_66: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_24, -1);  mul_24 = None
    unsqueeze_67: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
    mul_25: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_8, unsqueeze_67);  sub_8 = unsqueeze_67 = None
    unsqueeze_68: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg25_1, -1);  arg25_1 = None
    unsqueeze_69: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    mul_26: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_25, unsqueeze_69);  mul_25 = unsqueeze_69 = None
    unsqueeze_70: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg26_1, -1);  arg26_1 = None
    unsqueeze_71: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    add_20: "f32[4, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_26, unsqueeze_71);  mul_26 = unsqueeze_71 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:94, code: out = self.relu(out)
    relu_7: "f32[4, 128, 28, 28]" = torch.ops.aten.relu.default(add_20);  add_20 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:96, code: out = self.conv2(out)
    convolution_9: "f32[4, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_7, arg27_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_7 = arg27_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:97, code: out = self.bn2(out)
    unsqueeze_72: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg89_1, -1);  arg89_1 = None
    unsqueeze_73: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
    sub_9: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_73);  convolution_9 = unsqueeze_73 = None
    add_21: "f32[128]" = torch.ops.aten.add.Tensor(arg90_1, 1e-05);  arg90_1 = None
    sqrt_9: "f32[128]" = torch.ops.aten.sqrt.default(add_21);  add_21 = None
    reciprocal_9: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_9);  sqrt_9 = None
    mul_27: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_9, 1);  reciprocal_9 = None
    unsqueeze_74: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_27, -1);  mul_27 = None
    unsqueeze_75: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
    mul_28: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_9, unsqueeze_75);  sub_9 = unsqueeze_75 = None
    unsqueeze_76: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg28_1, -1);  arg28_1 = None
    unsqueeze_77: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
    mul_29: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_77);  mul_28 = unsqueeze_77 = None
    unsqueeze_78: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg29_1, -1);  arg29_1 = None
    unsqueeze_79: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    add_22: "f32[4, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_29, unsqueeze_79);  mul_29 = unsqueeze_79 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:102, code: out += identity
    add_23: "f32[4, 128, 28, 28]" = torch.ops.aten.add.Tensor(add_22, relu_6);  add_22 = relu_6 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:103, code: out = self.relu(out)
    relu_8: "f32[4, 128, 28, 28]" = torch.ops.aten.relu.default(add_23);  add_23 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:92, code: out = self.conv1(x)
    convolution_10: "f32[4, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_8, arg30_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg30_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:93, code: out = self.bn1(out)
    unsqueeze_80: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg92_1, -1);  arg92_1 = None
    unsqueeze_81: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
    sub_10: "f32[4, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_81);  convolution_10 = unsqueeze_81 = None
    add_24: "f32[256]" = torch.ops.aten.add.Tensor(arg93_1, 1e-05);  arg93_1 = None
    sqrt_10: "f32[256]" = torch.ops.aten.sqrt.default(add_24);  add_24 = None
    reciprocal_10: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_10);  sqrt_10 = None
    mul_30: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_10, 1);  reciprocal_10 = None
    unsqueeze_82: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_30, -1);  mul_30 = None
    unsqueeze_83: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    mul_31: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_10, unsqueeze_83);  sub_10 = unsqueeze_83 = None
    unsqueeze_84: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg31_1, -1);  arg31_1 = None
    unsqueeze_85: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    mul_32: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_31, unsqueeze_85);  mul_31 = unsqueeze_85 = None
    unsqueeze_86: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg32_1, -1);  arg32_1 = None
    unsqueeze_87: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
    add_25: "f32[4, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_32, unsqueeze_87);  mul_32 = unsqueeze_87 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:94, code: out = self.relu(out)
    relu_9: "f32[4, 256, 14, 14]" = torch.ops.aten.relu.default(add_25);  add_25 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:96, code: out = self.conv2(out)
    convolution_11: "f32[4, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_9, arg33_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_9 = arg33_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:97, code: out = self.bn2(out)
    unsqueeze_88: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg95_1, -1);  arg95_1 = None
    unsqueeze_89: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
    sub_11: "f32[4, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_89);  convolution_11 = unsqueeze_89 = None
    add_26: "f32[256]" = torch.ops.aten.add.Tensor(arg96_1, 1e-05);  arg96_1 = None
    sqrt_11: "f32[256]" = torch.ops.aten.sqrt.default(add_26);  add_26 = None
    reciprocal_11: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_11);  sqrt_11 = None
    mul_33: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_11, 1);  reciprocal_11 = None
    unsqueeze_90: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_33, -1);  mul_33 = None
    unsqueeze_91: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    mul_34: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_11, unsqueeze_91);  sub_11 = unsqueeze_91 = None
    unsqueeze_92: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg34_1, -1);  arg34_1 = None
    unsqueeze_93: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
    mul_35: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_34, unsqueeze_93);  mul_34 = unsqueeze_93 = None
    unsqueeze_94: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg35_1, -1);  arg35_1 = None
    unsqueeze_95: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
    add_27: "f32[4, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_35, unsqueeze_95);  mul_35 = unsqueeze_95 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:100, code: identity = self.downsample(x)
    convolution_12: "f32[4, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_8, arg36_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  relu_8 = arg36_1 = None
    unsqueeze_96: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg98_1, -1);  arg98_1 = None
    unsqueeze_97: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
    sub_12: "f32[4, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_97);  convolution_12 = unsqueeze_97 = None
    add_28: "f32[256]" = torch.ops.aten.add.Tensor(arg99_1, 1e-05);  arg99_1 = None
    sqrt_12: "f32[256]" = torch.ops.aten.sqrt.default(add_28);  add_28 = None
    reciprocal_12: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_12);  sqrt_12 = None
    mul_36: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_12, 1);  reciprocal_12 = None
    unsqueeze_98: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_36, -1);  mul_36 = None
    unsqueeze_99: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
    mul_37: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_12, unsqueeze_99);  sub_12 = unsqueeze_99 = None
    unsqueeze_100: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg37_1, -1);  arg37_1 = None
    unsqueeze_101: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
    mul_38: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_37, unsqueeze_101);  mul_37 = unsqueeze_101 = None
    unsqueeze_102: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg38_1, -1);  arg38_1 = None
    unsqueeze_103: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    add_29: "f32[4, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_38, unsqueeze_103);  mul_38 = unsqueeze_103 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:102, code: out += identity
    add_30: "f32[4, 256, 14, 14]" = torch.ops.aten.add.Tensor(add_27, add_29);  add_27 = add_29 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:103, code: out = self.relu(out)
    relu_10: "f32[4, 256, 14, 14]" = torch.ops.aten.relu.default(add_30);  add_30 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:92, code: out = self.conv1(x)
    convolution_13: "f32[4, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_10, arg39_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg39_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:93, code: out = self.bn1(out)
    unsqueeze_104: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg101_1, -1);  arg101_1 = None
    unsqueeze_105: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
    sub_13: "f32[4, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_105);  convolution_13 = unsqueeze_105 = None
    add_31: "f32[256]" = torch.ops.aten.add.Tensor(arg102_1, 1e-05);  arg102_1 = None
    sqrt_13: "f32[256]" = torch.ops.aten.sqrt.default(add_31);  add_31 = None
    reciprocal_13: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_13);  sqrt_13 = None
    mul_39: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_13, 1);  reciprocal_13 = None
    unsqueeze_106: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_39, -1);  mul_39 = None
    unsqueeze_107: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
    mul_40: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_13, unsqueeze_107);  sub_13 = unsqueeze_107 = None
    unsqueeze_108: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg40_1, -1);  arg40_1 = None
    unsqueeze_109: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
    mul_41: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_40, unsqueeze_109);  mul_40 = unsqueeze_109 = None
    unsqueeze_110: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg41_1, -1);  arg41_1 = None
    unsqueeze_111: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
    add_32: "f32[4, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_41, unsqueeze_111);  mul_41 = unsqueeze_111 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:94, code: out = self.relu(out)
    relu_11: "f32[4, 256, 14, 14]" = torch.ops.aten.relu.default(add_32);  add_32 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:96, code: out = self.conv2(out)
    convolution_14: "f32[4, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_11, arg42_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_11 = arg42_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:97, code: out = self.bn2(out)
    unsqueeze_112: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg104_1, -1);  arg104_1 = None
    unsqueeze_113: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
    sub_14: "f32[4, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_113);  convolution_14 = unsqueeze_113 = None
    add_33: "f32[256]" = torch.ops.aten.add.Tensor(arg105_1, 1e-05);  arg105_1 = None
    sqrt_14: "f32[256]" = torch.ops.aten.sqrt.default(add_33);  add_33 = None
    reciprocal_14: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_14);  sqrt_14 = None
    mul_42: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_14, 1);  reciprocal_14 = None
    unsqueeze_114: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_42, -1);  mul_42 = None
    unsqueeze_115: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
    mul_43: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_14, unsqueeze_115);  sub_14 = unsqueeze_115 = None
    unsqueeze_116: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg43_1, -1);  arg43_1 = None
    unsqueeze_117: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
    mul_44: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_43, unsqueeze_117);  mul_43 = unsqueeze_117 = None
    unsqueeze_118: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg44_1, -1);  arg44_1 = None
    unsqueeze_119: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
    add_34: "f32[4, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_44, unsqueeze_119);  mul_44 = unsqueeze_119 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:102, code: out += identity
    add_35: "f32[4, 256, 14, 14]" = torch.ops.aten.add.Tensor(add_34, relu_10);  add_34 = relu_10 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:103, code: out = self.relu(out)
    relu_12: "f32[4, 256, 14, 14]" = torch.ops.aten.relu.default(add_35);  add_35 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:92, code: out = self.conv1(x)
    convolution_15: "f32[4, 512, 7, 7]" = torch.ops.aten.convolution.default(relu_12, arg45_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg45_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:93, code: out = self.bn1(out)
    unsqueeze_120: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg107_1, -1);  arg107_1 = None
    unsqueeze_121: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
    sub_15: "f32[4, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_121);  convolution_15 = unsqueeze_121 = None
    add_36: "f32[512]" = torch.ops.aten.add.Tensor(arg108_1, 1e-05);  arg108_1 = None
    sqrt_15: "f32[512]" = torch.ops.aten.sqrt.default(add_36);  add_36 = None
    reciprocal_15: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_15);  sqrt_15 = None
    mul_45: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_15, 1);  reciprocal_15 = None
    unsqueeze_122: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_45, -1);  mul_45 = None
    unsqueeze_123: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
    mul_46: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(sub_15, unsqueeze_123);  sub_15 = unsqueeze_123 = None
    unsqueeze_124: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg46_1, -1);  arg46_1 = None
    unsqueeze_125: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
    mul_47: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(mul_46, unsqueeze_125);  mul_46 = unsqueeze_125 = None
    unsqueeze_126: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg47_1, -1);  arg47_1 = None
    unsqueeze_127: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
    add_37: "f32[4, 512, 7, 7]" = torch.ops.aten.add.Tensor(mul_47, unsqueeze_127);  mul_47 = unsqueeze_127 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:94, code: out = self.relu(out)
    relu_13: "f32[4, 512, 7, 7]" = torch.ops.aten.relu.default(add_37);  add_37 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:96, code: out = self.conv2(out)
    convolution_16: "f32[4, 512, 7, 7]" = torch.ops.aten.convolution.default(relu_13, arg48_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_13 = arg48_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:97, code: out = self.bn2(out)
    unsqueeze_128: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg110_1, -1);  arg110_1 = None
    unsqueeze_129: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
    sub_16: "f32[4, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_129);  convolution_16 = unsqueeze_129 = None
    add_38: "f32[512]" = torch.ops.aten.add.Tensor(arg111_1, 1e-05);  arg111_1 = None
    sqrt_16: "f32[512]" = torch.ops.aten.sqrt.default(add_38);  add_38 = None
    reciprocal_16: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_16);  sqrt_16 = None
    mul_48: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_16, 1);  reciprocal_16 = None
    unsqueeze_130: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_48, -1);  mul_48 = None
    unsqueeze_131: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
    mul_49: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(sub_16, unsqueeze_131);  sub_16 = unsqueeze_131 = None
    unsqueeze_132: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg49_1, -1);  arg49_1 = None
    unsqueeze_133: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
    mul_50: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_133);  mul_49 = unsqueeze_133 = None
    unsqueeze_134: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg50_1, -1);  arg50_1 = None
    unsqueeze_135: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
    add_39: "f32[4, 512, 7, 7]" = torch.ops.aten.add.Tensor(mul_50, unsqueeze_135);  mul_50 = unsqueeze_135 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:100, code: identity = self.downsample(x)
    convolution_17: "f32[4, 512, 7, 7]" = torch.ops.aten.convolution.default(relu_12, arg51_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  relu_12 = arg51_1 = None
    unsqueeze_136: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg113_1, -1);  arg113_1 = None
    unsqueeze_137: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
    sub_17: "f32[4, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_137);  convolution_17 = unsqueeze_137 = None
    add_40: "f32[512]" = torch.ops.aten.add.Tensor(arg114_1, 1e-05);  arg114_1 = None
    sqrt_17: "f32[512]" = torch.ops.aten.sqrt.default(add_40);  add_40 = None
    reciprocal_17: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_17);  sqrt_17 = None
    mul_51: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_17, 1);  reciprocal_17 = None
    unsqueeze_138: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_51, -1);  mul_51 = None
    unsqueeze_139: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
    mul_52: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(sub_17, unsqueeze_139);  sub_17 = unsqueeze_139 = None
    unsqueeze_140: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg52_1, -1);  arg52_1 = None
    unsqueeze_141: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
    mul_53: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(mul_52, unsqueeze_141);  mul_52 = unsqueeze_141 = None
    unsqueeze_142: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg53_1, -1);  arg53_1 = None
    unsqueeze_143: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
    add_41: "f32[4, 512, 7, 7]" = torch.ops.aten.add.Tensor(mul_53, unsqueeze_143);  mul_53 = unsqueeze_143 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:102, code: out += identity
    add_42: "f32[4, 512, 7, 7]" = torch.ops.aten.add.Tensor(add_39, add_41);  add_39 = add_41 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:103, code: out = self.relu(out)
    relu_14: "f32[4, 512, 7, 7]" = torch.ops.aten.relu.default(add_42);  add_42 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:92, code: out = self.conv1(x)
    convolution_18: "f32[4, 512, 7, 7]" = torch.ops.aten.convolution.default(relu_14, arg54_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg54_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:93, code: out = self.bn1(out)
    unsqueeze_144: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg116_1, -1);  arg116_1 = None
    unsqueeze_145: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
    sub_18: "f32[4, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_145);  convolution_18 = unsqueeze_145 = None
    add_43: "f32[512]" = torch.ops.aten.add.Tensor(arg117_1, 1e-05);  arg117_1 = None
    sqrt_18: "f32[512]" = torch.ops.aten.sqrt.default(add_43);  add_43 = None
    reciprocal_18: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_18);  sqrt_18 = None
    mul_54: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_18, 1);  reciprocal_18 = None
    unsqueeze_146: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_54, -1);  mul_54 = None
    unsqueeze_147: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
    mul_55: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(sub_18, unsqueeze_147);  sub_18 = unsqueeze_147 = None
    unsqueeze_148: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg55_1, -1);  arg55_1 = None
    unsqueeze_149: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
    mul_56: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(mul_55, unsqueeze_149);  mul_55 = unsqueeze_149 = None
    unsqueeze_150: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg56_1, -1);  arg56_1 = None
    unsqueeze_151: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
    add_44: "f32[4, 512, 7, 7]" = torch.ops.aten.add.Tensor(mul_56, unsqueeze_151);  mul_56 = unsqueeze_151 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:94, code: out = self.relu(out)
    relu_15: "f32[4, 512, 7, 7]" = torch.ops.aten.relu.default(add_44);  add_44 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:96, code: out = self.conv2(out)
    convolution_19: "f32[4, 512, 7, 7]" = torch.ops.aten.convolution.default(relu_15, arg57_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_15 = arg57_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:97, code: out = self.bn2(out)
    unsqueeze_152: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg119_1, -1);  arg119_1 = None
    unsqueeze_153: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
    sub_19: "f32[4, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_153);  convolution_19 = unsqueeze_153 = None
    add_45: "f32[512]" = torch.ops.aten.add.Tensor(arg120_1, 1e-05);  arg120_1 = None
    sqrt_19: "f32[512]" = torch.ops.aten.sqrt.default(add_45);  add_45 = None
    reciprocal_19: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_19);  sqrt_19 = None
    mul_57: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_19, 1);  reciprocal_19 = None
    unsqueeze_154: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_57, -1);  mul_57 = None
    unsqueeze_155: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
    mul_58: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(sub_19, unsqueeze_155);  sub_19 = unsqueeze_155 = None
    unsqueeze_156: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg58_1, -1);  arg58_1 = None
    unsqueeze_157: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
    mul_59: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(mul_58, unsqueeze_157);  mul_58 = unsqueeze_157 = None
    unsqueeze_158: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg59_1, -1);  arg59_1 = None
    unsqueeze_159: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
    add_46: "f32[4, 512, 7, 7]" = torch.ops.aten.add.Tensor(mul_59, unsqueeze_159);  mul_59 = unsqueeze_159 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:102, code: out += identity
    add_47: "f32[4, 512, 7, 7]" = torch.ops.aten.add.Tensor(add_46, relu_14);  add_46 = relu_14 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:103, code: out = self.relu(out)
    relu_16: "f32[4, 512, 7, 7]" = torch.ops.aten.relu.default(add_47);  add_47 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:278, code: x = self.avgpool(x)
    mean: "f32[4, 512, 1, 1]" = torch.ops.aten.mean.dim(relu_16, [-1, -2], True);  relu_16 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:279, code: x = torch.flatten(x, 1)
    view: "f32[4, 512]" = torch.ops.aten.reshape.default(mean, [4, 512]);  mean = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:280, code: x = self.fc(x)
    permute: "f32[512, 1000]" = torch.ops.aten.permute.default(arg60_1, [1, 0]);  arg60_1 = None
    addmm: "f32[4, 1000]" = torch.ops.aten.addmm.default(arg61_1, view, permute);  arg61_1 = view = permute = None
    return (addmm,)
    