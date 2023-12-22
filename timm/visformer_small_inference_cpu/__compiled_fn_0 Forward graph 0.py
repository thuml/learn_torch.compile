from __future__ import annotations



def forward(self, arg0_1: "f32[1, 192, 28, 28]", arg1_1: "f32[1, 384, 14, 14]", arg2_1: "f32[1, 768, 7, 7]", arg3_1: "f32[32, 3, 7, 7]", arg4_1: "f32[32]", arg5_1: "f32[32]", arg6_1: "f32[192, 32, 4, 4]", arg7_1: "f32[192]", arg8_1: "f32[192]", arg9_1: "f32[192]", arg10_1: "f32[192]", arg11_1: "f32[192]", arg12_1: "f32[384, 192, 1, 1]", arg13_1: "f32[384, 48, 3, 3]", arg14_1: "f32[192, 384, 1, 1]", arg15_1: "f32[192]", arg16_1: "f32[192]", arg17_1: "f32[384, 192, 1, 1]", arg18_1: "f32[384, 48, 3, 3]", arg19_1: "f32[192, 384, 1, 1]", arg20_1: "f32[192]", arg21_1: "f32[192]", arg22_1: "f32[384, 192, 1, 1]", arg23_1: "f32[384, 48, 3, 3]", arg24_1: "f32[192, 384, 1, 1]", arg25_1: "f32[192]", arg26_1: "f32[192]", arg27_1: "f32[384, 192, 1, 1]", arg28_1: "f32[384, 48, 3, 3]", arg29_1: "f32[192, 384, 1, 1]", arg30_1: "f32[192]", arg31_1: "f32[192]", arg32_1: "f32[384, 192, 1, 1]", arg33_1: "f32[384, 48, 3, 3]", arg34_1: "f32[192, 384, 1, 1]", arg35_1: "f32[192]", arg36_1: "f32[192]", arg37_1: "f32[384, 192, 1, 1]", arg38_1: "f32[384, 48, 3, 3]", arg39_1: "f32[192, 384, 1, 1]", arg40_1: "f32[192]", arg41_1: "f32[192]", arg42_1: "f32[384, 192, 1, 1]", arg43_1: "f32[384, 48, 3, 3]", arg44_1: "f32[192, 384, 1, 1]", arg45_1: "f32[384, 192, 2, 2]", arg46_1: "f32[384]", arg47_1: "f32[384]", arg48_1: "f32[384]", arg49_1: "f32[384]", arg50_1: "f32[384]", arg51_1: "f32[1152, 384, 1, 1]", arg52_1: "f32[384, 384, 1, 1]", arg53_1: "f32[384]", arg54_1: "f32[384]", arg55_1: "f32[1536, 384, 1, 1]", arg56_1: "f32[384, 1536, 1, 1]", arg57_1: "f32[384]", arg58_1: "f32[384]", arg59_1: "f32[1152, 384, 1, 1]", arg60_1: "f32[384, 384, 1, 1]", arg61_1: "f32[384]", arg62_1: "f32[384]", arg63_1: "f32[1536, 384, 1, 1]", arg64_1: "f32[384, 1536, 1, 1]", arg65_1: "f32[384]", arg66_1: "f32[384]", arg67_1: "f32[1152, 384, 1, 1]", arg68_1: "f32[384, 384, 1, 1]", arg69_1: "f32[384]", arg70_1: "f32[384]", arg71_1: "f32[1536, 384, 1, 1]", arg72_1: "f32[384, 1536, 1, 1]", arg73_1: "f32[384]", arg74_1: "f32[384]", arg75_1: "f32[1152, 384, 1, 1]", arg76_1: "f32[384, 384, 1, 1]", arg77_1: "f32[384]", arg78_1: "f32[384]", arg79_1: "f32[1536, 384, 1, 1]", arg80_1: "f32[384, 1536, 1, 1]", arg81_1: "f32[768, 384, 2, 2]", arg82_1: "f32[768]", arg83_1: "f32[768]", arg84_1: "f32[768]", arg85_1: "f32[768]", arg86_1: "f32[768]", arg87_1: "f32[2304, 768, 1, 1]", arg88_1: "f32[768, 768, 1, 1]", arg89_1: "f32[768]", arg90_1: "f32[768]", arg91_1: "f32[3072, 768, 1, 1]", arg92_1: "f32[768, 3072, 1, 1]", arg93_1: "f32[768]", arg94_1: "f32[768]", arg95_1: "f32[2304, 768, 1, 1]", arg96_1: "f32[768, 768, 1, 1]", arg97_1: "f32[768]", arg98_1: "f32[768]", arg99_1: "f32[3072, 768, 1, 1]", arg100_1: "f32[768, 3072, 1, 1]", arg101_1: "f32[768]", arg102_1: "f32[768]", arg103_1: "f32[2304, 768, 1, 1]", arg104_1: "f32[768, 768, 1, 1]", arg105_1: "f32[768]", arg106_1: "f32[768]", arg107_1: "f32[3072, 768, 1, 1]", arg108_1: "f32[768, 3072, 1, 1]", arg109_1: "f32[768]", arg110_1: "f32[768]", arg111_1: "f32[2304, 768, 1, 1]", arg112_1: "f32[768, 768, 1, 1]", arg113_1: "f32[768]", arg114_1: "f32[768]", arg115_1: "f32[3072, 768, 1, 1]", arg116_1: "f32[768, 3072, 1, 1]", arg117_1: "f32[768]", arg118_1: "f32[768]", arg119_1: "f32[1000, 768]", arg120_1: "f32[1000]", arg121_1: "f32[32]", arg122_1: "f32[32]", arg123_1: "i64[]", arg124_1: "f32[192]", arg125_1: "f32[192]", arg126_1: "i64[]", arg127_1: "f32[192]", arg128_1: "f32[192]", arg129_1: "i64[]", arg130_1: "f32[192]", arg131_1: "f32[192]", arg132_1: "i64[]", arg133_1: "f32[192]", arg134_1: "f32[192]", arg135_1: "i64[]", arg136_1: "f32[192]", arg137_1: "f32[192]", arg138_1: "i64[]", arg139_1: "f32[192]", arg140_1: "f32[192]", arg141_1: "i64[]", arg142_1: "f32[192]", arg143_1: "f32[192]", arg144_1: "i64[]", arg145_1: "f32[192]", arg146_1: "f32[192]", arg147_1: "i64[]", arg148_1: "f32[384]", arg149_1: "f32[384]", arg150_1: "i64[]", arg151_1: "f32[384]", arg152_1: "f32[384]", arg153_1: "i64[]", arg154_1: "f32[384]", arg155_1: "f32[384]", arg156_1: "i64[]", arg157_1: "f32[384]", arg158_1: "f32[384]", arg159_1: "i64[]", arg160_1: "f32[384]", arg161_1: "f32[384]", arg162_1: "i64[]", arg163_1: "f32[384]", arg164_1: "f32[384]", arg165_1: "i64[]", arg166_1: "f32[384]", arg167_1: "f32[384]", arg168_1: "i64[]", arg169_1: "f32[384]", arg170_1: "f32[384]", arg171_1: "i64[]", arg172_1: "f32[384]", arg173_1: "f32[384]", arg174_1: "i64[]", arg175_1: "f32[768]", arg176_1: "f32[768]", arg177_1: "i64[]", arg178_1: "f32[768]", arg179_1: "f32[768]", arg180_1: "i64[]", arg181_1: "f32[768]", arg182_1: "f32[768]", arg183_1: "i64[]", arg184_1: "f32[768]", arg185_1: "f32[768]", arg186_1: "i64[]", arg187_1: "f32[768]", arg188_1: "f32[768]", arg189_1: "i64[]", arg190_1: "f32[768]", arg191_1: "f32[768]", arg192_1: "i64[]", arg193_1: "f32[768]", arg194_1: "f32[768]", arg195_1: "i64[]", arg196_1: "f32[768]", arg197_1: "f32[768]", arg198_1: "i64[]", arg199_1: "f32[768]", arg200_1: "f32[768]", arg201_1: "i64[]", arg202_1: "f32[768]", arg203_1: "f32[768]", arg204_1: "i64[]", arg205_1: "f32[8, 3, 224, 224]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:396, code: x = self.stem(x)
    convolution: "f32[8, 32, 112, 112]" = torch.ops.aten.convolution.default(arg205_1, arg3_1, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1);  arg205_1 = arg3_1 = None
    convert_element_type: "f32[32]" = torch.ops.prims.convert_element_type.default(arg121_1, torch.float32);  arg121_1 = None
    convert_element_type_1: "f32[32]" = torch.ops.prims.convert_element_type.default(arg122_1, torch.float32);  arg122_1 = None
    add: "f32[32]" = torch.ops.aten.add.Tensor(convert_element_type_1, 1e-05);  convert_element_type_1 = None
    sqrt: "f32[32]" = torch.ops.aten.sqrt.default(add);  add = None
    reciprocal: "f32[32]" = torch.ops.aten.reciprocal.default(sqrt);  sqrt = None
    mul: "f32[32]" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
    unsqueeze: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type, -1);  convert_element_type = None
    unsqueeze_1: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    unsqueeze_2: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(mul, -1);  mul = None
    unsqueeze_3: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    sub: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1);  convolution = unsqueeze_1 = None
    mul_1: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub, unsqueeze_3);  sub = unsqueeze_3 = None
    unsqueeze_4: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
    unsqueeze_5: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_2: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None
    unsqueeze_6: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
    unsqueeze_7: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_1: "f32[8, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_2, unsqueeze_7);  mul_2 = unsqueeze_7 = None
    relu: "f32[8, 32, 112, 112]" = torch.ops.aten.relu.default(add_1);  add_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution_1: "f32[8, 192, 28, 28]" = torch.ops.aten.convolution.default(relu, arg6_1, arg7_1, [4, 4], [0, 0], [1, 1], False, [0, 0], 1);  relu = arg6_1 = arg7_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:92, code: x = self.norm(x)
    convert_element_type_2: "f32[192]" = torch.ops.prims.convert_element_type.default(arg124_1, torch.float32);  arg124_1 = None
    convert_element_type_3: "f32[192]" = torch.ops.prims.convert_element_type.default(arg125_1, torch.float32);  arg125_1 = None
    add_2: "f32[192]" = torch.ops.aten.add.Tensor(convert_element_type_3, 1e-05);  convert_element_type_3 = None
    sqrt_1: "f32[192]" = torch.ops.aten.sqrt.default(add_2);  add_2 = None
    reciprocal_1: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_1);  sqrt_1 = None
    mul_3: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_1, 1);  reciprocal_1 = None
    unsqueeze_8: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, -1);  convert_element_type_2 = None
    unsqueeze_9: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    unsqueeze_10: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_3, -1);  mul_3 = None
    unsqueeze_11: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    sub_1: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_9);  convolution_1 = unsqueeze_9 = None
    mul_4: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_1, unsqueeze_11);  sub_1 = unsqueeze_11 = None
    unsqueeze_12: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg8_1, -1);  arg8_1 = None
    unsqueeze_13: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_5: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(mul_4, unsqueeze_13);  mul_4 = unsqueeze_13 = None
    unsqueeze_14: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
    unsqueeze_15: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_3: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_5, unsqueeze_15);  mul_5 = unsqueeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:401, code: x = self.pos_drop(x + self.pos_embed1)
    add_4: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_3, arg0_1);  add_3 = arg0_1 = None
    clone: "f32[8, 192, 28, 28]" = torch.ops.aten.clone.default(add_4);  add_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    convert_element_type_4: "f32[192]" = torch.ops.prims.convert_element_type.default(arg127_1, torch.float32);  arg127_1 = None
    convert_element_type_5: "f32[192]" = torch.ops.prims.convert_element_type.default(arg128_1, torch.float32);  arg128_1 = None
    add_5: "f32[192]" = torch.ops.aten.add.Tensor(convert_element_type_5, 1e-05);  convert_element_type_5 = None
    sqrt_2: "f32[192]" = torch.ops.aten.sqrt.default(add_5);  add_5 = None
    reciprocal_2: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_2);  sqrt_2 = None
    mul_6: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_2, 1);  reciprocal_2 = None
    unsqueeze_16: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_4, -1);  convert_element_type_4 = None
    unsqueeze_17: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    unsqueeze_18: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_6, -1);  mul_6 = None
    unsqueeze_19: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    sub_2: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(clone, unsqueeze_17);  unsqueeze_17 = None
    mul_7: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_2, unsqueeze_19);  sub_2 = unsqueeze_19 = None
    unsqueeze_20: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
    unsqueeze_21: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_8: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_21);  mul_7 = unsqueeze_21 = None
    unsqueeze_22: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg11_1, -1);  arg11_1 = None
    unsqueeze_23: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_6: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_8, unsqueeze_23);  mul_8 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    convolution_2: "f32[8, 384, 28, 28]" = torch.ops.aten.convolution.default(add_6, arg12_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_6 = arg12_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_9: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_2, 0.5)
    mul_10: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_2, 0.7071067811865476);  convolution_2 = None
    erf: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_10);  mul_10 = None
    add_7: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_11: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_9, add_7);  mul_9 = add_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:64, code: x = self.drop1(x)
    clone_1: "f32[8, 384, 28, 28]" = torch.ops.aten.clone.default(mul_11);  mul_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:66, code: x = self.conv2(x)
    convolution_3: "f32[8, 384, 28, 28]" = torch.ops.aten.convolution.default(clone_1, arg13_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  clone_1 = arg13_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:67, code: x = self.act2(x)
    mul_12: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_3, 0.5)
    mul_13: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_3, 0.7071067811865476);  convolution_3 = None
    erf_1: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_13);  mul_13 = None
    add_8: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_14: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_12, add_8);  mul_12 = add_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_4: "f32[8, 192, 28, 28]" = torch.ops.aten.convolution.default(mul_14, arg14_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_14 = arg14_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:69, code: x = self.drop3(x)
    clone_2: "f32[8, 192, 28, 28]" = torch.ops.aten.clone.default(convolution_4);  convolution_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_9: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(clone, clone_2);  clone = clone_2 = None
    convert_element_type_6: "f32[192]" = torch.ops.prims.convert_element_type.default(arg130_1, torch.float32);  arg130_1 = None
    convert_element_type_7: "f32[192]" = torch.ops.prims.convert_element_type.default(arg131_1, torch.float32);  arg131_1 = None
    add_10: "f32[192]" = torch.ops.aten.add.Tensor(convert_element_type_7, 1e-05);  convert_element_type_7 = None
    sqrt_3: "f32[192]" = torch.ops.aten.sqrt.default(add_10);  add_10 = None
    reciprocal_3: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_3);  sqrt_3 = None
    mul_15: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_3, 1);  reciprocal_3 = None
    unsqueeze_24: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_6, -1);  convert_element_type_6 = None
    unsqueeze_25: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
    unsqueeze_26: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_15, -1);  mul_15 = None
    unsqueeze_27: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
    sub_3: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(add_9, unsqueeze_25);  unsqueeze_25 = None
    mul_16: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_3, unsqueeze_27);  sub_3 = unsqueeze_27 = None
    unsqueeze_28: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
    unsqueeze_29: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    mul_17: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(mul_16, unsqueeze_29);  mul_16 = unsqueeze_29 = None
    unsqueeze_30: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg16_1, -1);  arg16_1 = None
    unsqueeze_31: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
    add_11: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_17, unsqueeze_31);  mul_17 = unsqueeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    convolution_5: "f32[8, 384, 28, 28]" = torch.ops.aten.convolution.default(add_11, arg17_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_11 = arg17_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_18: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_5, 0.5)
    mul_19: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_5, 0.7071067811865476);  convolution_5 = None
    erf_2: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_19);  mul_19 = None
    add_12: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_20: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_18, add_12);  mul_18 = add_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:64, code: x = self.drop1(x)
    clone_3: "f32[8, 384, 28, 28]" = torch.ops.aten.clone.default(mul_20);  mul_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:66, code: x = self.conv2(x)
    convolution_6: "f32[8, 384, 28, 28]" = torch.ops.aten.convolution.default(clone_3, arg18_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  clone_3 = arg18_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:67, code: x = self.act2(x)
    mul_21: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_6, 0.5)
    mul_22: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_6, 0.7071067811865476);  convolution_6 = None
    erf_3: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_22);  mul_22 = None
    add_13: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_23: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_21, add_13);  mul_21 = add_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_7: "f32[8, 192, 28, 28]" = torch.ops.aten.convolution.default(mul_23, arg19_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_23 = arg19_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:69, code: x = self.drop3(x)
    clone_4: "f32[8, 192, 28, 28]" = torch.ops.aten.clone.default(convolution_7);  convolution_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_14: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_9, clone_4);  add_9 = clone_4 = None
    convert_element_type_8: "f32[192]" = torch.ops.prims.convert_element_type.default(arg133_1, torch.float32);  arg133_1 = None
    convert_element_type_9: "f32[192]" = torch.ops.prims.convert_element_type.default(arg134_1, torch.float32);  arg134_1 = None
    add_15: "f32[192]" = torch.ops.aten.add.Tensor(convert_element_type_9, 1e-05);  convert_element_type_9 = None
    sqrt_4: "f32[192]" = torch.ops.aten.sqrt.default(add_15);  add_15 = None
    reciprocal_4: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_4);  sqrt_4 = None
    mul_24: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_4, 1);  reciprocal_4 = None
    unsqueeze_32: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_8, -1);  convert_element_type_8 = None
    unsqueeze_33: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    unsqueeze_34: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_24, -1);  mul_24 = None
    unsqueeze_35: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
    sub_4: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(add_14, unsqueeze_33);  unsqueeze_33 = None
    mul_25: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_4, unsqueeze_35);  sub_4 = unsqueeze_35 = None
    unsqueeze_36: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg20_1, -1);  arg20_1 = None
    unsqueeze_37: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
    mul_26: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(mul_25, unsqueeze_37);  mul_25 = unsqueeze_37 = None
    unsqueeze_38: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg21_1, -1);  arg21_1 = None
    unsqueeze_39: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
    add_16: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_26, unsqueeze_39);  mul_26 = unsqueeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    convolution_8: "f32[8, 384, 28, 28]" = torch.ops.aten.convolution.default(add_16, arg22_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_16 = arg22_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_27: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_8, 0.5)
    mul_28: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_8, 0.7071067811865476);  convolution_8 = None
    erf_4: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_28);  mul_28 = None
    add_17: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_29: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_27, add_17);  mul_27 = add_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:64, code: x = self.drop1(x)
    clone_5: "f32[8, 384, 28, 28]" = torch.ops.aten.clone.default(mul_29);  mul_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:66, code: x = self.conv2(x)
    convolution_9: "f32[8, 384, 28, 28]" = torch.ops.aten.convolution.default(clone_5, arg23_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  clone_5 = arg23_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:67, code: x = self.act2(x)
    mul_30: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_9, 0.5)
    mul_31: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_9, 0.7071067811865476);  convolution_9 = None
    erf_5: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_31);  mul_31 = None
    add_18: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_32: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_30, add_18);  mul_30 = add_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_10: "f32[8, 192, 28, 28]" = torch.ops.aten.convolution.default(mul_32, arg24_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_32 = arg24_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:69, code: x = self.drop3(x)
    clone_6: "f32[8, 192, 28, 28]" = torch.ops.aten.clone.default(convolution_10);  convolution_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_19: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_14, clone_6);  add_14 = clone_6 = None
    convert_element_type_10: "f32[192]" = torch.ops.prims.convert_element_type.default(arg136_1, torch.float32);  arg136_1 = None
    convert_element_type_11: "f32[192]" = torch.ops.prims.convert_element_type.default(arg137_1, torch.float32);  arg137_1 = None
    add_20: "f32[192]" = torch.ops.aten.add.Tensor(convert_element_type_11, 1e-05);  convert_element_type_11 = None
    sqrt_5: "f32[192]" = torch.ops.aten.sqrt.default(add_20);  add_20 = None
    reciprocal_5: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_5);  sqrt_5 = None
    mul_33: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_5, 1);  reciprocal_5 = None
    unsqueeze_40: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_10, -1);  convert_element_type_10 = None
    unsqueeze_41: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    unsqueeze_42: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_33, -1);  mul_33 = None
    unsqueeze_43: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    sub_5: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(add_19, unsqueeze_41);  unsqueeze_41 = None
    mul_34: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_5, unsqueeze_43);  sub_5 = unsqueeze_43 = None
    unsqueeze_44: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg25_1, -1);  arg25_1 = None
    unsqueeze_45: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    mul_35: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(mul_34, unsqueeze_45);  mul_34 = unsqueeze_45 = None
    unsqueeze_46: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg26_1, -1);  arg26_1 = None
    unsqueeze_47: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
    add_21: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_35, unsqueeze_47);  mul_35 = unsqueeze_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    convolution_11: "f32[8, 384, 28, 28]" = torch.ops.aten.convolution.default(add_21, arg27_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_21 = arg27_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_36: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_11, 0.5)
    mul_37: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_11, 0.7071067811865476);  convolution_11 = None
    erf_6: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_37);  mul_37 = None
    add_22: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_38: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_36, add_22);  mul_36 = add_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:64, code: x = self.drop1(x)
    clone_7: "f32[8, 384, 28, 28]" = torch.ops.aten.clone.default(mul_38);  mul_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:66, code: x = self.conv2(x)
    convolution_12: "f32[8, 384, 28, 28]" = torch.ops.aten.convolution.default(clone_7, arg28_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  clone_7 = arg28_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:67, code: x = self.act2(x)
    mul_39: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_12, 0.5)
    mul_40: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_12, 0.7071067811865476);  convolution_12 = None
    erf_7: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_40);  mul_40 = None
    add_23: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_41: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_39, add_23);  mul_39 = add_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_13: "f32[8, 192, 28, 28]" = torch.ops.aten.convolution.default(mul_41, arg29_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_41 = arg29_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:69, code: x = self.drop3(x)
    clone_8: "f32[8, 192, 28, 28]" = torch.ops.aten.clone.default(convolution_13);  convolution_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_24: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_19, clone_8);  add_19 = clone_8 = None
    convert_element_type_12: "f32[192]" = torch.ops.prims.convert_element_type.default(arg139_1, torch.float32);  arg139_1 = None
    convert_element_type_13: "f32[192]" = torch.ops.prims.convert_element_type.default(arg140_1, torch.float32);  arg140_1 = None
    add_25: "f32[192]" = torch.ops.aten.add.Tensor(convert_element_type_13, 1e-05);  convert_element_type_13 = None
    sqrt_6: "f32[192]" = torch.ops.aten.sqrt.default(add_25);  add_25 = None
    reciprocal_6: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_6);  sqrt_6 = None
    mul_42: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_6, 1);  reciprocal_6 = None
    unsqueeze_48: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_12, -1);  convert_element_type_12 = None
    unsqueeze_49: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
    unsqueeze_50: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_42, -1);  mul_42 = None
    unsqueeze_51: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
    sub_6: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(add_24, unsqueeze_49);  unsqueeze_49 = None
    mul_43: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_6, unsqueeze_51);  sub_6 = unsqueeze_51 = None
    unsqueeze_52: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg30_1, -1);  arg30_1 = None
    unsqueeze_53: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    mul_44: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(mul_43, unsqueeze_53);  mul_43 = unsqueeze_53 = None
    unsqueeze_54: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg31_1, -1);  arg31_1 = None
    unsqueeze_55: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
    add_26: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_44, unsqueeze_55);  mul_44 = unsqueeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    convolution_14: "f32[8, 384, 28, 28]" = torch.ops.aten.convolution.default(add_26, arg32_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_26 = arg32_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_45: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_14, 0.5)
    mul_46: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_14, 0.7071067811865476);  convolution_14 = None
    erf_8: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_46);  mul_46 = None
    add_27: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_47: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_45, add_27);  mul_45 = add_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:64, code: x = self.drop1(x)
    clone_9: "f32[8, 384, 28, 28]" = torch.ops.aten.clone.default(mul_47);  mul_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:66, code: x = self.conv2(x)
    convolution_15: "f32[8, 384, 28, 28]" = torch.ops.aten.convolution.default(clone_9, arg33_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  clone_9 = arg33_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:67, code: x = self.act2(x)
    mul_48: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_15, 0.5)
    mul_49: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_15, 0.7071067811865476);  convolution_15 = None
    erf_9: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_49);  mul_49 = None
    add_28: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_50: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_48, add_28);  mul_48 = add_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_16: "f32[8, 192, 28, 28]" = torch.ops.aten.convolution.default(mul_50, arg34_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_50 = arg34_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:69, code: x = self.drop3(x)
    clone_10: "f32[8, 192, 28, 28]" = torch.ops.aten.clone.default(convolution_16);  convolution_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_29: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_24, clone_10);  add_24 = clone_10 = None
    convert_element_type_14: "f32[192]" = torch.ops.prims.convert_element_type.default(arg142_1, torch.float32);  arg142_1 = None
    convert_element_type_15: "f32[192]" = torch.ops.prims.convert_element_type.default(arg143_1, torch.float32);  arg143_1 = None
    add_30: "f32[192]" = torch.ops.aten.add.Tensor(convert_element_type_15, 1e-05);  convert_element_type_15 = None
    sqrt_7: "f32[192]" = torch.ops.aten.sqrt.default(add_30);  add_30 = None
    reciprocal_7: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_7);  sqrt_7 = None
    mul_51: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_7, 1);  reciprocal_7 = None
    unsqueeze_56: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_14, -1);  convert_element_type_14 = None
    unsqueeze_57: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    unsqueeze_58: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_51, -1);  mul_51 = None
    unsqueeze_59: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
    sub_7: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(add_29, unsqueeze_57);  unsqueeze_57 = None
    mul_52: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_7, unsqueeze_59);  sub_7 = unsqueeze_59 = None
    unsqueeze_60: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg35_1, -1);  arg35_1 = None
    unsqueeze_61: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
    mul_53: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(mul_52, unsqueeze_61);  mul_52 = unsqueeze_61 = None
    unsqueeze_62: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg36_1, -1);  arg36_1 = None
    unsqueeze_63: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
    add_31: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_53, unsqueeze_63);  mul_53 = unsqueeze_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    convolution_17: "f32[8, 384, 28, 28]" = torch.ops.aten.convolution.default(add_31, arg37_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_31 = arg37_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_54: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_17, 0.5)
    mul_55: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_17, 0.7071067811865476);  convolution_17 = None
    erf_10: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_55);  mul_55 = None
    add_32: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_56: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_54, add_32);  mul_54 = add_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:64, code: x = self.drop1(x)
    clone_11: "f32[8, 384, 28, 28]" = torch.ops.aten.clone.default(mul_56);  mul_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:66, code: x = self.conv2(x)
    convolution_18: "f32[8, 384, 28, 28]" = torch.ops.aten.convolution.default(clone_11, arg38_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  clone_11 = arg38_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:67, code: x = self.act2(x)
    mul_57: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_18, 0.5)
    mul_58: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_18, 0.7071067811865476);  convolution_18 = None
    erf_11: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_58);  mul_58 = None
    add_33: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_59: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_57, add_33);  mul_57 = add_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_19: "f32[8, 192, 28, 28]" = torch.ops.aten.convolution.default(mul_59, arg39_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_59 = arg39_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:69, code: x = self.drop3(x)
    clone_12: "f32[8, 192, 28, 28]" = torch.ops.aten.clone.default(convolution_19);  convolution_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_34: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_29, clone_12);  add_29 = clone_12 = None
    convert_element_type_16: "f32[192]" = torch.ops.prims.convert_element_type.default(arg145_1, torch.float32);  arg145_1 = None
    convert_element_type_17: "f32[192]" = torch.ops.prims.convert_element_type.default(arg146_1, torch.float32);  arg146_1 = None
    add_35: "f32[192]" = torch.ops.aten.add.Tensor(convert_element_type_17, 1e-05);  convert_element_type_17 = None
    sqrt_8: "f32[192]" = torch.ops.aten.sqrt.default(add_35);  add_35 = None
    reciprocal_8: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_8);  sqrt_8 = None
    mul_60: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_8, 1);  reciprocal_8 = None
    unsqueeze_64: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_16, -1);  convert_element_type_16 = None
    unsqueeze_65: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
    unsqueeze_66: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_60, -1);  mul_60 = None
    unsqueeze_67: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
    sub_8: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(add_34, unsqueeze_65);  unsqueeze_65 = None
    mul_61: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_8, unsqueeze_67);  sub_8 = unsqueeze_67 = None
    unsqueeze_68: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg40_1, -1);  arg40_1 = None
    unsqueeze_69: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    mul_62: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(mul_61, unsqueeze_69);  mul_61 = unsqueeze_69 = None
    unsqueeze_70: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg41_1, -1);  arg41_1 = None
    unsqueeze_71: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    add_36: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_62, unsqueeze_71);  mul_62 = unsqueeze_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    convolution_20: "f32[8, 384, 28, 28]" = torch.ops.aten.convolution.default(add_36, arg42_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_36 = arg42_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_63: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_20, 0.5)
    mul_64: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_20, 0.7071067811865476);  convolution_20 = None
    erf_12: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_64);  mul_64 = None
    add_37: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_65: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_63, add_37);  mul_63 = add_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:64, code: x = self.drop1(x)
    clone_13: "f32[8, 384, 28, 28]" = torch.ops.aten.clone.default(mul_65);  mul_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:66, code: x = self.conv2(x)
    convolution_21: "f32[8, 384, 28, 28]" = torch.ops.aten.convolution.default(clone_13, arg43_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  clone_13 = arg43_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:67, code: x = self.act2(x)
    mul_66: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_21, 0.5)
    mul_67: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_21, 0.7071067811865476);  convolution_21 = None
    erf_13: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_67);  mul_67 = None
    add_38: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    mul_68: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_66, add_38);  mul_66 = add_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_22: "f32[8, 192, 28, 28]" = torch.ops.aten.convolution.default(mul_68, arg44_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_68 = arg44_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:69, code: x = self.drop3(x)
    clone_14: "f32[8, 192, 28, 28]" = torch.ops.aten.clone.default(convolution_22);  convolution_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_39: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_34, clone_14);  add_34 = clone_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution_23: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(add_39, arg45_1, arg46_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  add_39 = arg45_1 = arg46_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:92, code: x = self.norm(x)
    convert_element_type_18: "f32[384]" = torch.ops.prims.convert_element_type.default(arg148_1, torch.float32);  arg148_1 = None
    convert_element_type_19: "f32[384]" = torch.ops.prims.convert_element_type.default(arg149_1, torch.float32);  arg149_1 = None
    add_40: "f32[384]" = torch.ops.aten.add.Tensor(convert_element_type_19, 1e-05);  convert_element_type_19 = None
    sqrt_9: "f32[384]" = torch.ops.aten.sqrt.default(add_40);  add_40 = None
    reciprocal_9: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_9);  sqrt_9 = None
    mul_69: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_9, 1);  reciprocal_9 = None
    unsqueeze_72: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_18, -1);  convert_element_type_18 = None
    unsqueeze_73: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
    unsqueeze_74: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_69, -1);  mul_69 = None
    unsqueeze_75: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
    sub_9: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_73);  convolution_23 = unsqueeze_73 = None
    mul_70: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_9, unsqueeze_75);  sub_9 = unsqueeze_75 = None
    unsqueeze_76: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg47_1, -1);  arg47_1 = None
    unsqueeze_77: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
    mul_71: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_70, unsqueeze_77);  mul_70 = unsqueeze_77 = None
    unsqueeze_78: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg48_1, -1);  arg48_1 = None
    unsqueeze_79: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    add_41: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_71, unsqueeze_79);  mul_71 = unsqueeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:411, code: x = self.pos_drop(x + self.pos_embed2)
    add_42: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_41, arg1_1);  add_41 = arg1_1 = None
    clone_15: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(add_42);  add_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    convert_element_type_20: "f32[384]" = torch.ops.prims.convert_element_type.default(arg151_1, torch.float32);  arg151_1 = None
    convert_element_type_21: "f32[384]" = torch.ops.prims.convert_element_type.default(arg152_1, torch.float32);  arg152_1 = None
    add_43: "f32[384]" = torch.ops.aten.add.Tensor(convert_element_type_21, 1e-05);  convert_element_type_21 = None
    sqrt_10: "f32[384]" = torch.ops.aten.sqrt.default(add_43);  add_43 = None
    reciprocal_10: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_10);  sqrt_10 = None
    mul_72: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_10, 1);  reciprocal_10 = None
    unsqueeze_80: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_20, -1);  convert_element_type_20 = None
    unsqueeze_81: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
    unsqueeze_82: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_72, -1);  mul_72 = None
    unsqueeze_83: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    sub_10: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(clone_15, unsqueeze_81);  unsqueeze_81 = None
    mul_73: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_10, unsqueeze_83);  sub_10 = unsqueeze_83 = None
    unsqueeze_84: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg49_1, -1);  arg49_1 = None
    unsqueeze_85: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    mul_74: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_73, unsqueeze_85);  mul_73 = unsqueeze_85 = None
    unsqueeze_86: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg50_1, -1);  arg50_1 = None
    unsqueeze_87: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
    add_44: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_74, unsqueeze_87);  mul_74 = unsqueeze_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:92, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
    convolution_24: "f32[8, 1152, 14, 14]" = torch.ops.aten.convolution.default(add_44, arg51_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_44 = arg51_1 = None
    view: "f32[8, 3, 6, 64, 196]" = torch.ops.aten.view.default(convolution_24, [8, 3, 6, 64, -1]);  convolution_24 = None
    permute: "f32[3, 8, 6, 196, 64]" = torch.ops.aten.permute.default(view, [1, 0, 2, 4, 3]);  view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:93, code: q, k, v = x.unbind(0)
    unbind = torch.ops.aten.unbind.int(permute);  permute = None
    getitem: "f32[8, 6, 196, 64]" = unbind[0]
    getitem_1: "f32[8, 6, 196, 64]" = unbind[1]
    getitem_2: "f32[8, 6, 196, 64]" = unbind[2];  unbind = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:101, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_1: "f32[8, 6, 64, 196]" = torch.ops.aten.permute.default(getitem_1, [0, 1, 3, 2]);  getitem_1 = None
    expand: "f32[8, 6, 196, 64]" = torch.ops.aten.expand.default(getitem, [8, 6, 196, 64]);  getitem = None
    clone_16: "f32[8, 6, 196, 64]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
    view_1: "f32[48, 196, 64]" = torch.ops.aten.view.default(clone_16, [48, 196, 64]);  clone_16 = None
    expand_1: "f32[8, 6, 64, 196]" = torch.ops.aten.expand.default(permute_1, [8, 6, 64, 196]);  permute_1 = None
    clone_17: "f32[8, 6, 64, 196]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
    view_2: "f32[48, 64, 196]" = torch.ops.aten.view.default(clone_17, [48, 64, 196]);  clone_17 = None
    bmm: "f32[48, 196, 196]" = torch.ops.aten.bmm.default(view_1, view_2);  view_1 = view_2 = None
    view_3: "f32[8, 6, 196, 196]" = torch.ops.aten.view.default(bmm, [8, 6, 196, 196]);  bmm = None
    mul_75: "f32[8, 6, 196, 196]" = torch.ops.aten.mul.Tensor(view_3, 0.125);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:102, code: attn = attn.softmax(dim=-1)
    amax: "f32[8, 6, 196, 1]" = torch.ops.aten.amax.default(mul_75, [-1], True)
    sub_11: "f32[8, 6, 196, 196]" = torch.ops.aten.sub.Tensor(mul_75, amax);  mul_75 = amax = None
    exp: "f32[8, 6, 196, 196]" = torch.ops.aten.exp.default(sub_11);  sub_11 = None
    sum_1: "f32[8, 6, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div: "f32[8, 6, 196, 196]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:103, code: attn = self.attn_drop(attn)
    clone_18: "f32[8, 6, 196, 196]" = torch.ops.aten.clone.default(div);  div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:104, code: x = attn @ v
    expand_2: "f32[8, 6, 196, 196]" = torch.ops.aten.expand.default(clone_18, [8, 6, 196, 196]);  clone_18 = None
    view_4: "f32[48, 196, 196]" = torch.ops.aten.view.default(expand_2, [48, 196, 196]);  expand_2 = None
    expand_3: "f32[8, 6, 196, 64]" = torch.ops.aten.expand.default(getitem_2, [8, 6, 196, 64]);  getitem_2 = None
    clone_19: "f32[8, 6, 196, 64]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
    view_5: "f32[48, 196, 64]" = torch.ops.aten.view.default(clone_19, [48, 196, 64]);  clone_19 = None
    bmm_1: "f32[48, 196, 64]" = torch.ops.aten.bmm.default(view_4, view_5);  view_4 = view_5 = None
    view_6: "f32[8, 6, 196, 64]" = torch.ops.aten.view.default(bmm_1, [8, 6, 196, 64]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:106, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
    permute_2: "f32[8, 6, 64, 196]" = torch.ops.aten.permute.default(view_6, [0, 1, 3, 2]);  view_6 = None
    clone_20: "f32[8, 6, 64, 196]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
    view_7: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(clone_20, [8, 384, 14, 14]);  clone_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:107, code: x = self.proj(x)
    convolution_25: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(view_7, arg52_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  view_7 = arg52_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:108, code: x = self.proj_drop(x)
    clone_21: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_25);  convolution_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_45: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(clone_15, clone_21);  clone_15 = clone_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    convert_element_type_22: "f32[384]" = torch.ops.prims.convert_element_type.default(arg154_1, torch.float32);  arg154_1 = None
    convert_element_type_23: "f32[384]" = torch.ops.prims.convert_element_type.default(arg155_1, torch.float32);  arg155_1 = None
    add_46: "f32[384]" = torch.ops.aten.add.Tensor(convert_element_type_23, 1e-05);  convert_element_type_23 = None
    sqrt_11: "f32[384]" = torch.ops.aten.sqrt.default(add_46);  add_46 = None
    reciprocal_11: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_11);  sqrt_11 = None
    mul_76: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_11, 1);  reciprocal_11 = None
    unsqueeze_88: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_22, -1);  convert_element_type_22 = None
    unsqueeze_89: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
    unsqueeze_90: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_76, -1);  mul_76 = None
    unsqueeze_91: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    sub_12: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(add_45, unsqueeze_89);  unsqueeze_89 = None
    mul_77: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_12, unsqueeze_91);  sub_12 = unsqueeze_91 = None
    unsqueeze_92: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg53_1, -1);  arg53_1 = None
    unsqueeze_93: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
    mul_78: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_77, unsqueeze_93);  mul_77 = unsqueeze_93 = None
    unsqueeze_94: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg54_1, -1);  arg54_1 = None
    unsqueeze_95: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
    add_47: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_78, unsqueeze_95);  mul_78 = unsqueeze_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    convolution_26: "f32[8, 1536, 14, 14]" = torch.ops.aten.convolution.default(add_47, arg55_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_47 = arg55_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_79: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_26, 0.5)
    mul_80: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_26, 0.7071067811865476);  convolution_26 = None
    erf_14: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_80);  mul_80 = None
    add_48: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    mul_81: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_79, add_48);  mul_79 = add_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:64, code: x = self.drop1(x)
    clone_22: "f32[8, 1536, 14, 14]" = torch.ops.aten.clone.default(mul_81);  mul_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_27: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(clone_22, arg56_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  clone_22 = arg56_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:69, code: x = self.drop3(x)
    clone_23: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_27);  convolution_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_49: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_45, clone_23);  add_45 = clone_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    convert_element_type_24: "f32[384]" = torch.ops.prims.convert_element_type.default(arg157_1, torch.float32);  arg157_1 = None
    convert_element_type_25: "f32[384]" = torch.ops.prims.convert_element_type.default(arg158_1, torch.float32);  arg158_1 = None
    add_50: "f32[384]" = torch.ops.aten.add.Tensor(convert_element_type_25, 1e-05);  convert_element_type_25 = None
    sqrt_12: "f32[384]" = torch.ops.aten.sqrt.default(add_50);  add_50 = None
    reciprocal_12: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_12);  sqrt_12 = None
    mul_82: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_12, 1);  reciprocal_12 = None
    unsqueeze_96: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_24, -1);  convert_element_type_24 = None
    unsqueeze_97: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
    unsqueeze_98: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_82, -1);  mul_82 = None
    unsqueeze_99: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
    sub_13: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(add_49, unsqueeze_97);  unsqueeze_97 = None
    mul_83: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_13, unsqueeze_99);  sub_13 = unsqueeze_99 = None
    unsqueeze_100: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg57_1, -1);  arg57_1 = None
    unsqueeze_101: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
    mul_84: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_83, unsqueeze_101);  mul_83 = unsqueeze_101 = None
    unsqueeze_102: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg58_1, -1);  arg58_1 = None
    unsqueeze_103: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    add_51: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_84, unsqueeze_103);  mul_84 = unsqueeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:92, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
    convolution_28: "f32[8, 1152, 14, 14]" = torch.ops.aten.convolution.default(add_51, arg59_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_51 = arg59_1 = None
    view_8: "f32[8, 3, 6, 64, 196]" = torch.ops.aten.view.default(convolution_28, [8, 3, 6, 64, -1]);  convolution_28 = None
    permute_3: "f32[3, 8, 6, 196, 64]" = torch.ops.aten.permute.default(view_8, [1, 0, 2, 4, 3]);  view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:93, code: q, k, v = x.unbind(0)
    unbind_1 = torch.ops.aten.unbind.int(permute_3);  permute_3 = None
    getitem_3: "f32[8, 6, 196, 64]" = unbind_1[0]
    getitem_4: "f32[8, 6, 196, 64]" = unbind_1[1]
    getitem_5: "f32[8, 6, 196, 64]" = unbind_1[2];  unbind_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:101, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_4: "f32[8, 6, 64, 196]" = torch.ops.aten.permute.default(getitem_4, [0, 1, 3, 2]);  getitem_4 = None
    expand_4: "f32[8, 6, 196, 64]" = torch.ops.aten.expand.default(getitem_3, [8, 6, 196, 64]);  getitem_3 = None
    clone_24: "f32[8, 6, 196, 64]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
    view_9: "f32[48, 196, 64]" = torch.ops.aten.view.default(clone_24, [48, 196, 64]);  clone_24 = None
    expand_5: "f32[8, 6, 64, 196]" = torch.ops.aten.expand.default(permute_4, [8, 6, 64, 196]);  permute_4 = None
    clone_25: "f32[8, 6, 64, 196]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
    view_10: "f32[48, 64, 196]" = torch.ops.aten.view.default(clone_25, [48, 64, 196]);  clone_25 = None
    bmm_2: "f32[48, 196, 196]" = torch.ops.aten.bmm.default(view_9, view_10);  view_9 = view_10 = None
    view_11: "f32[8, 6, 196, 196]" = torch.ops.aten.view.default(bmm_2, [8, 6, 196, 196]);  bmm_2 = None
    mul_85: "f32[8, 6, 196, 196]" = torch.ops.aten.mul.Tensor(view_11, 0.125);  view_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:102, code: attn = attn.softmax(dim=-1)
    amax_1: "f32[8, 6, 196, 1]" = torch.ops.aten.amax.default(mul_85, [-1], True)
    sub_14: "f32[8, 6, 196, 196]" = torch.ops.aten.sub.Tensor(mul_85, amax_1);  mul_85 = amax_1 = None
    exp_1: "f32[8, 6, 196, 196]" = torch.ops.aten.exp.default(sub_14);  sub_14 = None
    sum_2: "f32[8, 6, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_1: "f32[8, 6, 196, 196]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:103, code: attn = self.attn_drop(attn)
    clone_26: "f32[8, 6, 196, 196]" = torch.ops.aten.clone.default(div_1);  div_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:104, code: x = attn @ v
    expand_6: "f32[8, 6, 196, 196]" = torch.ops.aten.expand.default(clone_26, [8, 6, 196, 196]);  clone_26 = None
    view_12: "f32[48, 196, 196]" = torch.ops.aten.view.default(expand_6, [48, 196, 196]);  expand_6 = None
    expand_7: "f32[8, 6, 196, 64]" = torch.ops.aten.expand.default(getitem_5, [8, 6, 196, 64]);  getitem_5 = None
    clone_27: "f32[8, 6, 196, 64]" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
    view_13: "f32[48, 196, 64]" = torch.ops.aten.view.default(clone_27, [48, 196, 64]);  clone_27 = None
    bmm_3: "f32[48, 196, 64]" = torch.ops.aten.bmm.default(view_12, view_13);  view_12 = view_13 = None
    view_14: "f32[8, 6, 196, 64]" = torch.ops.aten.view.default(bmm_3, [8, 6, 196, 64]);  bmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:106, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
    permute_5: "f32[8, 6, 64, 196]" = torch.ops.aten.permute.default(view_14, [0, 1, 3, 2]);  view_14 = None
    clone_28: "f32[8, 6, 64, 196]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
    view_15: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(clone_28, [8, 384, 14, 14]);  clone_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:107, code: x = self.proj(x)
    convolution_29: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(view_15, arg60_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  view_15 = arg60_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:108, code: x = self.proj_drop(x)
    clone_29: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_29);  convolution_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_52: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_49, clone_29);  add_49 = clone_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    convert_element_type_26: "f32[384]" = torch.ops.prims.convert_element_type.default(arg160_1, torch.float32);  arg160_1 = None
    convert_element_type_27: "f32[384]" = torch.ops.prims.convert_element_type.default(arg161_1, torch.float32);  arg161_1 = None
    add_53: "f32[384]" = torch.ops.aten.add.Tensor(convert_element_type_27, 1e-05);  convert_element_type_27 = None
    sqrt_13: "f32[384]" = torch.ops.aten.sqrt.default(add_53);  add_53 = None
    reciprocal_13: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_13);  sqrt_13 = None
    mul_86: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_13, 1);  reciprocal_13 = None
    unsqueeze_104: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_26, -1);  convert_element_type_26 = None
    unsqueeze_105: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
    unsqueeze_106: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_86, -1);  mul_86 = None
    unsqueeze_107: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
    sub_15: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(add_52, unsqueeze_105);  unsqueeze_105 = None
    mul_87: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_15, unsqueeze_107);  sub_15 = unsqueeze_107 = None
    unsqueeze_108: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg61_1, -1);  arg61_1 = None
    unsqueeze_109: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
    mul_88: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_87, unsqueeze_109);  mul_87 = unsqueeze_109 = None
    unsqueeze_110: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg62_1, -1);  arg62_1 = None
    unsqueeze_111: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
    add_54: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_88, unsqueeze_111);  mul_88 = unsqueeze_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    convolution_30: "f32[8, 1536, 14, 14]" = torch.ops.aten.convolution.default(add_54, arg63_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_54 = arg63_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_89: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_30, 0.5)
    mul_90: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_30, 0.7071067811865476);  convolution_30 = None
    erf_15: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_90);  mul_90 = None
    add_55: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    mul_91: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_89, add_55);  mul_89 = add_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:64, code: x = self.drop1(x)
    clone_30: "f32[8, 1536, 14, 14]" = torch.ops.aten.clone.default(mul_91);  mul_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_31: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(clone_30, arg64_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  clone_30 = arg64_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:69, code: x = self.drop3(x)
    clone_31: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_31);  convolution_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_56: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_52, clone_31);  add_52 = clone_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    convert_element_type_28: "f32[384]" = torch.ops.prims.convert_element_type.default(arg163_1, torch.float32);  arg163_1 = None
    convert_element_type_29: "f32[384]" = torch.ops.prims.convert_element_type.default(arg164_1, torch.float32);  arg164_1 = None
    add_57: "f32[384]" = torch.ops.aten.add.Tensor(convert_element_type_29, 1e-05);  convert_element_type_29 = None
    sqrt_14: "f32[384]" = torch.ops.aten.sqrt.default(add_57);  add_57 = None
    reciprocal_14: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_14);  sqrt_14 = None
    mul_92: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_14, 1);  reciprocal_14 = None
    unsqueeze_112: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_28, -1);  convert_element_type_28 = None
    unsqueeze_113: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
    unsqueeze_114: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_92, -1);  mul_92 = None
    unsqueeze_115: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
    sub_16: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(add_56, unsqueeze_113);  unsqueeze_113 = None
    mul_93: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_16, unsqueeze_115);  sub_16 = unsqueeze_115 = None
    unsqueeze_116: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg65_1, -1);  arg65_1 = None
    unsqueeze_117: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
    mul_94: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_93, unsqueeze_117);  mul_93 = unsqueeze_117 = None
    unsqueeze_118: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg66_1, -1);  arg66_1 = None
    unsqueeze_119: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
    add_58: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_94, unsqueeze_119);  mul_94 = unsqueeze_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:92, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
    convolution_32: "f32[8, 1152, 14, 14]" = torch.ops.aten.convolution.default(add_58, arg67_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_58 = arg67_1 = None
    view_16: "f32[8, 3, 6, 64, 196]" = torch.ops.aten.view.default(convolution_32, [8, 3, 6, 64, -1]);  convolution_32 = None
    permute_6: "f32[3, 8, 6, 196, 64]" = torch.ops.aten.permute.default(view_16, [1, 0, 2, 4, 3]);  view_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:93, code: q, k, v = x.unbind(0)
    unbind_2 = torch.ops.aten.unbind.int(permute_6);  permute_6 = None
    getitem_6: "f32[8, 6, 196, 64]" = unbind_2[0]
    getitem_7: "f32[8, 6, 196, 64]" = unbind_2[1]
    getitem_8: "f32[8, 6, 196, 64]" = unbind_2[2];  unbind_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:101, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_7: "f32[8, 6, 64, 196]" = torch.ops.aten.permute.default(getitem_7, [0, 1, 3, 2]);  getitem_7 = None
    expand_8: "f32[8, 6, 196, 64]" = torch.ops.aten.expand.default(getitem_6, [8, 6, 196, 64]);  getitem_6 = None
    clone_32: "f32[8, 6, 196, 64]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
    view_17: "f32[48, 196, 64]" = torch.ops.aten.view.default(clone_32, [48, 196, 64]);  clone_32 = None
    expand_9: "f32[8, 6, 64, 196]" = torch.ops.aten.expand.default(permute_7, [8, 6, 64, 196]);  permute_7 = None
    clone_33: "f32[8, 6, 64, 196]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
    view_18: "f32[48, 64, 196]" = torch.ops.aten.view.default(clone_33, [48, 64, 196]);  clone_33 = None
    bmm_4: "f32[48, 196, 196]" = torch.ops.aten.bmm.default(view_17, view_18);  view_17 = view_18 = None
    view_19: "f32[8, 6, 196, 196]" = torch.ops.aten.view.default(bmm_4, [8, 6, 196, 196]);  bmm_4 = None
    mul_95: "f32[8, 6, 196, 196]" = torch.ops.aten.mul.Tensor(view_19, 0.125);  view_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:102, code: attn = attn.softmax(dim=-1)
    amax_2: "f32[8, 6, 196, 1]" = torch.ops.aten.amax.default(mul_95, [-1], True)
    sub_17: "f32[8, 6, 196, 196]" = torch.ops.aten.sub.Tensor(mul_95, amax_2);  mul_95 = amax_2 = None
    exp_2: "f32[8, 6, 196, 196]" = torch.ops.aten.exp.default(sub_17);  sub_17 = None
    sum_3: "f32[8, 6, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_2: "f32[8, 6, 196, 196]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:103, code: attn = self.attn_drop(attn)
    clone_34: "f32[8, 6, 196, 196]" = torch.ops.aten.clone.default(div_2);  div_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:104, code: x = attn @ v
    expand_10: "f32[8, 6, 196, 196]" = torch.ops.aten.expand.default(clone_34, [8, 6, 196, 196]);  clone_34 = None
    view_20: "f32[48, 196, 196]" = torch.ops.aten.view.default(expand_10, [48, 196, 196]);  expand_10 = None
    expand_11: "f32[8, 6, 196, 64]" = torch.ops.aten.expand.default(getitem_8, [8, 6, 196, 64]);  getitem_8 = None
    clone_35: "f32[8, 6, 196, 64]" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
    view_21: "f32[48, 196, 64]" = torch.ops.aten.view.default(clone_35, [48, 196, 64]);  clone_35 = None
    bmm_5: "f32[48, 196, 64]" = torch.ops.aten.bmm.default(view_20, view_21);  view_20 = view_21 = None
    view_22: "f32[8, 6, 196, 64]" = torch.ops.aten.view.default(bmm_5, [8, 6, 196, 64]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:106, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
    permute_8: "f32[8, 6, 64, 196]" = torch.ops.aten.permute.default(view_22, [0, 1, 3, 2]);  view_22 = None
    clone_36: "f32[8, 6, 64, 196]" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None
    view_23: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(clone_36, [8, 384, 14, 14]);  clone_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:107, code: x = self.proj(x)
    convolution_33: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(view_23, arg68_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  view_23 = arg68_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:108, code: x = self.proj_drop(x)
    clone_37: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_33);  convolution_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_59: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_56, clone_37);  add_56 = clone_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    convert_element_type_30: "f32[384]" = torch.ops.prims.convert_element_type.default(arg166_1, torch.float32);  arg166_1 = None
    convert_element_type_31: "f32[384]" = torch.ops.prims.convert_element_type.default(arg167_1, torch.float32);  arg167_1 = None
    add_60: "f32[384]" = torch.ops.aten.add.Tensor(convert_element_type_31, 1e-05);  convert_element_type_31 = None
    sqrt_15: "f32[384]" = torch.ops.aten.sqrt.default(add_60);  add_60 = None
    reciprocal_15: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_15);  sqrt_15 = None
    mul_96: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_15, 1);  reciprocal_15 = None
    unsqueeze_120: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_30, -1);  convert_element_type_30 = None
    unsqueeze_121: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
    unsqueeze_122: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_96, -1);  mul_96 = None
    unsqueeze_123: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
    sub_18: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(add_59, unsqueeze_121);  unsqueeze_121 = None
    mul_97: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_18, unsqueeze_123);  sub_18 = unsqueeze_123 = None
    unsqueeze_124: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg69_1, -1);  arg69_1 = None
    unsqueeze_125: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
    mul_98: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_97, unsqueeze_125);  mul_97 = unsqueeze_125 = None
    unsqueeze_126: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg70_1, -1);  arg70_1 = None
    unsqueeze_127: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
    add_61: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_98, unsqueeze_127);  mul_98 = unsqueeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    convolution_34: "f32[8, 1536, 14, 14]" = torch.ops.aten.convolution.default(add_61, arg71_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_61 = arg71_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_99: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_34, 0.5)
    mul_100: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_34, 0.7071067811865476);  convolution_34 = None
    erf_16: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_100);  mul_100 = None
    add_62: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    mul_101: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_99, add_62);  mul_99 = add_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:64, code: x = self.drop1(x)
    clone_38: "f32[8, 1536, 14, 14]" = torch.ops.aten.clone.default(mul_101);  mul_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_35: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(clone_38, arg72_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  clone_38 = arg72_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:69, code: x = self.drop3(x)
    clone_39: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_35);  convolution_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_63: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_59, clone_39);  add_59 = clone_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    convert_element_type_32: "f32[384]" = torch.ops.prims.convert_element_type.default(arg169_1, torch.float32);  arg169_1 = None
    convert_element_type_33: "f32[384]" = torch.ops.prims.convert_element_type.default(arg170_1, torch.float32);  arg170_1 = None
    add_64: "f32[384]" = torch.ops.aten.add.Tensor(convert_element_type_33, 1e-05);  convert_element_type_33 = None
    sqrt_16: "f32[384]" = torch.ops.aten.sqrt.default(add_64);  add_64 = None
    reciprocal_16: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_16);  sqrt_16 = None
    mul_102: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_16, 1);  reciprocal_16 = None
    unsqueeze_128: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_32, -1);  convert_element_type_32 = None
    unsqueeze_129: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
    unsqueeze_130: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_102, -1);  mul_102 = None
    unsqueeze_131: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
    sub_19: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(add_63, unsqueeze_129);  unsqueeze_129 = None
    mul_103: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_19, unsqueeze_131);  sub_19 = unsqueeze_131 = None
    unsqueeze_132: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg73_1, -1);  arg73_1 = None
    unsqueeze_133: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
    mul_104: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_103, unsqueeze_133);  mul_103 = unsqueeze_133 = None
    unsqueeze_134: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg74_1, -1);  arg74_1 = None
    unsqueeze_135: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
    add_65: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_104, unsqueeze_135);  mul_104 = unsqueeze_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:92, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
    convolution_36: "f32[8, 1152, 14, 14]" = torch.ops.aten.convolution.default(add_65, arg75_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_65 = arg75_1 = None
    view_24: "f32[8, 3, 6, 64, 196]" = torch.ops.aten.view.default(convolution_36, [8, 3, 6, 64, -1]);  convolution_36 = None
    permute_9: "f32[3, 8, 6, 196, 64]" = torch.ops.aten.permute.default(view_24, [1, 0, 2, 4, 3]);  view_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:93, code: q, k, v = x.unbind(0)
    unbind_3 = torch.ops.aten.unbind.int(permute_9);  permute_9 = None
    getitem_9: "f32[8, 6, 196, 64]" = unbind_3[0]
    getitem_10: "f32[8, 6, 196, 64]" = unbind_3[1]
    getitem_11: "f32[8, 6, 196, 64]" = unbind_3[2];  unbind_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:101, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_10: "f32[8, 6, 64, 196]" = torch.ops.aten.permute.default(getitem_10, [0, 1, 3, 2]);  getitem_10 = None
    expand_12: "f32[8, 6, 196, 64]" = torch.ops.aten.expand.default(getitem_9, [8, 6, 196, 64]);  getitem_9 = None
    clone_40: "f32[8, 6, 196, 64]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
    view_25: "f32[48, 196, 64]" = torch.ops.aten.view.default(clone_40, [48, 196, 64]);  clone_40 = None
    expand_13: "f32[8, 6, 64, 196]" = torch.ops.aten.expand.default(permute_10, [8, 6, 64, 196]);  permute_10 = None
    clone_41: "f32[8, 6, 64, 196]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
    view_26: "f32[48, 64, 196]" = torch.ops.aten.view.default(clone_41, [48, 64, 196]);  clone_41 = None
    bmm_6: "f32[48, 196, 196]" = torch.ops.aten.bmm.default(view_25, view_26);  view_25 = view_26 = None
    view_27: "f32[8, 6, 196, 196]" = torch.ops.aten.view.default(bmm_6, [8, 6, 196, 196]);  bmm_6 = None
    mul_105: "f32[8, 6, 196, 196]" = torch.ops.aten.mul.Tensor(view_27, 0.125);  view_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:102, code: attn = attn.softmax(dim=-1)
    amax_3: "f32[8, 6, 196, 1]" = torch.ops.aten.amax.default(mul_105, [-1], True)
    sub_20: "f32[8, 6, 196, 196]" = torch.ops.aten.sub.Tensor(mul_105, amax_3);  mul_105 = amax_3 = None
    exp_3: "f32[8, 6, 196, 196]" = torch.ops.aten.exp.default(sub_20);  sub_20 = None
    sum_4: "f32[8, 6, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_3: "f32[8, 6, 196, 196]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:103, code: attn = self.attn_drop(attn)
    clone_42: "f32[8, 6, 196, 196]" = torch.ops.aten.clone.default(div_3);  div_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:104, code: x = attn @ v
    expand_14: "f32[8, 6, 196, 196]" = torch.ops.aten.expand.default(clone_42, [8, 6, 196, 196]);  clone_42 = None
    view_28: "f32[48, 196, 196]" = torch.ops.aten.view.default(expand_14, [48, 196, 196]);  expand_14 = None
    expand_15: "f32[8, 6, 196, 64]" = torch.ops.aten.expand.default(getitem_11, [8, 6, 196, 64]);  getitem_11 = None
    clone_43: "f32[8, 6, 196, 64]" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
    view_29: "f32[48, 196, 64]" = torch.ops.aten.view.default(clone_43, [48, 196, 64]);  clone_43 = None
    bmm_7: "f32[48, 196, 64]" = torch.ops.aten.bmm.default(view_28, view_29);  view_28 = view_29 = None
    view_30: "f32[8, 6, 196, 64]" = torch.ops.aten.view.default(bmm_7, [8, 6, 196, 64]);  bmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:106, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
    permute_11: "f32[8, 6, 64, 196]" = torch.ops.aten.permute.default(view_30, [0, 1, 3, 2]);  view_30 = None
    clone_44: "f32[8, 6, 64, 196]" = torch.ops.aten.clone.default(permute_11, memory_format = torch.contiguous_format);  permute_11 = None
    view_31: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(clone_44, [8, 384, 14, 14]);  clone_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:107, code: x = self.proj(x)
    convolution_37: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(view_31, arg76_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  view_31 = arg76_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:108, code: x = self.proj_drop(x)
    clone_45: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_37);  convolution_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_66: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_63, clone_45);  add_63 = clone_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    convert_element_type_34: "f32[384]" = torch.ops.prims.convert_element_type.default(arg172_1, torch.float32);  arg172_1 = None
    convert_element_type_35: "f32[384]" = torch.ops.prims.convert_element_type.default(arg173_1, torch.float32);  arg173_1 = None
    add_67: "f32[384]" = torch.ops.aten.add.Tensor(convert_element_type_35, 1e-05);  convert_element_type_35 = None
    sqrt_17: "f32[384]" = torch.ops.aten.sqrt.default(add_67);  add_67 = None
    reciprocal_17: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_17);  sqrt_17 = None
    mul_106: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_17, 1);  reciprocal_17 = None
    unsqueeze_136: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_34, -1);  convert_element_type_34 = None
    unsqueeze_137: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
    unsqueeze_138: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_106, -1);  mul_106 = None
    unsqueeze_139: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
    sub_21: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(add_66, unsqueeze_137);  unsqueeze_137 = None
    mul_107: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_21, unsqueeze_139);  sub_21 = unsqueeze_139 = None
    unsqueeze_140: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg77_1, -1);  arg77_1 = None
    unsqueeze_141: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
    mul_108: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_107, unsqueeze_141);  mul_107 = unsqueeze_141 = None
    unsqueeze_142: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg78_1, -1);  arg78_1 = None
    unsqueeze_143: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
    add_68: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_108, unsqueeze_143);  mul_108 = unsqueeze_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    convolution_38: "f32[8, 1536, 14, 14]" = torch.ops.aten.convolution.default(add_68, arg79_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_68 = arg79_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_109: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_38, 0.5)
    mul_110: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_38, 0.7071067811865476);  convolution_38 = None
    erf_17: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_110);  mul_110 = None
    add_69: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    mul_111: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_109, add_69);  mul_109 = add_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:64, code: x = self.drop1(x)
    clone_46: "f32[8, 1536, 14, 14]" = torch.ops.aten.clone.default(mul_111);  mul_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_39: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(clone_46, arg80_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  clone_46 = arg80_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:69, code: x = self.drop3(x)
    clone_47: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_39);  convolution_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_70: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_66, clone_47);  add_66 = clone_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution_40: "f32[8, 768, 7, 7]" = torch.ops.aten.convolution.default(add_70, arg81_1, arg82_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  add_70 = arg81_1 = arg82_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:92, code: x = self.norm(x)
    convert_element_type_36: "f32[768]" = torch.ops.prims.convert_element_type.default(arg175_1, torch.float32);  arg175_1 = None
    convert_element_type_37: "f32[768]" = torch.ops.prims.convert_element_type.default(arg176_1, torch.float32);  arg176_1 = None
    add_71: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_37, 1e-05);  convert_element_type_37 = None
    sqrt_18: "f32[768]" = torch.ops.aten.sqrt.default(add_71);  add_71 = None
    reciprocal_18: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_18);  sqrt_18 = None
    mul_112: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_18, 1);  reciprocal_18 = None
    unsqueeze_144: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_36, -1);  convert_element_type_36 = None
    unsqueeze_145: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
    unsqueeze_146: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_112, -1);  mul_112 = None
    unsqueeze_147: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
    sub_22: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_145);  convolution_40 = unsqueeze_145 = None
    mul_113: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_22, unsqueeze_147);  sub_22 = unsqueeze_147 = None
    unsqueeze_148: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg83_1, -1);  arg83_1 = None
    unsqueeze_149: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
    mul_114: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(mul_113, unsqueeze_149);  mul_113 = unsqueeze_149 = None
    unsqueeze_150: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg84_1, -1);  arg84_1 = None
    unsqueeze_151: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
    add_72: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_114, unsqueeze_151);  mul_114 = unsqueeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:421, code: x = self.pos_drop(x + self.pos_embed3)
    add_73: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_72, arg2_1);  add_72 = arg2_1 = None
    clone_48: "f32[8, 768, 7, 7]" = torch.ops.aten.clone.default(add_73);  add_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    convert_element_type_38: "f32[768]" = torch.ops.prims.convert_element_type.default(arg178_1, torch.float32);  arg178_1 = None
    convert_element_type_39: "f32[768]" = torch.ops.prims.convert_element_type.default(arg179_1, torch.float32);  arg179_1 = None
    add_74: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_39, 1e-05);  convert_element_type_39 = None
    sqrt_19: "f32[768]" = torch.ops.aten.sqrt.default(add_74);  add_74 = None
    reciprocal_19: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_19);  sqrt_19 = None
    mul_115: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_19, 1);  reciprocal_19 = None
    unsqueeze_152: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_38, -1);  convert_element_type_38 = None
    unsqueeze_153: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
    unsqueeze_154: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_115, -1);  mul_115 = None
    unsqueeze_155: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
    sub_23: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(clone_48, unsqueeze_153);  unsqueeze_153 = None
    mul_116: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_23, unsqueeze_155);  sub_23 = unsqueeze_155 = None
    unsqueeze_156: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg85_1, -1);  arg85_1 = None
    unsqueeze_157: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
    mul_117: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(mul_116, unsqueeze_157);  mul_116 = unsqueeze_157 = None
    unsqueeze_158: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg86_1, -1);  arg86_1 = None
    unsqueeze_159: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
    add_75: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_117, unsqueeze_159);  mul_117 = unsqueeze_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:92, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
    convolution_41: "f32[8, 2304, 7, 7]" = torch.ops.aten.convolution.default(add_75, arg87_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_75 = arg87_1 = None
    view_32: "f32[8, 3, 6, 128, 49]" = torch.ops.aten.view.default(convolution_41, [8, 3, 6, 128, -1]);  convolution_41 = None
    permute_12: "f32[3, 8, 6, 49, 128]" = torch.ops.aten.permute.default(view_32, [1, 0, 2, 4, 3]);  view_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:93, code: q, k, v = x.unbind(0)
    unbind_4 = torch.ops.aten.unbind.int(permute_12);  permute_12 = None
    getitem_12: "f32[8, 6, 49, 128]" = unbind_4[0]
    getitem_13: "f32[8, 6, 49, 128]" = unbind_4[1]
    getitem_14: "f32[8, 6, 49, 128]" = unbind_4[2];  unbind_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:101, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_13: "f32[8, 6, 128, 49]" = torch.ops.aten.permute.default(getitem_13, [0, 1, 3, 2]);  getitem_13 = None
    expand_16: "f32[8, 6, 49, 128]" = torch.ops.aten.expand.default(getitem_12, [8, 6, 49, 128]);  getitem_12 = None
    clone_49: "f32[8, 6, 49, 128]" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
    view_33: "f32[48, 49, 128]" = torch.ops.aten.view.default(clone_49, [48, 49, 128]);  clone_49 = None
    expand_17: "f32[8, 6, 128, 49]" = torch.ops.aten.expand.default(permute_13, [8, 6, 128, 49]);  permute_13 = None
    clone_50: "f32[8, 6, 128, 49]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
    view_34: "f32[48, 128, 49]" = torch.ops.aten.view.default(clone_50, [48, 128, 49]);  clone_50 = None
    bmm_8: "f32[48, 49, 49]" = torch.ops.aten.bmm.default(view_33, view_34);  view_33 = view_34 = None
    view_35: "f32[8, 6, 49, 49]" = torch.ops.aten.view.default(bmm_8, [8, 6, 49, 49]);  bmm_8 = None
    mul_118: "f32[8, 6, 49, 49]" = torch.ops.aten.mul.Tensor(view_35, 0.08838834764831845);  view_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:102, code: attn = attn.softmax(dim=-1)
    amax_4: "f32[8, 6, 49, 1]" = torch.ops.aten.amax.default(mul_118, [-1], True)
    sub_24: "f32[8, 6, 49, 49]" = torch.ops.aten.sub.Tensor(mul_118, amax_4);  mul_118 = amax_4 = None
    exp_4: "f32[8, 6, 49, 49]" = torch.ops.aten.exp.default(sub_24);  sub_24 = None
    sum_5: "f32[8, 6, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_4: "f32[8, 6, 49, 49]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:103, code: attn = self.attn_drop(attn)
    clone_51: "f32[8, 6, 49, 49]" = torch.ops.aten.clone.default(div_4);  div_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:104, code: x = attn @ v
    expand_18: "f32[8, 6, 49, 49]" = torch.ops.aten.expand.default(clone_51, [8, 6, 49, 49]);  clone_51 = None
    view_36: "f32[48, 49, 49]" = torch.ops.aten.view.default(expand_18, [48, 49, 49]);  expand_18 = None
    expand_19: "f32[8, 6, 49, 128]" = torch.ops.aten.expand.default(getitem_14, [8, 6, 49, 128]);  getitem_14 = None
    clone_52: "f32[8, 6, 49, 128]" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
    view_37: "f32[48, 49, 128]" = torch.ops.aten.view.default(clone_52, [48, 49, 128]);  clone_52 = None
    bmm_9: "f32[48, 49, 128]" = torch.ops.aten.bmm.default(view_36, view_37);  view_36 = view_37 = None
    view_38: "f32[8, 6, 49, 128]" = torch.ops.aten.view.default(bmm_9, [8, 6, 49, 128]);  bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:106, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
    permute_14: "f32[8, 6, 128, 49]" = torch.ops.aten.permute.default(view_38, [0, 1, 3, 2]);  view_38 = None
    clone_53: "f32[8, 6, 128, 49]" = torch.ops.aten.clone.default(permute_14, memory_format = torch.contiguous_format);  permute_14 = None
    view_39: "f32[8, 768, 7, 7]" = torch.ops.aten.view.default(clone_53, [8, 768, 7, 7]);  clone_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:107, code: x = self.proj(x)
    convolution_42: "f32[8, 768, 7, 7]" = torch.ops.aten.convolution.default(view_39, arg88_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  view_39 = arg88_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:108, code: x = self.proj_drop(x)
    clone_54: "f32[8, 768, 7, 7]" = torch.ops.aten.clone.default(convolution_42);  convolution_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_76: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(clone_48, clone_54);  clone_48 = clone_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    convert_element_type_40: "f32[768]" = torch.ops.prims.convert_element_type.default(arg181_1, torch.float32);  arg181_1 = None
    convert_element_type_41: "f32[768]" = torch.ops.prims.convert_element_type.default(arg182_1, torch.float32);  arg182_1 = None
    add_77: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_41, 1e-05);  convert_element_type_41 = None
    sqrt_20: "f32[768]" = torch.ops.aten.sqrt.default(add_77);  add_77 = None
    reciprocal_20: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_20);  sqrt_20 = None
    mul_119: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_20, 1);  reciprocal_20 = None
    unsqueeze_160: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_40, -1);  convert_element_type_40 = None
    unsqueeze_161: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
    unsqueeze_162: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_119, -1);  mul_119 = None
    unsqueeze_163: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
    sub_25: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_76, unsqueeze_161);  unsqueeze_161 = None
    mul_120: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_25, unsqueeze_163);  sub_25 = unsqueeze_163 = None
    unsqueeze_164: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg89_1, -1);  arg89_1 = None
    unsqueeze_165: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
    mul_121: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(mul_120, unsqueeze_165);  mul_120 = unsqueeze_165 = None
    unsqueeze_166: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg90_1, -1);  arg90_1 = None
    unsqueeze_167: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
    add_78: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_121, unsqueeze_167);  mul_121 = unsqueeze_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    convolution_43: "f32[8, 3072, 7, 7]" = torch.ops.aten.convolution.default(add_78, arg91_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_78 = arg91_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_122: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_43, 0.5)
    mul_123: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_43, 0.7071067811865476);  convolution_43 = None
    erf_18: "f32[8, 3072, 7, 7]" = torch.ops.aten.erf.default(mul_123);  mul_123 = None
    add_79: "f32[8, 3072, 7, 7]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    mul_124: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(mul_122, add_79);  mul_122 = add_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:64, code: x = self.drop1(x)
    clone_55: "f32[8, 3072, 7, 7]" = torch.ops.aten.clone.default(mul_124);  mul_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_44: "f32[8, 768, 7, 7]" = torch.ops.aten.convolution.default(clone_55, arg92_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  clone_55 = arg92_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:69, code: x = self.drop3(x)
    clone_56: "f32[8, 768, 7, 7]" = torch.ops.aten.clone.default(convolution_44);  convolution_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_80: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_76, clone_56);  add_76 = clone_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    convert_element_type_42: "f32[768]" = torch.ops.prims.convert_element_type.default(arg184_1, torch.float32);  arg184_1 = None
    convert_element_type_43: "f32[768]" = torch.ops.prims.convert_element_type.default(arg185_1, torch.float32);  arg185_1 = None
    add_81: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_43, 1e-05);  convert_element_type_43 = None
    sqrt_21: "f32[768]" = torch.ops.aten.sqrt.default(add_81);  add_81 = None
    reciprocal_21: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_21);  sqrt_21 = None
    mul_125: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_21, 1);  reciprocal_21 = None
    unsqueeze_168: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_42, -1);  convert_element_type_42 = None
    unsqueeze_169: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
    unsqueeze_170: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_125, -1);  mul_125 = None
    unsqueeze_171: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
    sub_26: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_80, unsqueeze_169);  unsqueeze_169 = None
    mul_126: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_26, unsqueeze_171);  sub_26 = unsqueeze_171 = None
    unsqueeze_172: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg93_1, -1);  arg93_1 = None
    unsqueeze_173: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
    mul_127: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(mul_126, unsqueeze_173);  mul_126 = unsqueeze_173 = None
    unsqueeze_174: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg94_1, -1);  arg94_1 = None
    unsqueeze_175: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
    add_82: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_127, unsqueeze_175);  mul_127 = unsqueeze_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:92, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
    convolution_45: "f32[8, 2304, 7, 7]" = torch.ops.aten.convolution.default(add_82, arg95_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_82 = arg95_1 = None
    view_40: "f32[8, 3, 6, 128, 49]" = torch.ops.aten.view.default(convolution_45, [8, 3, 6, 128, -1]);  convolution_45 = None
    permute_15: "f32[3, 8, 6, 49, 128]" = torch.ops.aten.permute.default(view_40, [1, 0, 2, 4, 3]);  view_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:93, code: q, k, v = x.unbind(0)
    unbind_5 = torch.ops.aten.unbind.int(permute_15);  permute_15 = None
    getitem_15: "f32[8, 6, 49, 128]" = unbind_5[0]
    getitem_16: "f32[8, 6, 49, 128]" = unbind_5[1]
    getitem_17: "f32[8, 6, 49, 128]" = unbind_5[2];  unbind_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:101, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_16: "f32[8, 6, 128, 49]" = torch.ops.aten.permute.default(getitem_16, [0, 1, 3, 2]);  getitem_16 = None
    expand_20: "f32[8, 6, 49, 128]" = torch.ops.aten.expand.default(getitem_15, [8, 6, 49, 128]);  getitem_15 = None
    clone_57: "f32[8, 6, 49, 128]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
    view_41: "f32[48, 49, 128]" = torch.ops.aten.view.default(clone_57, [48, 49, 128]);  clone_57 = None
    expand_21: "f32[8, 6, 128, 49]" = torch.ops.aten.expand.default(permute_16, [8, 6, 128, 49]);  permute_16 = None
    clone_58: "f32[8, 6, 128, 49]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
    view_42: "f32[48, 128, 49]" = torch.ops.aten.view.default(clone_58, [48, 128, 49]);  clone_58 = None
    bmm_10: "f32[48, 49, 49]" = torch.ops.aten.bmm.default(view_41, view_42);  view_41 = view_42 = None
    view_43: "f32[8, 6, 49, 49]" = torch.ops.aten.view.default(bmm_10, [8, 6, 49, 49]);  bmm_10 = None
    mul_128: "f32[8, 6, 49, 49]" = torch.ops.aten.mul.Tensor(view_43, 0.08838834764831845);  view_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:102, code: attn = attn.softmax(dim=-1)
    amax_5: "f32[8, 6, 49, 1]" = torch.ops.aten.amax.default(mul_128, [-1], True)
    sub_27: "f32[8, 6, 49, 49]" = torch.ops.aten.sub.Tensor(mul_128, amax_5);  mul_128 = amax_5 = None
    exp_5: "f32[8, 6, 49, 49]" = torch.ops.aten.exp.default(sub_27);  sub_27 = None
    sum_6: "f32[8, 6, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_5: "f32[8, 6, 49, 49]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:103, code: attn = self.attn_drop(attn)
    clone_59: "f32[8, 6, 49, 49]" = torch.ops.aten.clone.default(div_5);  div_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:104, code: x = attn @ v
    expand_22: "f32[8, 6, 49, 49]" = torch.ops.aten.expand.default(clone_59, [8, 6, 49, 49]);  clone_59 = None
    view_44: "f32[48, 49, 49]" = torch.ops.aten.view.default(expand_22, [48, 49, 49]);  expand_22 = None
    expand_23: "f32[8, 6, 49, 128]" = torch.ops.aten.expand.default(getitem_17, [8, 6, 49, 128]);  getitem_17 = None
    clone_60: "f32[8, 6, 49, 128]" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
    view_45: "f32[48, 49, 128]" = torch.ops.aten.view.default(clone_60, [48, 49, 128]);  clone_60 = None
    bmm_11: "f32[48, 49, 128]" = torch.ops.aten.bmm.default(view_44, view_45);  view_44 = view_45 = None
    view_46: "f32[8, 6, 49, 128]" = torch.ops.aten.view.default(bmm_11, [8, 6, 49, 128]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:106, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
    permute_17: "f32[8, 6, 128, 49]" = torch.ops.aten.permute.default(view_46, [0, 1, 3, 2]);  view_46 = None
    clone_61: "f32[8, 6, 128, 49]" = torch.ops.aten.clone.default(permute_17, memory_format = torch.contiguous_format);  permute_17 = None
    view_47: "f32[8, 768, 7, 7]" = torch.ops.aten.view.default(clone_61, [8, 768, 7, 7]);  clone_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:107, code: x = self.proj(x)
    convolution_46: "f32[8, 768, 7, 7]" = torch.ops.aten.convolution.default(view_47, arg96_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  view_47 = arg96_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:108, code: x = self.proj_drop(x)
    clone_62: "f32[8, 768, 7, 7]" = torch.ops.aten.clone.default(convolution_46);  convolution_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_83: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_80, clone_62);  add_80 = clone_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    convert_element_type_44: "f32[768]" = torch.ops.prims.convert_element_type.default(arg187_1, torch.float32);  arg187_1 = None
    convert_element_type_45: "f32[768]" = torch.ops.prims.convert_element_type.default(arg188_1, torch.float32);  arg188_1 = None
    add_84: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_45, 1e-05);  convert_element_type_45 = None
    sqrt_22: "f32[768]" = torch.ops.aten.sqrt.default(add_84);  add_84 = None
    reciprocal_22: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_22);  sqrt_22 = None
    mul_129: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_22, 1);  reciprocal_22 = None
    unsqueeze_176: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_44, -1);  convert_element_type_44 = None
    unsqueeze_177: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
    unsqueeze_178: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_129, -1);  mul_129 = None
    unsqueeze_179: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
    sub_28: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_83, unsqueeze_177);  unsqueeze_177 = None
    mul_130: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_28, unsqueeze_179);  sub_28 = unsqueeze_179 = None
    unsqueeze_180: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg97_1, -1);  arg97_1 = None
    unsqueeze_181: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
    mul_131: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(mul_130, unsqueeze_181);  mul_130 = unsqueeze_181 = None
    unsqueeze_182: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg98_1, -1);  arg98_1 = None
    unsqueeze_183: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
    add_85: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_131, unsqueeze_183);  mul_131 = unsqueeze_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    convolution_47: "f32[8, 3072, 7, 7]" = torch.ops.aten.convolution.default(add_85, arg99_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_85 = arg99_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_132: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_47, 0.5)
    mul_133: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_47, 0.7071067811865476);  convolution_47 = None
    erf_19: "f32[8, 3072, 7, 7]" = torch.ops.aten.erf.default(mul_133);  mul_133 = None
    add_86: "f32[8, 3072, 7, 7]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    mul_134: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(mul_132, add_86);  mul_132 = add_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:64, code: x = self.drop1(x)
    clone_63: "f32[8, 3072, 7, 7]" = torch.ops.aten.clone.default(mul_134);  mul_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_48: "f32[8, 768, 7, 7]" = torch.ops.aten.convolution.default(clone_63, arg100_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  clone_63 = arg100_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:69, code: x = self.drop3(x)
    clone_64: "f32[8, 768, 7, 7]" = torch.ops.aten.clone.default(convolution_48);  convolution_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_87: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_83, clone_64);  add_83 = clone_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    convert_element_type_46: "f32[768]" = torch.ops.prims.convert_element_type.default(arg190_1, torch.float32);  arg190_1 = None
    convert_element_type_47: "f32[768]" = torch.ops.prims.convert_element_type.default(arg191_1, torch.float32);  arg191_1 = None
    add_88: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_47, 1e-05);  convert_element_type_47 = None
    sqrt_23: "f32[768]" = torch.ops.aten.sqrt.default(add_88);  add_88 = None
    reciprocal_23: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_23);  sqrt_23 = None
    mul_135: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_23, 1);  reciprocal_23 = None
    unsqueeze_184: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_46, -1);  convert_element_type_46 = None
    unsqueeze_185: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
    unsqueeze_186: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_135, -1);  mul_135 = None
    unsqueeze_187: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
    sub_29: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_87, unsqueeze_185);  unsqueeze_185 = None
    mul_136: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_29, unsqueeze_187);  sub_29 = unsqueeze_187 = None
    unsqueeze_188: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg101_1, -1);  arg101_1 = None
    unsqueeze_189: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
    mul_137: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(mul_136, unsqueeze_189);  mul_136 = unsqueeze_189 = None
    unsqueeze_190: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg102_1, -1);  arg102_1 = None
    unsqueeze_191: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
    add_89: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_137, unsqueeze_191);  mul_137 = unsqueeze_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:92, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
    convolution_49: "f32[8, 2304, 7, 7]" = torch.ops.aten.convolution.default(add_89, arg103_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_89 = arg103_1 = None
    view_48: "f32[8, 3, 6, 128, 49]" = torch.ops.aten.view.default(convolution_49, [8, 3, 6, 128, -1]);  convolution_49 = None
    permute_18: "f32[3, 8, 6, 49, 128]" = torch.ops.aten.permute.default(view_48, [1, 0, 2, 4, 3]);  view_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:93, code: q, k, v = x.unbind(0)
    unbind_6 = torch.ops.aten.unbind.int(permute_18);  permute_18 = None
    getitem_18: "f32[8, 6, 49, 128]" = unbind_6[0]
    getitem_19: "f32[8, 6, 49, 128]" = unbind_6[1]
    getitem_20: "f32[8, 6, 49, 128]" = unbind_6[2];  unbind_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:101, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_19: "f32[8, 6, 128, 49]" = torch.ops.aten.permute.default(getitem_19, [0, 1, 3, 2]);  getitem_19 = None
    expand_24: "f32[8, 6, 49, 128]" = torch.ops.aten.expand.default(getitem_18, [8, 6, 49, 128]);  getitem_18 = None
    clone_65: "f32[8, 6, 49, 128]" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
    view_49: "f32[48, 49, 128]" = torch.ops.aten.view.default(clone_65, [48, 49, 128]);  clone_65 = None
    expand_25: "f32[8, 6, 128, 49]" = torch.ops.aten.expand.default(permute_19, [8, 6, 128, 49]);  permute_19 = None
    clone_66: "f32[8, 6, 128, 49]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
    view_50: "f32[48, 128, 49]" = torch.ops.aten.view.default(clone_66, [48, 128, 49]);  clone_66 = None
    bmm_12: "f32[48, 49, 49]" = torch.ops.aten.bmm.default(view_49, view_50);  view_49 = view_50 = None
    view_51: "f32[8, 6, 49, 49]" = torch.ops.aten.view.default(bmm_12, [8, 6, 49, 49]);  bmm_12 = None
    mul_138: "f32[8, 6, 49, 49]" = torch.ops.aten.mul.Tensor(view_51, 0.08838834764831845);  view_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:102, code: attn = attn.softmax(dim=-1)
    amax_6: "f32[8, 6, 49, 1]" = torch.ops.aten.amax.default(mul_138, [-1], True)
    sub_30: "f32[8, 6, 49, 49]" = torch.ops.aten.sub.Tensor(mul_138, amax_6);  mul_138 = amax_6 = None
    exp_6: "f32[8, 6, 49, 49]" = torch.ops.aten.exp.default(sub_30);  sub_30 = None
    sum_7: "f32[8, 6, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_6: "f32[8, 6, 49, 49]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:103, code: attn = self.attn_drop(attn)
    clone_67: "f32[8, 6, 49, 49]" = torch.ops.aten.clone.default(div_6);  div_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:104, code: x = attn @ v
    expand_26: "f32[8, 6, 49, 49]" = torch.ops.aten.expand.default(clone_67, [8, 6, 49, 49]);  clone_67 = None
    view_52: "f32[48, 49, 49]" = torch.ops.aten.view.default(expand_26, [48, 49, 49]);  expand_26 = None
    expand_27: "f32[8, 6, 49, 128]" = torch.ops.aten.expand.default(getitem_20, [8, 6, 49, 128]);  getitem_20 = None
    clone_68: "f32[8, 6, 49, 128]" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
    view_53: "f32[48, 49, 128]" = torch.ops.aten.view.default(clone_68, [48, 49, 128]);  clone_68 = None
    bmm_13: "f32[48, 49, 128]" = torch.ops.aten.bmm.default(view_52, view_53);  view_52 = view_53 = None
    view_54: "f32[8, 6, 49, 128]" = torch.ops.aten.view.default(bmm_13, [8, 6, 49, 128]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:106, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
    permute_20: "f32[8, 6, 128, 49]" = torch.ops.aten.permute.default(view_54, [0, 1, 3, 2]);  view_54 = None
    clone_69: "f32[8, 6, 128, 49]" = torch.ops.aten.clone.default(permute_20, memory_format = torch.contiguous_format);  permute_20 = None
    view_55: "f32[8, 768, 7, 7]" = torch.ops.aten.view.default(clone_69, [8, 768, 7, 7]);  clone_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:107, code: x = self.proj(x)
    convolution_50: "f32[8, 768, 7, 7]" = torch.ops.aten.convolution.default(view_55, arg104_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  view_55 = arg104_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:108, code: x = self.proj_drop(x)
    clone_70: "f32[8, 768, 7, 7]" = torch.ops.aten.clone.default(convolution_50);  convolution_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_90: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_87, clone_70);  add_87 = clone_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    convert_element_type_48: "f32[768]" = torch.ops.prims.convert_element_type.default(arg193_1, torch.float32);  arg193_1 = None
    convert_element_type_49: "f32[768]" = torch.ops.prims.convert_element_type.default(arg194_1, torch.float32);  arg194_1 = None
    add_91: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_49, 1e-05);  convert_element_type_49 = None
    sqrt_24: "f32[768]" = torch.ops.aten.sqrt.default(add_91);  add_91 = None
    reciprocal_24: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_24);  sqrt_24 = None
    mul_139: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_24, 1);  reciprocal_24 = None
    unsqueeze_192: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_48, -1);  convert_element_type_48 = None
    unsqueeze_193: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
    unsqueeze_194: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_139, -1);  mul_139 = None
    unsqueeze_195: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
    sub_31: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_90, unsqueeze_193);  unsqueeze_193 = None
    mul_140: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_31, unsqueeze_195);  sub_31 = unsqueeze_195 = None
    unsqueeze_196: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg105_1, -1);  arg105_1 = None
    unsqueeze_197: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
    mul_141: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(mul_140, unsqueeze_197);  mul_140 = unsqueeze_197 = None
    unsqueeze_198: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg106_1, -1);  arg106_1 = None
    unsqueeze_199: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
    add_92: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_141, unsqueeze_199);  mul_141 = unsqueeze_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    convolution_51: "f32[8, 3072, 7, 7]" = torch.ops.aten.convolution.default(add_92, arg107_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_92 = arg107_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_142: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_51, 0.5)
    mul_143: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_51, 0.7071067811865476);  convolution_51 = None
    erf_20: "f32[8, 3072, 7, 7]" = torch.ops.aten.erf.default(mul_143);  mul_143 = None
    add_93: "f32[8, 3072, 7, 7]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    mul_144: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(mul_142, add_93);  mul_142 = add_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:64, code: x = self.drop1(x)
    clone_71: "f32[8, 3072, 7, 7]" = torch.ops.aten.clone.default(mul_144);  mul_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_52: "f32[8, 768, 7, 7]" = torch.ops.aten.convolution.default(clone_71, arg108_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  clone_71 = arg108_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:69, code: x = self.drop3(x)
    clone_72: "f32[8, 768, 7, 7]" = torch.ops.aten.clone.default(convolution_52);  convolution_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_94: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_90, clone_72);  add_90 = clone_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    convert_element_type_50: "f32[768]" = torch.ops.prims.convert_element_type.default(arg196_1, torch.float32);  arg196_1 = None
    convert_element_type_51: "f32[768]" = torch.ops.prims.convert_element_type.default(arg197_1, torch.float32);  arg197_1 = None
    add_95: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_51, 1e-05);  convert_element_type_51 = None
    sqrt_25: "f32[768]" = torch.ops.aten.sqrt.default(add_95);  add_95 = None
    reciprocal_25: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_25);  sqrt_25 = None
    mul_145: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_25, 1);  reciprocal_25 = None
    unsqueeze_200: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_50, -1);  convert_element_type_50 = None
    unsqueeze_201: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
    unsqueeze_202: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_145, -1);  mul_145 = None
    unsqueeze_203: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
    sub_32: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_94, unsqueeze_201);  unsqueeze_201 = None
    mul_146: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_32, unsqueeze_203);  sub_32 = unsqueeze_203 = None
    unsqueeze_204: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg109_1, -1);  arg109_1 = None
    unsqueeze_205: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
    mul_147: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(mul_146, unsqueeze_205);  mul_146 = unsqueeze_205 = None
    unsqueeze_206: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg110_1, -1);  arg110_1 = None
    unsqueeze_207: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
    add_96: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_147, unsqueeze_207);  mul_147 = unsqueeze_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:92, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
    convolution_53: "f32[8, 2304, 7, 7]" = torch.ops.aten.convolution.default(add_96, arg111_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_96 = arg111_1 = None
    view_56: "f32[8, 3, 6, 128, 49]" = torch.ops.aten.view.default(convolution_53, [8, 3, 6, 128, -1]);  convolution_53 = None
    permute_21: "f32[3, 8, 6, 49, 128]" = torch.ops.aten.permute.default(view_56, [1, 0, 2, 4, 3]);  view_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:93, code: q, k, v = x.unbind(0)
    unbind_7 = torch.ops.aten.unbind.int(permute_21);  permute_21 = None
    getitem_21: "f32[8, 6, 49, 128]" = unbind_7[0]
    getitem_22: "f32[8, 6, 49, 128]" = unbind_7[1]
    getitem_23: "f32[8, 6, 49, 128]" = unbind_7[2];  unbind_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:101, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_22: "f32[8, 6, 128, 49]" = torch.ops.aten.permute.default(getitem_22, [0, 1, 3, 2]);  getitem_22 = None
    expand_28: "f32[8, 6, 49, 128]" = torch.ops.aten.expand.default(getitem_21, [8, 6, 49, 128]);  getitem_21 = None
    clone_73: "f32[8, 6, 49, 128]" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
    view_57: "f32[48, 49, 128]" = torch.ops.aten.view.default(clone_73, [48, 49, 128]);  clone_73 = None
    expand_29: "f32[8, 6, 128, 49]" = torch.ops.aten.expand.default(permute_22, [8, 6, 128, 49]);  permute_22 = None
    clone_74: "f32[8, 6, 128, 49]" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
    view_58: "f32[48, 128, 49]" = torch.ops.aten.view.default(clone_74, [48, 128, 49]);  clone_74 = None
    bmm_14: "f32[48, 49, 49]" = torch.ops.aten.bmm.default(view_57, view_58);  view_57 = view_58 = None
    view_59: "f32[8, 6, 49, 49]" = torch.ops.aten.view.default(bmm_14, [8, 6, 49, 49]);  bmm_14 = None
    mul_148: "f32[8, 6, 49, 49]" = torch.ops.aten.mul.Tensor(view_59, 0.08838834764831845);  view_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:102, code: attn = attn.softmax(dim=-1)
    amax_7: "f32[8, 6, 49, 1]" = torch.ops.aten.amax.default(mul_148, [-1], True)
    sub_33: "f32[8, 6, 49, 49]" = torch.ops.aten.sub.Tensor(mul_148, amax_7);  mul_148 = amax_7 = None
    exp_7: "f32[8, 6, 49, 49]" = torch.ops.aten.exp.default(sub_33);  sub_33 = None
    sum_8: "f32[8, 6, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_7: "f32[8, 6, 49, 49]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:103, code: attn = self.attn_drop(attn)
    clone_75: "f32[8, 6, 49, 49]" = torch.ops.aten.clone.default(div_7);  div_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:104, code: x = attn @ v
    expand_30: "f32[8, 6, 49, 49]" = torch.ops.aten.expand.default(clone_75, [8, 6, 49, 49]);  clone_75 = None
    view_60: "f32[48, 49, 49]" = torch.ops.aten.view.default(expand_30, [48, 49, 49]);  expand_30 = None
    expand_31: "f32[8, 6, 49, 128]" = torch.ops.aten.expand.default(getitem_23, [8, 6, 49, 128]);  getitem_23 = None
    clone_76: "f32[8, 6, 49, 128]" = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
    view_61: "f32[48, 49, 128]" = torch.ops.aten.view.default(clone_76, [48, 49, 128]);  clone_76 = None
    bmm_15: "f32[48, 49, 128]" = torch.ops.aten.bmm.default(view_60, view_61);  view_60 = view_61 = None
    view_62: "f32[8, 6, 49, 128]" = torch.ops.aten.view.default(bmm_15, [8, 6, 49, 128]);  bmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:106, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
    permute_23: "f32[8, 6, 128, 49]" = torch.ops.aten.permute.default(view_62, [0, 1, 3, 2]);  view_62 = None
    clone_77: "f32[8, 6, 128, 49]" = torch.ops.aten.clone.default(permute_23, memory_format = torch.contiguous_format);  permute_23 = None
    view_63: "f32[8, 768, 7, 7]" = torch.ops.aten.view.default(clone_77, [8, 768, 7, 7]);  clone_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:107, code: x = self.proj(x)
    convolution_54: "f32[8, 768, 7, 7]" = torch.ops.aten.convolution.default(view_63, arg112_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  view_63 = arg112_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:108, code: x = self.proj_drop(x)
    clone_78: "f32[8, 768, 7, 7]" = torch.ops.aten.clone.default(convolution_54);  convolution_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_97: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_94, clone_78);  add_94 = clone_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    convert_element_type_52: "f32[768]" = torch.ops.prims.convert_element_type.default(arg199_1, torch.float32);  arg199_1 = None
    convert_element_type_53: "f32[768]" = torch.ops.prims.convert_element_type.default(arg200_1, torch.float32);  arg200_1 = None
    add_98: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_53, 1e-05);  convert_element_type_53 = None
    sqrt_26: "f32[768]" = torch.ops.aten.sqrt.default(add_98);  add_98 = None
    reciprocal_26: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_26);  sqrt_26 = None
    mul_149: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_26, 1);  reciprocal_26 = None
    unsqueeze_208: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_52, -1);  convert_element_type_52 = None
    unsqueeze_209: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
    unsqueeze_210: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_149, -1);  mul_149 = None
    unsqueeze_211: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
    sub_34: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_97, unsqueeze_209);  unsqueeze_209 = None
    mul_150: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_34, unsqueeze_211);  sub_34 = unsqueeze_211 = None
    unsqueeze_212: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg113_1, -1);  arg113_1 = None
    unsqueeze_213: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
    mul_151: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(mul_150, unsqueeze_213);  mul_150 = unsqueeze_213 = None
    unsqueeze_214: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg114_1, -1);  arg114_1 = None
    unsqueeze_215: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
    add_99: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_151, unsqueeze_215);  mul_151 = unsqueeze_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    convolution_55: "f32[8, 3072, 7, 7]" = torch.ops.aten.convolution.default(add_99, arg115_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_99 = arg115_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_152: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_55, 0.5)
    mul_153: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_55, 0.7071067811865476);  convolution_55 = None
    erf_21: "f32[8, 3072, 7, 7]" = torch.ops.aten.erf.default(mul_153);  mul_153 = None
    add_100: "f32[8, 3072, 7, 7]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    mul_154: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(mul_152, add_100);  mul_152 = add_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:64, code: x = self.drop1(x)
    clone_79: "f32[8, 3072, 7, 7]" = torch.ops.aten.clone.default(mul_154);  mul_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_56: "f32[8, 768, 7, 7]" = torch.ops.aten.convolution.default(clone_79, arg116_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  clone_79 = arg116_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:69, code: x = self.drop3(x)
    clone_80: "f32[8, 768, 7, 7]" = torch.ops.aten.clone.default(convolution_56);  convolution_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_101: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_97, clone_80);  add_97 = clone_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:427, code: x = self.norm(x)
    convert_element_type_54: "f32[768]" = torch.ops.prims.convert_element_type.default(arg202_1, torch.float32);  arg202_1 = None
    convert_element_type_55: "f32[768]" = torch.ops.prims.convert_element_type.default(arg203_1, torch.float32);  arg203_1 = None
    add_102: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_55, 1e-05);  convert_element_type_55 = None
    sqrt_27: "f32[768]" = torch.ops.aten.sqrt.default(add_102);  add_102 = None
    reciprocal_27: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_27);  sqrt_27 = None
    mul_155: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_27, 1);  reciprocal_27 = None
    unsqueeze_216: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_54, -1);  convert_element_type_54 = None
    unsqueeze_217: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
    unsqueeze_218: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_155, -1);  mul_155 = None
    unsqueeze_219: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, -1);  unsqueeze_218 = None
    sub_35: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_101, unsqueeze_217);  add_101 = unsqueeze_217 = None
    mul_156: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_35, unsqueeze_219);  sub_35 = unsqueeze_219 = None
    unsqueeze_220: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg117_1, -1);  arg117_1 = None
    unsqueeze_221: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, -1);  unsqueeze_220 = None
    mul_157: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(mul_156, unsqueeze_221);  mul_156 = unsqueeze_221 = None
    unsqueeze_222: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg118_1, -1);  arg118_1 = None
    unsqueeze_223: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, -1);  unsqueeze_222 = None
    add_103: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_157, unsqueeze_223);  mul_157 = unsqueeze_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean: "f32[8, 768, 1, 1]" = torch.ops.aten.mean.dim(add_103, [-1, -2], True);  add_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_64: "f32[8, 768]" = torch.ops.aten.view.default(mean, [8, 768]);  mean = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:432, code: x = self.head_drop(x)
    clone_81: "f32[8, 768]" = torch.ops.aten.clone.default(view_64);  view_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:433, code: return x if pre_logits else self.head(x)
    permute_24: "f32[768, 1000]" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
    addmm: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg120_1, clone_81, permute_24);  arg120_1 = clone_81 = permute_24 = None
    return (addmm,)
    