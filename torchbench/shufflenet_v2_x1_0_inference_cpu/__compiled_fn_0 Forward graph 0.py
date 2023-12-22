from __future__ import annotations



def forward(self, arg0_1: "f32[24, 3, 3, 3]", arg1_1: "f32[24]", arg2_1: "f32[24]", arg3_1: "f32[24, 1, 3, 3]", arg4_1: "f32[24]", arg5_1: "f32[24]", arg6_1: "f32[58, 24, 1, 1]", arg7_1: "f32[58]", arg8_1: "f32[58]", arg9_1: "f32[58, 24, 1, 1]", arg10_1: "f32[58]", arg11_1: "f32[58]", arg12_1: "f32[58, 1, 3, 3]", arg13_1: "f32[58]", arg14_1: "f32[58]", arg15_1: "f32[58, 58, 1, 1]", arg16_1: "f32[58]", arg17_1: "f32[58]", arg18_1: "f32[58, 58, 1, 1]", arg19_1: "f32[58]", arg20_1: "f32[58]", arg21_1: "f32[58, 1, 3, 3]", arg22_1: "f32[58]", arg23_1: "f32[58]", arg24_1: "f32[58, 58, 1, 1]", arg25_1: "f32[58]", arg26_1: "f32[58]", arg27_1: "f32[58, 58, 1, 1]", arg28_1: "f32[58]", arg29_1: "f32[58]", arg30_1: "f32[58, 1, 3, 3]", arg31_1: "f32[58]", arg32_1: "f32[58]", arg33_1: "f32[58, 58, 1, 1]", arg34_1: "f32[58]", arg35_1: "f32[58]", arg36_1: "f32[58, 58, 1, 1]", arg37_1: "f32[58]", arg38_1: "f32[58]", arg39_1: "f32[58, 1, 3, 3]", arg40_1: "f32[58]", arg41_1: "f32[58]", arg42_1: "f32[58, 58, 1, 1]", arg43_1: "f32[58]", arg44_1: "f32[58]", arg45_1: "f32[116, 1, 3, 3]", arg46_1: "f32[116]", arg47_1: "f32[116]", arg48_1: "f32[116, 116, 1, 1]", arg49_1: "f32[116]", arg50_1: "f32[116]", arg51_1: "f32[116, 116, 1, 1]", arg52_1: "f32[116]", arg53_1: "f32[116]", arg54_1: "f32[116, 1, 3, 3]", arg55_1: "f32[116]", arg56_1: "f32[116]", arg57_1: "f32[116, 116, 1, 1]", arg58_1: "f32[116]", arg59_1: "f32[116]", arg60_1: "f32[116, 116, 1, 1]", arg61_1: "f32[116]", arg62_1: "f32[116]", arg63_1: "f32[116, 1, 3, 3]", arg64_1: "f32[116]", arg65_1: "f32[116]", arg66_1: "f32[116, 116, 1, 1]", arg67_1: "f32[116]", arg68_1: "f32[116]", arg69_1: "f32[116, 116, 1, 1]", arg70_1: "f32[116]", arg71_1: "f32[116]", arg72_1: "f32[116, 1, 3, 3]", arg73_1: "f32[116]", arg74_1: "f32[116]", arg75_1: "f32[116, 116, 1, 1]", arg76_1: "f32[116]", arg77_1: "f32[116]", arg78_1: "f32[116, 116, 1, 1]", arg79_1: "f32[116]", arg80_1: "f32[116]", arg81_1: "f32[116, 1, 3, 3]", arg82_1: "f32[116]", arg83_1: "f32[116]", arg84_1: "f32[116, 116, 1, 1]", arg85_1: "f32[116]", arg86_1: "f32[116]", arg87_1: "f32[116, 116, 1, 1]", arg88_1: "f32[116]", arg89_1: "f32[116]", arg90_1: "f32[116, 1, 3, 3]", arg91_1: "f32[116]", arg92_1: "f32[116]", arg93_1: "f32[116, 116, 1, 1]", arg94_1: "f32[116]", arg95_1: "f32[116]", arg96_1: "f32[116, 116, 1, 1]", arg97_1: "f32[116]", arg98_1: "f32[116]", arg99_1: "f32[116, 1, 3, 3]", arg100_1: "f32[116]", arg101_1: "f32[116]", arg102_1: "f32[116, 116, 1, 1]", arg103_1: "f32[116]", arg104_1: "f32[116]", arg105_1: "f32[116, 116, 1, 1]", arg106_1: "f32[116]", arg107_1: "f32[116]", arg108_1: "f32[116, 1, 3, 3]", arg109_1: "f32[116]", arg110_1: "f32[116]", arg111_1: "f32[116, 116, 1, 1]", arg112_1: "f32[116]", arg113_1: "f32[116]", arg114_1: "f32[116, 116, 1, 1]", arg115_1: "f32[116]", arg116_1: "f32[116]", arg117_1: "f32[116, 1, 3, 3]", arg118_1: "f32[116]", arg119_1: "f32[116]", arg120_1: "f32[116, 116, 1, 1]", arg121_1: "f32[116]", arg122_1: "f32[116]", arg123_1: "f32[232, 1, 3, 3]", arg124_1: "f32[232]", arg125_1: "f32[232]", arg126_1: "f32[232, 232, 1, 1]", arg127_1: "f32[232]", arg128_1: "f32[232]", arg129_1: "f32[232, 232, 1, 1]", arg130_1: "f32[232]", arg131_1: "f32[232]", arg132_1: "f32[232, 1, 3, 3]", arg133_1: "f32[232]", arg134_1: "f32[232]", arg135_1: "f32[232, 232, 1, 1]", arg136_1: "f32[232]", arg137_1: "f32[232]", arg138_1: "f32[232, 232, 1, 1]", arg139_1: "f32[232]", arg140_1: "f32[232]", arg141_1: "f32[232, 1, 3, 3]", arg142_1: "f32[232]", arg143_1: "f32[232]", arg144_1: "f32[232, 232, 1, 1]", arg145_1: "f32[232]", arg146_1: "f32[232]", arg147_1: "f32[232, 232, 1, 1]", arg148_1: "f32[232]", arg149_1: "f32[232]", arg150_1: "f32[232, 1, 3, 3]", arg151_1: "f32[232]", arg152_1: "f32[232]", arg153_1: "f32[232, 232, 1, 1]", arg154_1: "f32[232]", arg155_1: "f32[232]", arg156_1: "f32[232, 232, 1, 1]", arg157_1: "f32[232]", arg158_1: "f32[232]", arg159_1: "f32[232, 1, 3, 3]", arg160_1: "f32[232]", arg161_1: "f32[232]", arg162_1: "f32[232, 232, 1, 1]", arg163_1: "f32[232]", arg164_1: "f32[232]", arg165_1: "f32[1024, 464, 1, 1]", arg166_1: "f32[1024]", arg167_1: "f32[1024]", arg168_1: "f32[1000, 1024]", arg169_1: "f32[1000]", arg170_1: "f32[24]", arg171_1: "f32[24]", arg172_1: "i64[]", arg173_1: "f32[24]", arg174_1: "f32[24]", arg175_1: "i64[]", arg176_1: "f32[58]", arg177_1: "f32[58]", arg178_1: "i64[]", arg179_1: "f32[58]", arg180_1: "f32[58]", arg181_1: "i64[]", arg182_1: "f32[58]", arg183_1: "f32[58]", arg184_1: "i64[]", arg185_1: "f32[58]", arg186_1: "f32[58]", arg187_1: "i64[]", arg188_1: "f32[58]", arg189_1: "f32[58]", arg190_1: "i64[]", arg191_1: "f32[58]", arg192_1: "f32[58]", arg193_1: "i64[]", arg194_1: "f32[58]", arg195_1: "f32[58]", arg196_1: "i64[]", arg197_1: "f32[58]", arg198_1: "f32[58]", arg199_1: "i64[]", arg200_1: "f32[58]", arg201_1: "f32[58]", arg202_1: "i64[]", arg203_1: "f32[58]", arg204_1: "f32[58]", arg205_1: "i64[]", arg206_1: "f32[58]", arg207_1: "f32[58]", arg208_1: "i64[]", arg209_1: "f32[58]", arg210_1: "f32[58]", arg211_1: "i64[]", arg212_1: "f32[58]", arg213_1: "f32[58]", arg214_1: "i64[]", arg215_1: "f32[116]", arg216_1: "f32[116]", arg217_1: "i64[]", arg218_1: "f32[116]", arg219_1: "f32[116]", arg220_1: "i64[]", arg221_1: "f32[116]", arg222_1: "f32[116]", arg223_1: "i64[]", arg224_1: "f32[116]", arg225_1: "f32[116]", arg226_1: "i64[]", arg227_1: "f32[116]", arg228_1: "f32[116]", arg229_1: "i64[]", arg230_1: "f32[116]", arg231_1: "f32[116]", arg232_1: "i64[]", arg233_1: "f32[116]", arg234_1: "f32[116]", arg235_1: "i64[]", arg236_1: "f32[116]", arg237_1: "f32[116]", arg238_1: "i64[]", arg239_1: "f32[116]", arg240_1: "f32[116]", arg241_1: "i64[]", arg242_1: "f32[116]", arg243_1: "f32[116]", arg244_1: "i64[]", arg245_1: "f32[116]", arg246_1: "f32[116]", arg247_1: "i64[]", arg248_1: "f32[116]", arg249_1: "f32[116]", arg250_1: "i64[]", arg251_1: "f32[116]", arg252_1: "f32[116]", arg253_1: "i64[]", arg254_1: "f32[116]", arg255_1: "f32[116]", arg256_1: "i64[]", arg257_1: "f32[116]", arg258_1: "f32[116]", arg259_1: "i64[]", arg260_1: "f32[116]", arg261_1: "f32[116]", arg262_1: "i64[]", arg263_1: "f32[116]", arg264_1: "f32[116]", arg265_1: "i64[]", arg266_1: "f32[116]", arg267_1: "f32[116]", arg268_1: "i64[]", arg269_1: "f32[116]", arg270_1: "f32[116]", arg271_1: "i64[]", arg272_1: "f32[116]", arg273_1: "f32[116]", arg274_1: "i64[]", arg275_1: "f32[116]", arg276_1: "f32[116]", arg277_1: "i64[]", arg278_1: "f32[116]", arg279_1: "f32[116]", arg280_1: "i64[]", arg281_1: "f32[116]", arg282_1: "f32[116]", arg283_1: "i64[]", arg284_1: "f32[116]", arg285_1: "f32[116]", arg286_1: "i64[]", arg287_1: "f32[116]", arg288_1: "f32[116]", arg289_1: "i64[]", arg290_1: "f32[116]", arg291_1: "f32[116]", arg292_1: "i64[]", arg293_1: "f32[232]", arg294_1: "f32[232]", arg295_1: "i64[]", arg296_1: "f32[232]", arg297_1: "f32[232]", arg298_1: "i64[]", arg299_1: "f32[232]", arg300_1: "f32[232]", arg301_1: "i64[]", arg302_1: "f32[232]", arg303_1: "f32[232]", arg304_1: "i64[]", arg305_1: "f32[232]", arg306_1: "f32[232]", arg307_1: "i64[]", arg308_1: "f32[232]", arg309_1: "f32[232]", arg310_1: "i64[]", arg311_1: "f32[232]", arg312_1: "f32[232]", arg313_1: "i64[]", arg314_1: "f32[232]", arg315_1: "f32[232]", arg316_1: "i64[]", arg317_1: "f32[232]", arg318_1: "f32[232]", arg319_1: "i64[]", arg320_1: "f32[232]", arg321_1: "f32[232]", arg322_1: "i64[]", arg323_1: "f32[232]", arg324_1: "f32[232]", arg325_1: "i64[]", arg326_1: "f32[232]", arg327_1: "f32[232]", arg328_1: "i64[]", arg329_1: "f32[232]", arg330_1: "f32[232]", arg331_1: "i64[]", arg332_1: "f32[232]", arg333_1: "f32[232]", arg334_1: "i64[]", arg335_1: "f32[1024]", arg336_1: "f32[1024]", arg337_1: "i64[]", arg338_1: "f32[4, 3, 224, 224]"):
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:155, code: x = self.conv1(x)
    convolution: "f32[4, 24, 112, 112]" = torch.ops.aten.convolution.default(arg338_1, arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg338_1 = arg0_1 = None
    convert_element_type: "f32[24]" = torch.ops.prims.convert_element_type.default(arg170_1, torch.float32);  arg170_1 = None
    convert_element_type_1: "f32[24]" = torch.ops.prims.convert_element_type.default(arg171_1, torch.float32);  arg171_1 = None
    add: "f32[24]" = torch.ops.aten.add.Tensor(convert_element_type_1, 1e-05);  convert_element_type_1 = None
    sqrt: "f32[24]" = torch.ops.aten.sqrt.default(add);  add = None
    reciprocal: "f32[24]" = torch.ops.aten.reciprocal.default(sqrt);  sqrt = None
    mul: "f32[24]" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
    unsqueeze: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type, -1);  convert_element_type = None
    unsqueeze_1: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    unsqueeze_2: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(mul, -1);  mul = None
    unsqueeze_3: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    sub: "f32[4, 24, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1);  convolution = unsqueeze_1 = None
    mul_1: "f32[4, 24, 112, 112]" = torch.ops.aten.mul.Tensor(sub, unsqueeze_3);  sub = unsqueeze_3 = None
    unsqueeze_4: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg1_1, -1);  arg1_1 = None
    unsqueeze_5: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_2: "f32[4, 24, 112, 112]" = torch.ops.aten.mul.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None
    unsqueeze_6: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
    unsqueeze_7: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_1: "f32[4, 24, 112, 112]" = torch.ops.aten.add.Tensor(mul_2, unsqueeze_7);  mul_2 = unsqueeze_7 = None
    relu: "f32[4, 24, 112, 112]" = torch.ops.aten.relu.default(add_1);  add_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:156, code: x = self.maxpool(x)
    max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices.default(relu, [3, 3], [2, 2], [1, 1]);  relu = None
    getitem: "f32[4, 24, 56, 56]" = max_pool2d_with_indices[0];  max_pool2d_with_indices = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:97, code: out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
    convolution_1: "f32[4, 24, 28, 28]" = torch.ops.aten.convolution.default(getitem, arg3_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 24);  arg3_1 = None
    convert_element_type_2: "f32[24]" = torch.ops.prims.convert_element_type.default(arg173_1, torch.float32);  arg173_1 = None
    convert_element_type_3: "f32[24]" = torch.ops.prims.convert_element_type.default(arg174_1, torch.float32);  arg174_1 = None
    add_2: "f32[24]" = torch.ops.aten.add.Tensor(convert_element_type_3, 1e-05);  convert_element_type_3 = None
    sqrt_1: "f32[24]" = torch.ops.aten.sqrt.default(add_2);  add_2 = None
    reciprocal_1: "f32[24]" = torch.ops.aten.reciprocal.default(sqrt_1);  sqrt_1 = None
    mul_3: "f32[24]" = torch.ops.aten.mul.Tensor(reciprocal_1, 1);  reciprocal_1 = None
    unsqueeze_8: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, -1);  convert_element_type_2 = None
    unsqueeze_9: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    unsqueeze_10: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(mul_3, -1);  mul_3 = None
    unsqueeze_11: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    sub_1: "f32[4, 24, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_9);  convolution_1 = unsqueeze_9 = None
    mul_4: "f32[4, 24, 28, 28]" = torch.ops.aten.mul.Tensor(sub_1, unsqueeze_11);  sub_1 = unsqueeze_11 = None
    unsqueeze_12: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
    unsqueeze_13: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_5: "f32[4, 24, 28, 28]" = torch.ops.aten.mul.Tensor(mul_4, unsqueeze_13);  mul_4 = unsqueeze_13 = None
    unsqueeze_14: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
    unsqueeze_15: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_3: "f32[4, 24, 28, 28]" = torch.ops.aten.add.Tensor(mul_5, unsqueeze_15);  mul_5 = unsqueeze_15 = None
    convolution_2: "f32[4, 58, 28, 28]" = torch.ops.aten.convolution.default(add_3, arg6_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_3 = arg6_1 = None
    convert_element_type_4: "f32[58]" = torch.ops.prims.convert_element_type.default(arg176_1, torch.float32);  arg176_1 = None
    convert_element_type_5: "f32[58]" = torch.ops.prims.convert_element_type.default(arg177_1, torch.float32);  arg177_1 = None
    add_4: "f32[58]" = torch.ops.aten.add.Tensor(convert_element_type_5, 1e-05);  convert_element_type_5 = None
    sqrt_2: "f32[58]" = torch.ops.aten.sqrt.default(add_4);  add_4 = None
    reciprocal_2: "f32[58]" = torch.ops.aten.reciprocal.default(sqrt_2);  sqrt_2 = None
    mul_6: "f32[58]" = torch.ops.aten.mul.Tensor(reciprocal_2, 1);  reciprocal_2 = None
    unsqueeze_16: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_4, -1);  convert_element_type_4 = None
    unsqueeze_17: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    unsqueeze_18: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(mul_6, -1);  mul_6 = None
    unsqueeze_19: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    sub_2: "f32[4, 58, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_17);  convolution_2 = unsqueeze_17 = None
    mul_7: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(sub_2, unsqueeze_19);  sub_2 = unsqueeze_19 = None
    unsqueeze_20: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
    unsqueeze_21: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_8: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_21);  mul_7 = unsqueeze_21 = None
    unsqueeze_22: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(arg8_1, -1);  arg8_1 = None
    unsqueeze_23: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_5: "f32[4, 58, 28, 28]" = torch.ops.aten.add.Tensor(mul_8, unsqueeze_23);  mul_8 = unsqueeze_23 = None
    relu_1: "f32[4, 58, 28, 28]" = torch.ops.aten.relu.default(add_5);  add_5 = None
    convolution_3: "f32[4, 58, 56, 56]" = torch.ops.aten.convolution.default(getitem, arg9_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem = arg9_1 = None
    convert_element_type_6: "f32[58]" = torch.ops.prims.convert_element_type.default(arg179_1, torch.float32);  arg179_1 = None
    convert_element_type_7: "f32[58]" = torch.ops.prims.convert_element_type.default(arg180_1, torch.float32);  arg180_1 = None
    add_6: "f32[58]" = torch.ops.aten.add.Tensor(convert_element_type_7, 1e-05);  convert_element_type_7 = None
    sqrt_3: "f32[58]" = torch.ops.aten.sqrt.default(add_6);  add_6 = None
    reciprocal_3: "f32[58]" = torch.ops.aten.reciprocal.default(sqrt_3);  sqrt_3 = None
    mul_9: "f32[58]" = torch.ops.aten.mul.Tensor(reciprocal_3, 1);  reciprocal_3 = None
    unsqueeze_24: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_6, -1);  convert_element_type_6 = None
    unsqueeze_25: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
    unsqueeze_26: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(mul_9, -1);  mul_9 = None
    unsqueeze_27: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
    sub_3: "f32[4, 58, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_25);  convolution_3 = unsqueeze_25 = None
    mul_10: "f32[4, 58, 56, 56]" = torch.ops.aten.mul.Tensor(sub_3, unsqueeze_27);  sub_3 = unsqueeze_27 = None
    unsqueeze_28: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
    unsqueeze_29: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    mul_11: "f32[4, 58, 56, 56]" = torch.ops.aten.mul.Tensor(mul_10, unsqueeze_29);  mul_10 = unsqueeze_29 = None
    unsqueeze_30: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(arg11_1, -1);  arg11_1 = None
    unsqueeze_31: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
    add_7: "f32[4, 58, 56, 56]" = torch.ops.aten.add.Tensor(mul_11, unsqueeze_31);  mul_11 = unsqueeze_31 = None
    relu_2: "f32[4, 58, 56, 56]" = torch.ops.aten.relu.default(add_7);  add_7 = None
    convolution_4: "f32[4, 58, 28, 28]" = torch.ops.aten.convolution.default(relu_2, arg12_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 58);  relu_2 = arg12_1 = None
    convert_element_type_8: "f32[58]" = torch.ops.prims.convert_element_type.default(arg182_1, torch.float32);  arg182_1 = None
    convert_element_type_9: "f32[58]" = torch.ops.prims.convert_element_type.default(arg183_1, torch.float32);  arg183_1 = None
    add_8: "f32[58]" = torch.ops.aten.add.Tensor(convert_element_type_9, 1e-05);  convert_element_type_9 = None
    sqrt_4: "f32[58]" = torch.ops.aten.sqrt.default(add_8);  add_8 = None
    reciprocal_4: "f32[58]" = torch.ops.aten.reciprocal.default(sqrt_4);  sqrt_4 = None
    mul_12: "f32[58]" = torch.ops.aten.mul.Tensor(reciprocal_4, 1);  reciprocal_4 = None
    unsqueeze_32: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_8, -1);  convert_element_type_8 = None
    unsqueeze_33: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    unsqueeze_34: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(mul_12, -1);  mul_12 = None
    unsqueeze_35: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
    sub_4: "f32[4, 58, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_33);  convolution_4 = unsqueeze_33 = None
    mul_13: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(sub_4, unsqueeze_35);  sub_4 = unsqueeze_35 = None
    unsqueeze_36: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(arg13_1, -1);  arg13_1 = None
    unsqueeze_37: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
    mul_14: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(mul_13, unsqueeze_37);  mul_13 = unsqueeze_37 = None
    unsqueeze_38: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
    unsqueeze_39: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
    add_9: "f32[4, 58, 28, 28]" = torch.ops.aten.add.Tensor(mul_14, unsqueeze_39);  mul_14 = unsqueeze_39 = None
    convolution_5: "f32[4, 58, 28, 28]" = torch.ops.aten.convolution.default(add_9, arg15_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_9 = arg15_1 = None
    convert_element_type_10: "f32[58]" = torch.ops.prims.convert_element_type.default(arg185_1, torch.float32);  arg185_1 = None
    convert_element_type_11: "f32[58]" = torch.ops.prims.convert_element_type.default(arg186_1, torch.float32);  arg186_1 = None
    add_10: "f32[58]" = torch.ops.aten.add.Tensor(convert_element_type_11, 1e-05);  convert_element_type_11 = None
    sqrt_5: "f32[58]" = torch.ops.aten.sqrt.default(add_10);  add_10 = None
    reciprocal_5: "f32[58]" = torch.ops.aten.reciprocal.default(sqrt_5);  sqrt_5 = None
    mul_15: "f32[58]" = torch.ops.aten.mul.Tensor(reciprocal_5, 1);  reciprocal_5 = None
    unsqueeze_40: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_10, -1);  convert_element_type_10 = None
    unsqueeze_41: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    unsqueeze_42: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(mul_15, -1);  mul_15 = None
    unsqueeze_43: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    sub_5: "f32[4, 58, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_41);  convolution_5 = unsqueeze_41 = None
    mul_16: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(sub_5, unsqueeze_43);  sub_5 = unsqueeze_43 = None
    unsqueeze_44: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(arg16_1, -1);  arg16_1 = None
    unsqueeze_45: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    mul_17: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(mul_16, unsqueeze_45);  mul_16 = unsqueeze_45 = None
    unsqueeze_46: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(arg17_1, -1);  arg17_1 = None
    unsqueeze_47: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
    add_11: "f32[4, 58, 28, 28]" = torch.ops.aten.add.Tensor(mul_17, unsqueeze_47);  mul_17 = unsqueeze_47 = None
    relu_3: "f32[4, 58, 28, 28]" = torch.ops.aten.relu.default(add_11);  add_11 = None
    cat: "f32[4, 116, 28, 28]" = torch.ops.aten.cat.default([relu_1, relu_3], 1);  relu_1 = relu_3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    view: "f32[4, 2, 58, 28, 28]" = torch.ops.aten.view.default(cat, [4, 2, 58, 28, 28]);  cat = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    permute: "f32[4, 58, 2, 28, 28]" = torch.ops.aten.permute.default(view, [0, 2, 1, 3, 4]);  view = None
    clone: "f32[4, 58, 2, 28, 28]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    view_1: "f32[4, 116, 28, 28]" = torch.ops.aten.view.default(clone, [4, 116, 28, 28]);  clone = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:94, code: x1, x2 = x.chunk(2, dim=1)
    split = torch.ops.aten.split.Tensor(view_1, 58, 1);  view_1 = None
    getitem_2: "f32[4, 58, 28, 28]" = split[0]
    getitem_3: "f32[4, 58, 28, 28]" = split[1];  split = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:95, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
    convolution_6: "f32[4, 58, 28, 28]" = torch.ops.aten.convolution.default(getitem_3, arg18_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_3 = arg18_1 = None
    convert_element_type_12: "f32[58]" = torch.ops.prims.convert_element_type.default(arg188_1, torch.float32);  arg188_1 = None
    convert_element_type_13: "f32[58]" = torch.ops.prims.convert_element_type.default(arg189_1, torch.float32);  arg189_1 = None
    add_12: "f32[58]" = torch.ops.aten.add.Tensor(convert_element_type_13, 1e-05);  convert_element_type_13 = None
    sqrt_6: "f32[58]" = torch.ops.aten.sqrt.default(add_12);  add_12 = None
    reciprocal_6: "f32[58]" = torch.ops.aten.reciprocal.default(sqrt_6);  sqrt_6 = None
    mul_18: "f32[58]" = torch.ops.aten.mul.Tensor(reciprocal_6, 1);  reciprocal_6 = None
    unsqueeze_48: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_12, -1);  convert_element_type_12 = None
    unsqueeze_49: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
    unsqueeze_50: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(mul_18, -1);  mul_18 = None
    unsqueeze_51: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
    sub_6: "f32[4, 58, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_49);  convolution_6 = unsqueeze_49 = None
    mul_19: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(sub_6, unsqueeze_51);  sub_6 = unsqueeze_51 = None
    unsqueeze_52: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(arg19_1, -1);  arg19_1 = None
    unsqueeze_53: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    mul_20: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(mul_19, unsqueeze_53);  mul_19 = unsqueeze_53 = None
    unsqueeze_54: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(arg20_1, -1);  arg20_1 = None
    unsqueeze_55: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
    add_13: "f32[4, 58, 28, 28]" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_55);  mul_20 = unsqueeze_55 = None
    relu_4: "f32[4, 58, 28, 28]" = torch.ops.aten.relu.default(add_13);  add_13 = None
    convolution_7: "f32[4, 58, 28, 28]" = torch.ops.aten.convolution.default(relu_4, arg21_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 58);  relu_4 = arg21_1 = None
    convert_element_type_14: "f32[58]" = torch.ops.prims.convert_element_type.default(arg191_1, torch.float32);  arg191_1 = None
    convert_element_type_15: "f32[58]" = torch.ops.prims.convert_element_type.default(arg192_1, torch.float32);  arg192_1 = None
    add_14: "f32[58]" = torch.ops.aten.add.Tensor(convert_element_type_15, 1e-05);  convert_element_type_15 = None
    sqrt_7: "f32[58]" = torch.ops.aten.sqrt.default(add_14);  add_14 = None
    reciprocal_7: "f32[58]" = torch.ops.aten.reciprocal.default(sqrt_7);  sqrt_7 = None
    mul_21: "f32[58]" = torch.ops.aten.mul.Tensor(reciprocal_7, 1);  reciprocal_7 = None
    unsqueeze_56: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_14, -1);  convert_element_type_14 = None
    unsqueeze_57: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    unsqueeze_58: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(mul_21, -1);  mul_21 = None
    unsqueeze_59: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
    sub_7: "f32[4, 58, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_57);  convolution_7 = unsqueeze_57 = None
    mul_22: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(sub_7, unsqueeze_59);  sub_7 = unsqueeze_59 = None
    unsqueeze_60: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(arg22_1, -1);  arg22_1 = None
    unsqueeze_61: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
    mul_23: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(mul_22, unsqueeze_61);  mul_22 = unsqueeze_61 = None
    unsqueeze_62: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(arg23_1, -1);  arg23_1 = None
    unsqueeze_63: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
    add_15: "f32[4, 58, 28, 28]" = torch.ops.aten.add.Tensor(mul_23, unsqueeze_63);  mul_23 = unsqueeze_63 = None
    convolution_8: "f32[4, 58, 28, 28]" = torch.ops.aten.convolution.default(add_15, arg24_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_15 = arg24_1 = None
    convert_element_type_16: "f32[58]" = torch.ops.prims.convert_element_type.default(arg194_1, torch.float32);  arg194_1 = None
    convert_element_type_17: "f32[58]" = torch.ops.prims.convert_element_type.default(arg195_1, torch.float32);  arg195_1 = None
    add_16: "f32[58]" = torch.ops.aten.add.Tensor(convert_element_type_17, 1e-05);  convert_element_type_17 = None
    sqrt_8: "f32[58]" = torch.ops.aten.sqrt.default(add_16);  add_16 = None
    reciprocal_8: "f32[58]" = torch.ops.aten.reciprocal.default(sqrt_8);  sqrt_8 = None
    mul_24: "f32[58]" = torch.ops.aten.mul.Tensor(reciprocal_8, 1);  reciprocal_8 = None
    unsqueeze_64: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_16, -1);  convert_element_type_16 = None
    unsqueeze_65: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
    unsqueeze_66: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(mul_24, -1);  mul_24 = None
    unsqueeze_67: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
    sub_8: "f32[4, 58, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_65);  convolution_8 = unsqueeze_65 = None
    mul_25: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(sub_8, unsqueeze_67);  sub_8 = unsqueeze_67 = None
    unsqueeze_68: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(arg25_1, -1);  arg25_1 = None
    unsqueeze_69: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    mul_26: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(mul_25, unsqueeze_69);  mul_25 = unsqueeze_69 = None
    unsqueeze_70: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(arg26_1, -1);  arg26_1 = None
    unsqueeze_71: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    add_17: "f32[4, 58, 28, 28]" = torch.ops.aten.add.Tensor(mul_26, unsqueeze_71);  mul_26 = unsqueeze_71 = None
    relu_5: "f32[4, 58, 28, 28]" = torch.ops.aten.relu.default(add_17);  add_17 = None
    cat_1: "f32[4, 116, 28, 28]" = torch.ops.aten.cat.default([getitem_2, relu_5], 1);  getitem_2 = relu_5 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    view_2: "f32[4, 2, 58, 28, 28]" = torch.ops.aten.view.default(cat_1, [4, 2, 58, 28, 28]);  cat_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    permute_1: "f32[4, 58, 2, 28, 28]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3, 4]);  view_2 = None
    clone_1: "f32[4, 58, 2, 28, 28]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    view_3: "f32[4, 116, 28, 28]" = torch.ops.aten.view.default(clone_1, [4, 116, 28, 28]);  clone_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:94, code: x1, x2 = x.chunk(2, dim=1)
    split_1 = torch.ops.aten.split.Tensor(view_3, 58, 1);  view_3 = None
    getitem_4: "f32[4, 58, 28, 28]" = split_1[0]
    getitem_5: "f32[4, 58, 28, 28]" = split_1[1];  split_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:95, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
    convolution_9: "f32[4, 58, 28, 28]" = torch.ops.aten.convolution.default(getitem_5, arg27_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_5 = arg27_1 = None
    convert_element_type_18: "f32[58]" = torch.ops.prims.convert_element_type.default(arg197_1, torch.float32);  arg197_1 = None
    convert_element_type_19: "f32[58]" = torch.ops.prims.convert_element_type.default(arg198_1, torch.float32);  arg198_1 = None
    add_18: "f32[58]" = torch.ops.aten.add.Tensor(convert_element_type_19, 1e-05);  convert_element_type_19 = None
    sqrt_9: "f32[58]" = torch.ops.aten.sqrt.default(add_18);  add_18 = None
    reciprocal_9: "f32[58]" = torch.ops.aten.reciprocal.default(sqrt_9);  sqrt_9 = None
    mul_27: "f32[58]" = torch.ops.aten.mul.Tensor(reciprocal_9, 1);  reciprocal_9 = None
    unsqueeze_72: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_18, -1);  convert_element_type_18 = None
    unsqueeze_73: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
    unsqueeze_74: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(mul_27, -1);  mul_27 = None
    unsqueeze_75: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
    sub_9: "f32[4, 58, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_73);  convolution_9 = unsqueeze_73 = None
    mul_28: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(sub_9, unsqueeze_75);  sub_9 = unsqueeze_75 = None
    unsqueeze_76: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(arg28_1, -1);  arg28_1 = None
    unsqueeze_77: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
    mul_29: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_77);  mul_28 = unsqueeze_77 = None
    unsqueeze_78: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(arg29_1, -1);  arg29_1 = None
    unsqueeze_79: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    add_19: "f32[4, 58, 28, 28]" = torch.ops.aten.add.Tensor(mul_29, unsqueeze_79);  mul_29 = unsqueeze_79 = None
    relu_6: "f32[4, 58, 28, 28]" = torch.ops.aten.relu.default(add_19);  add_19 = None
    convolution_10: "f32[4, 58, 28, 28]" = torch.ops.aten.convolution.default(relu_6, arg30_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 58);  relu_6 = arg30_1 = None
    convert_element_type_20: "f32[58]" = torch.ops.prims.convert_element_type.default(arg200_1, torch.float32);  arg200_1 = None
    convert_element_type_21: "f32[58]" = torch.ops.prims.convert_element_type.default(arg201_1, torch.float32);  arg201_1 = None
    add_20: "f32[58]" = torch.ops.aten.add.Tensor(convert_element_type_21, 1e-05);  convert_element_type_21 = None
    sqrt_10: "f32[58]" = torch.ops.aten.sqrt.default(add_20);  add_20 = None
    reciprocal_10: "f32[58]" = torch.ops.aten.reciprocal.default(sqrt_10);  sqrt_10 = None
    mul_30: "f32[58]" = torch.ops.aten.mul.Tensor(reciprocal_10, 1);  reciprocal_10 = None
    unsqueeze_80: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_20, -1);  convert_element_type_20 = None
    unsqueeze_81: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
    unsqueeze_82: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(mul_30, -1);  mul_30 = None
    unsqueeze_83: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    sub_10: "f32[4, 58, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_81);  convolution_10 = unsqueeze_81 = None
    mul_31: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(sub_10, unsqueeze_83);  sub_10 = unsqueeze_83 = None
    unsqueeze_84: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(arg31_1, -1);  arg31_1 = None
    unsqueeze_85: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    mul_32: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(mul_31, unsqueeze_85);  mul_31 = unsqueeze_85 = None
    unsqueeze_86: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(arg32_1, -1);  arg32_1 = None
    unsqueeze_87: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
    add_21: "f32[4, 58, 28, 28]" = torch.ops.aten.add.Tensor(mul_32, unsqueeze_87);  mul_32 = unsqueeze_87 = None
    convolution_11: "f32[4, 58, 28, 28]" = torch.ops.aten.convolution.default(add_21, arg33_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_21 = arg33_1 = None
    convert_element_type_22: "f32[58]" = torch.ops.prims.convert_element_type.default(arg203_1, torch.float32);  arg203_1 = None
    convert_element_type_23: "f32[58]" = torch.ops.prims.convert_element_type.default(arg204_1, torch.float32);  arg204_1 = None
    add_22: "f32[58]" = torch.ops.aten.add.Tensor(convert_element_type_23, 1e-05);  convert_element_type_23 = None
    sqrt_11: "f32[58]" = torch.ops.aten.sqrt.default(add_22);  add_22 = None
    reciprocal_11: "f32[58]" = torch.ops.aten.reciprocal.default(sqrt_11);  sqrt_11 = None
    mul_33: "f32[58]" = torch.ops.aten.mul.Tensor(reciprocal_11, 1);  reciprocal_11 = None
    unsqueeze_88: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_22, -1);  convert_element_type_22 = None
    unsqueeze_89: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
    unsqueeze_90: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(mul_33, -1);  mul_33 = None
    unsqueeze_91: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    sub_11: "f32[4, 58, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_89);  convolution_11 = unsqueeze_89 = None
    mul_34: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(sub_11, unsqueeze_91);  sub_11 = unsqueeze_91 = None
    unsqueeze_92: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(arg34_1, -1);  arg34_1 = None
    unsqueeze_93: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
    mul_35: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(mul_34, unsqueeze_93);  mul_34 = unsqueeze_93 = None
    unsqueeze_94: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(arg35_1, -1);  arg35_1 = None
    unsqueeze_95: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
    add_23: "f32[4, 58, 28, 28]" = torch.ops.aten.add.Tensor(mul_35, unsqueeze_95);  mul_35 = unsqueeze_95 = None
    relu_7: "f32[4, 58, 28, 28]" = torch.ops.aten.relu.default(add_23);  add_23 = None
    cat_2: "f32[4, 116, 28, 28]" = torch.ops.aten.cat.default([getitem_4, relu_7], 1);  getitem_4 = relu_7 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    view_4: "f32[4, 2, 58, 28, 28]" = torch.ops.aten.view.default(cat_2, [4, 2, 58, 28, 28]);  cat_2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    permute_2: "f32[4, 58, 2, 28, 28]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3, 4]);  view_4 = None
    clone_2: "f32[4, 58, 2, 28, 28]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    view_5: "f32[4, 116, 28, 28]" = torch.ops.aten.view.default(clone_2, [4, 116, 28, 28]);  clone_2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:94, code: x1, x2 = x.chunk(2, dim=1)
    split_2 = torch.ops.aten.split.Tensor(view_5, 58, 1);  view_5 = None
    getitem_6: "f32[4, 58, 28, 28]" = split_2[0]
    getitem_7: "f32[4, 58, 28, 28]" = split_2[1];  split_2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:95, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
    convolution_12: "f32[4, 58, 28, 28]" = torch.ops.aten.convolution.default(getitem_7, arg36_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_7 = arg36_1 = None
    convert_element_type_24: "f32[58]" = torch.ops.prims.convert_element_type.default(arg206_1, torch.float32);  arg206_1 = None
    convert_element_type_25: "f32[58]" = torch.ops.prims.convert_element_type.default(arg207_1, torch.float32);  arg207_1 = None
    add_24: "f32[58]" = torch.ops.aten.add.Tensor(convert_element_type_25, 1e-05);  convert_element_type_25 = None
    sqrt_12: "f32[58]" = torch.ops.aten.sqrt.default(add_24);  add_24 = None
    reciprocal_12: "f32[58]" = torch.ops.aten.reciprocal.default(sqrt_12);  sqrt_12 = None
    mul_36: "f32[58]" = torch.ops.aten.mul.Tensor(reciprocal_12, 1);  reciprocal_12 = None
    unsqueeze_96: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_24, -1);  convert_element_type_24 = None
    unsqueeze_97: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
    unsqueeze_98: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(mul_36, -1);  mul_36 = None
    unsqueeze_99: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
    sub_12: "f32[4, 58, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_97);  convolution_12 = unsqueeze_97 = None
    mul_37: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(sub_12, unsqueeze_99);  sub_12 = unsqueeze_99 = None
    unsqueeze_100: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(arg37_1, -1);  arg37_1 = None
    unsqueeze_101: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
    mul_38: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(mul_37, unsqueeze_101);  mul_37 = unsqueeze_101 = None
    unsqueeze_102: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(arg38_1, -1);  arg38_1 = None
    unsqueeze_103: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    add_25: "f32[4, 58, 28, 28]" = torch.ops.aten.add.Tensor(mul_38, unsqueeze_103);  mul_38 = unsqueeze_103 = None
    relu_8: "f32[4, 58, 28, 28]" = torch.ops.aten.relu.default(add_25);  add_25 = None
    convolution_13: "f32[4, 58, 28, 28]" = torch.ops.aten.convolution.default(relu_8, arg39_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 58);  relu_8 = arg39_1 = None
    convert_element_type_26: "f32[58]" = torch.ops.prims.convert_element_type.default(arg209_1, torch.float32);  arg209_1 = None
    convert_element_type_27: "f32[58]" = torch.ops.prims.convert_element_type.default(arg210_1, torch.float32);  arg210_1 = None
    add_26: "f32[58]" = torch.ops.aten.add.Tensor(convert_element_type_27, 1e-05);  convert_element_type_27 = None
    sqrt_13: "f32[58]" = torch.ops.aten.sqrt.default(add_26);  add_26 = None
    reciprocal_13: "f32[58]" = torch.ops.aten.reciprocal.default(sqrt_13);  sqrt_13 = None
    mul_39: "f32[58]" = torch.ops.aten.mul.Tensor(reciprocal_13, 1);  reciprocal_13 = None
    unsqueeze_104: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_26, -1);  convert_element_type_26 = None
    unsqueeze_105: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
    unsqueeze_106: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(mul_39, -1);  mul_39 = None
    unsqueeze_107: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
    sub_13: "f32[4, 58, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_105);  convolution_13 = unsqueeze_105 = None
    mul_40: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(sub_13, unsqueeze_107);  sub_13 = unsqueeze_107 = None
    unsqueeze_108: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(arg40_1, -1);  arg40_1 = None
    unsqueeze_109: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
    mul_41: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(mul_40, unsqueeze_109);  mul_40 = unsqueeze_109 = None
    unsqueeze_110: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(arg41_1, -1);  arg41_1 = None
    unsqueeze_111: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
    add_27: "f32[4, 58, 28, 28]" = torch.ops.aten.add.Tensor(mul_41, unsqueeze_111);  mul_41 = unsqueeze_111 = None
    convolution_14: "f32[4, 58, 28, 28]" = torch.ops.aten.convolution.default(add_27, arg42_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_27 = arg42_1 = None
    convert_element_type_28: "f32[58]" = torch.ops.prims.convert_element_type.default(arg212_1, torch.float32);  arg212_1 = None
    convert_element_type_29: "f32[58]" = torch.ops.prims.convert_element_type.default(arg213_1, torch.float32);  arg213_1 = None
    add_28: "f32[58]" = torch.ops.aten.add.Tensor(convert_element_type_29, 1e-05);  convert_element_type_29 = None
    sqrt_14: "f32[58]" = torch.ops.aten.sqrt.default(add_28);  add_28 = None
    reciprocal_14: "f32[58]" = torch.ops.aten.reciprocal.default(sqrt_14);  sqrt_14 = None
    mul_42: "f32[58]" = torch.ops.aten.mul.Tensor(reciprocal_14, 1);  reciprocal_14 = None
    unsqueeze_112: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_28, -1);  convert_element_type_28 = None
    unsqueeze_113: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
    unsqueeze_114: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(mul_42, -1);  mul_42 = None
    unsqueeze_115: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
    sub_14: "f32[4, 58, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_113);  convolution_14 = unsqueeze_113 = None
    mul_43: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(sub_14, unsqueeze_115);  sub_14 = unsqueeze_115 = None
    unsqueeze_116: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(arg43_1, -1);  arg43_1 = None
    unsqueeze_117: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
    mul_44: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(mul_43, unsqueeze_117);  mul_43 = unsqueeze_117 = None
    unsqueeze_118: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(arg44_1, -1);  arg44_1 = None
    unsqueeze_119: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
    add_29: "f32[4, 58, 28, 28]" = torch.ops.aten.add.Tensor(mul_44, unsqueeze_119);  mul_44 = unsqueeze_119 = None
    relu_9: "f32[4, 58, 28, 28]" = torch.ops.aten.relu.default(add_29);  add_29 = None
    cat_3: "f32[4, 116, 28, 28]" = torch.ops.aten.cat.default([getitem_6, relu_9], 1);  getitem_6 = relu_9 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    view_6: "f32[4, 2, 58, 28, 28]" = torch.ops.aten.view.default(cat_3, [4, 2, 58, 28, 28]);  cat_3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    permute_3: "f32[4, 58, 2, 28, 28]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3, 4]);  view_6 = None
    clone_3: "f32[4, 58, 2, 28, 28]" = torch.ops.aten.clone.default(permute_3, memory_format = torch.contiguous_format);  permute_3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    view_7: "f32[4, 116, 28, 28]" = torch.ops.aten.view.default(clone_3, [4, 116, 28, 28]);  clone_3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:97, code: out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
    convolution_15: "f32[4, 116, 14, 14]" = torch.ops.aten.convolution.default(view_7, arg45_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 116);  arg45_1 = None
    convert_element_type_30: "f32[116]" = torch.ops.prims.convert_element_type.default(arg215_1, torch.float32);  arg215_1 = None
    convert_element_type_31: "f32[116]" = torch.ops.prims.convert_element_type.default(arg216_1, torch.float32);  arg216_1 = None
    add_30: "f32[116]" = torch.ops.aten.add.Tensor(convert_element_type_31, 1e-05);  convert_element_type_31 = None
    sqrt_15: "f32[116]" = torch.ops.aten.sqrt.default(add_30);  add_30 = None
    reciprocal_15: "f32[116]" = torch.ops.aten.reciprocal.default(sqrt_15);  sqrt_15 = None
    mul_45: "f32[116]" = torch.ops.aten.mul.Tensor(reciprocal_15, 1);  reciprocal_15 = None
    unsqueeze_120: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_30, -1);  convert_element_type_30 = None
    unsqueeze_121: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
    unsqueeze_122: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(mul_45, -1);  mul_45 = None
    unsqueeze_123: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
    sub_15: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_121);  convolution_15 = unsqueeze_121 = None
    mul_46: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_15, unsqueeze_123);  sub_15 = unsqueeze_123 = None
    unsqueeze_124: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(arg46_1, -1);  arg46_1 = None
    unsqueeze_125: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
    mul_47: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(mul_46, unsqueeze_125);  mul_46 = unsqueeze_125 = None
    unsqueeze_126: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(arg47_1, -1);  arg47_1 = None
    unsqueeze_127: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
    add_31: "f32[4, 116, 14, 14]" = torch.ops.aten.add.Tensor(mul_47, unsqueeze_127);  mul_47 = unsqueeze_127 = None
    convolution_16: "f32[4, 116, 14, 14]" = torch.ops.aten.convolution.default(add_31, arg48_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_31 = arg48_1 = None
    convert_element_type_32: "f32[116]" = torch.ops.prims.convert_element_type.default(arg218_1, torch.float32);  arg218_1 = None
    convert_element_type_33: "f32[116]" = torch.ops.prims.convert_element_type.default(arg219_1, torch.float32);  arg219_1 = None
    add_32: "f32[116]" = torch.ops.aten.add.Tensor(convert_element_type_33, 1e-05);  convert_element_type_33 = None
    sqrt_16: "f32[116]" = torch.ops.aten.sqrt.default(add_32);  add_32 = None
    reciprocal_16: "f32[116]" = torch.ops.aten.reciprocal.default(sqrt_16);  sqrt_16 = None
    mul_48: "f32[116]" = torch.ops.aten.mul.Tensor(reciprocal_16, 1);  reciprocal_16 = None
    unsqueeze_128: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_32, -1);  convert_element_type_32 = None
    unsqueeze_129: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
    unsqueeze_130: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(mul_48, -1);  mul_48 = None
    unsqueeze_131: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
    sub_16: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_129);  convolution_16 = unsqueeze_129 = None
    mul_49: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_16, unsqueeze_131);  sub_16 = unsqueeze_131 = None
    unsqueeze_132: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(arg49_1, -1);  arg49_1 = None
    unsqueeze_133: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
    mul_50: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_133);  mul_49 = unsqueeze_133 = None
    unsqueeze_134: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(arg50_1, -1);  arg50_1 = None
    unsqueeze_135: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
    add_33: "f32[4, 116, 14, 14]" = torch.ops.aten.add.Tensor(mul_50, unsqueeze_135);  mul_50 = unsqueeze_135 = None
    relu_10: "f32[4, 116, 14, 14]" = torch.ops.aten.relu.default(add_33);  add_33 = None
    convolution_17: "f32[4, 116, 28, 28]" = torch.ops.aten.convolution.default(view_7, arg51_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  view_7 = arg51_1 = None
    convert_element_type_34: "f32[116]" = torch.ops.prims.convert_element_type.default(arg221_1, torch.float32);  arg221_1 = None
    convert_element_type_35: "f32[116]" = torch.ops.prims.convert_element_type.default(arg222_1, torch.float32);  arg222_1 = None
    add_34: "f32[116]" = torch.ops.aten.add.Tensor(convert_element_type_35, 1e-05);  convert_element_type_35 = None
    sqrt_17: "f32[116]" = torch.ops.aten.sqrt.default(add_34);  add_34 = None
    reciprocal_17: "f32[116]" = torch.ops.aten.reciprocal.default(sqrt_17);  sqrt_17 = None
    mul_51: "f32[116]" = torch.ops.aten.mul.Tensor(reciprocal_17, 1);  reciprocal_17 = None
    unsqueeze_136: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_34, -1);  convert_element_type_34 = None
    unsqueeze_137: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
    unsqueeze_138: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(mul_51, -1);  mul_51 = None
    unsqueeze_139: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
    sub_17: "f32[4, 116, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_137);  convolution_17 = unsqueeze_137 = None
    mul_52: "f32[4, 116, 28, 28]" = torch.ops.aten.mul.Tensor(sub_17, unsqueeze_139);  sub_17 = unsqueeze_139 = None
    unsqueeze_140: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(arg52_1, -1);  arg52_1 = None
    unsqueeze_141: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
    mul_53: "f32[4, 116, 28, 28]" = torch.ops.aten.mul.Tensor(mul_52, unsqueeze_141);  mul_52 = unsqueeze_141 = None
    unsqueeze_142: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(arg53_1, -1);  arg53_1 = None
    unsqueeze_143: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
    add_35: "f32[4, 116, 28, 28]" = torch.ops.aten.add.Tensor(mul_53, unsqueeze_143);  mul_53 = unsqueeze_143 = None
    relu_11: "f32[4, 116, 28, 28]" = torch.ops.aten.relu.default(add_35);  add_35 = None
    convolution_18: "f32[4, 116, 14, 14]" = torch.ops.aten.convolution.default(relu_11, arg54_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 116);  relu_11 = arg54_1 = None
    convert_element_type_36: "f32[116]" = torch.ops.prims.convert_element_type.default(arg224_1, torch.float32);  arg224_1 = None
    convert_element_type_37: "f32[116]" = torch.ops.prims.convert_element_type.default(arg225_1, torch.float32);  arg225_1 = None
    add_36: "f32[116]" = torch.ops.aten.add.Tensor(convert_element_type_37, 1e-05);  convert_element_type_37 = None
    sqrt_18: "f32[116]" = torch.ops.aten.sqrt.default(add_36);  add_36 = None
    reciprocal_18: "f32[116]" = torch.ops.aten.reciprocal.default(sqrt_18);  sqrt_18 = None
    mul_54: "f32[116]" = torch.ops.aten.mul.Tensor(reciprocal_18, 1);  reciprocal_18 = None
    unsqueeze_144: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_36, -1);  convert_element_type_36 = None
    unsqueeze_145: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
    unsqueeze_146: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(mul_54, -1);  mul_54 = None
    unsqueeze_147: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
    sub_18: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_145);  convolution_18 = unsqueeze_145 = None
    mul_55: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_18, unsqueeze_147);  sub_18 = unsqueeze_147 = None
    unsqueeze_148: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(arg55_1, -1);  arg55_1 = None
    unsqueeze_149: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
    mul_56: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(mul_55, unsqueeze_149);  mul_55 = unsqueeze_149 = None
    unsqueeze_150: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(arg56_1, -1);  arg56_1 = None
    unsqueeze_151: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
    add_37: "f32[4, 116, 14, 14]" = torch.ops.aten.add.Tensor(mul_56, unsqueeze_151);  mul_56 = unsqueeze_151 = None
    convolution_19: "f32[4, 116, 14, 14]" = torch.ops.aten.convolution.default(add_37, arg57_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_37 = arg57_1 = None
    convert_element_type_38: "f32[116]" = torch.ops.prims.convert_element_type.default(arg227_1, torch.float32);  arg227_1 = None
    convert_element_type_39: "f32[116]" = torch.ops.prims.convert_element_type.default(arg228_1, torch.float32);  arg228_1 = None
    add_38: "f32[116]" = torch.ops.aten.add.Tensor(convert_element_type_39, 1e-05);  convert_element_type_39 = None
    sqrt_19: "f32[116]" = torch.ops.aten.sqrt.default(add_38);  add_38 = None
    reciprocal_19: "f32[116]" = torch.ops.aten.reciprocal.default(sqrt_19);  sqrt_19 = None
    mul_57: "f32[116]" = torch.ops.aten.mul.Tensor(reciprocal_19, 1);  reciprocal_19 = None
    unsqueeze_152: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_38, -1);  convert_element_type_38 = None
    unsqueeze_153: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
    unsqueeze_154: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(mul_57, -1);  mul_57 = None
    unsqueeze_155: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
    sub_19: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_153);  convolution_19 = unsqueeze_153 = None
    mul_58: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_19, unsqueeze_155);  sub_19 = unsqueeze_155 = None
    unsqueeze_156: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(arg58_1, -1);  arg58_1 = None
    unsqueeze_157: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
    mul_59: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(mul_58, unsqueeze_157);  mul_58 = unsqueeze_157 = None
    unsqueeze_158: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(arg59_1, -1);  arg59_1 = None
    unsqueeze_159: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
    add_39: "f32[4, 116, 14, 14]" = torch.ops.aten.add.Tensor(mul_59, unsqueeze_159);  mul_59 = unsqueeze_159 = None
    relu_12: "f32[4, 116, 14, 14]" = torch.ops.aten.relu.default(add_39);  add_39 = None
    cat_4: "f32[4, 232, 14, 14]" = torch.ops.aten.cat.default([relu_10, relu_12], 1);  relu_10 = relu_12 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    view_8: "f32[4, 2, 116, 14, 14]" = torch.ops.aten.view.default(cat_4, [4, 2, 116, 14, 14]);  cat_4 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    permute_4: "f32[4, 116, 2, 14, 14]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3, 4]);  view_8 = None
    clone_4: "f32[4, 116, 2, 14, 14]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    view_9: "f32[4, 232, 14, 14]" = torch.ops.aten.view.default(clone_4, [4, 232, 14, 14]);  clone_4 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:94, code: x1, x2 = x.chunk(2, dim=1)
    split_3 = torch.ops.aten.split.Tensor(view_9, 116, 1);  view_9 = None
    getitem_8: "f32[4, 116, 14, 14]" = split_3[0]
    getitem_9: "f32[4, 116, 14, 14]" = split_3[1];  split_3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:95, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
    convolution_20: "f32[4, 116, 14, 14]" = torch.ops.aten.convolution.default(getitem_9, arg60_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_9 = arg60_1 = None
    convert_element_type_40: "f32[116]" = torch.ops.prims.convert_element_type.default(arg230_1, torch.float32);  arg230_1 = None
    convert_element_type_41: "f32[116]" = torch.ops.prims.convert_element_type.default(arg231_1, torch.float32);  arg231_1 = None
    add_40: "f32[116]" = torch.ops.aten.add.Tensor(convert_element_type_41, 1e-05);  convert_element_type_41 = None
    sqrt_20: "f32[116]" = torch.ops.aten.sqrt.default(add_40);  add_40 = None
    reciprocal_20: "f32[116]" = torch.ops.aten.reciprocal.default(sqrt_20);  sqrt_20 = None
    mul_60: "f32[116]" = torch.ops.aten.mul.Tensor(reciprocal_20, 1);  reciprocal_20 = None
    unsqueeze_160: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_40, -1);  convert_element_type_40 = None
    unsqueeze_161: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
    unsqueeze_162: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(mul_60, -1);  mul_60 = None
    unsqueeze_163: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
    sub_20: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_161);  convolution_20 = unsqueeze_161 = None
    mul_61: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_20, unsqueeze_163);  sub_20 = unsqueeze_163 = None
    unsqueeze_164: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(arg61_1, -1);  arg61_1 = None
    unsqueeze_165: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
    mul_62: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(mul_61, unsqueeze_165);  mul_61 = unsqueeze_165 = None
    unsqueeze_166: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(arg62_1, -1);  arg62_1 = None
    unsqueeze_167: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
    add_41: "f32[4, 116, 14, 14]" = torch.ops.aten.add.Tensor(mul_62, unsqueeze_167);  mul_62 = unsqueeze_167 = None
    relu_13: "f32[4, 116, 14, 14]" = torch.ops.aten.relu.default(add_41);  add_41 = None
    convolution_21: "f32[4, 116, 14, 14]" = torch.ops.aten.convolution.default(relu_13, arg63_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 116);  relu_13 = arg63_1 = None
    convert_element_type_42: "f32[116]" = torch.ops.prims.convert_element_type.default(arg233_1, torch.float32);  arg233_1 = None
    convert_element_type_43: "f32[116]" = torch.ops.prims.convert_element_type.default(arg234_1, torch.float32);  arg234_1 = None
    add_42: "f32[116]" = torch.ops.aten.add.Tensor(convert_element_type_43, 1e-05);  convert_element_type_43 = None
    sqrt_21: "f32[116]" = torch.ops.aten.sqrt.default(add_42);  add_42 = None
    reciprocal_21: "f32[116]" = torch.ops.aten.reciprocal.default(sqrt_21);  sqrt_21 = None
    mul_63: "f32[116]" = torch.ops.aten.mul.Tensor(reciprocal_21, 1);  reciprocal_21 = None
    unsqueeze_168: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_42, -1);  convert_element_type_42 = None
    unsqueeze_169: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
    unsqueeze_170: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(mul_63, -1);  mul_63 = None
    unsqueeze_171: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
    sub_21: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_169);  convolution_21 = unsqueeze_169 = None
    mul_64: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_21, unsqueeze_171);  sub_21 = unsqueeze_171 = None
    unsqueeze_172: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(arg64_1, -1);  arg64_1 = None
    unsqueeze_173: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
    mul_65: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(mul_64, unsqueeze_173);  mul_64 = unsqueeze_173 = None
    unsqueeze_174: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(arg65_1, -1);  arg65_1 = None
    unsqueeze_175: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
    add_43: "f32[4, 116, 14, 14]" = torch.ops.aten.add.Tensor(mul_65, unsqueeze_175);  mul_65 = unsqueeze_175 = None
    convolution_22: "f32[4, 116, 14, 14]" = torch.ops.aten.convolution.default(add_43, arg66_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_43 = arg66_1 = None
    convert_element_type_44: "f32[116]" = torch.ops.prims.convert_element_type.default(arg236_1, torch.float32);  arg236_1 = None
    convert_element_type_45: "f32[116]" = torch.ops.prims.convert_element_type.default(arg237_1, torch.float32);  arg237_1 = None
    add_44: "f32[116]" = torch.ops.aten.add.Tensor(convert_element_type_45, 1e-05);  convert_element_type_45 = None
    sqrt_22: "f32[116]" = torch.ops.aten.sqrt.default(add_44);  add_44 = None
    reciprocal_22: "f32[116]" = torch.ops.aten.reciprocal.default(sqrt_22);  sqrt_22 = None
    mul_66: "f32[116]" = torch.ops.aten.mul.Tensor(reciprocal_22, 1);  reciprocal_22 = None
    unsqueeze_176: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_44, -1);  convert_element_type_44 = None
    unsqueeze_177: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
    unsqueeze_178: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(mul_66, -1);  mul_66 = None
    unsqueeze_179: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
    sub_22: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_177);  convolution_22 = unsqueeze_177 = None
    mul_67: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_22, unsqueeze_179);  sub_22 = unsqueeze_179 = None
    unsqueeze_180: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(arg67_1, -1);  arg67_1 = None
    unsqueeze_181: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
    mul_68: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(mul_67, unsqueeze_181);  mul_67 = unsqueeze_181 = None
    unsqueeze_182: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(arg68_1, -1);  arg68_1 = None
    unsqueeze_183: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
    add_45: "f32[4, 116, 14, 14]" = torch.ops.aten.add.Tensor(mul_68, unsqueeze_183);  mul_68 = unsqueeze_183 = None
    relu_14: "f32[4, 116, 14, 14]" = torch.ops.aten.relu.default(add_45);  add_45 = None
    cat_5: "f32[4, 232, 14, 14]" = torch.ops.aten.cat.default([getitem_8, relu_14], 1);  getitem_8 = relu_14 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    view_10: "f32[4, 2, 116, 14, 14]" = torch.ops.aten.view.default(cat_5, [4, 2, 116, 14, 14]);  cat_5 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    permute_5: "f32[4, 116, 2, 14, 14]" = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3, 4]);  view_10 = None
    clone_5: "f32[4, 116, 2, 14, 14]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    view_11: "f32[4, 232, 14, 14]" = torch.ops.aten.view.default(clone_5, [4, 232, 14, 14]);  clone_5 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:94, code: x1, x2 = x.chunk(2, dim=1)
    split_4 = torch.ops.aten.split.Tensor(view_11, 116, 1);  view_11 = None
    getitem_10: "f32[4, 116, 14, 14]" = split_4[0]
    getitem_11: "f32[4, 116, 14, 14]" = split_4[1];  split_4 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:95, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
    convolution_23: "f32[4, 116, 14, 14]" = torch.ops.aten.convolution.default(getitem_11, arg69_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_11 = arg69_1 = None
    convert_element_type_46: "f32[116]" = torch.ops.prims.convert_element_type.default(arg239_1, torch.float32);  arg239_1 = None
    convert_element_type_47: "f32[116]" = torch.ops.prims.convert_element_type.default(arg240_1, torch.float32);  arg240_1 = None
    add_46: "f32[116]" = torch.ops.aten.add.Tensor(convert_element_type_47, 1e-05);  convert_element_type_47 = None
    sqrt_23: "f32[116]" = torch.ops.aten.sqrt.default(add_46);  add_46 = None
    reciprocal_23: "f32[116]" = torch.ops.aten.reciprocal.default(sqrt_23);  sqrt_23 = None
    mul_69: "f32[116]" = torch.ops.aten.mul.Tensor(reciprocal_23, 1);  reciprocal_23 = None
    unsqueeze_184: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_46, -1);  convert_element_type_46 = None
    unsqueeze_185: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
    unsqueeze_186: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(mul_69, -1);  mul_69 = None
    unsqueeze_187: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
    sub_23: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_185);  convolution_23 = unsqueeze_185 = None
    mul_70: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_23, unsqueeze_187);  sub_23 = unsqueeze_187 = None
    unsqueeze_188: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(arg70_1, -1);  arg70_1 = None
    unsqueeze_189: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
    mul_71: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(mul_70, unsqueeze_189);  mul_70 = unsqueeze_189 = None
    unsqueeze_190: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(arg71_1, -1);  arg71_1 = None
    unsqueeze_191: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
    add_47: "f32[4, 116, 14, 14]" = torch.ops.aten.add.Tensor(mul_71, unsqueeze_191);  mul_71 = unsqueeze_191 = None
    relu_15: "f32[4, 116, 14, 14]" = torch.ops.aten.relu.default(add_47);  add_47 = None
    convolution_24: "f32[4, 116, 14, 14]" = torch.ops.aten.convolution.default(relu_15, arg72_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 116);  relu_15 = arg72_1 = None
    convert_element_type_48: "f32[116]" = torch.ops.prims.convert_element_type.default(arg242_1, torch.float32);  arg242_1 = None
    convert_element_type_49: "f32[116]" = torch.ops.prims.convert_element_type.default(arg243_1, torch.float32);  arg243_1 = None
    add_48: "f32[116]" = torch.ops.aten.add.Tensor(convert_element_type_49, 1e-05);  convert_element_type_49 = None
    sqrt_24: "f32[116]" = torch.ops.aten.sqrt.default(add_48);  add_48 = None
    reciprocal_24: "f32[116]" = torch.ops.aten.reciprocal.default(sqrt_24);  sqrt_24 = None
    mul_72: "f32[116]" = torch.ops.aten.mul.Tensor(reciprocal_24, 1);  reciprocal_24 = None
    unsqueeze_192: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_48, -1);  convert_element_type_48 = None
    unsqueeze_193: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
    unsqueeze_194: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(mul_72, -1);  mul_72 = None
    unsqueeze_195: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
    sub_24: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_193);  convolution_24 = unsqueeze_193 = None
    mul_73: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_24, unsqueeze_195);  sub_24 = unsqueeze_195 = None
    unsqueeze_196: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(arg73_1, -1);  arg73_1 = None
    unsqueeze_197: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
    mul_74: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(mul_73, unsqueeze_197);  mul_73 = unsqueeze_197 = None
    unsqueeze_198: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(arg74_1, -1);  arg74_1 = None
    unsqueeze_199: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
    add_49: "f32[4, 116, 14, 14]" = torch.ops.aten.add.Tensor(mul_74, unsqueeze_199);  mul_74 = unsqueeze_199 = None
    convolution_25: "f32[4, 116, 14, 14]" = torch.ops.aten.convolution.default(add_49, arg75_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_49 = arg75_1 = None
    convert_element_type_50: "f32[116]" = torch.ops.prims.convert_element_type.default(arg245_1, torch.float32);  arg245_1 = None
    convert_element_type_51: "f32[116]" = torch.ops.prims.convert_element_type.default(arg246_1, torch.float32);  arg246_1 = None
    add_50: "f32[116]" = torch.ops.aten.add.Tensor(convert_element_type_51, 1e-05);  convert_element_type_51 = None
    sqrt_25: "f32[116]" = torch.ops.aten.sqrt.default(add_50);  add_50 = None
    reciprocal_25: "f32[116]" = torch.ops.aten.reciprocal.default(sqrt_25);  sqrt_25 = None
    mul_75: "f32[116]" = torch.ops.aten.mul.Tensor(reciprocal_25, 1);  reciprocal_25 = None
    unsqueeze_200: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_50, -1);  convert_element_type_50 = None
    unsqueeze_201: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
    unsqueeze_202: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(mul_75, -1);  mul_75 = None
    unsqueeze_203: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
    sub_25: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_201);  convolution_25 = unsqueeze_201 = None
    mul_76: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_25, unsqueeze_203);  sub_25 = unsqueeze_203 = None
    unsqueeze_204: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(arg76_1, -1);  arg76_1 = None
    unsqueeze_205: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
    mul_77: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(mul_76, unsqueeze_205);  mul_76 = unsqueeze_205 = None
    unsqueeze_206: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(arg77_1, -1);  arg77_1 = None
    unsqueeze_207: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
    add_51: "f32[4, 116, 14, 14]" = torch.ops.aten.add.Tensor(mul_77, unsqueeze_207);  mul_77 = unsqueeze_207 = None
    relu_16: "f32[4, 116, 14, 14]" = torch.ops.aten.relu.default(add_51);  add_51 = None
    cat_6: "f32[4, 232, 14, 14]" = torch.ops.aten.cat.default([getitem_10, relu_16], 1);  getitem_10 = relu_16 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    view_12: "f32[4, 2, 116, 14, 14]" = torch.ops.aten.view.default(cat_6, [4, 2, 116, 14, 14]);  cat_6 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    permute_6: "f32[4, 116, 2, 14, 14]" = torch.ops.aten.permute.default(view_12, [0, 2, 1, 3, 4]);  view_12 = None
    clone_6: "f32[4, 116, 2, 14, 14]" = torch.ops.aten.clone.default(permute_6, memory_format = torch.contiguous_format);  permute_6 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    view_13: "f32[4, 232, 14, 14]" = torch.ops.aten.view.default(clone_6, [4, 232, 14, 14]);  clone_6 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:94, code: x1, x2 = x.chunk(2, dim=1)
    split_5 = torch.ops.aten.split.Tensor(view_13, 116, 1);  view_13 = None
    getitem_12: "f32[4, 116, 14, 14]" = split_5[0]
    getitem_13: "f32[4, 116, 14, 14]" = split_5[1];  split_5 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:95, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
    convolution_26: "f32[4, 116, 14, 14]" = torch.ops.aten.convolution.default(getitem_13, arg78_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_13 = arg78_1 = None
    convert_element_type_52: "f32[116]" = torch.ops.prims.convert_element_type.default(arg248_1, torch.float32);  arg248_1 = None
    convert_element_type_53: "f32[116]" = torch.ops.prims.convert_element_type.default(arg249_1, torch.float32);  arg249_1 = None
    add_52: "f32[116]" = torch.ops.aten.add.Tensor(convert_element_type_53, 1e-05);  convert_element_type_53 = None
    sqrt_26: "f32[116]" = torch.ops.aten.sqrt.default(add_52);  add_52 = None
    reciprocal_26: "f32[116]" = torch.ops.aten.reciprocal.default(sqrt_26);  sqrt_26 = None
    mul_78: "f32[116]" = torch.ops.aten.mul.Tensor(reciprocal_26, 1);  reciprocal_26 = None
    unsqueeze_208: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_52, -1);  convert_element_type_52 = None
    unsqueeze_209: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
    unsqueeze_210: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(mul_78, -1);  mul_78 = None
    unsqueeze_211: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
    sub_26: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_209);  convolution_26 = unsqueeze_209 = None
    mul_79: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_26, unsqueeze_211);  sub_26 = unsqueeze_211 = None
    unsqueeze_212: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(arg79_1, -1);  arg79_1 = None
    unsqueeze_213: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
    mul_80: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(mul_79, unsqueeze_213);  mul_79 = unsqueeze_213 = None
    unsqueeze_214: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(arg80_1, -1);  arg80_1 = None
    unsqueeze_215: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
    add_53: "f32[4, 116, 14, 14]" = torch.ops.aten.add.Tensor(mul_80, unsqueeze_215);  mul_80 = unsqueeze_215 = None
    relu_17: "f32[4, 116, 14, 14]" = torch.ops.aten.relu.default(add_53);  add_53 = None
    convolution_27: "f32[4, 116, 14, 14]" = torch.ops.aten.convolution.default(relu_17, arg81_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 116);  relu_17 = arg81_1 = None
    convert_element_type_54: "f32[116]" = torch.ops.prims.convert_element_type.default(arg251_1, torch.float32);  arg251_1 = None
    convert_element_type_55: "f32[116]" = torch.ops.prims.convert_element_type.default(arg252_1, torch.float32);  arg252_1 = None
    add_54: "f32[116]" = torch.ops.aten.add.Tensor(convert_element_type_55, 1e-05);  convert_element_type_55 = None
    sqrt_27: "f32[116]" = torch.ops.aten.sqrt.default(add_54);  add_54 = None
    reciprocal_27: "f32[116]" = torch.ops.aten.reciprocal.default(sqrt_27);  sqrt_27 = None
    mul_81: "f32[116]" = torch.ops.aten.mul.Tensor(reciprocal_27, 1);  reciprocal_27 = None
    unsqueeze_216: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_54, -1);  convert_element_type_54 = None
    unsqueeze_217: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
    unsqueeze_218: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(mul_81, -1);  mul_81 = None
    unsqueeze_219: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, -1);  unsqueeze_218 = None
    sub_27: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_217);  convolution_27 = unsqueeze_217 = None
    mul_82: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_27, unsqueeze_219);  sub_27 = unsqueeze_219 = None
    unsqueeze_220: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(arg82_1, -1);  arg82_1 = None
    unsqueeze_221: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, -1);  unsqueeze_220 = None
    mul_83: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(mul_82, unsqueeze_221);  mul_82 = unsqueeze_221 = None
    unsqueeze_222: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(arg83_1, -1);  arg83_1 = None
    unsqueeze_223: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, -1);  unsqueeze_222 = None
    add_55: "f32[4, 116, 14, 14]" = torch.ops.aten.add.Tensor(mul_83, unsqueeze_223);  mul_83 = unsqueeze_223 = None
    convolution_28: "f32[4, 116, 14, 14]" = torch.ops.aten.convolution.default(add_55, arg84_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_55 = arg84_1 = None
    convert_element_type_56: "f32[116]" = torch.ops.prims.convert_element_type.default(arg254_1, torch.float32);  arg254_1 = None
    convert_element_type_57: "f32[116]" = torch.ops.prims.convert_element_type.default(arg255_1, torch.float32);  arg255_1 = None
    add_56: "f32[116]" = torch.ops.aten.add.Tensor(convert_element_type_57, 1e-05);  convert_element_type_57 = None
    sqrt_28: "f32[116]" = torch.ops.aten.sqrt.default(add_56);  add_56 = None
    reciprocal_28: "f32[116]" = torch.ops.aten.reciprocal.default(sqrt_28);  sqrt_28 = None
    mul_84: "f32[116]" = torch.ops.aten.mul.Tensor(reciprocal_28, 1);  reciprocal_28 = None
    unsqueeze_224: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_56, -1);  convert_element_type_56 = None
    unsqueeze_225: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, -1);  unsqueeze_224 = None
    unsqueeze_226: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(mul_84, -1);  mul_84 = None
    unsqueeze_227: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, -1);  unsqueeze_226 = None
    sub_28: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_225);  convolution_28 = unsqueeze_225 = None
    mul_85: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_28, unsqueeze_227);  sub_28 = unsqueeze_227 = None
    unsqueeze_228: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(arg85_1, -1);  arg85_1 = None
    unsqueeze_229: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, -1);  unsqueeze_228 = None
    mul_86: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(mul_85, unsqueeze_229);  mul_85 = unsqueeze_229 = None
    unsqueeze_230: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(arg86_1, -1);  arg86_1 = None
    unsqueeze_231: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, -1);  unsqueeze_230 = None
    add_57: "f32[4, 116, 14, 14]" = torch.ops.aten.add.Tensor(mul_86, unsqueeze_231);  mul_86 = unsqueeze_231 = None
    relu_18: "f32[4, 116, 14, 14]" = torch.ops.aten.relu.default(add_57);  add_57 = None
    cat_7: "f32[4, 232, 14, 14]" = torch.ops.aten.cat.default([getitem_12, relu_18], 1);  getitem_12 = relu_18 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    view_14: "f32[4, 2, 116, 14, 14]" = torch.ops.aten.view.default(cat_7, [4, 2, 116, 14, 14]);  cat_7 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    permute_7: "f32[4, 116, 2, 14, 14]" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3, 4]);  view_14 = None
    clone_7: "f32[4, 116, 2, 14, 14]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    view_15: "f32[4, 232, 14, 14]" = torch.ops.aten.view.default(clone_7, [4, 232, 14, 14]);  clone_7 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:94, code: x1, x2 = x.chunk(2, dim=1)
    split_6 = torch.ops.aten.split.Tensor(view_15, 116, 1);  view_15 = None
    getitem_14: "f32[4, 116, 14, 14]" = split_6[0]
    getitem_15: "f32[4, 116, 14, 14]" = split_6[1];  split_6 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:95, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
    convolution_29: "f32[4, 116, 14, 14]" = torch.ops.aten.convolution.default(getitem_15, arg87_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_15 = arg87_1 = None
    convert_element_type_58: "f32[116]" = torch.ops.prims.convert_element_type.default(arg257_1, torch.float32);  arg257_1 = None
    convert_element_type_59: "f32[116]" = torch.ops.prims.convert_element_type.default(arg258_1, torch.float32);  arg258_1 = None
    add_58: "f32[116]" = torch.ops.aten.add.Tensor(convert_element_type_59, 1e-05);  convert_element_type_59 = None
    sqrt_29: "f32[116]" = torch.ops.aten.sqrt.default(add_58);  add_58 = None
    reciprocal_29: "f32[116]" = torch.ops.aten.reciprocal.default(sqrt_29);  sqrt_29 = None
    mul_87: "f32[116]" = torch.ops.aten.mul.Tensor(reciprocal_29, 1);  reciprocal_29 = None
    unsqueeze_232: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_58, -1);  convert_element_type_58 = None
    unsqueeze_233: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, -1);  unsqueeze_232 = None
    unsqueeze_234: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(mul_87, -1);  mul_87 = None
    unsqueeze_235: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, -1);  unsqueeze_234 = None
    sub_29: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_233);  convolution_29 = unsqueeze_233 = None
    mul_88: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_29, unsqueeze_235);  sub_29 = unsqueeze_235 = None
    unsqueeze_236: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(arg88_1, -1);  arg88_1 = None
    unsqueeze_237: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, -1);  unsqueeze_236 = None
    mul_89: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(mul_88, unsqueeze_237);  mul_88 = unsqueeze_237 = None
    unsqueeze_238: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(arg89_1, -1);  arg89_1 = None
    unsqueeze_239: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, -1);  unsqueeze_238 = None
    add_59: "f32[4, 116, 14, 14]" = torch.ops.aten.add.Tensor(mul_89, unsqueeze_239);  mul_89 = unsqueeze_239 = None
    relu_19: "f32[4, 116, 14, 14]" = torch.ops.aten.relu.default(add_59);  add_59 = None
    convolution_30: "f32[4, 116, 14, 14]" = torch.ops.aten.convolution.default(relu_19, arg90_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 116);  relu_19 = arg90_1 = None
    convert_element_type_60: "f32[116]" = torch.ops.prims.convert_element_type.default(arg260_1, torch.float32);  arg260_1 = None
    convert_element_type_61: "f32[116]" = torch.ops.prims.convert_element_type.default(arg261_1, torch.float32);  arg261_1 = None
    add_60: "f32[116]" = torch.ops.aten.add.Tensor(convert_element_type_61, 1e-05);  convert_element_type_61 = None
    sqrt_30: "f32[116]" = torch.ops.aten.sqrt.default(add_60);  add_60 = None
    reciprocal_30: "f32[116]" = torch.ops.aten.reciprocal.default(sqrt_30);  sqrt_30 = None
    mul_90: "f32[116]" = torch.ops.aten.mul.Tensor(reciprocal_30, 1);  reciprocal_30 = None
    unsqueeze_240: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_60, -1);  convert_element_type_60 = None
    unsqueeze_241: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, -1);  unsqueeze_240 = None
    unsqueeze_242: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(mul_90, -1);  mul_90 = None
    unsqueeze_243: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, -1);  unsqueeze_242 = None
    sub_30: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_241);  convolution_30 = unsqueeze_241 = None
    mul_91: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_30, unsqueeze_243);  sub_30 = unsqueeze_243 = None
    unsqueeze_244: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(arg91_1, -1);  arg91_1 = None
    unsqueeze_245: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, -1);  unsqueeze_244 = None
    mul_92: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(mul_91, unsqueeze_245);  mul_91 = unsqueeze_245 = None
    unsqueeze_246: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(arg92_1, -1);  arg92_1 = None
    unsqueeze_247: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, -1);  unsqueeze_246 = None
    add_61: "f32[4, 116, 14, 14]" = torch.ops.aten.add.Tensor(mul_92, unsqueeze_247);  mul_92 = unsqueeze_247 = None
    convolution_31: "f32[4, 116, 14, 14]" = torch.ops.aten.convolution.default(add_61, arg93_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_61 = arg93_1 = None
    convert_element_type_62: "f32[116]" = torch.ops.prims.convert_element_type.default(arg263_1, torch.float32);  arg263_1 = None
    convert_element_type_63: "f32[116]" = torch.ops.prims.convert_element_type.default(arg264_1, torch.float32);  arg264_1 = None
    add_62: "f32[116]" = torch.ops.aten.add.Tensor(convert_element_type_63, 1e-05);  convert_element_type_63 = None
    sqrt_31: "f32[116]" = torch.ops.aten.sqrt.default(add_62);  add_62 = None
    reciprocal_31: "f32[116]" = torch.ops.aten.reciprocal.default(sqrt_31);  sqrt_31 = None
    mul_93: "f32[116]" = torch.ops.aten.mul.Tensor(reciprocal_31, 1);  reciprocal_31 = None
    unsqueeze_248: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_62, -1);  convert_element_type_62 = None
    unsqueeze_249: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, -1);  unsqueeze_248 = None
    unsqueeze_250: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(mul_93, -1);  mul_93 = None
    unsqueeze_251: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, -1);  unsqueeze_250 = None
    sub_31: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_249);  convolution_31 = unsqueeze_249 = None
    mul_94: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_31, unsqueeze_251);  sub_31 = unsqueeze_251 = None
    unsqueeze_252: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(arg94_1, -1);  arg94_1 = None
    unsqueeze_253: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, -1);  unsqueeze_252 = None
    mul_95: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(mul_94, unsqueeze_253);  mul_94 = unsqueeze_253 = None
    unsqueeze_254: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(arg95_1, -1);  arg95_1 = None
    unsqueeze_255: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, -1);  unsqueeze_254 = None
    add_63: "f32[4, 116, 14, 14]" = torch.ops.aten.add.Tensor(mul_95, unsqueeze_255);  mul_95 = unsqueeze_255 = None
    relu_20: "f32[4, 116, 14, 14]" = torch.ops.aten.relu.default(add_63);  add_63 = None
    cat_8: "f32[4, 232, 14, 14]" = torch.ops.aten.cat.default([getitem_14, relu_20], 1);  getitem_14 = relu_20 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    view_16: "f32[4, 2, 116, 14, 14]" = torch.ops.aten.view.default(cat_8, [4, 2, 116, 14, 14]);  cat_8 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    permute_8: "f32[4, 116, 2, 14, 14]" = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3, 4]);  view_16 = None
    clone_8: "f32[4, 116, 2, 14, 14]" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    view_17: "f32[4, 232, 14, 14]" = torch.ops.aten.view.default(clone_8, [4, 232, 14, 14]);  clone_8 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:94, code: x1, x2 = x.chunk(2, dim=1)
    split_7 = torch.ops.aten.split.Tensor(view_17, 116, 1);  view_17 = None
    getitem_16: "f32[4, 116, 14, 14]" = split_7[0]
    getitem_17: "f32[4, 116, 14, 14]" = split_7[1];  split_7 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:95, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
    convolution_32: "f32[4, 116, 14, 14]" = torch.ops.aten.convolution.default(getitem_17, arg96_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_17 = arg96_1 = None
    convert_element_type_64: "f32[116]" = torch.ops.prims.convert_element_type.default(arg266_1, torch.float32);  arg266_1 = None
    convert_element_type_65: "f32[116]" = torch.ops.prims.convert_element_type.default(arg267_1, torch.float32);  arg267_1 = None
    add_64: "f32[116]" = torch.ops.aten.add.Tensor(convert_element_type_65, 1e-05);  convert_element_type_65 = None
    sqrt_32: "f32[116]" = torch.ops.aten.sqrt.default(add_64);  add_64 = None
    reciprocal_32: "f32[116]" = torch.ops.aten.reciprocal.default(sqrt_32);  sqrt_32 = None
    mul_96: "f32[116]" = torch.ops.aten.mul.Tensor(reciprocal_32, 1);  reciprocal_32 = None
    unsqueeze_256: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_64, -1);  convert_element_type_64 = None
    unsqueeze_257: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, -1);  unsqueeze_256 = None
    unsqueeze_258: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(mul_96, -1);  mul_96 = None
    unsqueeze_259: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, -1);  unsqueeze_258 = None
    sub_32: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_257);  convolution_32 = unsqueeze_257 = None
    mul_97: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_32, unsqueeze_259);  sub_32 = unsqueeze_259 = None
    unsqueeze_260: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(arg97_1, -1);  arg97_1 = None
    unsqueeze_261: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, -1);  unsqueeze_260 = None
    mul_98: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(mul_97, unsqueeze_261);  mul_97 = unsqueeze_261 = None
    unsqueeze_262: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(arg98_1, -1);  arg98_1 = None
    unsqueeze_263: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, -1);  unsqueeze_262 = None
    add_65: "f32[4, 116, 14, 14]" = torch.ops.aten.add.Tensor(mul_98, unsqueeze_263);  mul_98 = unsqueeze_263 = None
    relu_21: "f32[4, 116, 14, 14]" = torch.ops.aten.relu.default(add_65);  add_65 = None
    convolution_33: "f32[4, 116, 14, 14]" = torch.ops.aten.convolution.default(relu_21, arg99_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 116);  relu_21 = arg99_1 = None
    convert_element_type_66: "f32[116]" = torch.ops.prims.convert_element_type.default(arg269_1, torch.float32);  arg269_1 = None
    convert_element_type_67: "f32[116]" = torch.ops.prims.convert_element_type.default(arg270_1, torch.float32);  arg270_1 = None
    add_66: "f32[116]" = torch.ops.aten.add.Tensor(convert_element_type_67, 1e-05);  convert_element_type_67 = None
    sqrt_33: "f32[116]" = torch.ops.aten.sqrt.default(add_66);  add_66 = None
    reciprocal_33: "f32[116]" = torch.ops.aten.reciprocal.default(sqrt_33);  sqrt_33 = None
    mul_99: "f32[116]" = torch.ops.aten.mul.Tensor(reciprocal_33, 1);  reciprocal_33 = None
    unsqueeze_264: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_66, -1);  convert_element_type_66 = None
    unsqueeze_265: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, -1);  unsqueeze_264 = None
    unsqueeze_266: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(mul_99, -1);  mul_99 = None
    unsqueeze_267: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, -1);  unsqueeze_266 = None
    sub_33: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_265);  convolution_33 = unsqueeze_265 = None
    mul_100: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_33, unsqueeze_267);  sub_33 = unsqueeze_267 = None
    unsqueeze_268: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(arg100_1, -1);  arg100_1 = None
    unsqueeze_269: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, -1);  unsqueeze_268 = None
    mul_101: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(mul_100, unsqueeze_269);  mul_100 = unsqueeze_269 = None
    unsqueeze_270: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(arg101_1, -1);  arg101_1 = None
    unsqueeze_271: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, -1);  unsqueeze_270 = None
    add_67: "f32[4, 116, 14, 14]" = torch.ops.aten.add.Tensor(mul_101, unsqueeze_271);  mul_101 = unsqueeze_271 = None
    convolution_34: "f32[4, 116, 14, 14]" = torch.ops.aten.convolution.default(add_67, arg102_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_67 = arg102_1 = None
    convert_element_type_68: "f32[116]" = torch.ops.prims.convert_element_type.default(arg272_1, torch.float32);  arg272_1 = None
    convert_element_type_69: "f32[116]" = torch.ops.prims.convert_element_type.default(arg273_1, torch.float32);  arg273_1 = None
    add_68: "f32[116]" = torch.ops.aten.add.Tensor(convert_element_type_69, 1e-05);  convert_element_type_69 = None
    sqrt_34: "f32[116]" = torch.ops.aten.sqrt.default(add_68);  add_68 = None
    reciprocal_34: "f32[116]" = torch.ops.aten.reciprocal.default(sqrt_34);  sqrt_34 = None
    mul_102: "f32[116]" = torch.ops.aten.mul.Tensor(reciprocal_34, 1);  reciprocal_34 = None
    unsqueeze_272: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_68, -1);  convert_element_type_68 = None
    unsqueeze_273: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, -1);  unsqueeze_272 = None
    unsqueeze_274: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(mul_102, -1);  mul_102 = None
    unsqueeze_275: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, -1);  unsqueeze_274 = None
    sub_34: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_273);  convolution_34 = unsqueeze_273 = None
    mul_103: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_34, unsqueeze_275);  sub_34 = unsqueeze_275 = None
    unsqueeze_276: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(arg103_1, -1);  arg103_1 = None
    unsqueeze_277: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, -1);  unsqueeze_276 = None
    mul_104: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(mul_103, unsqueeze_277);  mul_103 = unsqueeze_277 = None
    unsqueeze_278: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(arg104_1, -1);  arg104_1 = None
    unsqueeze_279: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, -1);  unsqueeze_278 = None
    add_69: "f32[4, 116, 14, 14]" = torch.ops.aten.add.Tensor(mul_104, unsqueeze_279);  mul_104 = unsqueeze_279 = None
    relu_22: "f32[4, 116, 14, 14]" = torch.ops.aten.relu.default(add_69);  add_69 = None
    cat_9: "f32[4, 232, 14, 14]" = torch.ops.aten.cat.default([getitem_16, relu_22], 1);  getitem_16 = relu_22 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    view_18: "f32[4, 2, 116, 14, 14]" = torch.ops.aten.view.default(cat_9, [4, 2, 116, 14, 14]);  cat_9 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    permute_9: "f32[4, 116, 2, 14, 14]" = torch.ops.aten.permute.default(view_18, [0, 2, 1, 3, 4]);  view_18 = None
    clone_9: "f32[4, 116, 2, 14, 14]" = torch.ops.aten.clone.default(permute_9, memory_format = torch.contiguous_format);  permute_9 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    view_19: "f32[4, 232, 14, 14]" = torch.ops.aten.view.default(clone_9, [4, 232, 14, 14]);  clone_9 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:94, code: x1, x2 = x.chunk(2, dim=1)
    split_8 = torch.ops.aten.split.Tensor(view_19, 116, 1);  view_19 = None
    getitem_18: "f32[4, 116, 14, 14]" = split_8[0]
    getitem_19: "f32[4, 116, 14, 14]" = split_8[1];  split_8 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:95, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
    convolution_35: "f32[4, 116, 14, 14]" = torch.ops.aten.convolution.default(getitem_19, arg105_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_19 = arg105_1 = None
    convert_element_type_70: "f32[116]" = torch.ops.prims.convert_element_type.default(arg275_1, torch.float32);  arg275_1 = None
    convert_element_type_71: "f32[116]" = torch.ops.prims.convert_element_type.default(arg276_1, torch.float32);  arg276_1 = None
    add_70: "f32[116]" = torch.ops.aten.add.Tensor(convert_element_type_71, 1e-05);  convert_element_type_71 = None
    sqrt_35: "f32[116]" = torch.ops.aten.sqrt.default(add_70);  add_70 = None
    reciprocal_35: "f32[116]" = torch.ops.aten.reciprocal.default(sqrt_35);  sqrt_35 = None
    mul_105: "f32[116]" = torch.ops.aten.mul.Tensor(reciprocal_35, 1);  reciprocal_35 = None
    unsqueeze_280: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_70, -1);  convert_element_type_70 = None
    unsqueeze_281: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, -1);  unsqueeze_280 = None
    unsqueeze_282: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(mul_105, -1);  mul_105 = None
    unsqueeze_283: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, -1);  unsqueeze_282 = None
    sub_35: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_281);  convolution_35 = unsqueeze_281 = None
    mul_106: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_35, unsqueeze_283);  sub_35 = unsqueeze_283 = None
    unsqueeze_284: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(arg106_1, -1);  arg106_1 = None
    unsqueeze_285: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, -1);  unsqueeze_284 = None
    mul_107: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(mul_106, unsqueeze_285);  mul_106 = unsqueeze_285 = None
    unsqueeze_286: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(arg107_1, -1);  arg107_1 = None
    unsqueeze_287: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, -1);  unsqueeze_286 = None
    add_71: "f32[4, 116, 14, 14]" = torch.ops.aten.add.Tensor(mul_107, unsqueeze_287);  mul_107 = unsqueeze_287 = None
    relu_23: "f32[4, 116, 14, 14]" = torch.ops.aten.relu.default(add_71);  add_71 = None
    convolution_36: "f32[4, 116, 14, 14]" = torch.ops.aten.convolution.default(relu_23, arg108_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 116);  relu_23 = arg108_1 = None
    convert_element_type_72: "f32[116]" = torch.ops.prims.convert_element_type.default(arg278_1, torch.float32);  arg278_1 = None
    convert_element_type_73: "f32[116]" = torch.ops.prims.convert_element_type.default(arg279_1, torch.float32);  arg279_1 = None
    add_72: "f32[116]" = torch.ops.aten.add.Tensor(convert_element_type_73, 1e-05);  convert_element_type_73 = None
    sqrt_36: "f32[116]" = torch.ops.aten.sqrt.default(add_72);  add_72 = None
    reciprocal_36: "f32[116]" = torch.ops.aten.reciprocal.default(sqrt_36);  sqrt_36 = None
    mul_108: "f32[116]" = torch.ops.aten.mul.Tensor(reciprocal_36, 1);  reciprocal_36 = None
    unsqueeze_288: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_72, -1);  convert_element_type_72 = None
    unsqueeze_289: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, -1);  unsqueeze_288 = None
    unsqueeze_290: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(mul_108, -1);  mul_108 = None
    unsqueeze_291: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, -1);  unsqueeze_290 = None
    sub_36: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_289);  convolution_36 = unsqueeze_289 = None
    mul_109: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_36, unsqueeze_291);  sub_36 = unsqueeze_291 = None
    unsqueeze_292: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(arg109_1, -1);  arg109_1 = None
    unsqueeze_293: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, -1);  unsqueeze_292 = None
    mul_110: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(mul_109, unsqueeze_293);  mul_109 = unsqueeze_293 = None
    unsqueeze_294: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(arg110_1, -1);  arg110_1 = None
    unsqueeze_295: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, -1);  unsqueeze_294 = None
    add_73: "f32[4, 116, 14, 14]" = torch.ops.aten.add.Tensor(mul_110, unsqueeze_295);  mul_110 = unsqueeze_295 = None
    convolution_37: "f32[4, 116, 14, 14]" = torch.ops.aten.convolution.default(add_73, arg111_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_73 = arg111_1 = None
    convert_element_type_74: "f32[116]" = torch.ops.prims.convert_element_type.default(arg281_1, torch.float32);  arg281_1 = None
    convert_element_type_75: "f32[116]" = torch.ops.prims.convert_element_type.default(arg282_1, torch.float32);  arg282_1 = None
    add_74: "f32[116]" = torch.ops.aten.add.Tensor(convert_element_type_75, 1e-05);  convert_element_type_75 = None
    sqrt_37: "f32[116]" = torch.ops.aten.sqrt.default(add_74);  add_74 = None
    reciprocal_37: "f32[116]" = torch.ops.aten.reciprocal.default(sqrt_37);  sqrt_37 = None
    mul_111: "f32[116]" = torch.ops.aten.mul.Tensor(reciprocal_37, 1);  reciprocal_37 = None
    unsqueeze_296: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_74, -1);  convert_element_type_74 = None
    unsqueeze_297: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, -1);  unsqueeze_296 = None
    unsqueeze_298: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(mul_111, -1);  mul_111 = None
    unsqueeze_299: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, -1);  unsqueeze_298 = None
    sub_37: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_297);  convolution_37 = unsqueeze_297 = None
    mul_112: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_37, unsqueeze_299);  sub_37 = unsqueeze_299 = None
    unsqueeze_300: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(arg112_1, -1);  arg112_1 = None
    unsqueeze_301: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, -1);  unsqueeze_300 = None
    mul_113: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(mul_112, unsqueeze_301);  mul_112 = unsqueeze_301 = None
    unsqueeze_302: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(arg113_1, -1);  arg113_1 = None
    unsqueeze_303: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, -1);  unsqueeze_302 = None
    add_75: "f32[4, 116, 14, 14]" = torch.ops.aten.add.Tensor(mul_113, unsqueeze_303);  mul_113 = unsqueeze_303 = None
    relu_24: "f32[4, 116, 14, 14]" = torch.ops.aten.relu.default(add_75);  add_75 = None
    cat_10: "f32[4, 232, 14, 14]" = torch.ops.aten.cat.default([getitem_18, relu_24], 1);  getitem_18 = relu_24 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    view_20: "f32[4, 2, 116, 14, 14]" = torch.ops.aten.view.default(cat_10, [4, 2, 116, 14, 14]);  cat_10 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    permute_10: "f32[4, 116, 2, 14, 14]" = torch.ops.aten.permute.default(view_20, [0, 2, 1, 3, 4]);  view_20 = None
    clone_10: "f32[4, 116, 2, 14, 14]" = torch.ops.aten.clone.default(permute_10, memory_format = torch.contiguous_format);  permute_10 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    view_21: "f32[4, 232, 14, 14]" = torch.ops.aten.view.default(clone_10, [4, 232, 14, 14]);  clone_10 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:94, code: x1, x2 = x.chunk(2, dim=1)
    split_9 = torch.ops.aten.split.Tensor(view_21, 116, 1);  view_21 = None
    getitem_20: "f32[4, 116, 14, 14]" = split_9[0]
    getitem_21: "f32[4, 116, 14, 14]" = split_9[1];  split_9 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:95, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
    convolution_38: "f32[4, 116, 14, 14]" = torch.ops.aten.convolution.default(getitem_21, arg114_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_21 = arg114_1 = None
    convert_element_type_76: "f32[116]" = torch.ops.prims.convert_element_type.default(arg284_1, torch.float32);  arg284_1 = None
    convert_element_type_77: "f32[116]" = torch.ops.prims.convert_element_type.default(arg285_1, torch.float32);  arg285_1 = None
    add_76: "f32[116]" = torch.ops.aten.add.Tensor(convert_element_type_77, 1e-05);  convert_element_type_77 = None
    sqrt_38: "f32[116]" = torch.ops.aten.sqrt.default(add_76);  add_76 = None
    reciprocal_38: "f32[116]" = torch.ops.aten.reciprocal.default(sqrt_38);  sqrt_38 = None
    mul_114: "f32[116]" = torch.ops.aten.mul.Tensor(reciprocal_38, 1);  reciprocal_38 = None
    unsqueeze_304: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_76, -1);  convert_element_type_76 = None
    unsqueeze_305: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, -1);  unsqueeze_304 = None
    unsqueeze_306: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(mul_114, -1);  mul_114 = None
    unsqueeze_307: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, -1);  unsqueeze_306 = None
    sub_38: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_305);  convolution_38 = unsqueeze_305 = None
    mul_115: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_38, unsqueeze_307);  sub_38 = unsqueeze_307 = None
    unsqueeze_308: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(arg115_1, -1);  arg115_1 = None
    unsqueeze_309: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, -1);  unsqueeze_308 = None
    mul_116: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(mul_115, unsqueeze_309);  mul_115 = unsqueeze_309 = None
    unsqueeze_310: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(arg116_1, -1);  arg116_1 = None
    unsqueeze_311: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, -1);  unsqueeze_310 = None
    add_77: "f32[4, 116, 14, 14]" = torch.ops.aten.add.Tensor(mul_116, unsqueeze_311);  mul_116 = unsqueeze_311 = None
    relu_25: "f32[4, 116, 14, 14]" = torch.ops.aten.relu.default(add_77);  add_77 = None
    convolution_39: "f32[4, 116, 14, 14]" = torch.ops.aten.convolution.default(relu_25, arg117_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 116);  relu_25 = arg117_1 = None
    convert_element_type_78: "f32[116]" = torch.ops.prims.convert_element_type.default(arg287_1, torch.float32);  arg287_1 = None
    convert_element_type_79: "f32[116]" = torch.ops.prims.convert_element_type.default(arg288_1, torch.float32);  arg288_1 = None
    add_78: "f32[116]" = torch.ops.aten.add.Tensor(convert_element_type_79, 1e-05);  convert_element_type_79 = None
    sqrt_39: "f32[116]" = torch.ops.aten.sqrt.default(add_78);  add_78 = None
    reciprocal_39: "f32[116]" = torch.ops.aten.reciprocal.default(sqrt_39);  sqrt_39 = None
    mul_117: "f32[116]" = torch.ops.aten.mul.Tensor(reciprocal_39, 1);  reciprocal_39 = None
    unsqueeze_312: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_78, -1);  convert_element_type_78 = None
    unsqueeze_313: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, -1);  unsqueeze_312 = None
    unsqueeze_314: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(mul_117, -1);  mul_117 = None
    unsqueeze_315: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, -1);  unsqueeze_314 = None
    sub_39: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_313);  convolution_39 = unsqueeze_313 = None
    mul_118: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_39, unsqueeze_315);  sub_39 = unsqueeze_315 = None
    unsqueeze_316: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(arg118_1, -1);  arg118_1 = None
    unsqueeze_317: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, -1);  unsqueeze_316 = None
    mul_119: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(mul_118, unsqueeze_317);  mul_118 = unsqueeze_317 = None
    unsqueeze_318: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(arg119_1, -1);  arg119_1 = None
    unsqueeze_319: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, -1);  unsqueeze_318 = None
    add_79: "f32[4, 116, 14, 14]" = torch.ops.aten.add.Tensor(mul_119, unsqueeze_319);  mul_119 = unsqueeze_319 = None
    convolution_40: "f32[4, 116, 14, 14]" = torch.ops.aten.convolution.default(add_79, arg120_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_79 = arg120_1 = None
    convert_element_type_80: "f32[116]" = torch.ops.prims.convert_element_type.default(arg290_1, torch.float32);  arg290_1 = None
    convert_element_type_81: "f32[116]" = torch.ops.prims.convert_element_type.default(arg291_1, torch.float32);  arg291_1 = None
    add_80: "f32[116]" = torch.ops.aten.add.Tensor(convert_element_type_81, 1e-05);  convert_element_type_81 = None
    sqrt_40: "f32[116]" = torch.ops.aten.sqrt.default(add_80);  add_80 = None
    reciprocal_40: "f32[116]" = torch.ops.aten.reciprocal.default(sqrt_40);  sqrt_40 = None
    mul_120: "f32[116]" = torch.ops.aten.mul.Tensor(reciprocal_40, 1);  reciprocal_40 = None
    unsqueeze_320: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_80, -1);  convert_element_type_80 = None
    unsqueeze_321: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, -1);  unsqueeze_320 = None
    unsqueeze_322: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(mul_120, -1);  mul_120 = None
    unsqueeze_323: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, -1);  unsqueeze_322 = None
    sub_40: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_321);  convolution_40 = unsqueeze_321 = None
    mul_121: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_40, unsqueeze_323);  sub_40 = unsqueeze_323 = None
    unsqueeze_324: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(arg121_1, -1);  arg121_1 = None
    unsqueeze_325: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, -1);  unsqueeze_324 = None
    mul_122: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(mul_121, unsqueeze_325);  mul_121 = unsqueeze_325 = None
    unsqueeze_326: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(arg122_1, -1);  arg122_1 = None
    unsqueeze_327: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, -1);  unsqueeze_326 = None
    add_81: "f32[4, 116, 14, 14]" = torch.ops.aten.add.Tensor(mul_122, unsqueeze_327);  mul_122 = unsqueeze_327 = None
    relu_26: "f32[4, 116, 14, 14]" = torch.ops.aten.relu.default(add_81);  add_81 = None
    cat_11: "f32[4, 232, 14, 14]" = torch.ops.aten.cat.default([getitem_20, relu_26], 1);  getitem_20 = relu_26 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    view_22: "f32[4, 2, 116, 14, 14]" = torch.ops.aten.view.default(cat_11, [4, 2, 116, 14, 14]);  cat_11 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    permute_11: "f32[4, 116, 2, 14, 14]" = torch.ops.aten.permute.default(view_22, [0, 2, 1, 3, 4]);  view_22 = None
    clone_11: "f32[4, 116, 2, 14, 14]" = torch.ops.aten.clone.default(permute_11, memory_format = torch.contiguous_format);  permute_11 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    view_23: "f32[4, 232, 14, 14]" = torch.ops.aten.view.default(clone_11, [4, 232, 14, 14]);  clone_11 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:97, code: out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
    convolution_41: "f32[4, 232, 7, 7]" = torch.ops.aten.convolution.default(view_23, arg123_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 232);  arg123_1 = None
    convert_element_type_82: "f32[232]" = torch.ops.prims.convert_element_type.default(arg293_1, torch.float32);  arg293_1 = None
    convert_element_type_83: "f32[232]" = torch.ops.prims.convert_element_type.default(arg294_1, torch.float32);  arg294_1 = None
    add_82: "f32[232]" = torch.ops.aten.add.Tensor(convert_element_type_83, 1e-05);  convert_element_type_83 = None
    sqrt_41: "f32[232]" = torch.ops.aten.sqrt.default(add_82);  add_82 = None
    reciprocal_41: "f32[232]" = torch.ops.aten.reciprocal.default(sqrt_41);  sqrt_41 = None
    mul_123: "f32[232]" = torch.ops.aten.mul.Tensor(reciprocal_41, 1);  reciprocal_41 = None
    unsqueeze_328: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_82, -1);  convert_element_type_82 = None
    unsqueeze_329: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, -1);  unsqueeze_328 = None
    unsqueeze_330: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(mul_123, -1);  mul_123 = None
    unsqueeze_331: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, -1);  unsqueeze_330 = None
    sub_41: "f32[4, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_329);  convolution_41 = unsqueeze_329 = None
    mul_124: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_41, unsqueeze_331);  sub_41 = unsqueeze_331 = None
    unsqueeze_332: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(arg124_1, -1);  arg124_1 = None
    unsqueeze_333: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, -1);  unsqueeze_332 = None
    mul_125: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(mul_124, unsqueeze_333);  mul_124 = unsqueeze_333 = None
    unsqueeze_334: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(arg125_1, -1);  arg125_1 = None
    unsqueeze_335: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, -1);  unsqueeze_334 = None
    add_83: "f32[4, 232, 7, 7]" = torch.ops.aten.add.Tensor(mul_125, unsqueeze_335);  mul_125 = unsqueeze_335 = None
    convolution_42: "f32[4, 232, 7, 7]" = torch.ops.aten.convolution.default(add_83, arg126_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_83 = arg126_1 = None
    convert_element_type_84: "f32[232]" = torch.ops.prims.convert_element_type.default(arg296_1, torch.float32);  arg296_1 = None
    convert_element_type_85: "f32[232]" = torch.ops.prims.convert_element_type.default(arg297_1, torch.float32);  arg297_1 = None
    add_84: "f32[232]" = torch.ops.aten.add.Tensor(convert_element_type_85, 1e-05);  convert_element_type_85 = None
    sqrt_42: "f32[232]" = torch.ops.aten.sqrt.default(add_84);  add_84 = None
    reciprocal_42: "f32[232]" = torch.ops.aten.reciprocal.default(sqrt_42);  sqrt_42 = None
    mul_126: "f32[232]" = torch.ops.aten.mul.Tensor(reciprocal_42, 1);  reciprocal_42 = None
    unsqueeze_336: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_84, -1);  convert_element_type_84 = None
    unsqueeze_337: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, -1);  unsqueeze_336 = None
    unsqueeze_338: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(mul_126, -1);  mul_126 = None
    unsqueeze_339: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, -1);  unsqueeze_338 = None
    sub_42: "f32[4, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_337);  convolution_42 = unsqueeze_337 = None
    mul_127: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_42, unsqueeze_339);  sub_42 = unsqueeze_339 = None
    unsqueeze_340: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(arg127_1, -1);  arg127_1 = None
    unsqueeze_341: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, -1);  unsqueeze_340 = None
    mul_128: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(mul_127, unsqueeze_341);  mul_127 = unsqueeze_341 = None
    unsqueeze_342: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(arg128_1, -1);  arg128_1 = None
    unsqueeze_343: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, -1);  unsqueeze_342 = None
    add_85: "f32[4, 232, 7, 7]" = torch.ops.aten.add.Tensor(mul_128, unsqueeze_343);  mul_128 = unsqueeze_343 = None
    relu_27: "f32[4, 232, 7, 7]" = torch.ops.aten.relu.default(add_85);  add_85 = None
    convolution_43: "f32[4, 232, 14, 14]" = torch.ops.aten.convolution.default(view_23, arg129_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  view_23 = arg129_1 = None
    convert_element_type_86: "f32[232]" = torch.ops.prims.convert_element_type.default(arg299_1, torch.float32);  arg299_1 = None
    convert_element_type_87: "f32[232]" = torch.ops.prims.convert_element_type.default(arg300_1, torch.float32);  arg300_1 = None
    add_86: "f32[232]" = torch.ops.aten.add.Tensor(convert_element_type_87, 1e-05);  convert_element_type_87 = None
    sqrt_43: "f32[232]" = torch.ops.aten.sqrt.default(add_86);  add_86 = None
    reciprocal_43: "f32[232]" = torch.ops.aten.reciprocal.default(sqrt_43);  sqrt_43 = None
    mul_129: "f32[232]" = torch.ops.aten.mul.Tensor(reciprocal_43, 1);  reciprocal_43 = None
    unsqueeze_344: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_86, -1);  convert_element_type_86 = None
    unsqueeze_345: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, -1);  unsqueeze_344 = None
    unsqueeze_346: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(mul_129, -1);  mul_129 = None
    unsqueeze_347: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, -1);  unsqueeze_346 = None
    sub_43: "f32[4, 232, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_345);  convolution_43 = unsqueeze_345 = None
    mul_130: "f32[4, 232, 14, 14]" = torch.ops.aten.mul.Tensor(sub_43, unsqueeze_347);  sub_43 = unsqueeze_347 = None
    unsqueeze_348: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(arg130_1, -1);  arg130_1 = None
    unsqueeze_349: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, -1);  unsqueeze_348 = None
    mul_131: "f32[4, 232, 14, 14]" = torch.ops.aten.mul.Tensor(mul_130, unsqueeze_349);  mul_130 = unsqueeze_349 = None
    unsqueeze_350: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(arg131_1, -1);  arg131_1 = None
    unsqueeze_351: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, -1);  unsqueeze_350 = None
    add_87: "f32[4, 232, 14, 14]" = torch.ops.aten.add.Tensor(mul_131, unsqueeze_351);  mul_131 = unsqueeze_351 = None
    relu_28: "f32[4, 232, 14, 14]" = torch.ops.aten.relu.default(add_87);  add_87 = None
    convolution_44: "f32[4, 232, 7, 7]" = torch.ops.aten.convolution.default(relu_28, arg132_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 232);  relu_28 = arg132_1 = None
    convert_element_type_88: "f32[232]" = torch.ops.prims.convert_element_type.default(arg302_1, torch.float32);  arg302_1 = None
    convert_element_type_89: "f32[232]" = torch.ops.prims.convert_element_type.default(arg303_1, torch.float32);  arg303_1 = None
    add_88: "f32[232]" = torch.ops.aten.add.Tensor(convert_element_type_89, 1e-05);  convert_element_type_89 = None
    sqrt_44: "f32[232]" = torch.ops.aten.sqrt.default(add_88);  add_88 = None
    reciprocal_44: "f32[232]" = torch.ops.aten.reciprocal.default(sqrt_44);  sqrt_44 = None
    mul_132: "f32[232]" = torch.ops.aten.mul.Tensor(reciprocal_44, 1);  reciprocal_44 = None
    unsqueeze_352: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_88, -1);  convert_element_type_88 = None
    unsqueeze_353: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, -1);  unsqueeze_352 = None
    unsqueeze_354: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(mul_132, -1);  mul_132 = None
    unsqueeze_355: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, -1);  unsqueeze_354 = None
    sub_44: "f32[4, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_353);  convolution_44 = unsqueeze_353 = None
    mul_133: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_44, unsqueeze_355);  sub_44 = unsqueeze_355 = None
    unsqueeze_356: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(arg133_1, -1);  arg133_1 = None
    unsqueeze_357: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, -1);  unsqueeze_356 = None
    mul_134: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(mul_133, unsqueeze_357);  mul_133 = unsqueeze_357 = None
    unsqueeze_358: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(arg134_1, -1);  arg134_1 = None
    unsqueeze_359: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, -1);  unsqueeze_358 = None
    add_89: "f32[4, 232, 7, 7]" = torch.ops.aten.add.Tensor(mul_134, unsqueeze_359);  mul_134 = unsqueeze_359 = None
    convolution_45: "f32[4, 232, 7, 7]" = torch.ops.aten.convolution.default(add_89, arg135_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_89 = arg135_1 = None
    convert_element_type_90: "f32[232]" = torch.ops.prims.convert_element_type.default(arg305_1, torch.float32);  arg305_1 = None
    convert_element_type_91: "f32[232]" = torch.ops.prims.convert_element_type.default(arg306_1, torch.float32);  arg306_1 = None
    add_90: "f32[232]" = torch.ops.aten.add.Tensor(convert_element_type_91, 1e-05);  convert_element_type_91 = None
    sqrt_45: "f32[232]" = torch.ops.aten.sqrt.default(add_90);  add_90 = None
    reciprocal_45: "f32[232]" = torch.ops.aten.reciprocal.default(sqrt_45);  sqrt_45 = None
    mul_135: "f32[232]" = torch.ops.aten.mul.Tensor(reciprocal_45, 1);  reciprocal_45 = None
    unsqueeze_360: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_90, -1);  convert_element_type_90 = None
    unsqueeze_361: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, -1);  unsqueeze_360 = None
    unsqueeze_362: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(mul_135, -1);  mul_135 = None
    unsqueeze_363: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, -1);  unsqueeze_362 = None
    sub_45: "f32[4, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_361);  convolution_45 = unsqueeze_361 = None
    mul_136: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_45, unsqueeze_363);  sub_45 = unsqueeze_363 = None
    unsqueeze_364: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(arg136_1, -1);  arg136_1 = None
    unsqueeze_365: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, -1);  unsqueeze_364 = None
    mul_137: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(mul_136, unsqueeze_365);  mul_136 = unsqueeze_365 = None
    unsqueeze_366: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(arg137_1, -1);  arg137_1 = None
    unsqueeze_367: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, -1);  unsqueeze_366 = None
    add_91: "f32[4, 232, 7, 7]" = torch.ops.aten.add.Tensor(mul_137, unsqueeze_367);  mul_137 = unsqueeze_367 = None
    relu_29: "f32[4, 232, 7, 7]" = torch.ops.aten.relu.default(add_91);  add_91 = None
    cat_12: "f32[4, 464, 7, 7]" = torch.ops.aten.cat.default([relu_27, relu_29], 1);  relu_27 = relu_29 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    view_24: "f32[4, 2, 232, 7, 7]" = torch.ops.aten.view.default(cat_12, [4, 2, 232, 7, 7]);  cat_12 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    permute_12: "f32[4, 232, 2, 7, 7]" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3, 4]);  view_24 = None
    clone_12: "f32[4, 232, 2, 7, 7]" = torch.ops.aten.clone.default(permute_12, memory_format = torch.contiguous_format);  permute_12 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    view_25: "f32[4, 464, 7, 7]" = torch.ops.aten.view.default(clone_12, [4, 464, 7, 7]);  clone_12 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:94, code: x1, x2 = x.chunk(2, dim=1)
    split_10 = torch.ops.aten.split.Tensor(view_25, 232, 1);  view_25 = None
    getitem_22: "f32[4, 232, 7, 7]" = split_10[0]
    getitem_23: "f32[4, 232, 7, 7]" = split_10[1];  split_10 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:95, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
    convolution_46: "f32[4, 232, 7, 7]" = torch.ops.aten.convolution.default(getitem_23, arg138_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_23 = arg138_1 = None
    convert_element_type_92: "f32[232]" = torch.ops.prims.convert_element_type.default(arg308_1, torch.float32);  arg308_1 = None
    convert_element_type_93: "f32[232]" = torch.ops.prims.convert_element_type.default(arg309_1, torch.float32);  arg309_1 = None
    add_92: "f32[232]" = torch.ops.aten.add.Tensor(convert_element_type_93, 1e-05);  convert_element_type_93 = None
    sqrt_46: "f32[232]" = torch.ops.aten.sqrt.default(add_92);  add_92 = None
    reciprocal_46: "f32[232]" = torch.ops.aten.reciprocal.default(sqrt_46);  sqrt_46 = None
    mul_138: "f32[232]" = torch.ops.aten.mul.Tensor(reciprocal_46, 1);  reciprocal_46 = None
    unsqueeze_368: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_92, -1);  convert_element_type_92 = None
    unsqueeze_369: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, -1);  unsqueeze_368 = None
    unsqueeze_370: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(mul_138, -1);  mul_138 = None
    unsqueeze_371: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, -1);  unsqueeze_370 = None
    sub_46: "f32[4, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_369);  convolution_46 = unsqueeze_369 = None
    mul_139: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_46, unsqueeze_371);  sub_46 = unsqueeze_371 = None
    unsqueeze_372: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(arg139_1, -1);  arg139_1 = None
    unsqueeze_373: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_372, -1);  unsqueeze_372 = None
    mul_140: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(mul_139, unsqueeze_373);  mul_139 = unsqueeze_373 = None
    unsqueeze_374: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(arg140_1, -1);  arg140_1 = None
    unsqueeze_375: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, -1);  unsqueeze_374 = None
    add_93: "f32[4, 232, 7, 7]" = torch.ops.aten.add.Tensor(mul_140, unsqueeze_375);  mul_140 = unsqueeze_375 = None
    relu_30: "f32[4, 232, 7, 7]" = torch.ops.aten.relu.default(add_93);  add_93 = None
    convolution_47: "f32[4, 232, 7, 7]" = torch.ops.aten.convolution.default(relu_30, arg141_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 232);  relu_30 = arg141_1 = None
    convert_element_type_94: "f32[232]" = torch.ops.prims.convert_element_type.default(arg311_1, torch.float32);  arg311_1 = None
    convert_element_type_95: "f32[232]" = torch.ops.prims.convert_element_type.default(arg312_1, torch.float32);  arg312_1 = None
    add_94: "f32[232]" = torch.ops.aten.add.Tensor(convert_element_type_95, 1e-05);  convert_element_type_95 = None
    sqrt_47: "f32[232]" = torch.ops.aten.sqrt.default(add_94);  add_94 = None
    reciprocal_47: "f32[232]" = torch.ops.aten.reciprocal.default(sqrt_47);  sqrt_47 = None
    mul_141: "f32[232]" = torch.ops.aten.mul.Tensor(reciprocal_47, 1);  reciprocal_47 = None
    unsqueeze_376: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_94, -1);  convert_element_type_94 = None
    unsqueeze_377: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, -1);  unsqueeze_376 = None
    unsqueeze_378: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(mul_141, -1);  mul_141 = None
    unsqueeze_379: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, -1);  unsqueeze_378 = None
    sub_47: "f32[4, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_377);  convolution_47 = unsqueeze_377 = None
    mul_142: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_47, unsqueeze_379);  sub_47 = unsqueeze_379 = None
    unsqueeze_380: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(arg142_1, -1);  arg142_1 = None
    unsqueeze_381: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, -1);  unsqueeze_380 = None
    mul_143: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(mul_142, unsqueeze_381);  mul_142 = unsqueeze_381 = None
    unsqueeze_382: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(arg143_1, -1);  arg143_1 = None
    unsqueeze_383: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, -1);  unsqueeze_382 = None
    add_95: "f32[4, 232, 7, 7]" = torch.ops.aten.add.Tensor(mul_143, unsqueeze_383);  mul_143 = unsqueeze_383 = None
    convolution_48: "f32[4, 232, 7, 7]" = torch.ops.aten.convolution.default(add_95, arg144_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_95 = arg144_1 = None
    convert_element_type_96: "f32[232]" = torch.ops.prims.convert_element_type.default(arg314_1, torch.float32);  arg314_1 = None
    convert_element_type_97: "f32[232]" = torch.ops.prims.convert_element_type.default(arg315_1, torch.float32);  arg315_1 = None
    add_96: "f32[232]" = torch.ops.aten.add.Tensor(convert_element_type_97, 1e-05);  convert_element_type_97 = None
    sqrt_48: "f32[232]" = torch.ops.aten.sqrt.default(add_96);  add_96 = None
    reciprocal_48: "f32[232]" = torch.ops.aten.reciprocal.default(sqrt_48);  sqrt_48 = None
    mul_144: "f32[232]" = torch.ops.aten.mul.Tensor(reciprocal_48, 1);  reciprocal_48 = None
    unsqueeze_384: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_96, -1);  convert_element_type_96 = None
    unsqueeze_385: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_384, -1);  unsqueeze_384 = None
    unsqueeze_386: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(mul_144, -1);  mul_144 = None
    unsqueeze_387: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, -1);  unsqueeze_386 = None
    sub_48: "f32[4, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_385);  convolution_48 = unsqueeze_385 = None
    mul_145: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_48, unsqueeze_387);  sub_48 = unsqueeze_387 = None
    unsqueeze_388: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(arg145_1, -1);  arg145_1 = None
    unsqueeze_389: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, -1);  unsqueeze_388 = None
    mul_146: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(mul_145, unsqueeze_389);  mul_145 = unsqueeze_389 = None
    unsqueeze_390: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(arg146_1, -1);  arg146_1 = None
    unsqueeze_391: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_390, -1);  unsqueeze_390 = None
    add_97: "f32[4, 232, 7, 7]" = torch.ops.aten.add.Tensor(mul_146, unsqueeze_391);  mul_146 = unsqueeze_391 = None
    relu_31: "f32[4, 232, 7, 7]" = torch.ops.aten.relu.default(add_97);  add_97 = None
    cat_13: "f32[4, 464, 7, 7]" = torch.ops.aten.cat.default([getitem_22, relu_31], 1);  getitem_22 = relu_31 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    view_26: "f32[4, 2, 232, 7, 7]" = torch.ops.aten.view.default(cat_13, [4, 2, 232, 7, 7]);  cat_13 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    permute_13: "f32[4, 232, 2, 7, 7]" = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3, 4]);  view_26 = None
    clone_13: "f32[4, 232, 2, 7, 7]" = torch.ops.aten.clone.default(permute_13, memory_format = torch.contiguous_format);  permute_13 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    view_27: "f32[4, 464, 7, 7]" = torch.ops.aten.view.default(clone_13, [4, 464, 7, 7]);  clone_13 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:94, code: x1, x2 = x.chunk(2, dim=1)
    split_11 = torch.ops.aten.split.Tensor(view_27, 232, 1);  view_27 = None
    getitem_24: "f32[4, 232, 7, 7]" = split_11[0]
    getitem_25: "f32[4, 232, 7, 7]" = split_11[1];  split_11 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:95, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
    convolution_49: "f32[4, 232, 7, 7]" = torch.ops.aten.convolution.default(getitem_25, arg147_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_25 = arg147_1 = None
    convert_element_type_98: "f32[232]" = torch.ops.prims.convert_element_type.default(arg317_1, torch.float32);  arg317_1 = None
    convert_element_type_99: "f32[232]" = torch.ops.prims.convert_element_type.default(arg318_1, torch.float32);  arg318_1 = None
    add_98: "f32[232]" = torch.ops.aten.add.Tensor(convert_element_type_99, 1e-05);  convert_element_type_99 = None
    sqrt_49: "f32[232]" = torch.ops.aten.sqrt.default(add_98);  add_98 = None
    reciprocal_49: "f32[232]" = torch.ops.aten.reciprocal.default(sqrt_49);  sqrt_49 = None
    mul_147: "f32[232]" = torch.ops.aten.mul.Tensor(reciprocal_49, 1);  reciprocal_49 = None
    unsqueeze_392: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_98, -1);  convert_element_type_98 = None
    unsqueeze_393: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, -1);  unsqueeze_392 = None
    unsqueeze_394: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(mul_147, -1);  mul_147 = None
    unsqueeze_395: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, -1);  unsqueeze_394 = None
    sub_49: "f32[4, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_393);  convolution_49 = unsqueeze_393 = None
    mul_148: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_49, unsqueeze_395);  sub_49 = unsqueeze_395 = None
    unsqueeze_396: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(arg148_1, -1);  arg148_1 = None
    unsqueeze_397: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_396, -1);  unsqueeze_396 = None
    mul_149: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(mul_148, unsqueeze_397);  mul_148 = unsqueeze_397 = None
    unsqueeze_398: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(arg149_1, -1);  arg149_1 = None
    unsqueeze_399: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, -1);  unsqueeze_398 = None
    add_99: "f32[4, 232, 7, 7]" = torch.ops.aten.add.Tensor(mul_149, unsqueeze_399);  mul_149 = unsqueeze_399 = None
    relu_32: "f32[4, 232, 7, 7]" = torch.ops.aten.relu.default(add_99);  add_99 = None
    convolution_50: "f32[4, 232, 7, 7]" = torch.ops.aten.convolution.default(relu_32, arg150_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 232);  relu_32 = arg150_1 = None
    convert_element_type_100: "f32[232]" = torch.ops.prims.convert_element_type.default(arg320_1, torch.float32);  arg320_1 = None
    convert_element_type_101: "f32[232]" = torch.ops.prims.convert_element_type.default(arg321_1, torch.float32);  arg321_1 = None
    add_100: "f32[232]" = torch.ops.aten.add.Tensor(convert_element_type_101, 1e-05);  convert_element_type_101 = None
    sqrt_50: "f32[232]" = torch.ops.aten.sqrt.default(add_100);  add_100 = None
    reciprocal_50: "f32[232]" = torch.ops.aten.reciprocal.default(sqrt_50);  sqrt_50 = None
    mul_150: "f32[232]" = torch.ops.aten.mul.Tensor(reciprocal_50, 1);  reciprocal_50 = None
    unsqueeze_400: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_100, -1);  convert_element_type_100 = None
    unsqueeze_401: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, -1);  unsqueeze_400 = None
    unsqueeze_402: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(mul_150, -1);  mul_150 = None
    unsqueeze_403: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, -1);  unsqueeze_402 = None
    sub_50: "f32[4, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_401);  convolution_50 = unsqueeze_401 = None
    mul_151: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_50, unsqueeze_403);  sub_50 = unsqueeze_403 = None
    unsqueeze_404: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(arg151_1, -1);  arg151_1 = None
    unsqueeze_405: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, -1);  unsqueeze_404 = None
    mul_152: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(mul_151, unsqueeze_405);  mul_151 = unsqueeze_405 = None
    unsqueeze_406: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(arg152_1, -1);  arg152_1 = None
    unsqueeze_407: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, -1);  unsqueeze_406 = None
    add_101: "f32[4, 232, 7, 7]" = torch.ops.aten.add.Tensor(mul_152, unsqueeze_407);  mul_152 = unsqueeze_407 = None
    convolution_51: "f32[4, 232, 7, 7]" = torch.ops.aten.convolution.default(add_101, arg153_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_101 = arg153_1 = None
    convert_element_type_102: "f32[232]" = torch.ops.prims.convert_element_type.default(arg323_1, torch.float32);  arg323_1 = None
    convert_element_type_103: "f32[232]" = torch.ops.prims.convert_element_type.default(arg324_1, torch.float32);  arg324_1 = None
    add_102: "f32[232]" = torch.ops.aten.add.Tensor(convert_element_type_103, 1e-05);  convert_element_type_103 = None
    sqrt_51: "f32[232]" = torch.ops.aten.sqrt.default(add_102);  add_102 = None
    reciprocal_51: "f32[232]" = torch.ops.aten.reciprocal.default(sqrt_51);  sqrt_51 = None
    mul_153: "f32[232]" = torch.ops.aten.mul.Tensor(reciprocal_51, 1);  reciprocal_51 = None
    unsqueeze_408: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_102, -1);  convert_element_type_102 = None
    unsqueeze_409: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_408, -1);  unsqueeze_408 = None
    unsqueeze_410: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(mul_153, -1);  mul_153 = None
    unsqueeze_411: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, -1);  unsqueeze_410 = None
    sub_51: "f32[4, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_409);  convolution_51 = unsqueeze_409 = None
    mul_154: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_51, unsqueeze_411);  sub_51 = unsqueeze_411 = None
    unsqueeze_412: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(arg154_1, -1);  arg154_1 = None
    unsqueeze_413: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, -1);  unsqueeze_412 = None
    mul_155: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(mul_154, unsqueeze_413);  mul_154 = unsqueeze_413 = None
    unsqueeze_414: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(arg155_1, -1);  arg155_1 = None
    unsqueeze_415: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, -1);  unsqueeze_414 = None
    add_103: "f32[4, 232, 7, 7]" = torch.ops.aten.add.Tensor(mul_155, unsqueeze_415);  mul_155 = unsqueeze_415 = None
    relu_33: "f32[4, 232, 7, 7]" = torch.ops.aten.relu.default(add_103);  add_103 = None
    cat_14: "f32[4, 464, 7, 7]" = torch.ops.aten.cat.default([getitem_24, relu_33], 1);  getitem_24 = relu_33 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    view_28: "f32[4, 2, 232, 7, 7]" = torch.ops.aten.view.default(cat_14, [4, 2, 232, 7, 7]);  cat_14 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    permute_14: "f32[4, 232, 2, 7, 7]" = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3, 4]);  view_28 = None
    clone_14: "f32[4, 232, 2, 7, 7]" = torch.ops.aten.clone.default(permute_14, memory_format = torch.contiguous_format);  permute_14 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    view_29: "f32[4, 464, 7, 7]" = torch.ops.aten.view.default(clone_14, [4, 464, 7, 7]);  clone_14 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:94, code: x1, x2 = x.chunk(2, dim=1)
    split_12 = torch.ops.aten.split.Tensor(view_29, 232, 1);  view_29 = None
    getitem_26: "f32[4, 232, 7, 7]" = split_12[0]
    getitem_27: "f32[4, 232, 7, 7]" = split_12[1];  split_12 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:95, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
    convolution_52: "f32[4, 232, 7, 7]" = torch.ops.aten.convolution.default(getitem_27, arg156_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_27 = arg156_1 = None
    convert_element_type_104: "f32[232]" = torch.ops.prims.convert_element_type.default(arg326_1, torch.float32);  arg326_1 = None
    convert_element_type_105: "f32[232]" = torch.ops.prims.convert_element_type.default(arg327_1, torch.float32);  arg327_1 = None
    add_104: "f32[232]" = torch.ops.aten.add.Tensor(convert_element_type_105, 1e-05);  convert_element_type_105 = None
    sqrt_52: "f32[232]" = torch.ops.aten.sqrt.default(add_104);  add_104 = None
    reciprocal_52: "f32[232]" = torch.ops.aten.reciprocal.default(sqrt_52);  sqrt_52 = None
    mul_156: "f32[232]" = torch.ops.aten.mul.Tensor(reciprocal_52, 1);  reciprocal_52 = None
    unsqueeze_416: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_104, -1);  convert_element_type_104 = None
    unsqueeze_417: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, -1);  unsqueeze_416 = None
    unsqueeze_418: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(mul_156, -1);  mul_156 = None
    unsqueeze_419: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, -1);  unsqueeze_418 = None
    sub_52: "f32[4, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_417);  convolution_52 = unsqueeze_417 = None
    mul_157: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_52, unsqueeze_419);  sub_52 = unsqueeze_419 = None
    unsqueeze_420: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(arg157_1, -1);  arg157_1 = None
    unsqueeze_421: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_420, -1);  unsqueeze_420 = None
    mul_158: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(mul_157, unsqueeze_421);  mul_157 = unsqueeze_421 = None
    unsqueeze_422: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(arg158_1, -1);  arg158_1 = None
    unsqueeze_423: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, -1);  unsqueeze_422 = None
    add_105: "f32[4, 232, 7, 7]" = torch.ops.aten.add.Tensor(mul_158, unsqueeze_423);  mul_158 = unsqueeze_423 = None
    relu_34: "f32[4, 232, 7, 7]" = torch.ops.aten.relu.default(add_105);  add_105 = None
    convolution_53: "f32[4, 232, 7, 7]" = torch.ops.aten.convolution.default(relu_34, arg159_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 232);  relu_34 = arg159_1 = None
    convert_element_type_106: "f32[232]" = torch.ops.prims.convert_element_type.default(arg329_1, torch.float32);  arg329_1 = None
    convert_element_type_107: "f32[232]" = torch.ops.prims.convert_element_type.default(arg330_1, torch.float32);  arg330_1 = None
    add_106: "f32[232]" = torch.ops.aten.add.Tensor(convert_element_type_107, 1e-05);  convert_element_type_107 = None
    sqrt_53: "f32[232]" = torch.ops.aten.sqrt.default(add_106);  add_106 = None
    reciprocal_53: "f32[232]" = torch.ops.aten.reciprocal.default(sqrt_53);  sqrt_53 = None
    mul_159: "f32[232]" = torch.ops.aten.mul.Tensor(reciprocal_53, 1);  reciprocal_53 = None
    unsqueeze_424: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_106, -1);  convert_element_type_106 = None
    unsqueeze_425: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_424, -1);  unsqueeze_424 = None
    unsqueeze_426: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(mul_159, -1);  mul_159 = None
    unsqueeze_427: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, -1);  unsqueeze_426 = None
    sub_53: "f32[4, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_425);  convolution_53 = unsqueeze_425 = None
    mul_160: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_53, unsqueeze_427);  sub_53 = unsqueeze_427 = None
    unsqueeze_428: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(arg160_1, -1);  arg160_1 = None
    unsqueeze_429: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, -1);  unsqueeze_428 = None
    mul_161: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(mul_160, unsqueeze_429);  mul_160 = unsqueeze_429 = None
    unsqueeze_430: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(arg161_1, -1);  arg161_1 = None
    unsqueeze_431: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, -1);  unsqueeze_430 = None
    add_107: "f32[4, 232, 7, 7]" = torch.ops.aten.add.Tensor(mul_161, unsqueeze_431);  mul_161 = unsqueeze_431 = None
    convolution_54: "f32[4, 232, 7, 7]" = torch.ops.aten.convolution.default(add_107, arg162_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_107 = arg162_1 = None
    convert_element_type_108: "f32[232]" = torch.ops.prims.convert_element_type.default(arg332_1, torch.float32);  arg332_1 = None
    convert_element_type_109: "f32[232]" = torch.ops.prims.convert_element_type.default(arg333_1, torch.float32);  arg333_1 = None
    add_108: "f32[232]" = torch.ops.aten.add.Tensor(convert_element_type_109, 1e-05);  convert_element_type_109 = None
    sqrt_54: "f32[232]" = torch.ops.aten.sqrt.default(add_108);  add_108 = None
    reciprocal_54: "f32[232]" = torch.ops.aten.reciprocal.default(sqrt_54);  sqrt_54 = None
    mul_162: "f32[232]" = torch.ops.aten.mul.Tensor(reciprocal_54, 1);  reciprocal_54 = None
    unsqueeze_432: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_108, -1);  convert_element_type_108 = None
    unsqueeze_433: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_432, -1);  unsqueeze_432 = None
    unsqueeze_434: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(mul_162, -1);  mul_162 = None
    unsqueeze_435: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, -1);  unsqueeze_434 = None
    sub_54: "f32[4, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_433);  convolution_54 = unsqueeze_433 = None
    mul_163: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_54, unsqueeze_435);  sub_54 = unsqueeze_435 = None
    unsqueeze_436: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(arg163_1, -1);  arg163_1 = None
    unsqueeze_437: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_436, -1);  unsqueeze_436 = None
    mul_164: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(mul_163, unsqueeze_437);  mul_163 = unsqueeze_437 = None
    unsqueeze_438: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(arg164_1, -1);  arg164_1 = None
    unsqueeze_439: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_438, -1);  unsqueeze_438 = None
    add_109: "f32[4, 232, 7, 7]" = torch.ops.aten.add.Tensor(mul_164, unsqueeze_439);  mul_164 = unsqueeze_439 = None
    relu_35: "f32[4, 232, 7, 7]" = torch.ops.aten.relu.default(add_109);  add_109 = None
    cat_15: "f32[4, 464, 7, 7]" = torch.ops.aten.cat.default([getitem_26, relu_35], 1);  getitem_26 = relu_35 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    view_30: "f32[4, 2, 232, 7, 7]" = torch.ops.aten.view.default(cat_15, [4, 2, 232, 7, 7]);  cat_15 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    permute_15: "f32[4, 232, 2, 7, 7]" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3, 4]);  view_30 = None
    clone_15: "f32[4, 232, 2, 7, 7]" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    view_31: "f32[4, 464, 7, 7]" = torch.ops.aten.view.default(clone_15, [4, 464, 7, 7]);  clone_15 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:160, code: x = self.conv5(x)
    convolution_55: "f32[4, 1024, 7, 7]" = torch.ops.aten.convolution.default(view_31, arg165_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  view_31 = arg165_1 = None
    convert_element_type_110: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg335_1, torch.float32);  arg335_1 = None
    convert_element_type_111: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg336_1, torch.float32);  arg336_1 = None
    add_110: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_111, 1e-05);  convert_element_type_111 = None
    sqrt_55: "f32[1024]" = torch.ops.aten.sqrt.default(add_110);  add_110 = None
    reciprocal_55: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_55);  sqrt_55 = None
    mul_165: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_55, 1);  reciprocal_55 = None
    unsqueeze_440: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_110, -1);  convert_element_type_110 = None
    unsqueeze_441: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, -1);  unsqueeze_440 = None
    unsqueeze_442: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_165, -1);  mul_165 = None
    unsqueeze_443: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_442, -1);  unsqueeze_442 = None
    sub_55: "f32[4, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_441);  convolution_55 = unsqueeze_441 = None
    mul_166: "f32[4, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub_55, unsqueeze_443);  sub_55 = unsqueeze_443 = None
    unsqueeze_444: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg166_1, -1);  arg166_1 = None
    unsqueeze_445: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_444, -1);  unsqueeze_444 = None
    mul_167: "f32[4, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(mul_166, unsqueeze_445);  mul_166 = unsqueeze_445 = None
    unsqueeze_446: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg167_1, -1);  arg167_1 = None
    unsqueeze_447: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, -1);  unsqueeze_446 = None
    add_111: "f32[4, 1024, 7, 7]" = torch.ops.aten.add.Tensor(mul_167, unsqueeze_447);  mul_167 = unsqueeze_447 = None
    relu_36: "f32[4, 1024, 7, 7]" = torch.ops.aten.relu.default(add_111);  add_111 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:161, code: x = x.mean([2, 3])  # globalpool
    mean: "f32[4, 1024]" = torch.ops.aten.mean.dim(relu_36, [2, 3]);  relu_36 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:162, code: x = self.fc(x)
    permute_16: "f32[1024, 1000]" = torch.ops.aten.permute.default(arg168_1, [1, 0]);  arg168_1 = None
    addmm: "f32[4, 1000]" = torch.ops.aten.addmm.default(arg169_1, mean, permute_16);  arg169_1 = mean = permute_16 = None
    return (addmm,)
    