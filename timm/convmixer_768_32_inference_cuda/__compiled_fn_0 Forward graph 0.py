from __future__ import annotations



def forward(self, arg0_1: "f32[768, 3, 7, 7]", arg1_1: "f32[768]", arg2_1: "f32[768]", arg3_1: "f32[768]", arg4_1: "f32[768, 1, 7, 7]", arg5_1: "f32[768]", arg6_1: "f32[768]", arg7_1: "f32[768]", arg8_1: "f32[768, 768, 1, 1]", arg9_1: "f32[768]", arg10_1: "f32[768]", arg11_1: "f32[768]", arg12_1: "f32[768, 1, 7, 7]", arg13_1: "f32[768]", arg14_1: "f32[768]", arg15_1: "f32[768]", arg16_1: "f32[768, 768, 1, 1]", arg17_1: "f32[768]", arg18_1: "f32[768]", arg19_1: "f32[768]", arg20_1: "f32[768, 1, 7, 7]", arg21_1: "f32[768]", arg22_1: "f32[768]", arg23_1: "f32[768]", arg24_1: "f32[768, 768, 1, 1]", arg25_1: "f32[768]", arg26_1: "f32[768]", arg27_1: "f32[768]", arg28_1: "f32[768, 1, 7, 7]", arg29_1: "f32[768]", arg30_1: "f32[768]", arg31_1: "f32[768]", arg32_1: "f32[768, 768, 1, 1]", arg33_1: "f32[768]", arg34_1: "f32[768]", arg35_1: "f32[768]", arg36_1: "f32[768, 1, 7, 7]", arg37_1: "f32[768]", arg38_1: "f32[768]", arg39_1: "f32[768]", arg40_1: "f32[768, 768, 1, 1]", arg41_1: "f32[768]", arg42_1: "f32[768]", arg43_1: "f32[768]", arg44_1: "f32[768, 1, 7, 7]", arg45_1: "f32[768]", arg46_1: "f32[768]", arg47_1: "f32[768]", arg48_1: "f32[768, 768, 1, 1]", arg49_1: "f32[768]", arg50_1: "f32[768]", arg51_1: "f32[768]", arg52_1: "f32[768, 1, 7, 7]", arg53_1: "f32[768]", arg54_1: "f32[768]", arg55_1: "f32[768]", arg56_1: "f32[768, 768, 1, 1]", arg57_1: "f32[768]", arg58_1: "f32[768]", arg59_1: "f32[768]", arg60_1: "f32[768, 1, 7, 7]", arg61_1: "f32[768]", arg62_1: "f32[768]", arg63_1: "f32[768]", arg64_1: "f32[768, 768, 1, 1]", arg65_1: "f32[768]", arg66_1: "f32[768]", arg67_1: "f32[768]", arg68_1: "f32[768, 1, 7, 7]", arg69_1: "f32[768]", arg70_1: "f32[768]", arg71_1: "f32[768]", arg72_1: "f32[768, 768, 1, 1]", arg73_1: "f32[768]", arg74_1: "f32[768]", arg75_1: "f32[768]", arg76_1: "f32[768, 1, 7, 7]", arg77_1: "f32[768]", arg78_1: "f32[768]", arg79_1: "f32[768]", arg80_1: "f32[768, 768, 1, 1]", arg81_1: "f32[768]", arg82_1: "f32[768]", arg83_1: "f32[768]", arg84_1: "f32[768, 1, 7, 7]", arg85_1: "f32[768]", arg86_1: "f32[768]", arg87_1: "f32[768]", arg88_1: "f32[768, 768, 1, 1]", arg89_1: "f32[768]", arg90_1: "f32[768]", arg91_1: "f32[768]", arg92_1: "f32[768, 1, 7, 7]", arg93_1: "f32[768]", arg94_1: "f32[768]", arg95_1: "f32[768]", arg96_1: "f32[768, 768, 1, 1]", arg97_1: "f32[768]", arg98_1: "f32[768]", arg99_1: "f32[768]", arg100_1: "f32[768, 1, 7, 7]", arg101_1: "f32[768]", arg102_1: "f32[768]", arg103_1: "f32[768]", arg104_1: "f32[768, 768, 1, 1]", arg105_1: "f32[768]", arg106_1: "f32[768]", arg107_1: "f32[768]", arg108_1: "f32[768, 1, 7, 7]", arg109_1: "f32[768]", arg110_1: "f32[768]", arg111_1: "f32[768]", arg112_1: "f32[768, 768, 1, 1]", arg113_1: "f32[768]", arg114_1: "f32[768]", arg115_1: "f32[768]", arg116_1: "f32[768, 1, 7, 7]", arg117_1: "f32[768]", arg118_1: "f32[768]", arg119_1: "f32[768]", arg120_1: "f32[768, 768, 1, 1]", arg121_1: "f32[768]", arg122_1: "f32[768]", arg123_1: "f32[768]", arg124_1: "f32[768, 1, 7, 7]", arg125_1: "f32[768]", arg126_1: "f32[768]", arg127_1: "f32[768]", arg128_1: "f32[768, 768, 1, 1]", arg129_1: "f32[768]", arg130_1: "f32[768]", arg131_1: "f32[768]", arg132_1: "f32[768, 1, 7, 7]", arg133_1: "f32[768]", arg134_1: "f32[768]", arg135_1: "f32[768]", arg136_1: "f32[768, 768, 1, 1]", arg137_1: "f32[768]", arg138_1: "f32[768]", arg139_1: "f32[768]", arg140_1: "f32[768, 1, 7, 7]", arg141_1: "f32[768]", arg142_1: "f32[768]", arg143_1: "f32[768]", arg144_1: "f32[768, 768, 1, 1]", arg145_1: "f32[768]", arg146_1: "f32[768]", arg147_1: "f32[768]", arg148_1: "f32[768, 1, 7, 7]", arg149_1: "f32[768]", arg150_1: "f32[768]", arg151_1: "f32[768]", arg152_1: "f32[768, 768, 1, 1]", arg153_1: "f32[768]", arg154_1: "f32[768]", arg155_1: "f32[768]", arg156_1: "f32[768, 1, 7, 7]", arg157_1: "f32[768]", arg158_1: "f32[768]", arg159_1: "f32[768]", arg160_1: "f32[768, 768, 1, 1]", arg161_1: "f32[768]", arg162_1: "f32[768]", arg163_1: "f32[768]", arg164_1: "f32[768, 1, 7, 7]", arg165_1: "f32[768]", arg166_1: "f32[768]", arg167_1: "f32[768]", arg168_1: "f32[768, 768, 1, 1]", arg169_1: "f32[768]", arg170_1: "f32[768]", arg171_1: "f32[768]", arg172_1: "f32[768, 1, 7, 7]", arg173_1: "f32[768]", arg174_1: "f32[768]", arg175_1: "f32[768]", arg176_1: "f32[768, 768, 1, 1]", arg177_1: "f32[768]", arg178_1: "f32[768]", arg179_1: "f32[768]", arg180_1: "f32[768, 1, 7, 7]", arg181_1: "f32[768]", arg182_1: "f32[768]", arg183_1: "f32[768]", arg184_1: "f32[768, 768, 1, 1]", arg185_1: "f32[768]", arg186_1: "f32[768]", arg187_1: "f32[768]", arg188_1: "f32[768, 1, 7, 7]", arg189_1: "f32[768]", arg190_1: "f32[768]", arg191_1: "f32[768]", arg192_1: "f32[768, 768, 1, 1]", arg193_1: "f32[768]", arg194_1: "f32[768]", arg195_1: "f32[768]", arg196_1: "f32[768, 1, 7, 7]", arg197_1: "f32[768]", arg198_1: "f32[768]", arg199_1: "f32[768]", arg200_1: "f32[768, 768, 1, 1]", arg201_1: "f32[768]", arg202_1: "f32[768]", arg203_1: "f32[768]", arg204_1: "f32[768, 1, 7, 7]", arg205_1: "f32[768]", arg206_1: "f32[768]", arg207_1: "f32[768]", arg208_1: "f32[768, 768, 1, 1]", arg209_1: "f32[768]", arg210_1: "f32[768]", arg211_1: "f32[768]", arg212_1: "f32[768, 1, 7, 7]", arg213_1: "f32[768]", arg214_1: "f32[768]", arg215_1: "f32[768]", arg216_1: "f32[768, 768, 1, 1]", arg217_1: "f32[768]", arg218_1: "f32[768]", arg219_1: "f32[768]", arg220_1: "f32[768, 1, 7, 7]", arg221_1: "f32[768]", arg222_1: "f32[768]", arg223_1: "f32[768]", arg224_1: "f32[768, 768, 1, 1]", arg225_1: "f32[768]", arg226_1: "f32[768]", arg227_1: "f32[768]", arg228_1: "f32[768, 1, 7, 7]", arg229_1: "f32[768]", arg230_1: "f32[768]", arg231_1: "f32[768]", arg232_1: "f32[768, 768, 1, 1]", arg233_1: "f32[768]", arg234_1: "f32[768]", arg235_1: "f32[768]", arg236_1: "f32[768, 1, 7, 7]", arg237_1: "f32[768]", arg238_1: "f32[768]", arg239_1: "f32[768]", arg240_1: "f32[768, 768, 1, 1]", arg241_1: "f32[768]", arg242_1: "f32[768]", arg243_1: "f32[768]", arg244_1: "f32[768, 1, 7, 7]", arg245_1: "f32[768]", arg246_1: "f32[768]", arg247_1: "f32[768]", arg248_1: "f32[768, 768, 1, 1]", arg249_1: "f32[768]", arg250_1: "f32[768]", arg251_1: "f32[768]", arg252_1: "f32[768, 1, 7, 7]", arg253_1: "f32[768]", arg254_1: "f32[768]", arg255_1: "f32[768]", arg256_1: "f32[768, 768, 1, 1]", arg257_1: "f32[768]", arg258_1: "f32[768]", arg259_1: "f32[768]", arg260_1: "f32[1000, 768]", arg261_1: "f32[1000]", arg262_1: "f32[768]", arg263_1: "f32[768]", arg264_1: "i64[]", arg265_1: "f32[768]", arg266_1: "f32[768]", arg267_1: "i64[]", arg268_1: "f32[768]", arg269_1: "f32[768]", arg270_1: "i64[]", arg271_1: "f32[768]", arg272_1: "f32[768]", arg273_1: "i64[]", arg274_1: "f32[768]", arg275_1: "f32[768]", arg276_1: "i64[]", arg277_1: "f32[768]", arg278_1: "f32[768]", arg279_1: "i64[]", arg280_1: "f32[768]", arg281_1: "f32[768]", arg282_1: "i64[]", arg283_1: "f32[768]", arg284_1: "f32[768]", arg285_1: "i64[]", arg286_1: "f32[768]", arg287_1: "f32[768]", arg288_1: "i64[]", arg289_1: "f32[768]", arg290_1: "f32[768]", arg291_1: "i64[]", arg292_1: "f32[768]", arg293_1: "f32[768]", arg294_1: "i64[]", arg295_1: "f32[768]", arg296_1: "f32[768]", arg297_1: "i64[]", arg298_1: "f32[768]", arg299_1: "f32[768]", arg300_1: "i64[]", arg301_1: "f32[768]", arg302_1: "f32[768]", arg303_1: "i64[]", arg304_1: "f32[768]", arg305_1: "f32[768]", arg306_1: "i64[]", arg307_1: "f32[768]", arg308_1: "f32[768]", arg309_1: "i64[]", arg310_1: "f32[768]", arg311_1: "f32[768]", arg312_1: "i64[]", arg313_1: "f32[768]", arg314_1: "f32[768]", arg315_1: "i64[]", arg316_1: "f32[768]", arg317_1: "f32[768]", arg318_1: "i64[]", arg319_1: "f32[768]", arg320_1: "f32[768]", arg321_1: "i64[]", arg322_1: "f32[768]", arg323_1: "f32[768]", arg324_1: "i64[]", arg325_1: "f32[768]", arg326_1: "f32[768]", arg327_1: "i64[]", arg328_1: "f32[768]", arg329_1: "f32[768]", arg330_1: "i64[]", arg331_1: "f32[768]", arg332_1: "f32[768]", arg333_1: "i64[]", arg334_1: "f32[768]", arg335_1: "f32[768]", arg336_1: "i64[]", arg337_1: "f32[768]", arg338_1: "f32[768]", arg339_1: "i64[]", arg340_1: "f32[768]", arg341_1: "f32[768]", arg342_1: "i64[]", arg343_1: "f32[768]", arg344_1: "f32[768]", arg345_1: "i64[]", arg346_1: "f32[768]", arg347_1: "f32[768]", arg348_1: "i64[]", arg349_1: "f32[768]", arg350_1: "f32[768]", arg351_1: "i64[]", arg352_1: "f32[768]", arg353_1: "f32[768]", arg354_1: "i64[]", arg355_1: "f32[768]", arg356_1: "f32[768]", arg357_1: "i64[]", arg358_1: "f32[768]", arg359_1: "f32[768]", arg360_1: "i64[]", arg361_1: "f32[768]", arg362_1: "f32[768]", arg363_1: "i64[]", arg364_1: "f32[768]", arg365_1: "f32[768]", arg366_1: "i64[]", arg367_1: "f32[768]", arg368_1: "f32[768]", arg369_1: "i64[]", arg370_1: "f32[768]", arg371_1: "f32[768]", arg372_1: "i64[]", arg373_1: "f32[768]", arg374_1: "f32[768]", arg375_1: "i64[]", arg376_1: "f32[768]", arg377_1: "f32[768]", arg378_1: "i64[]", arg379_1: "f32[768]", arg380_1: "f32[768]", arg381_1: "i64[]", arg382_1: "f32[768]", arg383_1: "f32[768]", arg384_1: "i64[]", arg385_1: "f32[768]", arg386_1: "f32[768]", arg387_1: "i64[]", arg388_1: "f32[768]", arg389_1: "f32[768]", arg390_1: "i64[]", arg391_1: "f32[768]", arg392_1: "f32[768]", arg393_1: "i64[]", arg394_1: "f32[768]", arg395_1: "f32[768]", arg396_1: "i64[]", arg397_1: "f32[768]", arg398_1: "f32[768]", arg399_1: "i64[]", arg400_1: "f32[768]", arg401_1: "f32[768]", arg402_1: "i64[]", arg403_1: "f32[768]", arg404_1: "f32[768]", arg405_1: "i64[]", arg406_1: "f32[768]", arg407_1: "f32[768]", arg408_1: "i64[]", arg409_1: "f32[768]", arg410_1: "f32[768]", arg411_1: "i64[]", arg412_1: "f32[768]", arg413_1: "f32[768]", arg414_1: "i64[]", arg415_1: "f32[768]", arg416_1: "f32[768]", arg417_1: "i64[]", arg418_1: "f32[768]", arg419_1: "f32[768]", arg420_1: "i64[]", arg421_1: "f32[768]", arg422_1: "f32[768]", arg423_1: "i64[]", arg424_1: "f32[768]", arg425_1: "f32[768]", arg426_1: "i64[]", arg427_1: "f32[768]", arg428_1: "f32[768]", arg429_1: "i64[]", arg430_1: "f32[768]", arg431_1: "f32[768]", arg432_1: "i64[]", arg433_1: "f32[768]", arg434_1: "f32[768]", arg435_1: "i64[]", arg436_1: "f32[768]", arg437_1: "f32[768]", arg438_1: "i64[]", arg439_1: "f32[768]", arg440_1: "f32[768]", arg441_1: "i64[]", arg442_1: "f32[768]", arg443_1: "f32[768]", arg444_1: "i64[]", arg445_1: "f32[768]", arg446_1: "f32[768]", arg447_1: "i64[]", arg448_1: "f32[768]", arg449_1: "f32[768]", arg450_1: "i64[]", arg451_1: "f32[768]", arg452_1: "f32[768]", arg453_1: "i64[]", arg454_1: "f32[768]", arg455_1: "f32[768]", arg456_1: "i64[]", arg457_1: "f32[8, 3, 224, 224]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:85, code: x = self.stem(x)
    convolution: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(arg457_1, arg0_1, arg1_1, [7, 7], [0, 0], [1, 1], False, [0, 0], 1);  arg457_1 = arg0_1 = arg1_1 = None
    relu: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution);  convolution = None
    convert_element_type: "f32[768]" = torch.ops.prims.convert_element_type.default(arg262_1, torch.float32);  arg262_1 = None
    convert_element_type_1: "f32[768]" = torch.ops.prims.convert_element_type.default(arg263_1, torch.float32);  arg263_1 = None
    add: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_1, 1e-05);  convert_element_type_1 = None
    sqrt: "f32[768]" = torch.ops.aten.sqrt.default(add);  add = None
    reciprocal: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt);  sqrt = None
    mul: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
    unsqueeze: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type, -1);  convert_element_type = None
    unsqueeze_1: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    unsqueeze_2: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul, -1);  mul = None
    unsqueeze_3: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    sub: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu, unsqueeze_1);  relu = unsqueeze_1 = None
    mul_1: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub, unsqueeze_3);  sub = unsqueeze_3 = None
    unsqueeze_4: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
    unsqueeze_5: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_2: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None
    unsqueeze_6: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, -1);  arg3_1 = None
    unsqueeze_7: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_1: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_2, unsqueeze_7);  mul_2 = unsqueeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_1: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_1, arg4_1, arg5_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  arg4_1 = arg5_1 = None
    relu_1: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_1);  convolution_1 = None
    convert_element_type_2: "f32[768]" = torch.ops.prims.convert_element_type.default(arg265_1, torch.float32);  arg265_1 = None
    convert_element_type_3: "f32[768]" = torch.ops.prims.convert_element_type.default(arg266_1, torch.float32);  arg266_1 = None
    add_2: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_3, 1e-05);  convert_element_type_3 = None
    sqrt_1: "f32[768]" = torch.ops.aten.sqrt.default(add_2);  add_2 = None
    reciprocal_1: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_1);  sqrt_1 = None
    mul_3: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_1, 1);  reciprocal_1 = None
    unsqueeze_8: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, -1);  convert_element_type_2 = None
    unsqueeze_9: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    unsqueeze_10: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_3, -1);  mul_3 = None
    unsqueeze_11: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    sub_1: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_1, unsqueeze_9);  relu_1 = unsqueeze_9 = None
    mul_4: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_1, unsqueeze_11);  sub_1 = unsqueeze_11 = None
    unsqueeze_12: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg6_1, -1);  arg6_1 = None
    unsqueeze_13: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_5: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_4, unsqueeze_13);  mul_4 = unsqueeze_13 = None
    unsqueeze_14: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
    unsqueeze_15: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_3: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_5, unsqueeze_15);  mul_5 = unsqueeze_15 = None
    add_4: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_3, add_1);  add_3 = add_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_2: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_4, arg8_1, arg9_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_4 = arg8_1 = arg9_1 = None
    relu_2: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_2);  convolution_2 = None
    convert_element_type_4: "f32[768]" = torch.ops.prims.convert_element_type.default(arg268_1, torch.float32);  arg268_1 = None
    convert_element_type_5: "f32[768]" = torch.ops.prims.convert_element_type.default(arg269_1, torch.float32);  arg269_1 = None
    add_5: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_5, 1e-05);  convert_element_type_5 = None
    sqrt_2: "f32[768]" = torch.ops.aten.sqrt.default(add_5);  add_5 = None
    reciprocal_2: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_2);  sqrt_2 = None
    mul_6: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_2, 1);  reciprocal_2 = None
    unsqueeze_16: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_4, -1);  convert_element_type_4 = None
    unsqueeze_17: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    unsqueeze_18: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_6, -1);  mul_6 = None
    unsqueeze_19: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    sub_2: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_2, unsqueeze_17);  relu_2 = unsqueeze_17 = None
    mul_7: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_2, unsqueeze_19);  sub_2 = unsqueeze_19 = None
    unsqueeze_20: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
    unsqueeze_21: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_8: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_21);  mul_7 = unsqueeze_21 = None
    unsqueeze_22: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg11_1, -1);  arg11_1 = None
    unsqueeze_23: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_6: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_8, unsqueeze_23);  mul_8 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_3: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_6, arg12_1, arg13_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  arg12_1 = arg13_1 = None
    relu_3: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_3);  convolution_3 = None
    convert_element_type_6: "f32[768]" = torch.ops.prims.convert_element_type.default(arg271_1, torch.float32);  arg271_1 = None
    convert_element_type_7: "f32[768]" = torch.ops.prims.convert_element_type.default(arg272_1, torch.float32);  arg272_1 = None
    add_7: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_7, 1e-05);  convert_element_type_7 = None
    sqrt_3: "f32[768]" = torch.ops.aten.sqrt.default(add_7);  add_7 = None
    reciprocal_3: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_3);  sqrt_3 = None
    mul_9: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_3, 1);  reciprocal_3 = None
    unsqueeze_24: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_6, -1);  convert_element_type_6 = None
    unsqueeze_25: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
    unsqueeze_26: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_9, -1);  mul_9 = None
    unsqueeze_27: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
    sub_3: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_3, unsqueeze_25);  relu_3 = unsqueeze_25 = None
    mul_10: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_3, unsqueeze_27);  sub_3 = unsqueeze_27 = None
    unsqueeze_28: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
    unsqueeze_29: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    mul_11: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_10, unsqueeze_29);  mul_10 = unsqueeze_29 = None
    unsqueeze_30: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
    unsqueeze_31: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
    add_8: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_11, unsqueeze_31);  mul_11 = unsqueeze_31 = None
    add_9: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_8, add_6);  add_8 = add_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_4: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_9, arg16_1, arg17_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_9 = arg16_1 = arg17_1 = None
    relu_4: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_4);  convolution_4 = None
    convert_element_type_8: "f32[768]" = torch.ops.prims.convert_element_type.default(arg274_1, torch.float32);  arg274_1 = None
    convert_element_type_9: "f32[768]" = torch.ops.prims.convert_element_type.default(arg275_1, torch.float32);  arg275_1 = None
    add_10: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_9, 1e-05);  convert_element_type_9 = None
    sqrt_4: "f32[768]" = torch.ops.aten.sqrt.default(add_10);  add_10 = None
    reciprocal_4: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_4);  sqrt_4 = None
    mul_12: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_4, 1);  reciprocal_4 = None
    unsqueeze_32: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_8, -1);  convert_element_type_8 = None
    unsqueeze_33: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    unsqueeze_34: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_12, -1);  mul_12 = None
    unsqueeze_35: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
    sub_4: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_4, unsqueeze_33);  relu_4 = unsqueeze_33 = None
    mul_13: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_4, unsqueeze_35);  sub_4 = unsqueeze_35 = None
    unsqueeze_36: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg18_1, -1);  arg18_1 = None
    unsqueeze_37: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
    mul_14: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_13, unsqueeze_37);  mul_13 = unsqueeze_37 = None
    unsqueeze_38: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg19_1, -1);  arg19_1 = None
    unsqueeze_39: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
    add_11: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_14, unsqueeze_39);  mul_14 = unsqueeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_5: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_11, arg20_1, arg21_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  arg20_1 = arg21_1 = None
    relu_5: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_5);  convolution_5 = None
    convert_element_type_10: "f32[768]" = torch.ops.prims.convert_element_type.default(arg277_1, torch.float32);  arg277_1 = None
    convert_element_type_11: "f32[768]" = torch.ops.prims.convert_element_type.default(arg278_1, torch.float32);  arg278_1 = None
    add_12: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_11, 1e-05);  convert_element_type_11 = None
    sqrt_5: "f32[768]" = torch.ops.aten.sqrt.default(add_12);  add_12 = None
    reciprocal_5: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_5);  sqrt_5 = None
    mul_15: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_5, 1);  reciprocal_5 = None
    unsqueeze_40: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_10, -1);  convert_element_type_10 = None
    unsqueeze_41: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    unsqueeze_42: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_15, -1);  mul_15 = None
    unsqueeze_43: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    sub_5: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_5, unsqueeze_41);  relu_5 = unsqueeze_41 = None
    mul_16: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_5, unsqueeze_43);  sub_5 = unsqueeze_43 = None
    unsqueeze_44: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg22_1, -1);  arg22_1 = None
    unsqueeze_45: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    mul_17: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_16, unsqueeze_45);  mul_16 = unsqueeze_45 = None
    unsqueeze_46: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg23_1, -1);  arg23_1 = None
    unsqueeze_47: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
    add_13: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_17, unsqueeze_47);  mul_17 = unsqueeze_47 = None
    add_14: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_13, add_11);  add_13 = add_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_6: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_14, arg24_1, arg25_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_14 = arg24_1 = arg25_1 = None
    relu_6: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_6);  convolution_6 = None
    convert_element_type_12: "f32[768]" = torch.ops.prims.convert_element_type.default(arg280_1, torch.float32);  arg280_1 = None
    convert_element_type_13: "f32[768]" = torch.ops.prims.convert_element_type.default(arg281_1, torch.float32);  arg281_1 = None
    add_15: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_13, 1e-05);  convert_element_type_13 = None
    sqrt_6: "f32[768]" = torch.ops.aten.sqrt.default(add_15);  add_15 = None
    reciprocal_6: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_6);  sqrt_6 = None
    mul_18: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_6, 1);  reciprocal_6 = None
    unsqueeze_48: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_12, -1);  convert_element_type_12 = None
    unsqueeze_49: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
    unsqueeze_50: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_18, -1);  mul_18 = None
    unsqueeze_51: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
    sub_6: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_6, unsqueeze_49);  relu_6 = unsqueeze_49 = None
    mul_19: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_6, unsqueeze_51);  sub_6 = unsqueeze_51 = None
    unsqueeze_52: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg26_1, -1);  arg26_1 = None
    unsqueeze_53: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    mul_20: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_19, unsqueeze_53);  mul_19 = unsqueeze_53 = None
    unsqueeze_54: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg27_1, -1);  arg27_1 = None
    unsqueeze_55: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
    add_16: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_55);  mul_20 = unsqueeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_7: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_16, arg28_1, arg29_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  arg28_1 = arg29_1 = None
    relu_7: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_7);  convolution_7 = None
    convert_element_type_14: "f32[768]" = torch.ops.prims.convert_element_type.default(arg283_1, torch.float32);  arg283_1 = None
    convert_element_type_15: "f32[768]" = torch.ops.prims.convert_element_type.default(arg284_1, torch.float32);  arg284_1 = None
    add_17: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_15, 1e-05);  convert_element_type_15 = None
    sqrt_7: "f32[768]" = torch.ops.aten.sqrt.default(add_17);  add_17 = None
    reciprocal_7: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_7);  sqrt_7 = None
    mul_21: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_7, 1);  reciprocal_7 = None
    unsqueeze_56: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_14, -1);  convert_element_type_14 = None
    unsqueeze_57: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    unsqueeze_58: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_21, -1);  mul_21 = None
    unsqueeze_59: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
    sub_7: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_7, unsqueeze_57);  relu_7 = unsqueeze_57 = None
    mul_22: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_7, unsqueeze_59);  sub_7 = unsqueeze_59 = None
    unsqueeze_60: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg30_1, -1);  arg30_1 = None
    unsqueeze_61: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
    mul_23: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_22, unsqueeze_61);  mul_22 = unsqueeze_61 = None
    unsqueeze_62: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg31_1, -1);  arg31_1 = None
    unsqueeze_63: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
    add_18: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_23, unsqueeze_63);  mul_23 = unsqueeze_63 = None
    add_19: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_18, add_16);  add_18 = add_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_8: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_19, arg32_1, arg33_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_19 = arg32_1 = arg33_1 = None
    relu_8: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_8);  convolution_8 = None
    convert_element_type_16: "f32[768]" = torch.ops.prims.convert_element_type.default(arg286_1, torch.float32);  arg286_1 = None
    convert_element_type_17: "f32[768]" = torch.ops.prims.convert_element_type.default(arg287_1, torch.float32);  arg287_1 = None
    add_20: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_17, 1e-05);  convert_element_type_17 = None
    sqrt_8: "f32[768]" = torch.ops.aten.sqrt.default(add_20);  add_20 = None
    reciprocal_8: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_8);  sqrt_8 = None
    mul_24: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_8, 1);  reciprocal_8 = None
    unsqueeze_64: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_16, -1);  convert_element_type_16 = None
    unsqueeze_65: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
    unsqueeze_66: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_24, -1);  mul_24 = None
    unsqueeze_67: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
    sub_8: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_8, unsqueeze_65);  relu_8 = unsqueeze_65 = None
    mul_25: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_8, unsqueeze_67);  sub_8 = unsqueeze_67 = None
    unsqueeze_68: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg34_1, -1);  arg34_1 = None
    unsqueeze_69: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    mul_26: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_25, unsqueeze_69);  mul_25 = unsqueeze_69 = None
    unsqueeze_70: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg35_1, -1);  arg35_1 = None
    unsqueeze_71: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    add_21: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_26, unsqueeze_71);  mul_26 = unsqueeze_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_9: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_21, arg36_1, arg37_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  arg36_1 = arg37_1 = None
    relu_9: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_9);  convolution_9 = None
    convert_element_type_18: "f32[768]" = torch.ops.prims.convert_element_type.default(arg289_1, torch.float32);  arg289_1 = None
    convert_element_type_19: "f32[768]" = torch.ops.prims.convert_element_type.default(arg290_1, torch.float32);  arg290_1 = None
    add_22: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_19, 1e-05);  convert_element_type_19 = None
    sqrt_9: "f32[768]" = torch.ops.aten.sqrt.default(add_22);  add_22 = None
    reciprocal_9: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_9);  sqrt_9 = None
    mul_27: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_9, 1);  reciprocal_9 = None
    unsqueeze_72: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_18, -1);  convert_element_type_18 = None
    unsqueeze_73: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
    unsqueeze_74: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_27, -1);  mul_27 = None
    unsqueeze_75: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
    sub_9: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_9, unsqueeze_73);  relu_9 = unsqueeze_73 = None
    mul_28: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_9, unsqueeze_75);  sub_9 = unsqueeze_75 = None
    unsqueeze_76: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg38_1, -1);  arg38_1 = None
    unsqueeze_77: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
    mul_29: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_77);  mul_28 = unsqueeze_77 = None
    unsqueeze_78: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg39_1, -1);  arg39_1 = None
    unsqueeze_79: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    add_23: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_29, unsqueeze_79);  mul_29 = unsqueeze_79 = None
    add_24: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_23, add_21);  add_23 = add_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_10: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_24, arg40_1, arg41_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_24 = arg40_1 = arg41_1 = None
    relu_10: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_10);  convolution_10 = None
    convert_element_type_20: "f32[768]" = torch.ops.prims.convert_element_type.default(arg292_1, torch.float32);  arg292_1 = None
    convert_element_type_21: "f32[768]" = torch.ops.prims.convert_element_type.default(arg293_1, torch.float32);  arg293_1 = None
    add_25: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_21, 1e-05);  convert_element_type_21 = None
    sqrt_10: "f32[768]" = torch.ops.aten.sqrt.default(add_25);  add_25 = None
    reciprocal_10: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_10);  sqrt_10 = None
    mul_30: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_10, 1);  reciprocal_10 = None
    unsqueeze_80: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_20, -1);  convert_element_type_20 = None
    unsqueeze_81: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
    unsqueeze_82: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_30, -1);  mul_30 = None
    unsqueeze_83: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    sub_10: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_10, unsqueeze_81);  relu_10 = unsqueeze_81 = None
    mul_31: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_10, unsqueeze_83);  sub_10 = unsqueeze_83 = None
    unsqueeze_84: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg42_1, -1);  arg42_1 = None
    unsqueeze_85: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    mul_32: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_31, unsqueeze_85);  mul_31 = unsqueeze_85 = None
    unsqueeze_86: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg43_1, -1);  arg43_1 = None
    unsqueeze_87: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
    add_26: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_32, unsqueeze_87);  mul_32 = unsqueeze_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_11: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_26, arg44_1, arg45_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  arg44_1 = arg45_1 = None
    relu_11: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_11);  convolution_11 = None
    convert_element_type_22: "f32[768]" = torch.ops.prims.convert_element_type.default(arg295_1, torch.float32);  arg295_1 = None
    convert_element_type_23: "f32[768]" = torch.ops.prims.convert_element_type.default(arg296_1, torch.float32);  arg296_1 = None
    add_27: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_23, 1e-05);  convert_element_type_23 = None
    sqrt_11: "f32[768]" = torch.ops.aten.sqrt.default(add_27);  add_27 = None
    reciprocal_11: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_11);  sqrt_11 = None
    mul_33: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_11, 1);  reciprocal_11 = None
    unsqueeze_88: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_22, -1);  convert_element_type_22 = None
    unsqueeze_89: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
    unsqueeze_90: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_33, -1);  mul_33 = None
    unsqueeze_91: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    sub_11: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_11, unsqueeze_89);  relu_11 = unsqueeze_89 = None
    mul_34: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_11, unsqueeze_91);  sub_11 = unsqueeze_91 = None
    unsqueeze_92: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg46_1, -1);  arg46_1 = None
    unsqueeze_93: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
    mul_35: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_34, unsqueeze_93);  mul_34 = unsqueeze_93 = None
    unsqueeze_94: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg47_1, -1);  arg47_1 = None
    unsqueeze_95: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
    add_28: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_35, unsqueeze_95);  mul_35 = unsqueeze_95 = None
    add_29: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_28, add_26);  add_28 = add_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_12: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_29, arg48_1, arg49_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_29 = arg48_1 = arg49_1 = None
    relu_12: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_12);  convolution_12 = None
    convert_element_type_24: "f32[768]" = torch.ops.prims.convert_element_type.default(arg298_1, torch.float32);  arg298_1 = None
    convert_element_type_25: "f32[768]" = torch.ops.prims.convert_element_type.default(arg299_1, torch.float32);  arg299_1 = None
    add_30: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_25, 1e-05);  convert_element_type_25 = None
    sqrt_12: "f32[768]" = torch.ops.aten.sqrt.default(add_30);  add_30 = None
    reciprocal_12: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_12);  sqrt_12 = None
    mul_36: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_12, 1);  reciprocal_12 = None
    unsqueeze_96: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_24, -1);  convert_element_type_24 = None
    unsqueeze_97: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
    unsqueeze_98: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_36, -1);  mul_36 = None
    unsqueeze_99: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
    sub_12: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_12, unsqueeze_97);  relu_12 = unsqueeze_97 = None
    mul_37: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_12, unsqueeze_99);  sub_12 = unsqueeze_99 = None
    unsqueeze_100: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg50_1, -1);  arg50_1 = None
    unsqueeze_101: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
    mul_38: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_37, unsqueeze_101);  mul_37 = unsqueeze_101 = None
    unsqueeze_102: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg51_1, -1);  arg51_1 = None
    unsqueeze_103: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    add_31: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_38, unsqueeze_103);  mul_38 = unsqueeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_13: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_31, arg52_1, arg53_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  arg52_1 = arg53_1 = None
    relu_13: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_13);  convolution_13 = None
    convert_element_type_26: "f32[768]" = torch.ops.prims.convert_element_type.default(arg301_1, torch.float32);  arg301_1 = None
    convert_element_type_27: "f32[768]" = torch.ops.prims.convert_element_type.default(arg302_1, torch.float32);  arg302_1 = None
    add_32: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_27, 1e-05);  convert_element_type_27 = None
    sqrt_13: "f32[768]" = torch.ops.aten.sqrt.default(add_32);  add_32 = None
    reciprocal_13: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_13);  sqrt_13 = None
    mul_39: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_13, 1);  reciprocal_13 = None
    unsqueeze_104: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_26, -1);  convert_element_type_26 = None
    unsqueeze_105: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
    unsqueeze_106: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_39, -1);  mul_39 = None
    unsqueeze_107: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
    sub_13: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_13, unsqueeze_105);  relu_13 = unsqueeze_105 = None
    mul_40: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_13, unsqueeze_107);  sub_13 = unsqueeze_107 = None
    unsqueeze_108: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg54_1, -1);  arg54_1 = None
    unsqueeze_109: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
    mul_41: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_40, unsqueeze_109);  mul_40 = unsqueeze_109 = None
    unsqueeze_110: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg55_1, -1);  arg55_1 = None
    unsqueeze_111: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
    add_33: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_41, unsqueeze_111);  mul_41 = unsqueeze_111 = None
    add_34: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_33, add_31);  add_33 = add_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_14: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_34, arg56_1, arg57_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_34 = arg56_1 = arg57_1 = None
    relu_14: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_14);  convolution_14 = None
    convert_element_type_28: "f32[768]" = torch.ops.prims.convert_element_type.default(arg304_1, torch.float32);  arg304_1 = None
    convert_element_type_29: "f32[768]" = torch.ops.prims.convert_element_type.default(arg305_1, torch.float32);  arg305_1 = None
    add_35: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_29, 1e-05);  convert_element_type_29 = None
    sqrt_14: "f32[768]" = torch.ops.aten.sqrt.default(add_35);  add_35 = None
    reciprocal_14: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_14);  sqrt_14 = None
    mul_42: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_14, 1);  reciprocal_14 = None
    unsqueeze_112: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_28, -1);  convert_element_type_28 = None
    unsqueeze_113: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
    unsqueeze_114: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_42, -1);  mul_42 = None
    unsqueeze_115: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
    sub_14: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_14, unsqueeze_113);  relu_14 = unsqueeze_113 = None
    mul_43: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_14, unsqueeze_115);  sub_14 = unsqueeze_115 = None
    unsqueeze_116: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg58_1, -1);  arg58_1 = None
    unsqueeze_117: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
    mul_44: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_43, unsqueeze_117);  mul_43 = unsqueeze_117 = None
    unsqueeze_118: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg59_1, -1);  arg59_1 = None
    unsqueeze_119: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
    add_36: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_44, unsqueeze_119);  mul_44 = unsqueeze_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_15: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_36, arg60_1, arg61_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  arg60_1 = arg61_1 = None
    relu_15: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_15);  convolution_15 = None
    convert_element_type_30: "f32[768]" = torch.ops.prims.convert_element_type.default(arg307_1, torch.float32);  arg307_1 = None
    convert_element_type_31: "f32[768]" = torch.ops.prims.convert_element_type.default(arg308_1, torch.float32);  arg308_1 = None
    add_37: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_31, 1e-05);  convert_element_type_31 = None
    sqrt_15: "f32[768]" = torch.ops.aten.sqrt.default(add_37);  add_37 = None
    reciprocal_15: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_15);  sqrt_15 = None
    mul_45: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_15, 1);  reciprocal_15 = None
    unsqueeze_120: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_30, -1);  convert_element_type_30 = None
    unsqueeze_121: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
    unsqueeze_122: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_45, -1);  mul_45 = None
    unsqueeze_123: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
    sub_15: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_15, unsqueeze_121);  relu_15 = unsqueeze_121 = None
    mul_46: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_15, unsqueeze_123);  sub_15 = unsqueeze_123 = None
    unsqueeze_124: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg62_1, -1);  arg62_1 = None
    unsqueeze_125: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
    mul_47: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_46, unsqueeze_125);  mul_46 = unsqueeze_125 = None
    unsqueeze_126: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg63_1, -1);  arg63_1 = None
    unsqueeze_127: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
    add_38: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_47, unsqueeze_127);  mul_47 = unsqueeze_127 = None
    add_39: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_38, add_36);  add_38 = add_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_16: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_39, arg64_1, arg65_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_39 = arg64_1 = arg65_1 = None
    relu_16: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_16);  convolution_16 = None
    convert_element_type_32: "f32[768]" = torch.ops.prims.convert_element_type.default(arg310_1, torch.float32);  arg310_1 = None
    convert_element_type_33: "f32[768]" = torch.ops.prims.convert_element_type.default(arg311_1, torch.float32);  arg311_1 = None
    add_40: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_33, 1e-05);  convert_element_type_33 = None
    sqrt_16: "f32[768]" = torch.ops.aten.sqrt.default(add_40);  add_40 = None
    reciprocal_16: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_16);  sqrt_16 = None
    mul_48: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_16, 1);  reciprocal_16 = None
    unsqueeze_128: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_32, -1);  convert_element_type_32 = None
    unsqueeze_129: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
    unsqueeze_130: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_48, -1);  mul_48 = None
    unsqueeze_131: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
    sub_16: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_16, unsqueeze_129);  relu_16 = unsqueeze_129 = None
    mul_49: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_16, unsqueeze_131);  sub_16 = unsqueeze_131 = None
    unsqueeze_132: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg66_1, -1);  arg66_1 = None
    unsqueeze_133: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
    mul_50: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_133);  mul_49 = unsqueeze_133 = None
    unsqueeze_134: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg67_1, -1);  arg67_1 = None
    unsqueeze_135: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
    add_41: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_50, unsqueeze_135);  mul_50 = unsqueeze_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_17: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_41, arg68_1, arg69_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  arg68_1 = arg69_1 = None
    relu_17: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_17);  convolution_17 = None
    convert_element_type_34: "f32[768]" = torch.ops.prims.convert_element_type.default(arg313_1, torch.float32);  arg313_1 = None
    convert_element_type_35: "f32[768]" = torch.ops.prims.convert_element_type.default(arg314_1, torch.float32);  arg314_1 = None
    add_42: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_35, 1e-05);  convert_element_type_35 = None
    sqrt_17: "f32[768]" = torch.ops.aten.sqrt.default(add_42);  add_42 = None
    reciprocal_17: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_17);  sqrt_17 = None
    mul_51: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_17, 1);  reciprocal_17 = None
    unsqueeze_136: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_34, -1);  convert_element_type_34 = None
    unsqueeze_137: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
    unsqueeze_138: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_51, -1);  mul_51 = None
    unsqueeze_139: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
    sub_17: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_17, unsqueeze_137);  relu_17 = unsqueeze_137 = None
    mul_52: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_17, unsqueeze_139);  sub_17 = unsqueeze_139 = None
    unsqueeze_140: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg70_1, -1);  arg70_1 = None
    unsqueeze_141: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
    mul_53: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_52, unsqueeze_141);  mul_52 = unsqueeze_141 = None
    unsqueeze_142: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg71_1, -1);  arg71_1 = None
    unsqueeze_143: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
    add_43: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_53, unsqueeze_143);  mul_53 = unsqueeze_143 = None
    add_44: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_43, add_41);  add_43 = add_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_18: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_44, arg72_1, arg73_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_44 = arg72_1 = arg73_1 = None
    relu_18: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_18);  convolution_18 = None
    convert_element_type_36: "f32[768]" = torch.ops.prims.convert_element_type.default(arg316_1, torch.float32);  arg316_1 = None
    convert_element_type_37: "f32[768]" = torch.ops.prims.convert_element_type.default(arg317_1, torch.float32);  arg317_1 = None
    add_45: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_37, 1e-05);  convert_element_type_37 = None
    sqrt_18: "f32[768]" = torch.ops.aten.sqrt.default(add_45);  add_45 = None
    reciprocal_18: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_18);  sqrt_18 = None
    mul_54: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_18, 1);  reciprocal_18 = None
    unsqueeze_144: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_36, -1);  convert_element_type_36 = None
    unsqueeze_145: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
    unsqueeze_146: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_54, -1);  mul_54 = None
    unsqueeze_147: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
    sub_18: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_18, unsqueeze_145);  relu_18 = unsqueeze_145 = None
    mul_55: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_18, unsqueeze_147);  sub_18 = unsqueeze_147 = None
    unsqueeze_148: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg74_1, -1);  arg74_1 = None
    unsqueeze_149: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
    mul_56: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_55, unsqueeze_149);  mul_55 = unsqueeze_149 = None
    unsqueeze_150: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg75_1, -1);  arg75_1 = None
    unsqueeze_151: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
    add_46: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_56, unsqueeze_151);  mul_56 = unsqueeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_19: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_46, arg76_1, arg77_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  arg76_1 = arg77_1 = None
    relu_19: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_19);  convolution_19 = None
    convert_element_type_38: "f32[768]" = torch.ops.prims.convert_element_type.default(arg319_1, torch.float32);  arg319_1 = None
    convert_element_type_39: "f32[768]" = torch.ops.prims.convert_element_type.default(arg320_1, torch.float32);  arg320_1 = None
    add_47: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_39, 1e-05);  convert_element_type_39 = None
    sqrt_19: "f32[768]" = torch.ops.aten.sqrt.default(add_47);  add_47 = None
    reciprocal_19: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_19);  sqrt_19 = None
    mul_57: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_19, 1);  reciprocal_19 = None
    unsqueeze_152: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_38, -1);  convert_element_type_38 = None
    unsqueeze_153: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
    unsqueeze_154: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_57, -1);  mul_57 = None
    unsqueeze_155: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
    sub_19: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_19, unsqueeze_153);  relu_19 = unsqueeze_153 = None
    mul_58: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_19, unsqueeze_155);  sub_19 = unsqueeze_155 = None
    unsqueeze_156: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg78_1, -1);  arg78_1 = None
    unsqueeze_157: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
    mul_59: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_58, unsqueeze_157);  mul_58 = unsqueeze_157 = None
    unsqueeze_158: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg79_1, -1);  arg79_1 = None
    unsqueeze_159: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
    add_48: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_59, unsqueeze_159);  mul_59 = unsqueeze_159 = None
    add_49: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_48, add_46);  add_48 = add_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_20: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_49, arg80_1, arg81_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_49 = arg80_1 = arg81_1 = None
    relu_20: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_20);  convolution_20 = None
    convert_element_type_40: "f32[768]" = torch.ops.prims.convert_element_type.default(arg322_1, torch.float32);  arg322_1 = None
    convert_element_type_41: "f32[768]" = torch.ops.prims.convert_element_type.default(arg323_1, torch.float32);  arg323_1 = None
    add_50: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_41, 1e-05);  convert_element_type_41 = None
    sqrt_20: "f32[768]" = torch.ops.aten.sqrt.default(add_50);  add_50 = None
    reciprocal_20: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_20);  sqrt_20 = None
    mul_60: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_20, 1);  reciprocal_20 = None
    unsqueeze_160: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_40, -1);  convert_element_type_40 = None
    unsqueeze_161: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
    unsqueeze_162: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_60, -1);  mul_60 = None
    unsqueeze_163: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
    sub_20: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_20, unsqueeze_161);  relu_20 = unsqueeze_161 = None
    mul_61: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_20, unsqueeze_163);  sub_20 = unsqueeze_163 = None
    unsqueeze_164: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg82_1, -1);  arg82_1 = None
    unsqueeze_165: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
    mul_62: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_61, unsqueeze_165);  mul_61 = unsqueeze_165 = None
    unsqueeze_166: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg83_1, -1);  arg83_1 = None
    unsqueeze_167: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
    add_51: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_62, unsqueeze_167);  mul_62 = unsqueeze_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_21: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_51, arg84_1, arg85_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  arg84_1 = arg85_1 = None
    relu_21: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_21);  convolution_21 = None
    convert_element_type_42: "f32[768]" = torch.ops.prims.convert_element_type.default(arg325_1, torch.float32);  arg325_1 = None
    convert_element_type_43: "f32[768]" = torch.ops.prims.convert_element_type.default(arg326_1, torch.float32);  arg326_1 = None
    add_52: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_43, 1e-05);  convert_element_type_43 = None
    sqrt_21: "f32[768]" = torch.ops.aten.sqrt.default(add_52);  add_52 = None
    reciprocal_21: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_21);  sqrt_21 = None
    mul_63: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_21, 1);  reciprocal_21 = None
    unsqueeze_168: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_42, -1);  convert_element_type_42 = None
    unsqueeze_169: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
    unsqueeze_170: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_63, -1);  mul_63 = None
    unsqueeze_171: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
    sub_21: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_21, unsqueeze_169);  relu_21 = unsqueeze_169 = None
    mul_64: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_21, unsqueeze_171);  sub_21 = unsqueeze_171 = None
    unsqueeze_172: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg86_1, -1);  arg86_1 = None
    unsqueeze_173: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
    mul_65: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_64, unsqueeze_173);  mul_64 = unsqueeze_173 = None
    unsqueeze_174: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg87_1, -1);  arg87_1 = None
    unsqueeze_175: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
    add_53: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_65, unsqueeze_175);  mul_65 = unsqueeze_175 = None
    add_54: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_53, add_51);  add_53 = add_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_22: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_54, arg88_1, arg89_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_54 = arg88_1 = arg89_1 = None
    relu_22: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_22);  convolution_22 = None
    convert_element_type_44: "f32[768]" = torch.ops.prims.convert_element_type.default(arg328_1, torch.float32);  arg328_1 = None
    convert_element_type_45: "f32[768]" = torch.ops.prims.convert_element_type.default(arg329_1, torch.float32);  arg329_1 = None
    add_55: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_45, 1e-05);  convert_element_type_45 = None
    sqrt_22: "f32[768]" = torch.ops.aten.sqrt.default(add_55);  add_55 = None
    reciprocal_22: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_22);  sqrt_22 = None
    mul_66: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_22, 1);  reciprocal_22 = None
    unsqueeze_176: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_44, -1);  convert_element_type_44 = None
    unsqueeze_177: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
    unsqueeze_178: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_66, -1);  mul_66 = None
    unsqueeze_179: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
    sub_22: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_22, unsqueeze_177);  relu_22 = unsqueeze_177 = None
    mul_67: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_22, unsqueeze_179);  sub_22 = unsqueeze_179 = None
    unsqueeze_180: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg90_1, -1);  arg90_1 = None
    unsqueeze_181: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
    mul_68: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_67, unsqueeze_181);  mul_67 = unsqueeze_181 = None
    unsqueeze_182: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg91_1, -1);  arg91_1 = None
    unsqueeze_183: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
    add_56: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_68, unsqueeze_183);  mul_68 = unsqueeze_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_23: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_56, arg92_1, arg93_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  arg92_1 = arg93_1 = None
    relu_23: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_23);  convolution_23 = None
    convert_element_type_46: "f32[768]" = torch.ops.prims.convert_element_type.default(arg331_1, torch.float32);  arg331_1 = None
    convert_element_type_47: "f32[768]" = torch.ops.prims.convert_element_type.default(arg332_1, torch.float32);  arg332_1 = None
    add_57: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_47, 1e-05);  convert_element_type_47 = None
    sqrt_23: "f32[768]" = torch.ops.aten.sqrt.default(add_57);  add_57 = None
    reciprocal_23: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_23);  sqrt_23 = None
    mul_69: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_23, 1);  reciprocal_23 = None
    unsqueeze_184: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_46, -1);  convert_element_type_46 = None
    unsqueeze_185: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
    unsqueeze_186: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_69, -1);  mul_69 = None
    unsqueeze_187: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
    sub_23: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_23, unsqueeze_185);  relu_23 = unsqueeze_185 = None
    mul_70: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_23, unsqueeze_187);  sub_23 = unsqueeze_187 = None
    unsqueeze_188: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg94_1, -1);  arg94_1 = None
    unsqueeze_189: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
    mul_71: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_70, unsqueeze_189);  mul_70 = unsqueeze_189 = None
    unsqueeze_190: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg95_1, -1);  arg95_1 = None
    unsqueeze_191: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
    add_58: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_71, unsqueeze_191);  mul_71 = unsqueeze_191 = None
    add_59: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_58, add_56);  add_58 = add_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_24: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_59, arg96_1, arg97_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_59 = arg96_1 = arg97_1 = None
    relu_24: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_24);  convolution_24 = None
    convert_element_type_48: "f32[768]" = torch.ops.prims.convert_element_type.default(arg334_1, torch.float32);  arg334_1 = None
    convert_element_type_49: "f32[768]" = torch.ops.prims.convert_element_type.default(arg335_1, torch.float32);  arg335_1 = None
    add_60: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_49, 1e-05);  convert_element_type_49 = None
    sqrt_24: "f32[768]" = torch.ops.aten.sqrt.default(add_60);  add_60 = None
    reciprocal_24: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_24);  sqrt_24 = None
    mul_72: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_24, 1);  reciprocal_24 = None
    unsqueeze_192: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_48, -1);  convert_element_type_48 = None
    unsqueeze_193: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
    unsqueeze_194: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_72, -1);  mul_72 = None
    unsqueeze_195: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
    sub_24: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_24, unsqueeze_193);  relu_24 = unsqueeze_193 = None
    mul_73: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_24, unsqueeze_195);  sub_24 = unsqueeze_195 = None
    unsqueeze_196: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg98_1, -1);  arg98_1 = None
    unsqueeze_197: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
    mul_74: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_73, unsqueeze_197);  mul_73 = unsqueeze_197 = None
    unsqueeze_198: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg99_1, -1);  arg99_1 = None
    unsqueeze_199: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
    add_61: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_74, unsqueeze_199);  mul_74 = unsqueeze_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_25: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_61, arg100_1, arg101_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  arg100_1 = arg101_1 = None
    relu_25: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_25);  convolution_25 = None
    convert_element_type_50: "f32[768]" = torch.ops.prims.convert_element_type.default(arg337_1, torch.float32);  arg337_1 = None
    convert_element_type_51: "f32[768]" = torch.ops.prims.convert_element_type.default(arg338_1, torch.float32);  arg338_1 = None
    add_62: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_51, 1e-05);  convert_element_type_51 = None
    sqrt_25: "f32[768]" = torch.ops.aten.sqrt.default(add_62);  add_62 = None
    reciprocal_25: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_25);  sqrt_25 = None
    mul_75: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_25, 1);  reciprocal_25 = None
    unsqueeze_200: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_50, -1);  convert_element_type_50 = None
    unsqueeze_201: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
    unsqueeze_202: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_75, -1);  mul_75 = None
    unsqueeze_203: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
    sub_25: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_25, unsqueeze_201);  relu_25 = unsqueeze_201 = None
    mul_76: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_25, unsqueeze_203);  sub_25 = unsqueeze_203 = None
    unsqueeze_204: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg102_1, -1);  arg102_1 = None
    unsqueeze_205: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
    mul_77: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_76, unsqueeze_205);  mul_76 = unsqueeze_205 = None
    unsqueeze_206: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg103_1, -1);  arg103_1 = None
    unsqueeze_207: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
    add_63: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_77, unsqueeze_207);  mul_77 = unsqueeze_207 = None
    add_64: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_63, add_61);  add_63 = add_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_26: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_64, arg104_1, arg105_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_64 = arg104_1 = arg105_1 = None
    relu_26: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_26);  convolution_26 = None
    convert_element_type_52: "f32[768]" = torch.ops.prims.convert_element_type.default(arg340_1, torch.float32);  arg340_1 = None
    convert_element_type_53: "f32[768]" = torch.ops.prims.convert_element_type.default(arg341_1, torch.float32);  arg341_1 = None
    add_65: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_53, 1e-05);  convert_element_type_53 = None
    sqrt_26: "f32[768]" = torch.ops.aten.sqrt.default(add_65);  add_65 = None
    reciprocal_26: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_26);  sqrt_26 = None
    mul_78: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_26, 1);  reciprocal_26 = None
    unsqueeze_208: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_52, -1);  convert_element_type_52 = None
    unsqueeze_209: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
    unsqueeze_210: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_78, -1);  mul_78 = None
    unsqueeze_211: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
    sub_26: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_26, unsqueeze_209);  relu_26 = unsqueeze_209 = None
    mul_79: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_26, unsqueeze_211);  sub_26 = unsqueeze_211 = None
    unsqueeze_212: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg106_1, -1);  arg106_1 = None
    unsqueeze_213: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
    mul_80: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_79, unsqueeze_213);  mul_79 = unsqueeze_213 = None
    unsqueeze_214: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg107_1, -1);  arg107_1 = None
    unsqueeze_215: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
    add_66: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_80, unsqueeze_215);  mul_80 = unsqueeze_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_27: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_66, arg108_1, arg109_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  arg108_1 = arg109_1 = None
    relu_27: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_27);  convolution_27 = None
    convert_element_type_54: "f32[768]" = torch.ops.prims.convert_element_type.default(arg343_1, torch.float32);  arg343_1 = None
    convert_element_type_55: "f32[768]" = torch.ops.prims.convert_element_type.default(arg344_1, torch.float32);  arg344_1 = None
    add_67: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_55, 1e-05);  convert_element_type_55 = None
    sqrt_27: "f32[768]" = torch.ops.aten.sqrt.default(add_67);  add_67 = None
    reciprocal_27: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_27);  sqrt_27 = None
    mul_81: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_27, 1);  reciprocal_27 = None
    unsqueeze_216: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_54, -1);  convert_element_type_54 = None
    unsqueeze_217: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
    unsqueeze_218: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_81, -1);  mul_81 = None
    unsqueeze_219: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, -1);  unsqueeze_218 = None
    sub_27: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_27, unsqueeze_217);  relu_27 = unsqueeze_217 = None
    mul_82: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_27, unsqueeze_219);  sub_27 = unsqueeze_219 = None
    unsqueeze_220: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg110_1, -1);  arg110_1 = None
    unsqueeze_221: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, -1);  unsqueeze_220 = None
    mul_83: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_82, unsqueeze_221);  mul_82 = unsqueeze_221 = None
    unsqueeze_222: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg111_1, -1);  arg111_1 = None
    unsqueeze_223: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, -1);  unsqueeze_222 = None
    add_68: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_83, unsqueeze_223);  mul_83 = unsqueeze_223 = None
    add_69: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_68, add_66);  add_68 = add_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_28: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_69, arg112_1, arg113_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_69 = arg112_1 = arg113_1 = None
    relu_28: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_28);  convolution_28 = None
    convert_element_type_56: "f32[768]" = torch.ops.prims.convert_element_type.default(arg346_1, torch.float32);  arg346_1 = None
    convert_element_type_57: "f32[768]" = torch.ops.prims.convert_element_type.default(arg347_1, torch.float32);  arg347_1 = None
    add_70: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_57, 1e-05);  convert_element_type_57 = None
    sqrt_28: "f32[768]" = torch.ops.aten.sqrt.default(add_70);  add_70 = None
    reciprocal_28: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_28);  sqrt_28 = None
    mul_84: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_28, 1);  reciprocal_28 = None
    unsqueeze_224: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_56, -1);  convert_element_type_56 = None
    unsqueeze_225: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, -1);  unsqueeze_224 = None
    unsqueeze_226: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_84, -1);  mul_84 = None
    unsqueeze_227: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, -1);  unsqueeze_226 = None
    sub_28: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_28, unsqueeze_225);  relu_28 = unsqueeze_225 = None
    mul_85: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_28, unsqueeze_227);  sub_28 = unsqueeze_227 = None
    unsqueeze_228: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg114_1, -1);  arg114_1 = None
    unsqueeze_229: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, -1);  unsqueeze_228 = None
    mul_86: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_85, unsqueeze_229);  mul_85 = unsqueeze_229 = None
    unsqueeze_230: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg115_1, -1);  arg115_1 = None
    unsqueeze_231: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, -1);  unsqueeze_230 = None
    add_71: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_86, unsqueeze_231);  mul_86 = unsqueeze_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_29: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_71, arg116_1, arg117_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  arg116_1 = arg117_1 = None
    relu_29: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_29);  convolution_29 = None
    convert_element_type_58: "f32[768]" = torch.ops.prims.convert_element_type.default(arg349_1, torch.float32);  arg349_1 = None
    convert_element_type_59: "f32[768]" = torch.ops.prims.convert_element_type.default(arg350_1, torch.float32);  arg350_1 = None
    add_72: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_59, 1e-05);  convert_element_type_59 = None
    sqrt_29: "f32[768]" = torch.ops.aten.sqrt.default(add_72);  add_72 = None
    reciprocal_29: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_29);  sqrt_29 = None
    mul_87: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_29, 1);  reciprocal_29 = None
    unsqueeze_232: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_58, -1);  convert_element_type_58 = None
    unsqueeze_233: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, -1);  unsqueeze_232 = None
    unsqueeze_234: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_87, -1);  mul_87 = None
    unsqueeze_235: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, -1);  unsqueeze_234 = None
    sub_29: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_29, unsqueeze_233);  relu_29 = unsqueeze_233 = None
    mul_88: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_29, unsqueeze_235);  sub_29 = unsqueeze_235 = None
    unsqueeze_236: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg118_1, -1);  arg118_1 = None
    unsqueeze_237: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, -1);  unsqueeze_236 = None
    mul_89: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_88, unsqueeze_237);  mul_88 = unsqueeze_237 = None
    unsqueeze_238: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg119_1, -1);  arg119_1 = None
    unsqueeze_239: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, -1);  unsqueeze_238 = None
    add_73: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_89, unsqueeze_239);  mul_89 = unsqueeze_239 = None
    add_74: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_73, add_71);  add_73 = add_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_30: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_74, arg120_1, arg121_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_74 = arg120_1 = arg121_1 = None
    relu_30: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_30);  convolution_30 = None
    convert_element_type_60: "f32[768]" = torch.ops.prims.convert_element_type.default(arg352_1, torch.float32);  arg352_1 = None
    convert_element_type_61: "f32[768]" = torch.ops.prims.convert_element_type.default(arg353_1, torch.float32);  arg353_1 = None
    add_75: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_61, 1e-05);  convert_element_type_61 = None
    sqrt_30: "f32[768]" = torch.ops.aten.sqrt.default(add_75);  add_75 = None
    reciprocal_30: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_30);  sqrt_30 = None
    mul_90: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_30, 1);  reciprocal_30 = None
    unsqueeze_240: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_60, -1);  convert_element_type_60 = None
    unsqueeze_241: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, -1);  unsqueeze_240 = None
    unsqueeze_242: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_90, -1);  mul_90 = None
    unsqueeze_243: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, -1);  unsqueeze_242 = None
    sub_30: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_30, unsqueeze_241);  relu_30 = unsqueeze_241 = None
    mul_91: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_30, unsqueeze_243);  sub_30 = unsqueeze_243 = None
    unsqueeze_244: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg122_1, -1);  arg122_1 = None
    unsqueeze_245: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, -1);  unsqueeze_244 = None
    mul_92: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_91, unsqueeze_245);  mul_91 = unsqueeze_245 = None
    unsqueeze_246: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg123_1, -1);  arg123_1 = None
    unsqueeze_247: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, -1);  unsqueeze_246 = None
    add_76: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_92, unsqueeze_247);  mul_92 = unsqueeze_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_31: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_76, arg124_1, arg125_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  arg124_1 = arg125_1 = None
    relu_31: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_31);  convolution_31 = None
    convert_element_type_62: "f32[768]" = torch.ops.prims.convert_element_type.default(arg355_1, torch.float32);  arg355_1 = None
    convert_element_type_63: "f32[768]" = torch.ops.prims.convert_element_type.default(arg356_1, torch.float32);  arg356_1 = None
    add_77: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_63, 1e-05);  convert_element_type_63 = None
    sqrt_31: "f32[768]" = torch.ops.aten.sqrt.default(add_77);  add_77 = None
    reciprocal_31: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_31);  sqrt_31 = None
    mul_93: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_31, 1);  reciprocal_31 = None
    unsqueeze_248: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_62, -1);  convert_element_type_62 = None
    unsqueeze_249: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, -1);  unsqueeze_248 = None
    unsqueeze_250: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_93, -1);  mul_93 = None
    unsqueeze_251: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, -1);  unsqueeze_250 = None
    sub_31: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_31, unsqueeze_249);  relu_31 = unsqueeze_249 = None
    mul_94: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_31, unsqueeze_251);  sub_31 = unsqueeze_251 = None
    unsqueeze_252: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg126_1, -1);  arg126_1 = None
    unsqueeze_253: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, -1);  unsqueeze_252 = None
    mul_95: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_94, unsqueeze_253);  mul_94 = unsqueeze_253 = None
    unsqueeze_254: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg127_1, -1);  arg127_1 = None
    unsqueeze_255: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, -1);  unsqueeze_254 = None
    add_78: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_95, unsqueeze_255);  mul_95 = unsqueeze_255 = None
    add_79: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_78, add_76);  add_78 = add_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_32: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_79, arg128_1, arg129_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_79 = arg128_1 = arg129_1 = None
    relu_32: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_32);  convolution_32 = None
    convert_element_type_64: "f32[768]" = torch.ops.prims.convert_element_type.default(arg358_1, torch.float32);  arg358_1 = None
    convert_element_type_65: "f32[768]" = torch.ops.prims.convert_element_type.default(arg359_1, torch.float32);  arg359_1 = None
    add_80: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_65, 1e-05);  convert_element_type_65 = None
    sqrt_32: "f32[768]" = torch.ops.aten.sqrt.default(add_80);  add_80 = None
    reciprocal_32: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_32);  sqrt_32 = None
    mul_96: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_32, 1);  reciprocal_32 = None
    unsqueeze_256: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_64, -1);  convert_element_type_64 = None
    unsqueeze_257: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, -1);  unsqueeze_256 = None
    unsqueeze_258: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_96, -1);  mul_96 = None
    unsqueeze_259: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, -1);  unsqueeze_258 = None
    sub_32: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_32, unsqueeze_257);  relu_32 = unsqueeze_257 = None
    mul_97: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_32, unsqueeze_259);  sub_32 = unsqueeze_259 = None
    unsqueeze_260: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg130_1, -1);  arg130_1 = None
    unsqueeze_261: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, -1);  unsqueeze_260 = None
    mul_98: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_97, unsqueeze_261);  mul_97 = unsqueeze_261 = None
    unsqueeze_262: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg131_1, -1);  arg131_1 = None
    unsqueeze_263: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, -1);  unsqueeze_262 = None
    add_81: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_98, unsqueeze_263);  mul_98 = unsqueeze_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_33: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_81, arg132_1, arg133_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  arg132_1 = arg133_1 = None
    relu_33: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_33);  convolution_33 = None
    convert_element_type_66: "f32[768]" = torch.ops.prims.convert_element_type.default(arg361_1, torch.float32);  arg361_1 = None
    convert_element_type_67: "f32[768]" = torch.ops.prims.convert_element_type.default(arg362_1, torch.float32);  arg362_1 = None
    add_82: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_67, 1e-05);  convert_element_type_67 = None
    sqrt_33: "f32[768]" = torch.ops.aten.sqrt.default(add_82);  add_82 = None
    reciprocal_33: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_33);  sqrt_33 = None
    mul_99: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_33, 1);  reciprocal_33 = None
    unsqueeze_264: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_66, -1);  convert_element_type_66 = None
    unsqueeze_265: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, -1);  unsqueeze_264 = None
    unsqueeze_266: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_99, -1);  mul_99 = None
    unsqueeze_267: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, -1);  unsqueeze_266 = None
    sub_33: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_33, unsqueeze_265);  relu_33 = unsqueeze_265 = None
    mul_100: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_33, unsqueeze_267);  sub_33 = unsqueeze_267 = None
    unsqueeze_268: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg134_1, -1);  arg134_1 = None
    unsqueeze_269: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, -1);  unsqueeze_268 = None
    mul_101: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_100, unsqueeze_269);  mul_100 = unsqueeze_269 = None
    unsqueeze_270: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg135_1, -1);  arg135_1 = None
    unsqueeze_271: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, -1);  unsqueeze_270 = None
    add_83: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_101, unsqueeze_271);  mul_101 = unsqueeze_271 = None
    add_84: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_83, add_81);  add_83 = add_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_34: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_84, arg136_1, arg137_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_84 = arg136_1 = arg137_1 = None
    relu_34: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_34);  convolution_34 = None
    convert_element_type_68: "f32[768]" = torch.ops.prims.convert_element_type.default(arg364_1, torch.float32);  arg364_1 = None
    convert_element_type_69: "f32[768]" = torch.ops.prims.convert_element_type.default(arg365_1, torch.float32);  arg365_1 = None
    add_85: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_69, 1e-05);  convert_element_type_69 = None
    sqrt_34: "f32[768]" = torch.ops.aten.sqrt.default(add_85);  add_85 = None
    reciprocal_34: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_34);  sqrt_34 = None
    mul_102: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_34, 1);  reciprocal_34 = None
    unsqueeze_272: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_68, -1);  convert_element_type_68 = None
    unsqueeze_273: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, -1);  unsqueeze_272 = None
    unsqueeze_274: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_102, -1);  mul_102 = None
    unsqueeze_275: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, -1);  unsqueeze_274 = None
    sub_34: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_34, unsqueeze_273);  relu_34 = unsqueeze_273 = None
    mul_103: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_34, unsqueeze_275);  sub_34 = unsqueeze_275 = None
    unsqueeze_276: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg138_1, -1);  arg138_1 = None
    unsqueeze_277: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, -1);  unsqueeze_276 = None
    mul_104: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_103, unsqueeze_277);  mul_103 = unsqueeze_277 = None
    unsqueeze_278: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg139_1, -1);  arg139_1 = None
    unsqueeze_279: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, -1);  unsqueeze_278 = None
    add_86: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_104, unsqueeze_279);  mul_104 = unsqueeze_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_35: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_86, arg140_1, arg141_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  arg140_1 = arg141_1 = None
    relu_35: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_35);  convolution_35 = None
    convert_element_type_70: "f32[768]" = torch.ops.prims.convert_element_type.default(arg367_1, torch.float32);  arg367_1 = None
    convert_element_type_71: "f32[768]" = torch.ops.prims.convert_element_type.default(arg368_1, torch.float32);  arg368_1 = None
    add_87: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_71, 1e-05);  convert_element_type_71 = None
    sqrt_35: "f32[768]" = torch.ops.aten.sqrt.default(add_87);  add_87 = None
    reciprocal_35: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_35);  sqrt_35 = None
    mul_105: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_35, 1);  reciprocal_35 = None
    unsqueeze_280: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_70, -1);  convert_element_type_70 = None
    unsqueeze_281: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, -1);  unsqueeze_280 = None
    unsqueeze_282: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_105, -1);  mul_105 = None
    unsqueeze_283: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, -1);  unsqueeze_282 = None
    sub_35: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_35, unsqueeze_281);  relu_35 = unsqueeze_281 = None
    mul_106: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_35, unsqueeze_283);  sub_35 = unsqueeze_283 = None
    unsqueeze_284: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg142_1, -1);  arg142_1 = None
    unsqueeze_285: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, -1);  unsqueeze_284 = None
    mul_107: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_106, unsqueeze_285);  mul_106 = unsqueeze_285 = None
    unsqueeze_286: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg143_1, -1);  arg143_1 = None
    unsqueeze_287: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, -1);  unsqueeze_286 = None
    add_88: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_107, unsqueeze_287);  mul_107 = unsqueeze_287 = None
    add_89: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_88, add_86);  add_88 = add_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_36: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_89, arg144_1, arg145_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_89 = arg144_1 = arg145_1 = None
    relu_36: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_36);  convolution_36 = None
    convert_element_type_72: "f32[768]" = torch.ops.prims.convert_element_type.default(arg370_1, torch.float32);  arg370_1 = None
    convert_element_type_73: "f32[768]" = torch.ops.prims.convert_element_type.default(arg371_1, torch.float32);  arg371_1 = None
    add_90: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_73, 1e-05);  convert_element_type_73 = None
    sqrt_36: "f32[768]" = torch.ops.aten.sqrt.default(add_90);  add_90 = None
    reciprocal_36: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_36);  sqrt_36 = None
    mul_108: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_36, 1);  reciprocal_36 = None
    unsqueeze_288: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_72, -1);  convert_element_type_72 = None
    unsqueeze_289: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, -1);  unsqueeze_288 = None
    unsqueeze_290: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_108, -1);  mul_108 = None
    unsqueeze_291: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, -1);  unsqueeze_290 = None
    sub_36: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_36, unsqueeze_289);  relu_36 = unsqueeze_289 = None
    mul_109: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_36, unsqueeze_291);  sub_36 = unsqueeze_291 = None
    unsqueeze_292: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg146_1, -1);  arg146_1 = None
    unsqueeze_293: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, -1);  unsqueeze_292 = None
    mul_110: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_109, unsqueeze_293);  mul_109 = unsqueeze_293 = None
    unsqueeze_294: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg147_1, -1);  arg147_1 = None
    unsqueeze_295: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, -1);  unsqueeze_294 = None
    add_91: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_110, unsqueeze_295);  mul_110 = unsqueeze_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_37: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_91, arg148_1, arg149_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  arg148_1 = arg149_1 = None
    relu_37: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_37);  convolution_37 = None
    convert_element_type_74: "f32[768]" = torch.ops.prims.convert_element_type.default(arg373_1, torch.float32);  arg373_1 = None
    convert_element_type_75: "f32[768]" = torch.ops.prims.convert_element_type.default(arg374_1, torch.float32);  arg374_1 = None
    add_92: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_75, 1e-05);  convert_element_type_75 = None
    sqrt_37: "f32[768]" = torch.ops.aten.sqrt.default(add_92);  add_92 = None
    reciprocal_37: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_37);  sqrt_37 = None
    mul_111: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_37, 1);  reciprocal_37 = None
    unsqueeze_296: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_74, -1);  convert_element_type_74 = None
    unsqueeze_297: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, -1);  unsqueeze_296 = None
    unsqueeze_298: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_111, -1);  mul_111 = None
    unsqueeze_299: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, -1);  unsqueeze_298 = None
    sub_37: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_37, unsqueeze_297);  relu_37 = unsqueeze_297 = None
    mul_112: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_37, unsqueeze_299);  sub_37 = unsqueeze_299 = None
    unsqueeze_300: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg150_1, -1);  arg150_1 = None
    unsqueeze_301: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, -1);  unsqueeze_300 = None
    mul_113: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_112, unsqueeze_301);  mul_112 = unsqueeze_301 = None
    unsqueeze_302: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg151_1, -1);  arg151_1 = None
    unsqueeze_303: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, -1);  unsqueeze_302 = None
    add_93: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_113, unsqueeze_303);  mul_113 = unsqueeze_303 = None
    add_94: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_93, add_91);  add_93 = add_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_38: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_94, arg152_1, arg153_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_94 = arg152_1 = arg153_1 = None
    relu_38: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_38);  convolution_38 = None
    convert_element_type_76: "f32[768]" = torch.ops.prims.convert_element_type.default(arg376_1, torch.float32);  arg376_1 = None
    convert_element_type_77: "f32[768]" = torch.ops.prims.convert_element_type.default(arg377_1, torch.float32);  arg377_1 = None
    add_95: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_77, 1e-05);  convert_element_type_77 = None
    sqrt_38: "f32[768]" = torch.ops.aten.sqrt.default(add_95);  add_95 = None
    reciprocal_38: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_38);  sqrt_38 = None
    mul_114: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_38, 1);  reciprocal_38 = None
    unsqueeze_304: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_76, -1);  convert_element_type_76 = None
    unsqueeze_305: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, -1);  unsqueeze_304 = None
    unsqueeze_306: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_114, -1);  mul_114 = None
    unsqueeze_307: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, -1);  unsqueeze_306 = None
    sub_38: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_38, unsqueeze_305);  relu_38 = unsqueeze_305 = None
    mul_115: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_38, unsqueeze_307);  sub_38 = unsqueeze_307 = None
    unsqueeze_308: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg154_1, -1);  arg154_1 = None
    unsqueeze_309: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, -1);  unsqueeze_308 = None
    mul_116: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_115, unsqueeze_309);  mul_115 = unsqueeze_309 = None
    unsqueeze_310: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg155_1, -1);  arg155_1 = None
    unsqueeze_311: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, -1);  unsqueeze_310 = None
    add_96: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_116, unsqueeze_311);  mul_116 = unsqueeze_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_39: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_96, arg156_1, arg157_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  arg156_1 = arg157_1 = None
    relu_39: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_39);  convolution_39 = None
    convert_element_type_78: "f32[768]" = torch.ops.prims.convert_element_type.default(arg379_1, torch.float32);  arg379_1 = None
    convert_element_type_79: "f32[768]" = torch.ops.prims.convert_element_type.default(arg380_1, torch.float32);  arg380_1 = None
    add_97: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_79, 1e-05);  convert_element_type_79 = None
    sqrt_39: "f32[768]" = torch.ops.aten.sqrt.default(add_97);  add_97 = None
    reciprocal_39: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_39);  sqrt_39 = None
    mul_117: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_39, 1);  reciprocal_39 = None
    unsqueeze_312: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_78, -1);  convert_element_type_78 = None
    unsqueeze_313: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, -1);  unsqueeze_312 = None
    unsqueeze_314: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_117, -1);  mul_117 = None
    unsqueeze_315: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, -1);  unsqueeze_314 = None
    sub_39: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_39, unsqueeze_313);  relu_39 = unsqueeze_313 = None
    mul_118: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_39, unsqueeze_315);  sub_39 = unsqueeze_315 = None
    unsqueeze_316: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg158_1, -1);  arg158_1 = None
    unsqueeze_317: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, -1);  unsqueeze_316 = None
    mul_119: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_118, unsqueeze_317);  mul_118 = unsqueeze_317 = None
    unsqueeze_318: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg159_1, -1);  arg159_1 = None
    unsqueeze_319: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, -1);  unsqueeze_318 = None
    add_98: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_119, unsqueeze_319);  mul_119 = unsqueeze_319 = None
    add_99: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_98, add_96);  add_98 = add_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_40: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_99, arg160_1, arg161_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_99 = arg160_1 = arg161_1 = None
    relu_40: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_40);  convolution_40 = None
    convert_element_type_80: "f32[768]" = torch.ops.prims.convert_element_type.default(arg382_1, torch.float32);  arg382_1 = None
    convert_element_type_81: "f32[768]" = torch.ops.prims.convert_element_type.default(arg383_1, torch.float32);  arg383_1 = None
    add_100: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_81, 1e-05);  convert_element_type_81 = None
    sqrt_40: "f32[768]" = torch.ops.aten.sqrt.default(add_100);  add_100 = None
    reciprocal_40: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_40);  sqrt_40 = None
    mul_120: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_40, 1);  reciprocal_40 = None
    unsqueeze_320: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_80, -1);  convert_element_type_80 = None
    unsqueeze_321: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, -1);  unsqueeze_320 = None
    unsqueeze_322: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_120, -1);  mul_120 = None
    unsqueeze_323: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, -1);  unsqueeze_322 = None
    sub_40: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_40, unsqueeze_321);  relu_40 = unsqueeze_321 = None
    mul_121: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_40, unsqueeze_323);  sub_40 = unsqueeze_323 = None
    unsqueeze_324: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg162_1, -1);  arg162_1 = None
    unsqueeze_325: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, -1);  unsqueeze_324 = None
    mul_122: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_121, unsqueeze_325);  mul_121 = unsqueeze_325 = None
    unsqueeze_326: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg163_1, -1);  arg163_1 = None
    unsqueeze_327: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, -1);  unsqueeze_326 = None
    add_101: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_122, unsqueeze_327);  mul_122 = unsqueeze_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_41: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_101, arg164_1, arg165_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  arg164_1 = arg165_1 = None
    relu_41: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_41);  convolution_41 = None
    convert_element_type_82: "f32[768]" = torch.ops.prims.convert_element_type.default(arg385_1, torch.float32);  arg385_1 = None
    convert_element_type_83: "f32[768]" = torch.ops.prims.convert_element_type.default(arg386_1, torch.float32);  arg386_1 = None
    add_102: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_83, 1e-05);  convert_element_type_83 = None
    sqrt_41: "f32[768]" = torch.ops.aten.sqrt.default(add_102);  add_102 = None
    reciprocal_41: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_41);  sqrt_41 = None
    mul_123: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_41, 1);  reciprocal_41 = None
    unsqueeze_328: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_82, -1);  convert_element_type_82 = None
    unsqueeze_329: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, -1);  unsqueeze_328 = None
    unsqueeze_330: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_123, -1);  mul_123 = None
    unsqueeze_331: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, -1);  unsqueeze_330 = None
    sub_41: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_41, unsqueeze_329);  relu_41 = unsqueeze_329 = None
    mul_124: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_41, unsqueeze_331);  sub_41 = unsqueeze_331 = None
    unsqueeze_332: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg166_1, -1);  arg166_1 = None
    unsqueeze_333: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, -1);  unsqueeze_332 = None
    mul_125: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_124, unsqueeze_333);  mul_124 = unsqueeze_333 = None
    unsqueeze_334: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg167_1, -1);  arg167_1 = None
    unsqueeze_335: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, -1);  unsqueeze_334 = None
    add_103: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_125, unsqueeze_335);  mul_125 = unsqueeze_335 = None
    add_104: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_103, add_101);  add_103 = add_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_42: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_104, arg168_1, arg169_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_104 = arg168_1 = arg169_1 = None
    relu_42: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_42);  convolution_42 = None
    convert_element_type_84: "f32[768]" = torch.ops.prims.convert_element_type.default(arg388_1, torch.float32);  arg388_1 = None
    convert_element_type_85: "f32[768]" = torch.ops.prims.convert_element_type.default(arg389_1, torch.float32);  arg389_1 = None
    add_105: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_85, 1e-05);  convert_element_type_85 = None
    sqrt_42: "f32[768]" = torch.ops.aten.sqrt.default(add_105);  add_105 = None
    reciprocal_42: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_42);  sqrt_42 = None
    mul_126: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_42, 1);  reciprocal_42 = None
    unsqueeze_336: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_84, -1);  convert_element_type_84 = None
    unsqueeze_337: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, -1);  unsqueeze_336 = None
    unsqueeze_338: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_126, -1);  mul_126 = None
    unsqueeze_339: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, -1);  unsqueeze_338 = None
    sub_42: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_42, unsqueeze_337);  relu_42 = unsqueeze_337 = None
    mul_127: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_42, unsqueeze_339);  sub_42 = unsqueeze_339 = None
    unsqueeze_340: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg170_1, -1);  arg170_1 = None
    unsqueeze_341: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, -1);  unsqueeze_340 = None
    mul_128: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_127, unsqueeze_341);  mul_127 = unsqueeze_341 = None
    unsqueeze_342: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg171_1, -1);  arg171_1 = None
    unsqueeze_343: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, -1);  unsqueeze_342 = None
    add_106: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_128, unsqueeze_343);  mul_128 = unsqueeze_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_43: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_106, arg172_1, arg173_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  arg172_1 = arg173_1 = None
    relu_43: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_43);  convolution_43 = None
    convert_element_type_86: "f32[768]" = torch.ops.prims.convert_element_type.default(arg391_1, torch.float32);  arg391_1 = None
    convert_element_type_87: "f32[768]" = torch.ops.prims.convert_element_type.default(arg392_1, torch.float32);  arg392_1 = None
    add_107: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_87, 1e-05);  convert_element_type_87 = None
    sqrt_43: "f32[768]" = torch.ops.aten.sqrt.default(add_107);  add_107 = None
    reciprocal_43: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_43);  sqrt_43 = None
    mul_129: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_43, 1);  reciprocal_43 = None
    unsqueeze_344: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_86, -1);  convert_element_type_86 = None
    unsqueeze_345: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, -1);  unsqueeze_344 = None
    unsqueeze_346: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_129, -1);  mul_129 = None
    unsqueeze_347: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, -1);  unsqueeze_346 = None
    sub_43: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_43, unsqueeze_345);  relu_43 = unsqueeze_345 = None
    mul_130: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_43, unsqueeze_347);  sub_43 = unsqueeze_347 = None
    unsqueeze_348: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg174_1, -1);  arg174_1 = None
    unsqueeze_349: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, -1);  unsqueeze_348 = None
    mul_131: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_130, unsqueeze_349);  mul_130 = unsqueeze_349 = None
    unsqueeze_350: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg175_1, -1);  arg175_1 = None
    unsqueeze_351: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, -1);  unsqueeze_350 = None
    add_108: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_131, unsqueeze_351);  mul_131 = unsqueeze_351 = None
    add_109: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_108, add_106);  add_108 = add_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_44: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_109, arg176_1, arg177_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_109 = arg176_1 = arg177_1 = None
    relu_44: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_44);  convolution_44 = None
    convert_element_type_88: "f32[768]" = torch.ops.prims.convert_element_type.default(arg394_1, torch.float32);  arg394_1 = None
    convert_element_type_89: "f32[768]" = torch.ops.prims.convert_element_type.default(arg395_1, torch.float32);  arg395_1 = None
    add_110: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_89, 1e-05);  convert_element_type_89 = None
    sqrt_44: "f32[768]" = torch.ops.aten.sqrt.default(add_110);  add_110 = None
    reciprocal_44: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_44);  sqrt_44 = None
    mul_132: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_44, 1);  reciprocal_44 = None
    unsqueeze_352: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_88, -1);  convert_element_type_88 = None
    unsqueeze_353: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, -1);  unsqueeze_352 = None
    unsqueeze_354: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_132, -1);  mul_132 = None
    unsqueeze_355: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, -1);  unsqueeze_354 = None
    sub_44: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_44, unsqueeze_353);  relu_44 = unsqueeze_353 = None
    mul_133: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_44, unsqueeze_355);  sub_44 = unsqueeze_355 = None
    unsqueeze_356: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg178_1, -1);  arg178_1 = None
    unsqueeze_357: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, -1);  unsqueeze_356 = None
    mul_134: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_133, unsqueeze_357);  mul_133 = unsqueeze_357 = None
    unsqueeze_358: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg179_1, -1);  arg179_1 = None
    unsqueeze_359: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, -1);  unsqueeze_358 = None
    add_111: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_134, unsqueeze_359);  mul_134 = unsqueeze_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_45: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_111, arg180_1, arg181_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  arg180_1 = arg181_1 = None
    relu_45: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_45);  convolution_45 = None
    convert_element_type_90: "f32[768]" = torch.ops.prims.convert_element_type.default(arg397_1, torch.float32);  arg397_1 = None
    convert_element_type_91: "f32[768]" = torch.ops.prims.convert_element_type.default(arg398_1, torch.float32);  arg398_1 = None
    add_112: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_91, 1e-05);  convert_element_type_91 = None
    sqrt_45: "f32[768]" = torch.ops.aten.sqrt.default(add_112);  add_112 = None
    reciprocal_45: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_45);  sqrt_45 = None
    mul_135: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_45, 1);  reciprocal_45 = None
    unsqueeze_360: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_90, -1);  convert_element_type_90 = None
    unsqueeze_361: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, -1);  unsqueeze_360 = None
    unsqueeze_362: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_135, -1);  mul_135 = None
    unsqueeze_363: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, -1);  unsqueeze_362 = None
    sub_45: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_45, unsqueeze_361);  relu_45 = unsqueeze_361 = None
    mul_136: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_45, unsqueeze_363);  sub_45 = unsqueeze_363 = None
    unsqueeze_364: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg182_1, -1);  arg182_1 = None
    unsqueeze_365: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, -1);  unsqueeze_364 = None
    mul_137: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_136, unsqueeze_365);  mul_136 = unsqueeze_365 = None
    unsqueeze_366: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg183_1, -1);  arg183_1 = None
    unsqueeze_367: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, -1);  unsqueeze_366 = None
    add_113: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_137, unsqueeze_367);  mul_137 = unsqueeze_367 = None
    add_114: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_113, add_111);  add_113 = add_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_46: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_114, arg184_1, arg185_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_114 = arg184_1 = arg185_1 = None
    relu_46: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_46);  convolution_46 = None
    convert_element_type_92: "f32[768]" = torch.ops.prims.convert_element_type.default(arg400_1, torch.float32);  arg400_1 = None
    convert_element_type_93: "f32[768]" = torch.ops.prims.convert_element_type.default(arg401_1, torch.float32);  arg401_1 = None
    add_115: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_93, 1e-05);  convert_element_type_93 = None
    sqrt_46: "f32[768]" = torch.ops.aten.sqrt.default(add_115);  add_115 = None
    reciprocal_46: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_46);  sqrt_46 = None
    mul_138: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_46, 1);  reciprocal_46 = None
    unsqueeze_368: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_92, -1);  convert_element_type_92 = None
    unsqueeze_369: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, -1);  unsqueeze_368 = None
    unsqueeze_370: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_138, -1);  mul_138 = None
    unsqueeze_371: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, -1);  unsqueeze_370 = None
    sub_46: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_46, unsqueeze_369);  relu_46 = unsqueeze_369 = None
    mul_139: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_46, unsqueeze_371);  sub_46 = unsqueeze_371 = None
    unsqueeze_372: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg186_1, -1);  arg186_1 = None
    unsqueeze_373: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_372, -1);  unsqueeze_372 = None
    mul_140: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_139, unsqueeze_373);  mul_139 = unsqueeze_373 = None
    unsqueeze_374: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg187_1, -1);  arg187_1 = None
    unsqueeze_375: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, -1);  unsqueeze_374 = None
    add_116: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_140, unsqueeze_375);  mul_140 = unsqueeze_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_47: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_116, arg188_1, arg189_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  arg188_1 = arg189_1 = None
    relu_47: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_47);  convolution_47 = None
    convert_element_type_94: "f32[768]" = torch.ops.prims.convert_element_type.default(arg403_1, torch.float32);  arg403_1 = None
    convert_element_type_95: "f32[768]" = torch.ops.prims.convert_element_type.default(arg404_1, torch.float32);  arg404_1 = None
    add_117: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_95, 1e-05);  convert_element_type_95 = None
    sqrt_47: "f32[768]" = torch.ops.aten.sqrt.default(add_117);  add_117 = None
    reciprocal_47: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_47);  sqrt_47 = None
    mul_141: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_47, 1);  reciprocal_47 = None
    unsqueeze_376: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_94, -1);  convert_element_type_94 = None
    unsqueeze_377: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, -1);  unsqueeze_376 = None
    unsqueeze_378: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_141, -1);  mul_141 = None
    unsqueeze_379: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, -1);  unsqueeze_378 = None
    sub_47: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_47, unsqueeze_377);  relu_47 = unsqueeze_377 = None
    mul_142: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_47, unsqueeze_379);  sub_47 = unsqueeze_379 = None
    unsqueeze_380: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg190_1, -1);  arg190_1 = None
    unsqueeze_381: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, -1);  unsqueeze_380 = None
    mul_143: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_142, unsqueeze_381);  mul_142 = unsqueeze_381 = None
    unsqueeze_382: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg191_1, -1);  arg191_1 = None
    unsqueeze_383: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, -1);  unsqueeze_382 = None
    add_118: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_143, unsqueeze_383);  mul_143 = unsqueeze_383 = None
    add_119: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_118, add_116);  add_118 = add_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_48: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_119, arg192_1, arg193_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_119 = arg192_1 = arg193_1 = None
    relu_48: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_48);  convolution_48 = None
    convert_element_type_96: "f32[768]" = torch.ops.prims.convert_element_type.default(arg406_1, torch.float32);  arg406_1 = None
    convert_element_type_97: "f32[768]" = torch.ops.prims.convert_element_type.default(arg407_1, torch.float32);  arg407_1 = None
    add_120: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_97, 1e-05);  convert_element_type_97 = None
    sqrt_48: "f32[768]" = torch.ops.aten.sqrt.default(add_120);  add_120 = None
    reciprocal_48: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_48);  sqrt_48 = None
    mul_144: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_48, 1);  reciprocal_48 = None
    unsqueeze_384: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_96, -1);  convert_element_type_96 = None
    unsqueeze_385: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_384, -1);  unsqueeze_384 = None
    unsqueeze_386: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_144, -1);  mul_144 = None
    unsqueeze_387: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, -1);  unsqueeze_386 = None
    sub_48: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_48, unsqueeze_385);  relu_48 = unsqueeze_385 = None
    mul_145: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_48, unsqueeze_387);  sub_48 = unsqueeze_387 = None
    unsqueeze_388: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg194_1, -1);  arg194_1 = None
    unsqueeze_389: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, -1);  unsqueeze_388 = None
    mul_146: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_145, unsqueeze_389);  mul_145 = unsqueeze_389 = None
    unsqueeze_390: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg195_1, -1);  arg195_1 = None
    unsqueeze_391: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_390, -1);  unsqueeze_390 = None
    add_121: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_146, unsqueeze_391);  mul_146 = unsqueeze_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_49: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_121, arg196_1, arg197_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  arg196_1 = arg197_1 = None
    relu_49: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_49);  convolution_49 = None
    convert_element_type_98: "f32[768]" = torch.ops.prims.convert_element_type.default(arg409_1, torch.float32);  arg409_1 = None
    convert_element_type_99: "f32[768]" = torch.ops.prims.convert_element_type.default(arg410_1, torch.float32);  arg410_1 = None
    add_122: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_99, 1e-05);  convert_element_type_99 = None
    sqrt_49: "f32[768]" = torch.ops.aten.sqrt.default(add_122);  add_122 = None
    reciprocal_49: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_49);  sqrt_49 = None
    mul_147: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_49, 1);  reciprocal_49 = None
    unsqueeze_392: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_98, -1);  convert_element_type_98 = None
    unsqueeze_393: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, -1);  unsqueeze_392 = None
    unsqueeze_394: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_147, -1);  mul_147 = None
    unsqueeze_395: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, -1);  unsqueeze_394 = None
    sub_49: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_49, unsqueeze_393);  relu_49 = unsqueeze_393 = None
    mul_148: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_49, unsqueeze_395);  sub_49 = unsqueeze_395 = None
    unsqueeze_396: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg198_1, -1);  arg198_1 = None
    unsqueeze_397: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_396, -1);  unsqueeze_396 = None
    mul_149: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_148, unsqueeze_397);  mul_148 = unsqueeze_397 = None
    unsqueeze_398: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg199_1, -1);  arg199_1 = None
    unsqueeze_399: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, -1);  unsqueeze_398 = None
    add_123: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_149, unsqueeze_399);  mul_149 = unsqueeze_399 = None
    add_124: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_123, add_121);  add_123 = add_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_50: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_124, arg200_1, arg201_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_124 = arg200_1 = arg201_1 = None
    relu_50: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_50);  convolution_50 = None
    convert_element_type_100: "f32[768]" = torch.ops.prims.convert_element_type.default(arg412_1, torch.float32);  arg412_1 = None
    convert_element_type_101: "f32[768]" = torch.ops.prims.convert_element_type.default(arg413_1, torch.float32);  arg413_1 = None
    add_125: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_101, 1e-05);  convert_element_type_101 = None
    sqrt_50: "f32[768]" = torch.ops.aten.sqrt.default(add_125);  add_125 = None
    reciprocal_50: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_50);  sqrt_50 = None
    mul_150: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_50, 1);  reciprocal_50 = None
    unsqueeze_400: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_100, -1);  convert_element_type_100 = None
    unsqueeze_401: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, -1);  unsqueeze_400 = None
    unsqueeze_402: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_150, -1);  mul_150 = None
    unsqueeze_403: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, -1);  unsqueeze_402 = None
    sub_50: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_50, unsqueeze_401);  relu_50 = unsqueeze_401 = None
    mul_151: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_50, unsqueeze_403);  sub_50 = unsqueeze_403 = None
    unsqueeze_404: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg202_1, -1);  arg202_1 = None
    unsqueeze_405: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, -1);  unsqueeze_404 = None
    mul_152: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_151, unsqueeze_405);  mul_151 = unsqueeze_405 = None
    unsqueeze_406: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg203_1, -1);  arg203_1 = None
    unsqueeze_407: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, -1);  unsqueeze_406 = None
    add_126: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_152, unsqueeze_407);  mul_152 = unsqueeze_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_51: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_126, arg204_1, arg205_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  arg204_1 = arg205_1 = None
    relu_51: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_51);  convolution_51 = None
    convert_element_type_102: "f32[768]" = torch.ops.prims.convert_element_type.default(arg415_1, torch.float32);  arg415_1 = None
    convert_element_type_103: "f32[768]" = torch.ops.prims.convert_element_type.default(arg416_1, torch.float32);  arg416_1 = None
    add_127: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_103, 1e-05);  convert_element_type_103 = None
    sqrt_51: "f32[768]" = torch.ops.aten.sqrt.default(add_127);  add_127 = None
    reciprocal_51: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_51);  sqrt_51 = None
    mul_153: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_51, 1);  reciprocal_51 = None
    unsqueeze_408: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_102, -1);  convert_element_type_102 = None
    unsqueeze_409: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_408, -1);  unsqueeze_408 = None
    unsqueeze_410: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_153, -1);  mul_153 = None
    unsqueeze_411: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, -1);  unsqueeze_410 = None
    sub_51: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_51, unsqueeze_409);  relu_51 = unsqueeze_409 = None
    mul_154: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_51, unsqueeze_411);  sub_51 = unsqueeze_411 = None
    unsqueeze_412: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg206_1, -1);  arg206_1 = None
    unsqueeze_413: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, -1);  unsqueeze_412 = None
    mul_155: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_154, unsqueeze_413);  mul_154 = unsqueeze_413 = None
    unsqueeze_414: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg207_1, -1);  arg207_1 = None
    unsqueeze_415: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, -1);  unsqueeze_414 = None
    add_128: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_155, unsqueeze_415);  mul_155 = unsqueeze_415 = None
    add_129: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_128, add_126);  add_128 = add_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_52: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_129, arg208_1, arg209_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_129 = arg208_1 = arg209_1 = None
    relu_52: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_52);  convolution_52 = None
    convert_element_type_104: "f32[768]" = torch.ops.prims.convert_element_type.default(arg418_1, torch.float32);  arg418_1 = None
    convert_element_type_105: "f32[768]" = torch.ops.prims.convert_element_type.default(arg419_1, torch.float32);  arg419_1 = None
    add_130: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_105, 1e-05);  convert_element_type_105 = None
    sqrt_52: "f32[768]" = torch.ops.aten.sqrt.default(add_130);  add_130 = None
    reciprocal_52: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_52);  sqrt_52 = None
    mul_156: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_52, 1);  reciprocal_52 = None
    unsqueeze_416: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_104, -1);  convert_element_type_104 = None
    unsqueeze_417: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, -1);  unsqueeze_416 = None
    unsqueeze_418: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_156, -1);  mul_156 = None
    unsqueeze_419: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, -1);  unsqueeze_418 = None
    sub_52: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_52, unsqueeze_417);  relu_52 = unsqueeze_417 = None
    mul_157: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_52, unsqueeze_419);  sub_52 = unsqueeze_419 = None
    unsqueeze_420: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg210_1, -1);  arg210_1 = None
    unsqueeze_421: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_420, -1);  unsqueeze_420 = None
    mul_158: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_157, unsqueeze_421);  mul_157 = unsqueeze_421 = None
    unsqueeze_422: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg211_1, -1);  arg211_1 = None
    unsqueeze_423: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, -1);  unsqueeze_422 = None
    add_131: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_158, unsqueeze_423);  mul_158 = unsqueeze_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_53: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_131, arg212_1, arg213_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  arg212_1 = arg213_1 = None
    relu_53: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_53);  convolution_53 = None
    convert_element_type_106: "f32[768]" = torch.ops.prims.convert_element_type.default(arg421_1, torch.float32);  arg421_1 = None
    convert_element_type_107: "f32[768]" = torch.ops.prims.convert_element_type.default(arg422_1, torch.float32);  arg422_1 = None
    add_132: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_107, 1e-05);  convert_element_type_107 = None
    sqrt_53: "f32[768]" = torch.ops.aten.sqrt.default(add_132);  add_132 = None
    reciprocal_53: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_53);  sqrt_53 = None
    mul_159: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_53, 1);  reciprocal_53 = None
    unsqueeze_424: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_106, -1);  convert_element_type_106 = None
    unsqueeze_425: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_424, -1);  unsqueeze_424 = None
    unsqueeze_426: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_159, -1);  mul_159 = None
    unsqueeze_427: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, -1);  unsqueeze_426 = None
    sub_53: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_53, unsqueeze_425);  relu_53 = unsqueeze_425 = None
    mul_160: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_53, unsqueeze_427);  sub_53 = unsqueeze_427 = None
    unsqueeze_428: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg214_1, -1);  arg214_1 = None
    unsqueeze_429: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, -1);  unsqueeze_428 = None
    mul_161: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_160, unsqueeze_429);  mul_160 = unsqueeze_429 = None
    unsqueeze_430: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg215_1, -1);  arg215_1 = None
    unsqueeze_431: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, -1);  unsqueeze_430 = None
    add_133: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_161, unsqueeze_431);  mul_161 = unsqueeze_431 = None
    add_134: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_133, add_131);  add_133 = add_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_54: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_134, arg216_1, arg217_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_134 = arg216_1 = arg217_1 = None
    relu_54: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_54);  convolution_54 = None
    convert_element_type_108: "f32[768]" = torch.ops.prims.convert_element_type.default(arg424_1, torch.float32);  arg424_1 = None
    convert_element_type_109: "f32[768]" = torch.ops.prims.convert_element_type.default(arg425_1, torch.float32);  arg425_1 = None
    add_135: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_109, 1e-05);  convert_element_type_109 = None
    sqrt_54: "f32[768]" = torch.ops.aten.sqrt.default(add_135);  add_135 = None
    reciprocal_54: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_54);  sqrt_54 = None
    mul_162: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_54, 1);  reciprocal_54 = None
    unsqueeze_432: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_108, -1);  convert_element_type_108 = None
    unsqueeze_433: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_432, -1);  unsqueeze_432 = None
    unsqueeze_434: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_162, -1);  mul_162 = None
    unsqueeze_435: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, -1);  unsqueeze_434 = None
    sub_54: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_54, unsqueeze_433);  relu_54 = unsqueeze_433 = None
    mul_163: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_54, unsqueeze_435);  sub_54 = unsqueeze_435 = None
    unsqueeze_436: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg218_1, -1);  arg218_1 = None
    unsqueeze_437: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_436, -1);  unsqueeze_436 = None
    mul_164: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_163, unsqueeze_437);  mul_163 = unsqueeze_437 = None
    unsqueeze_438: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg219_1, -1);  arg219_1 = None
    unsqueeze_439: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_438, -1);  unsqueeze_438 = None
    add_136: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_164, unsqueeze_439);  mul_164 = unsqueeze_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_55: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_136, arg220_1, arg221_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  arg220_1 = arg221_1 = None
    relu_55: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_55);  convolution_55 = None
    convert_element_type_110: "f32[768]" = torch.ops.prims.convert_element_type.default(arg427_1, torch.float32);  arg427_1 = None
    convert_element_type_111: "f32[768]" = torch.ops.prims.convert_element_type.default(arg428_1, torch.float32);  arg428_1 = None
    add_137: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_111, 1e-05);  convert_element_type_111 = None
    sqrt_55: "f32[768]" = torch.ops.aten.sqrt.default(add_137);  add_137 = None
    reciprocal_55: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_55);  sqrt_55 = None
    mul_165: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_55, 1);  reciprocal_55 = None
    unsqueeze_440: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_110, -1);  convert_element_type_110 = None
    unsqueeze_441: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, -1);  unsqueeze_440 = None
    unsqueeze_442: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_165, -1);  mul_165 = None
    unsqueeze_443: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_442, -1);  unsqueeze_442 = None
    sub_55: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_55, unsqueeze_441);  relu_55 = unsqueeze_441 = None
    mul_166: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_55, unsqueeze_443);  sub_55 = unsqueeze_443 = None
    unsqueeze_444: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg222_1, -1);  arg222_1 = None
    unsqueeze_445: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_444, -1);  unsqueeze_444 = None
    mul_167: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_166, unsqueeze_445);  mul_166 = unsqueeze_445 = None
    unsqueeze_446: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg223_1, -1);  arg223_1 = None
    unsqueeze_447: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, -1);  unsqueeze_446 = None
    add_138: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_167, unsqueeze_447);  mul_167 = unsqueeze_447 = None
    add_139: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_138, add_136);  add_138 = add_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_56: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_139, arg224_1, arg225_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_139 = arg224_1 = arg225_1 = None
    relu_56: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_56);  convolution_56 = None
    convert_element_type_112: "f32[768]" = torch.ops.prims.convert_element_type.default(arg430_1, torch.float32);  arg430_1 = None
    convert_element_type_113: "f32[768]" = torch.ops.prims.convert_element_type.default(arg431_1, torch.float32);  arg431_1 = None
    add_140: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_113, 1e-05);  convert_element_type_113 = None
    sqrt_56: "f32[768]" = torch.ops.aten.sqrt.default(add_140);  add_140 = None
    reciprocal_56: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_56);  sqrt_56 = None
    mul_168: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_56, 1);  reciprocal_56 = None
    unsqueeze_448: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_112, -1);  convert_element_type_112 = None
    unsqueeze_449: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_448, -1);  unsqueeze_448 = None
    unsqueeze_450: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_168, -1);  mul_168 = None
    unsqueeze_451: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_450, -1);  unsqueeze_450 = None
    sub_56: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_56, unsqueeze_449);  relu_56 = unsqueeze_449 = None
    mul_169: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_56, unsqueeze_451);  sub_56 = unsqueeze_451 = None
    unsqueeze_452: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg226_1, -1);  arg226_1 = None
    unsqueeze_453: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, -1);  unsqueeze_452 = None
    mul_170: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_169, unsqueeze_453);  mul_169 = unsqueeze_453 = None
    unsqueeze_454: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg227_1, -1);  arg227_1 = None
    unsqueeze_455: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_454, -1);  unsqueeze_454 = None
    add_141: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_170, unsqueeze_455);  mul_170 = unsqueeze_455 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_57: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_141, arg228_1, arg229_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  arg228_1 = arg229_1 = None
    relu_57: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_57);  convolution_57 = None
    convert_element_type_114: "f32[768]" = torch.ops.prims.convert_element_type.default(arg433_1, torch.float32);  arg433_1 = None
    convert_element_type_115: "f32[768]" = torch.ops.prims.convert_element_type.default(arg434_1, torch.float32);  arg434_1 = None
    add_142: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_115, 1e-05);  convert_element_type_115 = None
    sqrt_57: "f32[768]" = torch.ops.aten.sqrt.default(add_142);  add_142 = None
    reciprocal_57: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_57);  sqrt_57 = None
    mul_171: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_57, 1);  reciprocal_57 = None
    unsqueeze_456: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_114, -1);  convert_element_type_114 = None
    unsqueeze_457: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_456, -1);  unsqueeze_456 = None
    unsqueeze_458: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_171, -1);  mul_171 = None
    unsqueeze_459: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, -1);  unsqueeze_458 = None
    sub_57: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_57, unsqueeze_457);  relu_57 = unsqueeze_457 = None
    mul_172: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_57, unsqueeze_459);  sub_57 = unsqueeze_459 = None
    unsqueeze_460: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg230_1, -1);  arg230_1 = None
    unsqueeze_461: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_460, -1);  unsqueeze_460 = None
    mul_173: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_172, unsqueeze_461);  mul_172 = unsqueeze_461 = None
    unsqueeze_462: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg231_1, -1);  arg231_1 = None
    unsqueeze_463: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_462, -1);  unsqueeze_462 = None
    add_143: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_173, unsqueeze_463);  mul_173 = unsqueeze_463 = None
    add_144: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_143, add_141);  add_143 = add_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_58: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_144, arg232_1, arg233_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_144 = arg232_1 = arg233_1 = None
    relu_58: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_58);  convolution_58 = None
    convert_element_type_116: "f32[768]" = torch.ops.prims.convert_element_type.default(arg436_1, torch.float32);  arg436_1 = None
    convert_element_type_117: "f32[768]" = torch.ops.prims.convert_element_type.default(arg437_1, torch.float32);  arg437_1 = None
    add_145: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_117, 1e-05);  convert_element_type_117 = None
    sqrt_58: "f32[768]" = torch.ops.aten.sqrt.default(add_145);  add_145 = None
    reciprocal_58: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_58);  sqrt_58 = None
    mul_174: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_58, 1);  reciprocal_58 = None
    unsqueeze_464: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_116, -1);  convert_element_type_116 = None
    unsqueeze_465: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, -1);  unsqueeze_464 = None
    unsqueeze_466: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_174, -1);  mul_174 = None
    unsqueeze_467: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_466, -1);  unsqueeze_466 = None
    sub_58: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_58, unsqueeze_465);  relu_58 = unsqueeze_465 = None
    mul_175: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_58, unsqueeze_467);  sub_58 = unsqueeze_467 = None
    unsqueeze_468: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg234_1, -1);  arg234_1 = None
    unsqueeze_469: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_468, -1);  unsqueeze_468 = None
    mul_176: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_175, unsqueeze_469);  mul_175 = unsqueeze_469 = None
    unsqueeze_470: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg235_1, -1);  arg235_1 = None
    unsqueeze_471: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, -1);  unsqueeze_470 = None
    add_146: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_176, unsqueeze_471);  mul_176 = unsqueeze_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_59: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_146, arg236_1, arg237_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  arg236_1 = arg237_1 = None
    relu_59: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_59);  convolution_59 = None
    convert_element_type_118: "f32[768]" = torch.ops.prims.convert_element_type.default(arg439_1, torch.float32);  arg439_1 = None
    convert_element_type_119: "f32[768]" = torch.ops.prims.convert_element_type.default(arg440_1, torch.float32);  arg440_1 = None
    add_147: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_119, 1e-05);  convert_element_type_119 = None
    sqrt_59: "f32[768]" = torch.ops.aten.sqrt.default(add_147);  add_147 = None
    reciprocal_59: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_59);  sqrt_59 = None
    mul_177: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_59, 1);  reciprocal_59 = None
    unsqueeze_472: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_118, -1);  convert_element_type_118 = None
    unsqueeze_473: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_472, -1);  unsqueeze_472 = None
    unsqueeze_474: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_177, -1);  mul_177 = None
    unsqueeze_475: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_474, -1);  unsqueeze_474 = None
    sub_59: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_59, unsqueeze_473);  relu_59 = unsqueeze_473 = None
    mul_178: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_59, unsqueeze_475);  sub_59 = unsqueeze_475 = None
    unsqueeze_476: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg238_1, -1);  arg238_1 = None
    unsqueeze_477: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, -1);  unsqueeze_476 = None
    mul_179: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_178, unsqueeze_477);  mul_178 = unsqueeze_477 = None
    unsqueeze_478: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg239_1, -1);  arg239_1 = None
    unsqueeze_479: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_478, -1);  unsqueeze_478 = None
    add_148: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_179, unsqueeze_479);  mul_179 = unsqueeze_479 = None
    add_149: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_148, add_146);  add_148 = add_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_60: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_149, arg240_1, arg241_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_149 = arg240_1 = arg241_1 = None
    relu_60: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_60);  convolution_60 = None
    convert_element_type_120: "f32[768]" = torch.ops.prims.convert_element_type.default(arg442_1, torch.float32);  arg442_1 = None
    convert_element_type_121: "f32[768]" = torch.ops.prims.convert_element_type.default(arg443_1, torch.float32);  arg443_1 = None
    add_150: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_121, 1e-05);  convert_element_type_121 = None
    sqrt_60: "f32[768]" = torch.ops.aten.sqrt.default(add_150);  add_150 = None
    reciprocal_60: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_60);  sqrt_60 = None
    mul_180: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_60, 1);  reciprocal_60 = None
    unsqueeze_480: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_120, -1);  convert_element_type_120 = None
    unsqueeze_481: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_480, -1);  unsqueeze_480 = None
    unsqueeze_482: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_180, -1);  mul_180 = None
    unsqueeze_483: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, -1);  unsqueeze_482 = None
    sub_60: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_60, unsqueeze_481);  relu_60 = unsqueeze_481 = None
    mul_181: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_60, unsqueeze_483);  sub_60 = unsqueeze_483 = None
    unsqueeze_484: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg242_1, -1);  arg242_1 = None
    unsqueeze_485: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_484, -1);  unsqueeze_484 = None
    mul_182: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_181, unsqueeze_485);  mul_181 = unsqueeze_485 = None
    unsqueeze_486: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg243_1, -1);  arg243_1 = None
    unsqueeze_487: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_486, -1);  unsqueeze_486 = None
    add_151: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_182, unsqueeze_487);  mul_182 = unsqueeze_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_61: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_151, arg244_1, arg245_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  arg244_1 = arg245_1 = None
    relu_61: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_61);  convolution_61 = None
    convert_element_type_122: "f32[768]" = torch.ops.prims.convert_element_type.default(arg445_1, torch.float32);  arg445_1 = None
    convert_element_type_123: "f32[768]" = torch.ops.prims.convert_element_type.default(arg446_1, torch.float32);  arg446_1 = None
    add_152: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_123, 1e-05);  convert_element_type_123 = None
    sqrt_61: "f32[768]" = torch.ops.aten.sqrt.default(add_152);  add_152 = None
    reciprocal_61: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_61);  sqrt_61 = None
    mul_183: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_61, 1);  reciprocal_61 = None
    unsqueeze_488: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_122, -1);  convert_element_type_122 = None
    unsqueeze_489: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, -1);  unsqueeze_488 = None
    unsqueeze_490: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_183, -1);  mul_183 = None
    unsqueeze_491: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_490, -1);  unsqueeze_490 = None
    sub_61: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_61, unsqueeze_489);  relu_61 = unsqueeze_489 = None
    mul_184: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_61, unsqueeze_491);  sub_61 = unsqueeze_491 = None
    unsqueeze_492: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg246_1, -1);  arg246_1 = None
    unsqueeze_493: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_492, -1);  unsqueeze_492 = None
    mul_185: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_184, unsqueeze_493);  mul_184 = unsqueeze_493 = None
    unsqueeze_494: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg247_1, -1);  arg247_1 = None
    unsqueeze_495: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, -1);  unsqueeze_494 = None
    add_153: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_185, unsqueeze_495);  mul_185 = unsqueeze_495 = None
    add_154: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_153, add_151);  add_153 = add_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_62: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_154, arg248_1, arg249_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_154 = arg248_1 = arg249_1 = None
    relu_62: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_62);  convolution_62 = None
    convert_element_type_124: "f32[768]" = torch.ops.prims.convert_element_type.default(arg448_1, torch.float32);  arg448_1 = None
    convert_element_type_125: "f32[768]" = torch.ops.prims.convert_element_type.default(arg449_1, torch.float32);  arg449_1 = None
    add_155: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_125, 1e-05);  convert_element_type_125 = None
    sqrt_62: "f32[768]" = torch.ops.aten.sqrt.default(add_155);  add_155 = None
    reciprocal_62: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_62);  sqrt_62 = None
    mul_186: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_62, 1);  reciprocal_62 = None
    unsqueeze_496: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_124, -1);  convert_element_type_124 = None
    unsqueeze_497: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_496, -1);  unsqueeze_496 = None
    unsqueeze_498: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_186, -1);  mul_186 = None
    unsqueeze_499: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_498, -1);  unsqueeze_498 = None
    sub_62: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_62, unsqueeze_497);  relu_62 = unsqueeze_497 = None
    mul_187: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_62, unsqueeze_499);  sub_62 = unsqueeze_499 = None
    unsqueeze_500: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg250_1, -1);  arg250_1 = None
    unsqueeze_501: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_500, -1);  unsqueeze_500 = None
    mul_188: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_187, unsqueeze_501);  mul_187 = unsqueeze_501 = None
    unsqueeze_502: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg251_1, -1);  arg251_1 = None
    unsqueeze_503: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_502, -1);  unsqueeze_502 = None
    add_156: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_188, unsqueeze_503);  mul_188 = unsqueeze_503 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_63: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_156, arg252_1, arg253_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  arg252_1 = arg253_1 = None
    relu_63: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_63);  convolution_63 = None
    convert_element_type_126: "f32[768]" = torch.ops.prims.convert_element_type.default(arg451_1, torch.float32);  arg451_1 = None
    convert_element_type_127: "f32[768]" = torch.ops.prims.convert_element_type.default(arg452_1, torch.float32);  arg452_1 = None
    add_157: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_127, 1e-05);  convert_element_type_127 = None
    sqrt_63: "f32[768]" = torch.ops.aten.sqrt.default(add_157);  add_157 = None
    reciprocal_63: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_63);  sqrt_63 = None
    mul_189: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_63, 1);  reciprocal_63 = None
    unsqueeze_504: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_126, -1);  convert_element_type_126 = None
    unsqueeze_505: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_504, -1);  unsqueeze_504 = None
    unsqueeze_506: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_189, -1);  mul_189 = None
    unsqueeze_507: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_506, -1);  unsqueeze_506 = None
    sub_63: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_63, unsqueeze_505);  relu_63 = unsqueeze_505 = None
    mul_190: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_63, unsqueeze_507);  sub_63 = unsqueeze_507 = None
    unsqueeze_508: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg254_1, -1);  arg254_1 = None
    unsqueeze_509: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_508, -1);  unsqueeze_508 = None
    mul_191: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_190, unsqueeze_509);  mul_190 = unsqueeze_509 = None
    unsqueeze_510: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg255_1, -1);  arg255_1 = None
    unsqueeze_511: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_510, -1);  unsqueeze_510 = None
    add_158: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_191, unsqueeze_511);  mul_191 = unsqueeze_511 = None
    add_159: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_158, add_156);  add_158 = add_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_64: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_159, arg256_1, arg257_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_159 = arg256_1 = arg257_1 = None
    relu_64: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_64);  convolution_64 = None
    convert_element_type_128: "f32[768]" = torch.ops.prims.convert_element_type.default(arg454_1, torch.float32);  arg454_1 = None
    convert_element_type_129: "f32[768]" = torch.ops.prims.convert_element_type.default(arg455_1, torch.float32);  arg455_1 = None
    add_160: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_129, 1e-05);  convert_element_type_129 = None
    sqrt_64: "f32[768]" = torch.ops.aten.sqrt.default(add_160);  add_160 = None
    reciprocal_64: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_64);  sqrt_64 = None
    mul_192: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_64, 1);  reciprocal_64 = None
    unsqueeze_512: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_128, -1);  convert_element_type_128 = None
    unsqueeze_513: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_512, -1);  unsqueeze_512 = None
    unsqueeze_514: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_192, -1);  mul_192 = None
    unsqueeze_515: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_514, -1);  unsqueeze_514 = None
    sub_64: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_64, unsqueeze_513);  relu_64 = unsqueeze_513 = None
    mul_193: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_515);  sub_64 = unsqueeze_515 = None
    unsqueeze_516: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg258_1, -1);  arg258_1 = None
    unsqueeze_517: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_516, -1);  unsqueeze_516 = None
    mul_194: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_193, unsqueeze_517);  mul_193 = unsqueeze_517 = None
    unsqueeze_518: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg259_1, -1);  arg259_1 = None
    unsqueeze_519: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_518, -1);  unsqueeze_518 = None
    add_161: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_194, unsqueeze_519);  mul_194 = unsqueeze_519 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean: "f32[8, 768, 1, 1]" = torch.ops.aten.mean.dim(add_161, [-1, -2], True);  add_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view: "f32[8, 768]" = torch.ops.aten.view.default(mean, [8, 768]);  mean = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:94, code: x = self.head_drop(x)
    clone: "f32[8, 768]" = torch.ops.aten.clone.default(view);  view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:95, code: return x if pre_logits else self.head(x)
    permute: "f32[768, 1000]" = torch.ops.aten.permute.default(arg260_1, [1, 0]);  arg260_1 = None
    addmm: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg261_1, clone, permute);  arg261_1 = clone = permute = None
    return (addmm,)
    