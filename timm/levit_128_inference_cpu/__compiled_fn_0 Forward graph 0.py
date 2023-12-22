from __future__ import annotations



def forward(self, arg0_1: "f32[4, 196]", arg1_1: "f32[4, 196]", arg2_1: "f32[4, 196]", arg3_1: "f32[4, 196]", arg4_1: "f32[8, 196]", arg5_1: "f32[8, 49]", arg6_1: "f32[8, 49]", arg7_1: "f32[8, 49]", arg8_1: "f32[8, 49]", arg9_1: "f32[16, 49]", arg10_1: "f32[12, 16]", arg11_1: "f32[12, 16]", arg12_1: "f32[12, 16]", arg13_1: "f32[12, 16]", arg14_1: "f32[16, 3, 3, 3]", arg15_1: "f32[16]", arg16_1: "f32[16]", arg17_1: "f32[32, 16, 3, 3]", arg18_1: "f32[32]", arg19_1: "f32[32]", arg20_1: "f32[64, 32, 3, 3]", arg21_1: "f32[64]", arg22_1: "f32[64]", arg23_1: "f32[128, 64, 3, 3]", arg24_1: "f32[128]", arg25_1: "f32[128]", arg26_1: "f32[256, 128]", arg27_1: "f32[256]", arg28_1: "f32[256]", arg29_1: "f32[128, 128]", arg30_1: "f32[128]", arg31_1: "f32[128]", arg32_1: "f32[256, 128]", arg33_1: "f32[256]", arg34_1: "f32[256]", arg35_1: "f32[128, 256]", arg36_1: "f32[128]", arg37_1: "f32[128]", arg38_1: "f32[256, 128]", arg39_1: "f32[256]", arg40_1: "f32[256]", arg41_1: "f32[128, 128]", arg42_1: "f32[128]", arg43_1: "f32[128]", arg44_1: "f32[256, 128]", arg45_1: "f32[256]", arg46_1: "f32[256]", arg47_1: "f32[128, 256]", arg48_1: "f32[128]", arg49_1: "f32[128]", arg50_1: "f32[256, 128]", arg51_1: "f32[256]", arg52_1: "f32[256]", arg53_1: "f32[128, 128]", arg54_1: "f32[128]", arg55_1: "f32[128]", arg56_1: "f32[256, 128]", arg57_1: "f32[256]", arg58_1: "f32[256]", arg59_1: "f32[128, 256]", arg60_1: "f32[128]", arg61_1: "f32[128]", arg62_1: "f32[256, 128]", arg63_1: "f32[256]", arg64_1: "f32[256]", arg65_1: "f32[128, 128]", arg66_1: "f32[128]", arg67_1: "f32[128]", arg68_1: "f32[256, 128]", arg69_1: "f32[256]", arg70_1: "f32[256]", arg71_1: "f32[128, 256]", arg72_1: "f32[128]", arg73_1: "f32[128]", arg74_1: "f32[640, 128]", arg75_1: "f32[640]", arg76_1: "f32[640]", arg77_1: "f32[128, 128]", arg78_1: "f32[128]", arg79_1: "f32[128]", arg80_1: "f32[256, 512]", arg81_1: "f32[256]", arg82_1: "f32[256]", arg83_1: "f32[512, 256]", arg84_1: "f32[512]", arg85_1: "f32[512]", arg86_1: "f32[256, 512]", arg87_1: "f32[256]", arg88_1: "f32[256]", arg89_1: "f32[512, 256]", arg90_1: "f32[512]", arg91_1: "f32[512]", arg92_1: "f32[256, 256]", arg93_1: "f32[256]", arg94_1: "f32[256]", arg95_1: "f32[512, 256]", arg96_1: "f32[512]", arg97_1: "f32[512]", arg98_1: "f32[256, 512]", arg99_1: "f32[256]", arg100_1: "f32[256]", arg101_1: "f32[512, 256]", arg102_1: "f32[512]", arg103_1: "f32[512]", arg104_1: "f32[256, 256]", arg105_1: "f32[256]", arg106_1: "f32[256]", arg107_1: "f32[512, 256]", arg108_1: "f32[512]", arg109_1: "f32[512]", arg110_1: "f32[256, 512]", arg111_1: "f32[256]", arg112_1: "f32[256]", arg113_1: "f32[512, 256]", arg114_1: "f32[512]", arg115_1: "f32[512]", arg116_1: "f32[256, 256]", arg117_1: "f32[256]", arg118_1: "f32[256]", arg119_1: "f32[512, 256]", arg120_1: "f32[512]", arg121_1: "f32[512]", arg122_1: "f32[256, 512]", arg123_1: "f32[256]", arg124_1: "f32[256]", arg125_1: "f32[512, 256]", arg126_1: "f32[512]", arg127_1: "f32[512]", arg128_1: "f32[256, 256]", arg129_1: "f32[256]", arg130_1: "f32[256]", arg131_1: "f32[512, 256]", arg132_1: "f32[512]", arg133_1: "f32[512]", arg134_1: "f32[256, 512]", arg135_1: "f32[256]", arg136_1: "f32[256]", arg137_1: "f32[1280, 256]", arg138_1: "f32[1280]", arg139_1: "f32[1280]", arg140_1: "f32[256, 256]", arg141_1: "f32[256]", arg142_1: "f32[256]", arg143_1: "f32[384, 1024]", arg144_1: "f32[384]", arg145_1: "f32[384]", arg146_1: "f32[768, 384]", arg147_1: "f32[768]", arg148_1: "f32[768]", arg149_1: "f32[384, 768]", arg150_1: "f32[384]", arg151_1: "f32[384]", arg152_1: "f32[768, 384]", arg153_1: "f32[768]", arg154_1: "f32[768]", arg155_1: "f32[384, 384]", arg156_1: "f32[384]", arg157_1: "f32[384]", arg158_1: "f32[768, 384]", arg159_1: "f32[768]", arg160_1: "f32[768]", arg161_1: "f32[384, 768]", arg162_1: "f32[384]", arg163_1: "f32[384]", arg164_1: "f32[768, 384]", arg165_1: "f32[768]", arg166_1: "f32[768]", arg167_1: "f32[384, 384]", arg168_1: "f32[384]", arg169_1: "f32[384]", arg170_1: "f32[768, 384]", arg171_1: "f32[768]", arg172_1: "f32[768]", arg173_1: "f32[384, 768]", arg174_1: "f32[384]", arg175_1: "f32[384]", arg176_1: "f32[768, 384]", arg177_1: "f32[768]", arg178_1: "f32[768]", arg179_1: "f32[384, 384]", arg180_1: "f32[384]", arg181_1: "f32[384]", arg182_1: "f32[768, 384]", arg183_1: "f32[768]", arg184_1: "f32[768]", arg185_1: "f32[384, 768]", arg186_1: "f32[384]", arg187_1: "f32[384]", arg188_1: "f32[768, 384]", arg189_1: "f32[768]", arg190_1: "f32[768]", arg191_1: "f32[384, 384]", arg192_1: "f32[384]", arg193_1: "f32[384]", arg194_1: "f32[768, 384]", arg195_1: "f32[768]", arg196_1: "f32[768]", arg197_1: "f32[384, 768]", arg198_1: "f32[384]", arg199_1: "f32[384]", arg200_1: "f32[384]", arg201_1: "f32[384]", arg202_1: "f32[1000, 384]", arg203_1: "f32[1000]", arg204_1: "f32[384]", arg205_1: "f32[384]", arg206_1: "f32[1000, 384]", arg207_1: "f32[1000]", arg208_1: "i64[196, 196]", arg209_1: "i64[196, 196]", arg210_1: "i64[196, 196]", arg211_1: "i64[196, 196]", arg212_1: "i64[49, 196]", arg213_1: "i64[49, 49]", arg214_1: "i64[49, 49]", arg215_1: "i64[49, 49]", arg216_1: "i64[49, 49]", arg217_1: "i64[16, 49]", arg218_1: "i64[16, 16]", arg219_1: "i64[16, 16]", arg220_1: "i64[16, 16]", arg221_1: "i64[16, 16]", arg222_1: "f32[16]", arg223_1: "f32[16]", arg224_1: "i64[]", arg225_1: "f32[32]", arg226_1: "f32[32]", arg227_1: "i64[]", arg228_1: "f32[64]", arg229_1: "f32[64]", arg230_1: "i64[]", arg231_1: "f32[128]", arg232_1: "f32[128]", arg233_1: "i64[]", arg234_1: "f32[256]", arg235_1: "f32[256]", arg236_1: "i64[]", arg237_1: "f32[128]", arg238_1: "f32[128]", arg239_1: "i64[]", arg240_1: "f32[256]", arg241_1: "f32[256]", arg242_1: "i64[]", arg243_1: "f32[128]", arg244_1: "f32[128]", arg245_1: "i64[]", arg246_1: "f32[256]", arg247_1: "f32[256]", arg248_1: "i64[]", arg249_1: "f32[128]", arg250_1: "f32[128]", arg251_1: "i64[]", arg252_1: "f32[256]", arg253_1: "f32[256]", arg254_1: "i64[]", arg255_1: "f32[128]", arg256_1: "f32[128]", arg257_1: "i64[]", arg258_1: "f32[256]", arg259_1: "f32[256]", arg260_1: "i64[]", arg261_1: "f32[128]", arg262_1: "f32[128]", arg263_1: "i64[]", arg264_1: "f32[256]", arg265_1: "f32[256]", arg266_1: "i64[]", arg267_1: "f32[128]", arg268_1: "f32[128]", arg269_1: "i64[]", arg270_1: "f32[256]", arg271_1: "f32[256]", arg272_1: "i64[]", arg273_1: "f32[128]", arg274_1: "f32[128]", arg275_1: "i64[]", arg276_1: "f32[256]", arg277_1: "f32[256]", arg278_1: "i64[]", arg279_1: "f32[128]", arg280_1: "f32[128]", arg281_1: "i64[]", arg282_1: "f32[640]", arg283_1: "f32[640]", arg284_1: "i64[]", arg285_1: "f32[128]", arg286_1: "f32[128]", arg287_1: "i64[]", arg288_1: "f32[256]", arg289_1: "f32[256]", arg290_1: "i64[]", arg291_1: "f32[512]", arg292_1: "f32[512]", arg293_1: "i64[]", arg294_1: "f32[256]", arg295_1: "f32[256]", arg296_1: "i64[]", arg297_1: "f32[512]", arg298_1: "f32[512]", arg299_1: "i64[]", arg300_1: "f32[256]", arg301_1: "f32[256]", arg302_1: "i64[]", arg303_1: "f32[512]", arg304_1: "f32[512]", arg305_1: "i64[]", arg306_1: "f32[256]", arg307_1: "f32[256]", arg308_1: "i64[]", arg309_1: "f32[512]", arg310_1: "f32[512]", arg311_1: "i64[]", arg312_1: "f32[256]", arg313_1: "f32[256]", arg314_1: "i64[]", arg315_1: "f32[512]", arg316_1: "f32[512]", arg317_1: "i64[]", arg318_1: "f32[256]", arg319_1: "f32[256]", arg320_1: "i64[]", arg321_1: "f32[512]", arg322_1: "f32[512]", arg323_1: "i64[]", arg324_1: "f32[256]", arg325_1: "f32[256]", arg326_1: "i64[]", arg327_1: "f32[512]", arg328_1: "f32[512]", arg329_1: "i64[]", arg330_1: "f32[256]", arg331_1: "f32[256]", arg332_1: "i64[]", arg333_1: "f32[512]", arg334_1: "f32[512]", arg335_1: "i64[]", arg336_1: "f32[256]", arg337_1: "f32[256]", arg338_1: "i64[]", arg339_1: "f32[512]", arg340_1: "f32[512]", arg341_1: "i64[]", arg342_1: "f32[256]", arg343_1: "f32[256]", arg344_1: "i64[]", arg345_1: "f32[1280]", arg346_1: "f32[1280]", arg347_1: "i64[]", arg348_1: "f32[256]", arg349_1: "f32[256]", arg350_1: "i64[]", arg351_1: "f32[384]", arg352_1: "f32[384]", arg353_1: "i64[]", arg354_1: "f32[768]", arg355_1: "f32[768]", arg356_1: "i64[]", arg357_1: "f32[384]", arg358_1: "f32[384]", arg359_1: "i64[]", arg360_1: "f32[768]", arg361_1: "f32[768]", arg362_1: "i64[]", arg363_1: "f32[384]", arg364_1: "f32[384]", arg365_1: "i64[]", arg366_1: "f32[768]", arg367_1: "f32[768]", arg368_1: "i64[]", arg369_1: "f32[384]", arg370_1: "f32[384]", arg371_1: "i64[]", arg372_1: "f32[768]", arg373_1: "f32[768]", arg374_1: "i64[]", arg375_1: "f32[384]", arg376_1: "f32[384]", arg377_1: "i64[]", arg378_1: "f32[768]", arg379_1: "f32[768]", arg380_1: "i64[]", arg381_1: "f32[384]", arg382_1: "f32[384]", arg383_1: "i64[]", arg384_1: "f32[768]", arg385_1: "f32[768]", arg386_1: "i64[]", arg387_1: "f32[384]", arg388_1: "f32[384]", arg389_1: "i64[]", arg390_1: "f32[768]", arg391_1: "f32[768]", arg392_1: "i64[]", arg393_1: "f32[384]", arg394_1: "f32[384]", arg395_1: "i64[]", arg396_1: "f32[768]", arg397_1: "f32[768]", arg398_1: "i64[]", arg399_1: "f32[384]", arg400_1: "f32[384]", arg401_1: "i64[]", arg402_1: "f32[768]", arg403_1: "f32[768]", arg404_1: "i64[]", arg405_1: "f32[384]", arg406_1: "f32[384]", arg407_1: "i64[]", arg408_1: "f32[384]", arg409_1: "f32[384]", arg410_1: "i64[]", arg411_1: "f32[384]", arg412_1: "f32[384]", arg413_1: "i64[]", arg414_1: "f32[8, 3, 224, 224]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:65, code: return self.bn(self.linear(x))
    convolution: "f32[8, 16, 112, 112]" = torch.ops.aten.convolution.default(arg414_1, arg14_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg414_1 = arg14_1 = None
    convert_element_type: "f32[16]" = torch.ops.prims.convert_element_type.default(arg222_1, torch.float32);  arg222_1 = None
    convert_element_type_1: "f32[16]" = torch.ops.prims.convert_element_type.default(arg223_1, torch.float32);  arg223_1 = None
    add: "f32[16]" = torch.ops.aten.add.Tensor(convert_element_type_1, 1e-05);  convert_element_type_1 = None
    sqrt: "f32[16]" = torch.ops.aten.sqrt.default(add);  add = None
    reciprocal: "f32[16]" = torch.ops.aten.reciprocal.default(sqrt);  sqrt = None
    mul: "f32[16]" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
    unsqueeze: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type, -1);  convert_element_type = None
    unsqueeze_1: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    unsqueeze_2: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(mul, -1);  mul = None
    unsqueeze_3: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    sub: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1);  convolution = unsqueeze_1 = None
    mul_1: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub, unsqueeze_3);  sub = unsqueeze_3 = None
    unsqueeze_4: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
    unsqueeze_5: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_2: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None
    unsqueeze_6: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg16_1, -1);  arg16_1 = None
    unsqueeze_7: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_1: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(mul_2, unsqueeze_7);  mul_2 = unsqueeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:637, code: x = self.stem(x)
    add_2: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(add_1, 3)
    clamp_min: "f32[8, 16, 112, 112]" = torch.ops.aten.clamp_min.default(add_2, 0);  add_2 = None
    clamp_max: "f32[8, 16, 112, 112]" = torch.ops.aten.clamp_max.default(clamp_min, 6);  clamp_min = None
    mul_3: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(add_1, clamp_max);  add_1 = clamp_max = None
    div: "f32[8, 16, 112, 112]" = torch.ops.aten.div.Tensor(mul_3, 6);  mul_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:65, code: return self.bn(self.linear(x))
    convolution_1: "f32[8, 32, 56, 56]" = torch.ops.aten.convolution.default(div, arg17_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  div = arg17_1 = None
    convert_element_type_2: "f32[32]" = torch.ops.prims.convert_element_type.default(arg225_1, torch.float32);  arg225_1 = None
    convert_element_type_3: "f32[32]" = torch.ops.prims.convert_element_type.default(arg226_1, torch.float32);  arg226_1 = None
    add_3: "f32[32]" = torch.ops.aten.add.Tensor(convert_element_type_3, 1e-05);  convert_element_type_3 = None
    sqrt_1: "f32[32]" = torch.ops.aten.sqrt.default(add_3);  add_3 = None
    reciprocal_1: "f32[32]" = torch.ops.aten.reciprocal.default(sqrt_1);  sqrt_1 = None
    mul_4: "f32[32]" = torch.ops.aten.mul.Tensor(reciprocal_1, 1);  reciprocal_1 = None
    unsqueeze_8: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, -1);  convert_element_type_2 = None
    unsqueeze_9: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    unsqueeze_10: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(mul_4, -1);  mul_4 = None
    unsqueeze_11: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    sub_1: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_9);  convolution_1 = unsqueeze_9 = None
    mul_5: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_1, unsqueeze_11);  sub_1 = unsqueeze_11 = None
    unsqueeze_12: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg18_1, -1);  arg18_1 = None
    unsqueeze_13: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_6: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(mul_5, unsqueeze_13);  mul_5 = unsqueeze_13 = None
    unsqueeze_14: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg19_1, -1);  arg19_1 = None
    unsqueeze_15: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_4: "f32[8, 32, 56, 56]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_15);  mul_6 = unsqueeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:637, code: x = self.stem(x)
    add_5: "f32[8, 32, 56, 56]" = torch.ops.aten.add.Tensor(add_4, 3)
    clamp_min_1: "f32[8, 32, 56, 56]" = torch.ops.aten.clamp_min.default(add_5, 0);  add_5 = None
    clamp_max_1: "f32[8, 32, 56, 56]" = torch.ops.aten.clamp_max.default(clamp_min_1, 6);  clamp_min_1 = None
    mul_7: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(add_4, clamp_max_1);  add_4 = clamp_max_1 = None
    div_1: "f32[8, 32, 56, 56]" = torch.ops.aten.div.Tensor(mul_7, 6);  mul_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:65, code: return self.bn(self.linear(x))
    convolution_2: "f32[8, 64, 28, 28]" = torch.ops.aten.convolution.default(div_1, arg20_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  div_1 = arg20_1 = None
    convert_element_type_4: "f32[64]" = torch.ops.prims.convert_element_type.default(arg228_1, torch.float32);  arg228_1 = None
    convert_element_type_5: "f32[64]" = torch.ops.prims.convert_element_type.default(arg229_1, torch.float32);  arg229_1 = None
    add_6: "f32[64]" = torch.ops.aten.add.Tensor(convert_element_type_5, 1e-05);  convert_element_type_5 = None
    sqrt_2: "f32[64]" = torch.ops.aten.sqrt.default(add_6);  add_6 = None
    reciprocal_2: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_2);  sqrt_2 = None
    mul_8: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_2, 1);  reciprocal_2 = None
    unsqueeze_16: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_4, -1);  convert_element_type_4 = None
    unsqueeze_17: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    unsqueeze_18: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_8, -1);  mul_8 = None
    unsqueeze_19: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    sub_2: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_17);  convolution_2 = unsqueeze_17 = None
    mul_9: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_2, unsqueeze_19);  sub_2 = unsqueeze_19 = None
    unsqueeze_20: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg21_1, -1);  arg21_1 = None
    unsqueeze_21: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_10: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(mul_9, unsqueeze_21);  mul_9 = unsqueeze_21 = None
    unsqueeze_22: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg22_1, -1);  arg22_1 = None
    unsqueeze_23: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_7: "f32[8, 64, 28, 28]" = torch.ops.aten.add.Tensor(mul_10, unsqueeze_23);  mul_10 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:637, code: x = self.stem(x)
    add_8: "f32[8, 64, 28, 28]" = torch.ops.aten.add.Tensor(add_7, 3)
    clamp_min_2: "f32[8, 64, 28, 28]" = torch.ops.aten.clamp_min.default(add_8, 0);  add_8 = None
    clamp_max_2: "f32[8, 64, 28, 28]" = torch.ops.aten.clamp_max.default(clamp_min_2, 6);  clamp_min_2 = None
    mul_11: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(add_7, clamp_max_2);  add_7 = clamp_max_2 = None
    div_2: "f32[8, 64, 28, 28]" = torch.ops.aten.div.Tensor(mul_11, 6);  mul_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:65, code: return self.bn(self.linear(x))
    convolution_3: "f32[8, 128, 14, 14]" = torch.ops.aten.convolution.default(div_2, arg23_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  div_2 = arg23_1 = None
    convert_element_type_6: "f32[128]" = torch.ops.prims.convert_element_type.default(arg231_1, torch.float32);  arg231_1 = None
    convert_element_type_7: "f32[128]" = torch.ops.prims.convert_element_type.default(arg232_1, torch.float32);  arg232_1 = None
    add_9: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_7, 1e-05);  convert_element_type_7 = None
    sqrt_3: "f32[128]" = torch.ops.aten.sqrt.default(add_9);  add_9 = None
    reciprocal_3: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_3);  sqrt_3 = None
    mul_12: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_3, 1);  reciprocal_3 = None
    unsqueeze_24: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_6, -1);  convert_element_type_6 = None
    unsqueeze_25: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
    unsqueeze_26: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_12, -1);  mul_12 = None
    unsqueeze_27: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
    sub_3: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_25);  convolution_3 = unsqueeze_25 = None
    mul_13: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_3, unsqueeze_27);  sub_3 = unsqueeze_27 = None
    unsqueeze_28: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg24_1, -1);  arg24_1 = None
    unsqueeze_29: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    mul_14: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_13, unsqueeze_29);  mul_13 = unsqueeze_29 = None
    unsqueeze_30: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg25_1, -1);  arg25_1 = None
    unsqueeze_31: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
    add_10: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_14, unsqueeze_31);  mul_14 = unsqueeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:639, code: x = x.flatten(2).transpose(1, 2)
    view: "f32[8, 128, 196]" = torch.ops.aten.view.default(add_10, [8, 128, 196]);  add_10 = None
    permute: "f32[8, 196, 128]" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_1: "f32[128, 256]" = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
    clone: "f32[8, 196, 128]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format)
    view_1: "f32[1568, 128]" = torch.ops.aten.view.default(clone, [1568, 128]);  clone = None
    mm: "f32[1568, 256]" = torch.ops.aten.mm.default(view_1, permute_1);  view_1 = permute_1 = None
    view_2: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm, [8, 196, 256]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_3: "f32[1568, 256]" = torch.ops.aten.view.default(view_2, [1568, 256]);  view_2 = None
    convert_element_type_8: "f32[256]" = torch.ops.prims.convert_element_type.default(arg234_1, torch.float32);  arg234_1 = None
    convert_element_type_9: "f32[256]" = torch.ops.prims.convert_element_type.default(arg235_1, torch.float32);  arg235_1 = None
    add_11: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_9, 1e-05);  convert_element_type_9 = None
    sqrt_4: "f32[256]" = torch.ops.aten.sqrt.default(add_11);  add_11 = None
    reciprocal_4: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_4);  sqrt_4 = None
    mul_15: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_4, 1);  reciprocal_4 = None
    sub_4: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_3, convert_element_type_8);  view_3 = convert_element_type_8 = None
    mul_16: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_4, mul_15);  sub_4 = mul_15 = None
    mul_17: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(mul_16, arg27_1);  mul_16 = arg27_1 = None
    add_12: "f32[1568, 256]" = torch.ops.aten.add.Tensor(mul_17, arg28_1);  mul_17 = arg28_1 = None
    view_4: "f32[8, 196, 256]" = torch.ops.aten.view.default(add_12, [8, 196, 256]);  add_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    view_5: "f32[8, 196, 4, 64]" = torch.ops.aten.view.default(view_4, [8, 196, 4, -1]);  view_4 = None
    split_with_sizes = torch.ops.aten.split_with_sizes.default(view_5, [16, 16, 32], 3);  view_5 = None
    getitem: "f32[8, 196, 4, 16]" = split_with_sizes[0]
    getitem_1: "f32[8, 196, 4, 16]" = split_with_sizes[1]
    getitem_2: "f32[8, 196, 4, 32]" = split_with_sizes[2];  split_with_sizes = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    permute_2: "f32[8, 4, 196, 16]" = torch.ops.aten.permute.default(getitem, [0, 2, 1, 3]);  getitem = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    permute_3: "f32[8, 4, 16, 196]" = torch.ops.aten.permute.default(getitem_1, [0, 2, 3, 1]);  getitem_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    permute_4: "f32[8, 4, 196, 32]" = torch.ops.aten.permute.default(getitem_2, [0, 2, 1, 3]);  getitem_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    expand: "f32[8, 4, 196, 16]" = torch.ops.aten.expand.default(permute_2, [8, 4, 196, 16]);  permute_2 = None
    clone_1: "f32[8, 4, 196, 16]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
    view_6: "f32[32, 196, 16]" = torch.ops.aten.view.default(clone_1, [32, 196, 16]);  clone_1 = None
    expand_1: "f32[8, 4, 16, 196]" = torch.ops.aten.expand.default(permute_3, [8, 4, 16, 196]);  permute_3 = None
    clone_2: "f32[8, 4, 16, 196]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
    view_7: "f32[32, 16, 196]" = torch.ops.aten.view.default(clone_2, [32, 16, 196]);  clone_2 = None
    bmm: "f32[32, 196, 196]" = torch.ops.aten.bmm.default(view_6, view_7);  view_6 = view_7 = None
    view_8: "f32[8, 4, 196, 196]" = torch.ops.aten.view.default(bmm, [8, 4, 196, 196]);  bmm = None
    mul_18: "f32[8, 4, 196, 196]" = torch.ops.aten.mul.Tensor(view_8, 0.25);  view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:215, code: self.attention_bias_cache[device_key] = self.attention_biases[:, self.attention_bias_idxs]
    slice_1: "f32[4, 196]" = torch.ops.aten.slice.Tensor(arg0_1, 0, 0, 9223372036854775807);  arg0_1 = None
    index: "f32[4, 196, 196]" = torch.ops.aten.index.Tensor(slice_1, [None, arg208_1]);  slice_1 = arg208_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    add_13: "f32[8, 4, 196, 196]" = torch.ops.aten.add.Tensor(mul_18, index);  mul_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    amax: "f32[8, 4, 196, 1]" = torch.ops.aten.amax.default(add_13, [-1], True)
    sub_5: "f32[8, 4, 196, 196]" = torch.ops.aten.sub.Tensor(add_13, amax);  add_13 = amax = None
    exp: "f32[8, 4, 196, 196]" = torch.ops.aten.exp.default(sub_5);  sub_5 = None
    sum_1: "f32[8, 4, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div_3: "f32[8, 4, 196, 196]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    expand_2: "f32[8, 4, 196, 196]" = torch.ops.aten.expand.default(div_3, [8, 4, 196, 196]);  div_3 = None
    view_9: "f32[32, 196, 196]" = torch.ops.aten.view.default(expand_2, [32, 196, 196]);  expand_2 = None
    expand_3: "f32[8, 4, 196, 32]" = torch.ops.aten.expand.default(permute_4, [8, 4, 196, 32]);  permute_4 = None
    clone_3: "f32[8, 4, 196, 32]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
    view_10: "f32[32, 196, 32]" = torch.ops.aten.view.default(clone_3, [32, 196, 32]);  clone_3 = None
    bmm_1: "f32[32, 196, 32]" = torch.ops.aten.bmm.default(view_9, view_10);  view_9 = view_10 = None
    view_11: "f32[8, 4, 196, 32]" = torch.ops.aten.view.default(bmm_1, [8, 4, 196, 32]);  bmm_1 = None
    permute_5: "f32[8, 196, 4, 32]" = torch.ops.aten.permute.default(view_11, [0, 2, 1, 3]);  view_11 = None
    clone_4: "f32[8, 196, 4, 32]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
    view_12: "f32[8, 196, 128]" = torch.ops.aten.view.default(clone_4, [8, 196, 128]);  clone_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    add_14: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(view_12, 3)
    clamp_min_3: "f32[8, 196, 128]" = torch.ops.aten.clamp_min.default(add_14, 0);  add_14 = None
    clamp_max_3: "f32[8, 196, 128]" = torch.ops.aten.clamp_max.default(clamp_min_3, 6);  clamp_min_3 = None
    mul_19: "f32[8, 196, 128]" = torch.ops.aten.mul.Tensor(view_12, clamp_max_3);  view_12 = clamp_max_3 = None
    div_4: "f32[8, 196, 128]" = torch.ops.aten.div.Tensor(mul_19, 6);  mul_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_6: "f32[128, 128]" = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
    view_13: "f32[1568, 128]" = torch.ops.aten.view.default(div_4, [1568, 128]);  div_4 = None
    mm_1: "f32[1568, 128]" = torch.ops.aten.mm.default(view_13, permute_6);  view_13 = permute_6 = None
    view_14: "f32[8, 196, 128]" = torch.ops.aten.view.default(mm_1, [8, 196, 128]);  mm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_15: "f32[1568, 128]" = torch.ops.aten.view.default(view_14, [1568, 128]);  view_14 = None
    convert_element_type_10: "f32[128]" = torch.ops.prims.convert_element_type.default(arg237_1, torch.float32);  arg237_1 = None
    convert_element_type_11: "f32[128]" = torch.ops.prims.convert_element_type.default(arg238_1, torch.float32);  arg238_1 = None
    add_15: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_11, 1e-05);  convert_element_type_11 = None
    sqrt_5: "f32[128]" = torch.ops.aten.sqrt.default(add_15);  add_15 = None
    reciprocal_5: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_5);  sqrt_5 = None
    mul_20: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_5, 1);  reciprocal_5 = None
    sub_6: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_15, convert_element_type_10);  view_15 = convert_element_type_10 = None
    mul_21: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_6, mul_20);  sub_6 = mul_20 = None
    mul_22: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(mul_21, arg30_1);  mul_21 = arg30_1 = None
    add_16: "f32[1568, 128]" = torch.ops.aten.add.Tensor(mul_22, arg31_1);  mul_22 = arg31_1 = None
    view_16: "f32[8, 196, 128]" = torch.ops.aten.view.default(add_16, [8, 196, 128]);  add_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:456, code: x = x + self.drop_path1(self.attn(x))
    add_17: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(permute, view_16);  permute = view_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_7: "f32[128, 256]" = torch.ops.aten.permute.default(arg32_1, [1, 0]);  arg32_1 = None
    clone_5: "f32[8, 196, 128]" = torch.ops.aten.clone.default(add_17, memory_format = torch.contiguous_format)
    view_17: "f32[1568, 128]" = torch.ops.aten.view.default(clone_5, [1568, 128]);  clone_5 = None
    mm_2: "f32[1568, 256]" = torch.ops.aten.mm.default(view_17, permute_7);  view_17 = permute_7 = None
    view_18: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_2, [8, 196, 256]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_19: "f32[1568, 256]" = torch.ops.aten.view.default(view_18, [1568, 256]);  view_18 = None
    convert_element_type_12: "f32[256]" = torch.ops.prims.convert_element_type.default(arg240_1, torch.float32);  arg240_1 = None
    convert_element_type_13: "f32[256]" = torch.ops.prims.convert_element_type.default(arg241_1, torch.float32);  arg241_1 = None
    add_18: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_13, 1e-05);  convert_element_type_13 = None
    sqrt_6: "f32[256]" = torch.ops.aten.sqrt.default(add_18);  add_18 = None
    reciprocal_6: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_6);  sqrt_6 = None
    mul_23: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_6, 1);  reciprocal_6 = None
    sub_7: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_19, convert_element_type_12);  view_19 = convert_element_type_12 = None
    mul_24: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_7, mul_23);  sub_7 = mul_23 = None
    mul_25: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(mul_24, arg33_1);  mul_24 = arg33_1 = None
    add_19: "f32[1568, 256]" = torch.ops.aten.add.Tensor(mul_25, arg34_1);  mul_25 = arg34_1 = None
    view_20: "f32[8, 196, 256]" = torch.ops.aten.view.default(add_19, [8, 196, 256]);  add_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    add_20: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(view_20, 3)
    clamp_min_4: "f32[8, 196, 256]" = torch.ops.aten.clamp_min.default(add_20, 0);  add_20 = None
    clamp_max_4: "f32[8, 196, 256]" = torch.ops.aten.clamp_max.default(clamp_min_4, 6);  clamp_min_4 = None
    mul_26: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_20, clamp_max_4);  view_20 = clamp_max_4 = None
    div_5: "f32[8, 196, 256]" = torch.ops.aten.div.Tensor(mul_26, 6);  mul_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:369, code: x = self.drop(x)
    clone_6: "f32[8, 196, 256]" = torch.ops.aten.clone.default(div_5);  div_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_8: "f32[256, 128]" = torch.ops.aten.permute.default(arg35_1, [1, 0]);  arg35_1 = None
    view_21: "f32[1568, 256]" = torch.ops.aten.view.default(clone_6, [1568, 256]);  clone_6 = None
    mm_3: "f32[1568, 128]" = torch.ops.aten.mm.default(view_21, permute_8);  view_21 = permute_8 = None
    view_22: "f32[8, 196, 128]" = torch.ops.aten.view.default(mm_3, [8, 196, 128]);  mm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_23: "f32[1568, 128]" = torch.ops.aten.view.default(view_22, [1568, 128]);  view_22 = None
    convert_element_type_14: "f32[128]" = torch.ops.prims.convert_element_type.default(arg243_1, torch.float32);  arg243_1 = None
    convert_element_type_15: "f32[128]" = torch.ops.prims.convert_element_type.default(arg244_1, torch.float32);  arg244_1 = None
    add_21: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_15, 1e-05);  convert_element_type_15 = None
    sqrt_7: "f32[128]" = torch.ops.aten.sqrt.default(add_21);  add_21 = None
    reciprocal_7: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_7);  sqrt_7 = None
    mul_27: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_7, 1);  reciprocal_7 = None
    sub_8: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_23, convert_element_type_14);  view_23 = convert_element_type_14 = None
    mul_28: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_8, mul_27);  sub_8 = mul_27 = None
    mul_29: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(mul_28, arg36_1);  mul_28 = arg36_1 = None
    add_22: "f32[1568, 128]" = torch.ops.aten.add.Tensor(mul_29, arg37_1);  mul_29 = arg37_1 = None
    view_24: "f32[8, 196, 128]" = torch.ops.aten.view.default(add_22, [8, 196, 128]);  add_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:457, code: x = x + self.drop_path2(self.mlp(x))
    add_23: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(add_17, view_24);  add_17 = view_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_9: "f32[128, 256]" = torch.ops.aten.permute.default(arg38_1, [1, 0]);  arg38_1 = None
    clone_7: "f32[8, 196, 128]" = torch.ops.aten.clone.default(add_23, memory_format = torch.contiguous_format)
    view_25: "f32[1568, 128]" = torch.ops.aten.view.default(clone_7, [1568, 128]);  clone_7 = None
    mm_4: "f32[1568, 256]" = torch.ops.aten.mm.default(view_25, permute_9);  view_25 = permute_9 = None
    view_26: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_4, [8, 196, 256]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_27: "f32[1568, 256]" = torch.ops.aten.view.default(view_26, [1568, 256]);  view_26 = None
    convert_element_type_16: "f32[256]" = torch.ops.prims.convert_element_type.default(arg246_1, torch.float32);  arg246_1 = None
    convert_element_type_17: "f32[256]" = torch.ops.prims.convert_element_type.default(arg247_1, torch.float32);  arg247_1 = None
    add_24: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_17, 1e-05);  convert_element_type_17 = None
    sqrt_8: "f32[256]" = torch.ops.aten.sqrt.default(add_24);  add_24 = None
    reciprocal_8: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_8);  sqrt_8 = None
    mul_30: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_8, 1);  reciprocal_8 = None
    sub_9: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_27, convert_element_type_16);  view_27 = convert_element_type_16 = None
    mul_31: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_9, mul_30);  sub_9 = mul_30 = None
    mul_32: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(mul_31, arg39_1);  mul_31 = arg39_1 = None
    add_25: "f32[1568, 256]" = torch.ops.aten.add.Tensor(mul_32, arg40_1);  mul_32 = arg40_1 = None
    view_28: "f32[8, 196, 256]" = torch.ops.aten.view.default(add_25, [8, 196, 256]);  add_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    view_29: "f32[8, 196, 4, 64]" = torch.ops.aten.view.default(view_28, [8, 196, 4, -1]);  view_28 = None
    split_with_sizes_1 = torch.ops.aten.split_with_sizes.default(view_29, [16, 16, 32], 3);  view_29 = None
    getitem_3: "f32[8, 196, 4, 16]" = split_with_sizes_1[0]
    getitem_4: "f32[8, 196, 4, 16]" = split_with_sizes_1[1]
    getitem_5: "f32[8, 196, 4, 32]" = split_with_sizes_1[2];  split_with_sizes_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    permute_10: "f32[8, 4, 196, 16]" = torch.ops.aten.permute.default(getitem_3, [0, 2, 1, 3]);  getitem_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    permute_11: "f32[8, 4, 16, 196]" = torch.ops.aten.permute.default(getitem_4, [0, 2, 3, 1]);  getitem_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    permute_12: "f32[8, 4, 196, 32]" = torch.ops.aten.permute.default(getitem_5, [0, 2, 1, 3]);  getitem_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    expand_4: "f32[8, 4, 196, 16]" = torch.ops.aten.expand.default(permute_10, [8, 4, 196, 16]);  permute_10 = None
    clone_8: "f32[8, 4, 196, 16]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
    view_30: "f32[32, 196, 16]" = torch.ops.aten.view.default(clone_8, [32, 196, 16]);  clone_8 = None
    expand_5: "f32[8, 4, 16, 196]" = torch.ops.aten.expand.default(permute_11, [8, 4, 16, 196]);  permute_11 = None
    clone_9: "f32[8, 4, 16, 196]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
    view_31: "f32[32, 16, 196]" = torch.ops.aten.view.default(clone_9, [32, 16, 196]);  clone_9 = None
    bmm_2: "f32[32, 196, 196]" = torch.ops.aten.bmm.default(view_30, view_31);  view_30 = view_31 = None
    view_32: "f32[8, 4, 196, 196]" = torch.ops.aten.view.default(bmm_2, [8, 4, 196, 196]);  bmm_2 = None
    mul_33: "f32[8, 4, 196, 196]" = torch.ops.aten.mul.Tensor(view_32, 0.25);  view_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:215, code: self.attention_bias_cache[device_key] = self.attention_biases[:, self.attention_bias_idxs]
    slice_2: "f32[4, 196]" = torch.ops.aten.slice.Tensor(arg1_1, 0, 0, 9223372036854775807);  arg1_1 = None
    index_1: "f32[4, 196, 196]" = torch.ops.aten.index.Tensor(slice_2, [None, arg209_1]);  slice_2 = arg209_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    add_26: "f32[8, 4, 196, 196]" = torch.ops.aten.add.Tensor(mul_33, index_1);  mul_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    amax_1: "f32[8, 4, 196, 1]" = torch.ops.aten.amax.default(add_26, [-1], True)
    sub_10: "f32[8, 4, 196, 196]" = torch.ops.aten.sub.Tensor(add_26, amax_1);  add_26 = amax_1 = None
    exp_1: "f32[8, 4, 196, 196]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
    sum_2: "f32[8, 4, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_6: "f32[8, 4, 196, 196]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    expand_6: "f32[8, 4, 196, 196]" = torch.ops.aten.expand.default(div_6, [8, 4, 196, 196]);  div_6 = None
    view_33: "f32[32, 196, 196]" = torch.ops.aten.view.default(expand_6, [32, 196, 196]);  expand_6 = None
    expand_7: "f32[8, 4, 196, 32]" = torch.ops.aten.expand.default(permute_12, [8, 4, 196, 32]);  permute_12 = None
    clone_10: "f32[8, 4, 196, 32]" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
    view_34: "f32[32, 196, 32]" = torch.ops.aten.view.default(clone_10, [32, 196, 32]);  clone_10 = None
    bmm_3: "f32[32, 196, 32]" = torch.ops.aten.bmm.default(view_33, view_34);  view_33 = view_34 = None
    view_35: "f32[8, 4, 196, 32]" = torch.ops.aten.view.default(bmm_3, [8, 4, 196, 32]);  bmm_3 = None
    permute_13: "f32[8, 196, 4, 32]" = torch.ops.aten.permute.default(view_35, [0, 2, 1, 3]);  view_35 = None
    clone_11: "f32[8, 196, 4, 32]" = torch.ops.aten.clone.default(permute_13, memory_format = torch.contiguous_format);  permute_13 = None
    view_36: "f32[8, 196, 128]" = torch.ops.aten.view.default(clone_11, [8, 196, 128]);  clone_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    add_27: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(view_36, 3)
    clamp_min_5: "f32[8, 196, 128]" = torch.ops.aten.clamp_min.default(add_27, 0);  add_27 = None
    clamp_max_5: "f32[8, 196, 128]" = torch.ops.aten.clamp_max.default(clamp_min_5, 6);  clamp_min_5 = None
    mul_34: "f32[8, 196, 128]" = torch.ops.aten.mul.Tensor(view_36, clamp_max_5);  view_36 = clamp_max_5 = None
    div_7: "f32[8, 196, 128]" = torch.ops.aten.div.Tensor(mul_34, 6);  mul_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_14: "f32[128, 128]" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
    view_37: "f32[1568, 128]" = torch.ops.aten.view.default(div_7, [1568, 128]);  div_7 = None
    mm_5: "f32[1568, 128]" = torch.ops.aten.mm.default(view_37, permute_14);  view_37 = permute_14 = None
    view_38: "f32[8, 196, 128]" = torch.ops.aten.view.default(mm_5, [8, 196, 128]);  mm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_39: "f32[1568, 128]" = torch.ops.aten.view.default(view_38, [1568, 128]);  view_38 = None
    convert_element_type_18: "f32[128]" = torch.ops.prims.convert_element_type.default(arg249_1, torch.float32);  arg249_1 = None
    convert_element_type_19: "f32[128]" = torch.ops.prims.convert_element_type.default(arg250_1, torch.float32);  arg250_1 = None
    add_28: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_19, 1e-05);  convert_element_type_19 = None
    sqrt_9: "f32[128]" = torch.ops.aten.sqrt.default(add_28);  add_28 = None
    reciprocal_9: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_9);  sqrt_9 = None
    mul_35: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_9, 1);  reciprocal_9 = None
    sub_11: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_39, convert_element_type_18);  view_39 = convert_element_type_18 = None
    mul_36: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_11, mul_35);  sub_11 = mul_35 = None
    mul_37: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(mul_36, arg42_1);  mul_36 = arg42_1 = None
    add_29: "f32[1568, 128]" = torch.ops.aten.add.Tensor(mul_37, arg43_1);  mul_37 = arg43_1 = None
    view_40: "f32[8, 196, 128]" = torch.ops.aten.view.default(add_29, [8, 196, 128]);  add_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:456, code: x = x + self.drop_path1(self.attn(x))
    add_30: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(add_23, view_40);  add_23 = view_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_15: "f32[128, 256]" = torch.ops.aten.permute.default(arg44_1, [1, 0]);  arg44_1 = None
    clone_12: "f32[8, 196, 128]" = torch.ops.aten.clone.default(add_30, memory_format = torch.contiguous_format)
    view_41: "f32[1568, 128]" = torch.ops.aten.view.default(clone_12, [1568, 128]);  clone_12 = None
    mm_6: "f32[1568, 256]" = torch.ops.aten.mm.default(view_41, permute_15);  view_41 = permute_15 = None
    view_42: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_6, [8, 196, 256]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_43: "f32[1568, 256]" = torch.ops.aten.view.default(view_42, [1568, 256]);  view_42 = None
    convert_element_type_20: "f32[256]" = torch.ops.prims.convert_element_type.default(arg252_1, torch.float32);  arg252_1 = None
    convert_element_type_21: "f32[256]" = torch.ops.prims.convert_element_type.default(arg253_1, torch.float32);  arg253_1 = None
    add_31: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_21, 1e-05);  convert_element_type_21 = None
    sqrt_10: "f32[256]" = torch.ops.aten.sqrt.default(add_31);  add_31 = None
    reciprocal_10: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_10);  sqrt_10 = None
    mul_38: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_10, 1);  reciprocal_10 = None
    sub_12: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_43, convert_element_type_20);  view_43 = convert_element_type_20 = None
    mul_39: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_12, mul_38);  sub_12 = mul_38 = None
    mul_40: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(mul_39, arg45_1);  mul_39 = arg45_1 = None
    add_32: "f32[1568, 256]" = torch.ops.aten.add.Tensor(mul_40, arg46_1);  mul_40 = arg46_1 = None
    view_44: "f32[8, 196, 256]" = torch.ops.aten.view.default(add_32, [8, 196, 256]);  add_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    add_33: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(view_44, 3)
    clamp_min_6: "f32[8, 196, 256]" = torch.ops.aten.clamp_min.default(add_33, 0);  add_33 = None
    clamp_max_6: "f32[8, 196, 256]" = torch.ops.aten.clamp_max.default(clamp_min_6, 6);  clamp_min_6 = None
    mul_41: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_44, clamp_max_6);  view_44 = clamp_max_6 = None
    div_8: "f32[8, 196, 256]" = torch.ops.aten.div.Tensor(mul_41, 6);  mul_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:369, code: x = self.drop(x)
    clone_13: "f32[8, 196, 256]" = torch.ops.aten.clone.default(div_8);  div_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_16: "f32[256, 128]" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
    view_45: "f32[1568, 256]" = torch.ops.aten.view.default(clone_13, [1568, 256]);  clone_13 = None
    mm_7: "f32[1568, 128]" = torch.ops.aten.mm.default(view_45, permute_16);  view_45 = permute_16 = None
    view_46: "f32[8, 196, 128]" = torch.ops.aten.view.default(mm_7, [8, 196, 128]);  mm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_47: "f32[1568, 128]" = torch.ops.aten.view.default(view_46, [1568, 128]);  view_46 = None
    convert_element_type_22: "f32[128]" = torch.ops.prims.convert_element_type.default(arg255_1, torch.float32);  arg255_1 = None
    convert_element_type_23: "f32[128]" = torch.ops.prims.convert_element_type.default(arg256_1, torch.float32);  arg256_1 = None
    add_34: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_23, 1e-05);  convert_element_type_23 = None
    sqrt_11: "f32[128]" = torch.ops.aten.sqrt.default(add_34);  add_34 = None
    reciprocal_11: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_11);  sqrt_11 = None
    mul_42: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_11, 1);  reciprocal_11 = None
    sub_13: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_47, convert_element_type_22);  view_47 = convert_element_type_22 = None
    mul_43: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_13, mul_42);  sub_13 = mul_42 = None
    mul_44: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(mul_43, arg48_1);  mul_43 = arg48_1 = None
    add_35: "f32[1568, 128]" = torch.ops.aten.add.Tensor(mul_44, arg49_1);  mul_44 = arg49_1 = None
    view_48: "f32[8, 196, 128]" = torch.ops.aten.view.default(add_35, [8, 196, 128]);  add_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:457, code: x = x + self.drop_path2(self.mlp(x))
    add_36: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(add_30, view_48);  add_30 = view_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_17: "f32[128, 256]" = torch.ops.aten.permute.default(arg50_1, [1, 0]);  arg50_1 = None
    clone_14: "f32[8, 196, 128]" = torch.ops.aten.clone.default(add_36, memory_format = torch.contiguous_format)
    view_49: "f32[1568, 128]" = torch.ops.aten.view.default(clone_14, [1568, 128]);  clone_14 = None
    mm_8: "f32[1568, 256]" = torch.ops.aten.mm.default(view_49, permute_17);  view_49 = permute_17 = None
    view_50: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_8, [8, 196, 256]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_51: "f32[1568, 256]" = torch.ops.aten.view.default(view_50, [1568, 256]);  view_50 = None
    convert_element_type_24: "f32[256]" = torch.ops.prims.convert_element_type.default(arg258_1, torch.float32);  arg258_1 = None
    convert_element_type_25: "f32[256]" = torch.ops.prims.convert_element_type.default(arg259_1, torch.float32);  arg259_1 = None
    add_37: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_25, 1e-05);  convert_element_type_25 = None
    sqrt_12: "f32[256]" = torch.ops.aten.sqrt.default(add_37);  add_37 = None
    reciprocal_12: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_12);  sqrt_12 = None
    mul_45: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_12, 1);  reciprocal_12 = None
    sub_14: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_51, convert_element_type_24);  view_51 = convert_element_type_24 = None
    mul_46: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_14, mul_45);  sub_14 = mul_45 = None
    mul_47: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(mul_46, arg51_1);  mul_46 = arg51_1 = None
    add_38: "f32[1568, 256]" = torch.ops.aten.add.Tensor(mul_47, arg52_1);  mul_47 = arg52_1 = None
    view_52: "f32[8, 196, 256]" = torch.ops.aten.view.default(add_38, [8, 196, 256]);  add_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    view_53: "f32[8, 196, 4, 64]" = torch.ops.aten.view.default(view_52, [8, 196, 4, -1]);  view_52 = None
    split_with_sizes_2 = torch.ops.aten.split_with_sizes.default(view_53, [16, 16, 32], 3);  view_53 = None
    getitem_6: "f32[8, 196, 4, 16]" = split_with_sizes_2[0]
    getitem_7: "f32[8, 196, 4, 16]" = split_with_sizes_2[1]
    getitem_8: "f32[8, 196, 4, 32]" = split_with_sizes_2[2];  split_with_sizes_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    permute_18: "f32[8, 4, 196, 16]" = torch.ops.aten.permute.default(getitem_6, [0, 2, 1, 3]);  getitem_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    permute_19: "f32[8, 4, 16, 196]" = torch.ops.aten.permute.default(getitem_7, [0, 2, 3, 1]);  getitem_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    permute_20: "f32[8, 4, 196, 32]" = torch.ops.aten.permute.default(getitem_8, [0, 2, 1, 3]);  getitem_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    expand_8: "f32[8, 4, 196, 16]" = torch.ops.aten.expand.default(permute_18, [8, 4, 196, 16]);  permute_18 = None
    clone_15: "f32[8, 4, 196, 16]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
    view_54: "f32[32, 196, 16]" = torch.ops.aten.view.default(clone_15, [32, 196, 16]);  clone_15 = None
    expand_9: "f32[8, 4, 16, 196]" = torch.ops.aten.expand.default(permute_19, [8, 4, 16, 196]);  permute_19 = None
    clone_16: "f32[8, 4, 16, 196]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
    view_55: "f32[32, 16, 196]" = torch.ops.aten.view.default(clone_16, [32, 16, 196]);  clone_16 = None
    bmm_4: "f32[32, 196, 196]" = torch.ops.aten.bmm.default(view_54, view_55);  view_54 = view_55 = None
    view_56: "f32[8, 4, 196, 196]" = torch.ops.aten.view.default(bmm_4, [8, 4, 196, 196]);  bmm_4 = None
    mul_48: "f32[8, 4, 196, 196]" = torch.ops.aten.mul.Tensor(view_56, 0.25);  view_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:215, code: self.attention_bias_cache[device_key] = self.attention_biases[:, self.attention_bias_idxs]
    slice_3: "f32[4, 196]" = torch.ops.aten.slice.Tensor(arg2_1, 0, 0, 9223372036854775807);  arg2_1 = None
    index_2: "f32[4, 196, 196]" = torch.ops.aten.index.Tensor(slice_3, [None, arg210_1]);  slice_3 = arg210_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    add_39: "f32[8, 4, 196, 196]" = torch.ops.aten.add.Tensor(mul_48, index_2);  mul_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    amax_2: "f32[8, 4, 196, 1]" = torch.ops.aten.amax.default(add_39, [-1], True)
    sub_15: "f32[8, 4, 196, 196]" = torch.ops.aten.sub.Tensor(add_39, amax_2);  add_39 = amax_2 = None
    exp_2: "f32[8, 4, 196, 196]" = torch.ops.aten.exp.default(sub_15);  sub_15 = None
    sum_3: "f32[8, 4, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_9: "f32[8, 4, 196, 196]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    expand_10: "f32[8, 4, 196, 196]" = torch.ops.aten.expand.default(div_9, [8, 4, 196, 196]);  div_9 = None
    view_57: "f32[32, 196, 196]" = torch.ops.aten.view.default(expand_10, [32, 196, 196]);  expand_10 = None
    expand_11: "f32[8, 4, 196, 32]" = torch.ops.aten.expand.default(permute_20, [8, 4, 196, 32]);  permute_20 = None
    clone_17: "f32[8, 4, 196, 32]" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
    view_58: "f32[32, 196, 32]" = torch.ops.aten.view.default(clone_17, [32, 196, 32]);  clone_17 = None
    bmm_5: "f32[32, 196, 32]" = torch.ops.aten.bmm.default(view_57, view_58);  view_57 = view_58 = None
    view_59: "f32[8, 4, 196, 32]" = torch.ops.aten.view.default(bmm_5, [8, 4, 196, 32]);  bmm_5 = None
    permute_21: "f32[8, 196, 4, 32]" = torch.ops.aten.permute.default(view_59, [0, 2, 1, 3]);  view_59 = None
    clone_18: "f32[8, 196, 4, 32]" = torch.ops.aten.clone.default(permute_21, memory_format = torch.contiguous_format);  permute_21 = None
    view_60: "f32[8, 196, 128]" = torch.ops.aten.view.default(clone_18, [8, 196, 128]);  clone_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    add_40: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(view_60, 3)
    clamp_min_7: "f32[8, 196, 128]" = torch.ops.aten.clamp_min.default(add_40, 0);  add_40 = None
    clamp_max_7: "f32[8, 196, 128]" = torch.ops.aten.clamp_max.default(clamp_min_7, 6);  clamp_min_7 = None
    mul_49: "f32[8, 196, 128]" = torch.ops.aten.mul.Tensor(view_60, clamp_max_7);  view_60 = clamp_max_7 = None
    div_10: "f32[8, 196, 128]" = torch.ops.aten.div.Tensor(mul_49, 6);  mul_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_22: "f32[128, 128]" = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
    view_61: "f32[1568, 128]" = torch.ops.aten.view.default(div_10, [1568, 128]);  div_10 = None
    mm_9: "f32[1568, 128]" = torch.ops.aten.mm.default(view_61, permute_22);  view_61 = permute_22 = None
    view_62: "f32[8, 196, 128]" = torch.ops.aten.view.default(mm_9, [8, 196, 128]);  mm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_63: "f32[1568, 128]" = torch.ops.aten.view.default(view_62, [1568, 128]);  view_62 = None
    convert_element_type_26: "f32[128]" = torch.ops.prims.convert_element_type.default(arg261_1, torch.float32);  arg261_1 = None
    convert_element_type_27: "f32[128]" = torch.ops.prims.convert_element_type.default(arg262_1, torch.float32);  arg262_1 = None
    add_41: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_27, 1e-05);  convert_element_type_27 = None
    sqrt_13: "f32[128]" = torch.ops.aten.sqrt.default(add_41);  add_41 = None
    reciprocal_13: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_13);  sqrt_13 = None
    mul_50: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_13, 1);  reciprocal_13 = None
    sub_16: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_63, convert_element_type_26);  view_63 = convert_element_type_26 = None
    mul_51: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_16, mul_50);  sub_16 = mul_50 = None
    mul_52: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(mul_51, arg54_1);  mul_51 = arg54_1 = None
    add_42: "f32[1568, 128]" = torch.ops.aten.add.Tensor(mul_52, arg55_1);  mul_52 = arg55_1 = None
    view_64: "f32[8, 196, 128]" = torch.ops.aten.view.default(add_42, [8, 196, 128]);  add_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:456, code: x = x + self.drop_path1(self.attn(x))
    add_43: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(add_36, view_64);  add_36 = view_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_23: "f32[128, 256]" = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
    clone_19: "f32[8, 196, 128]" = torch.ops.aten.clone.default(add_43, memory_format = torch.contiguous_format)
    view_65: "f32[1568, 128]" = torch.ops.aten.view.default(clone_19, [1568, 128]);  clone_19 = None
    mm_10: "f32[1568, 256]" = torch.ops.aten.mm.default(view_65, permute_23);  view_65 = permute_23 = None
    view_66: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_10, [8, 196, 256]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_67: "f32[1568, 256]" = torch.ops.aten.view.default(view_66, [1568, 256]);  view_66 = None
    convert_element_type_28: "f32[256]" = torch.ops.prims.convert_element_type.default(arg264_1, torch.float32);  arg264_1 = None
    convert_element_type_29: "f32[256]" = torch.ops.prims.convert_element_type.default(arg265_1, torch.float32);  arg265_1 = None
    add_44: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_29, 1e-05);  convert_element_type_29 = None
    sqrt_14: "f32[256]" = torch.ops.aten.sqrt.default(add_44);  add_44 = None
    reciprocal_14: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_14);  sqrt_14 = None
    mul_53: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_14, 1);  reciprocal_14 = None
    sub_17: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_67, convert_element_type_28);  view_67 = convert_element_type_28 = None
    mul_54: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_17, mul_53);  sub_17 = mul_53 = None
    mul_55: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(mul_54, arg57_1);  mul_54 = arg57_1 = None
    add_45: "f32[1568, 256]" = torch.ops.aten.add.Tensor(mul_55, arg58_1);  mul_55 = arg58_1 = None
    view_68: "f32[8, 196, 256]" = torch.ops.aten.view.default(add_45, [8, 196, 256]);  add_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    add_46: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(view_68, 3)
    clamp_min_8: "f32[8, 196, 256]" = torch.ops.aten.clamp_min.default(add_46, 0);  add_46 = None
    clamp_max_8: "f32[8, 196, 256]" = torch.ops.aten.clamp_max.default(clamp_min_8, 6);  clamp_min_8 = None
    mul_56: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_68, clamp_max_8);  view_68 = clamp_max_8 = None
    div_11: "f32[8, 196, 256]" = torch.ops.aten.div.Tensor(mul_56, 6);  mul_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:369, code: x = self.drop(x)
    clone_20: "f32[8, 196, 256]" = torch.ops.aten.clone.default(div_11);  div_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_24: "f32[256, 128]" = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
    view_69: "f32[1568, 256]" = torch.ops.aten.view.default(clone_20, [1568, 256]);  clone_20 = None
    mm_11: "f32[1568, 128]" = torch.ops.aten.mm.default(view_69, permute_24);  view_69 = permute_24 = None
    view_70: "f32[8, 196, 128]" = torch.ops.aten.view.default(mm_11, [8, 196, 128]);  mm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_71: "f32[1568, 128]" = torch.ops.aten.view.default(view_70, [1568, 128]);  view_70 = None
    convert_element_type_30: "f32[128]" = torch.ops.prims.convert_element_type.default(arg267_1, torch.float32);  arg267_1 = None
    convert_element_type_31: "f32[128]" = torch.ops.prims.convert_element_type.default(arg268_1, torch.float32);  arg268_1 = None
    add_47: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_31, 1e-05);  convert_element_type_31 = None
    sqrt_15: "f32[128]" = torch.ops.aten.sqrt.default(add_47);  add_47 = None
    reciprocal_15: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_15);  sqrt_15 = None
    mul_57: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_15, 1);  reciprocal_15 = None
    sub_18: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_71, convert_element_type_30);  view_71 = convert_element_type_30 = None
    mul_58: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_18, mul_57);  sub_18 = mul_57 = None
    mul_59: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(mul_58, arg60_1);  mul_58 = arg60_1 = None
    add_48: "f32[1568, 128]" = torch.ops.aten.add.Tensor(mul_59, arg61_1);  mul_59 = arg61_1 = None
    view_72: "f32[8, 196, 128]" = torch.ops.aten.view.default(add_48, [8, 196, 128]);  add_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:457, code: x = x + self.drop_path2(self.mlp(x))
    add_49: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(add_43, view_72);  add_43 = view_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_25: "f32[128, 256]" = torch.ops.aten.permute.default(arg62_1, [1, 0]);  arg62_1 = None
    clone_21: "f32[8, 196, 128]" = torch.ops.aten.clone.default(add_49, memory_format = torch.contiguous_format)
    view_73: "f32[1568, 128]" = torch.ops.aten.view.default(clone_21, [1568, 128]);  clone_21 = None
    mm_12: "f32[1568, 256]" = torch.ops.aten.mm.default(view_73, permute_25);  view_73 = permute_25 = None
    view_74: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_12, [8, 196, 256]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_75: "f32[1568, 256]" = torch.ops.aten.view.default(view_74, [1568, 256]);  view_74 = None
    convert_element_type_32: "f32[256]" = torch.ops.prims.convert_element_type.default(arg270_1, torch.float32);  arg270_1 = None
    convert_element_type_33: "f32[256]" = torch.ops.prims.convert_element_type.default(arg271_1, torch.float32);  arg271_1 = None
    add_50: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_33, 1e-05);  convert_element_type_33 = None
    sqrt_16: "f32[256]" = torch.ops.aten.sqrt.default(add_50);  add_50 = None
    reciprocal_16: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_16);  sqrt_16 = None
    mul_60: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_16, 1);  reciprocal_16 = None
    sub_19: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_75, convert_element_type_32);  view_75 = convert_element_type_32 = None
    mul_61: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_19, mul_60);  sub_19 = mul_60 = None
    mul_62: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(mul_61, arg63_1);  mul_61 = arg63_1 = None
    add_51: "f32[1568, 256]" = torch.ops.aten.add.Tensor(mul_62, arg64_1);  mul_62 = arg64_1 = None
    view_76: "f32[8, 196, 256]" = torch.ops.aten.view.default(add_51, [8, 196, 256]);  add_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    view_77: "f32[8, 196, 4, 64]" = torch.ops.aten.view.default(view_76, [8, 196, 4, -1]);  view_76 = None
    split_with_sizes_3 = torch.ops.aten.split_with_sizes.default(view_77, [16, 16, 32], 3);  view_77 = None
    getitem_9: "f32[8, 196, 4, 16]" = split_with_sizes_3[0]
    getitem_10: "f32[8, 196, 4, 16]" = split_with_sizes_3[1]
    getitem_11: "f32[8, 196, 4, 32]" = split_with_sizes_3[2];  split_with_sizes_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    permute_26: "f32[8, 4, 196, 16]" = torch.ops.aten.permute.default(getitem_9, [0, 2, 1, 3]);  getitem_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    permute_27: "f32[8, 4, 16, 196]" = torch.ops.aten.permute.default(getitem_10, [0, 2, 3, 1]);  getitem_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    permute_28: "f32[8, 4, 196, 32]" = torch.ops.aten.permute.default(getitem_11, [0, 2, 1, 3]);  getitem_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    expand_12: "f32[8, 4, 196, 16]" = torch.ops.aten.expand.default(permute_26, [8, 4, 196, 16]);  permute_26 = None
    clone_22: "f32[8, 4, 196, 16]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
    view_78: "f32[32, 196, 16]" = torch.ops.aten.view.default(clone_22, [32, 196, 16]);  clone_22 = None
    expand_13: "f32[8, 4, 16, 196]" = torch.ops.aten.expand.default(permute_27, [8, 4, 16, 196]);  permute_27 = None
    clone_23: "f32[8, 4, 16, 196]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
    view_79: "f32[32, 16, 196]" = torch.ops.aten.view.default(clone_23, [32, 16, 196]);  clone_23 = None
    bmm_6: "f32[32, 196, 196]" = torch.ops.aten.bmm.default(view_78, view_79);  view_78 = view_79 = None
    view_80: "f32[8, 4, 196, 196]" = torch.ops.aten.view.default(bmm_6, [8, 4, 196, 196]);  bmm_6 = None
    mul_63: "f32[8, 4, 196, 196]" = torch.ops.aten.mul.Tensor(view_80, 0.25);  view_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:215, code: self.attention_bias_cache[device_key] = self.attention_biases[:, self.attention_bias_idxs]
    slice_4: "f32[4, 196]" = torch.ops.aten.slice.Tensor(arg3_1, 0, 0, 9223372036854775807);  arg3_1 = None
    index_3: "f32[4, 196, 196]" = torch.ops.aten.index.Tensor(slice_4, [None, arg211_1]);  slice_4 = arg211_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    add_52: "f32[8, 4, 196, 196]" = torch.ops.aten.add.Tensor(mul_63, index_3);  mul_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    amax_3: "f32[8, 4, 196, 1]" = torch.ops.aten.amax.default(add_52, [-1], True)
    sub_20: "f32[8, 4, 196, 196]" = torch.ops.aten.sub.Tensor(add_52, amax_3);  add_52 = amax_3 = None
    exp_3: "f32[8, 4, 196, 196]" = torch.ops.aten.exp.default(sub_20);  sub_20 = None
    sum_4: "f32[8, 4, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_12: "f32[8, 4, 196, 196]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    expand_14: "f32[8, 4, 196, 196]" = torch.ops.aten.expand.default(div_12, [8, 4, 196, 196]);  div_12 = None
    view_81: "f32[32, 196, 196]" = torch.ops.aten.view.default(expand_14, [32, 196, 196]);  expand_14 = None
    expand_15: "f32[8, 4, 196, 32]" = torch.ops.aten.expand.default(permute_28, [8, 4, 196, 32]);  permute_28 = None
    clone_24: "f32[8, 4, 196, 32]" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
    view_82: "f32[32, 196, 32]" = torch.ops.aten.view.default(clone_24, [32, 196, 32]);  clone_24 = None
    bmm_7: "f32[32, 196, 32]" = torch.ops.aten.bmm.default(view_81, view_82);  view_81 = view_82 = None
    view_83: "f32[8, 4, 196, 32]" = torch.ops.aten.view.default(bmm_7, [8, 4, 196, 32]);  bmm_7 = None
    permute_29: "f32[8, 196, 4, 32]" = torch.ops.aten.permute.default(view_83, [0, 2, 1, 3]);  view_83 = None
    clone_25: "f32[8, 196, 4, 32]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
    view_84: "f32[8, 196, 128]" = torch.ops.aten.view.default(clone_25, [8, 196, 128]);  clone_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    add_53: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(view_84, 3)
    clamp_min_9: "f32[8, 196, 128]" = torch.ops.aten.clamp_min.default(add_53, 0);  add_53 = None
    clamp_max_9: "f32[8, 196, 128]" = torch.ops.aten.clamp_max.default(clamp_min_9, 6);  clamp_min_9 = None
    mul_64: "f32[8, 196, 128]" = torch.ops.aten.mul.Tensor(view_84, clamp_max_9);  view_84 = clamp_max_9 = None
    div_13: "f32[8, 196, 128]" = torch.ops.aten.div.Tensor(mul_64, 6);  mul_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_30: "f32[128, 128]" = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
    view_85: "f32[1568, 128]" = torch.ops.aten.view.default(div_13, [1568, 128]);  div_13 = None
    mm_13: "f32[1568, 128]" = torch.ops.aten.mm.default(view_85, permute_30);  view_85 = permute_30 = None
    view_86: "f32[8, 196, 128]" = torch.ops.aten.view.default(mm_13, [8, 196, 128]);  mm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_87: "f32[1568, 128]" = torch.ops.aten.view.default(view_86, [1568, 128]);  view_86 = None
    convert_element_type_34: "f32[128]" = torch.ops.prims.convert_element_type.default(arg273_1, torch.float32);  arg273_1 = None
    convert_element_type_35: "f32[128]" = torch.ops.prims.convert_element_type.default(arg274_1, torch.float32);  arg274_1 = None
    add_54: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_35, 1e-05);  convert_element_type_35 = None
    sqrt_17: "f32[128]" = torch.ops.aten.sqrt.default(add_54);  add_54 = None
    reciprocal_17: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_17);  sqrt_17 = None
    mul_65: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_17, 1);  reciprocal_17 = None
    sub_21: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_87, convert_element_type_34);  view_87 = convert_element_type_34 = None
    mul_66: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_21, mul_65);  sub_21 = mul_65 = None
    mul_67: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(mul_66, arg66_1);  mul_66 = arg66_1 = None
    add_55: "f32[1568, 128]" = torch.ops.aten.add.Tensor(mul_67, arg67_1);  mul_67 = arg67_1 = None
    view_88: "f32[8, 196, 128]" = torch.ops.aten.view.default(add_55, [8, 196, 128]);  add_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:456, code: x = x + self.drop_path1(self.attn(x))
    add_56: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(add_49, view_88);  add_49 = view_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_31: "f32[128, 256]" = torch.ops.aten.permute.default(arg68_1, [1, 0]);  arg68_1 = None
    clone_26: "f32[8, 196, 128]" = torch.ops.aten.clone.default(add_56, memory_format = torch.contiguous_format)
    view_89: "f32[1568, 128]" = torch.ops.aten.view.default(clone_26, [1568, 128]);  clone_26 = None
    mm_14: "f32[1568, 256]" = torch.ops.aten.mm.default(view_89, permute_31);  view_89 = permute_31 = None
    view_90: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_14, [8, 196, 256]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_91: "f32[1568, 256]" = torch.ops.aten.view.default(view_90, [1568, 256]);  view_90 = None
    convert_element_type_36: "f32[256]" = torch.ops.prims.convert_element_type.default(arg276_1, torch.float32);  arg276_1 = None
    convert_element_type_37: "f32[256]" = torch.ops.prims.convert_element_type.default(arg277_1, torch.float32);  arg277_1 = None
    add_57: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_37, 1e-05);  convert_element_type_37 = None
    sqrt_18: "f32[256]" = torch.ops.aten.sqrt.default(add_57);  add_57 = None
    reciprocal_18: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_18);  sqrt_18 = None
    mul_68: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_18, 1);  reciprocal_18 = None
    sub_22: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_91, convert_element_type_36);  view_91 = convert_element_type_36 = None
    mul_69: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_22, mul_68);  sub_22 = mul_68 = None
    mul_70: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(mul_69, arg69_1);  mul_69 = arg69_1 = None
    add_58: "f32[1568, 256]" = torch.ops.aten.add.Tensor(mul_70, arg70_1);  mul_70 = arg70_1 = None
    view_92: "f32[8, 196, 256]" = torch.ops.aten.view.default(add_58, [8, 196, 256]);  add_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    add_59: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(view_92, 3)
    clamp_min_10: "f32[8, 196, 256]" = torch.ops.aten.clamp_min.default(add_59, 0);  add_59 = None
    clamp_max_10: "f32[8, 196, 256]" = torch.ops.aten.clamp_max.default(clamp_min_10, 6);  clamp_min_10 = None
    mul_71: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_92, clamp_max_10);  view_92 = clamp_max_10 = None
    div_14: "f32[8, 196, 256]" = torch.ops.aten.div.Tensor(mul_71, 6);  mul_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:369, code: x = self.drop(x)
    clone_27: "f32[8, 196, 256]" = torch.ops.aten.clone.default(div_14);  div_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_32: "f32[256, 128]" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
    view_93: "f32[1568, 256]" = torch.ops.aten.view.default(clone_27, [1568, 256]);  clone_27 = None
    mm_15: "f32[1568, 128]" = torch.ops.aten.mm.default(view_93, permute_32);  view_93 = permute_32 = None
    view_94: "f32[8, 196, 128]" = torch.ops.aten.view.default(mm_15, [8, 196, 128]);  mm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_95: "f32[1568, 128]" = torch.ops.aten.view.default(view_94, [1568, 128]);  view_94 = None
    convert_element_type_38: "f32[128]" = torch.ops.prims.convert_element_type.default(arg279_1, torch.float32);  arg279_1 = None
    convert_element_type_39: "f32[128]" = torch.ops.prims.convert_element_type.default(arg280_1, torch.float32);  arg280_1 = None
    add_60: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_39, 1e-05);  convert_element_type_39 = None
    sqrt_19: "f32[128]" = torch.ops.aten.sqrt.default(add_60);  add_60 = None
    reciprocal_19: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_19);  sqrt_19 = None
    mul_72: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_19, 1);  reciprocal_19 = None
    sub_23: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_95, convert_element_type_38);  view_95 = convert_element_type_38 = None
    mul_73: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_23, mul_72);  sub_23 = mul_72 = None
    mul_74: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(mul_73, arg72_1);  mul_73 = arg72_1 = None
    add_61: "f32[1568, 128]" = torch.ops.aten.add.Tensor(mul_74, arg73_1);  mul_74 = arg73_1 = None
    view_96: "f32[8, 196, 128]" = torch.ops.aten.view.default(add_61, [8, 196, 128]);  add_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:457, code: x = x + self.drop_path2(self.mlp(x))
    add_62: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(add_56, view_96);  add_56 = view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_33: "f32[128, 640]" = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
    clone_28: "f32[8, 196, 128]" = torch.ops.aten.clone.default(add_62, memory_format = torch.contiguous_format)
    view_97: "f32[1568, 128]" = torch.ops.aten.view.default(clone_28, [1568, 128]);  clone_28 = None
    mm_16: "f32[1568, 640]" = torch.ops.aten.mm.default(view_97, permute_33);  view_97 = permute_33 = None
    view_98: "f32[8, 196, 640]" = torch.ops.aten.view.default(mm_16, [8, 196, 640]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_99: "f32[1568, 640]" = torch.ops.aten.view.default(view_98, [1568, 640]);  view_98 = None
    convert_element_type_40: "f32[640]" = torch.ops.prims.convert_element_type.default(arg282_1, torch.float32);  arg282_1 = None
    convert_element_type_41: "f32[640]" = torch.ops.prims.convert_element_type.default(arg283_1, torch.float32);  arg283_1 = None
    add_63: "f32[640]" = torch.ops.aten.add.Tensor(convert_element_type_41, 1e-05);  convert_element_type_41 = None
    sqrt_20: "f32[640]" = torch.ops.aten.sqrt.default(add_63);  add_63 = None
    reciprocal_20: "f32[640]" = torch.ops.aten.reciprocal.default(sqrt_20);  sqrt_20 = None
    mul_75: "f32[640]" = torch.ops.aten.mul.Tensor(reciprocal_20, 1);  reciprocal_20 = None
    sub_24: "f32[1568, 640]" = torch.ops.aten.sub.Tensor(view_99, convert_element_type_40);  view_99 = convert_element_type_40 = None
    mul_76: "f32[1568, 640]" = torch.ops.aten.mul.Tensor(sub_24, mul_75);  sub_24 = mul_75 = None
    mul_77: "f32[1568, 640]" = torch.ops.aten.mul.Tensor(mul_76, arg75_1);  mul_76 = arg75_1 = None
    add_64: "f32[1568, 640]" = torch.ops.aten.add.Tensor(mul_77, arg76_1);  mul_77 = arg76_1 = None
    view_100: "f32[8, 196, 640]" = torch.ops.aten.view.default(add_64, [8, 196, 640]);  add_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:331, code: k, v = self.kv(x).view(B, N, self.num_heads, -1).split([self.key_dim, self.val_dim], dim=3)
    view_101: "f32[8, 196, 8, 80]" = torch.ops.aten.view.default(view_100, [8, 196, 8, -1]);  view_100 = None
    split_with_sizes_4 = torch.ops.aten.split_with_sizes.default(view_101, [16, 64], 3);  view_101 = None
    getitem_12: "f32[8, 196, 8, 16]" = split_with_sizes_4[0]
    getitem_13: "f32[8, 196, 8, 64]" = split_with_sizes_4[1];  split_with_sizes_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:332, code: k = k.permute(0, 2, 3, 1)  # BHCN
    permute_34: "f32[8, 8, 16, 196]" = torch.ops.aten.permute.default(getitem_12, [0, 2, 3, 1]);  getitem_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:333, code: v = v.permute(0, 2, 1, 3)  # BHNC
    permute_35: "f32[8, 8, 196, 64]" = torch.ops.aten.permute.default(getitem_13, [0, 2, 1, 3]);  getitem_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:157, code: x = x.view(B, self.resolution[0], self.resolution[1], C)
    view_102: "f32[8, 14, 14, 128]" = torch.ops.aten.view.default(add_62, [8, 14, 14, 128]);  add_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:161, code: x = x[:, ::self.stride, ::self.stride]
    slice_5: "f32[8, 14, 14, 128]" = torch.ops.aten.slice.Tensor(view_102, 0, 0, 9223372036854775807);  view_102 = None
    slice_6: "f32[8, 7, 14, 128]" = torch.ops.aten.slice.Tensor(slice_5, 1, 0, 9223372036854775807, 2);  slice_5 = None
    slice_7: "f32[8, 7, 7, 128]" = torch.ops.aten.slice.Tensor(slice_6, 2, 0, 9223372036854775807, 2);  slice_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:162, code: return x.reshape(B, -1, C)
    clone_29: "f32[8, 7, 7, 128]" = torch.ops.aten.clone.default(slice_7, memory_format = torch.contiguous_format);  slice_7 = None
    view_103: "f32[8, 49, 128]" = torch.ops.aten.view.default(clone_29, [8, 49, 128]);  clone_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_36: "f32[128, 128]" = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
    view_104: "f32[392, 128]" = torch.ops.aten.view.default(view_103, [392, 128]);  view_103 = None
    mm_17: "f32[392, 128]" = torch.ops.aten.mm.default(view_104, permute_36);  view_104 = permute_36 = None
    view_105: "f32[8, 49, 128]" = torch.ops.aten.view.default(mm_17, [8, 49, 128]);  mm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_106: "f32[392, 128]" = torch.ops.aten.view.default(view_105, [392, 128]);  view_105 = None
    convert_element_type_42: "f32[128]" = torch.ops.prims.convert_element_type.default(arg285_1, torch.float32);  arg285_1 = None
    convert_element_type_43: "f32[128]" = torch.ops.prims.convert_element_type.default(arg286_1, torch.float32);  arg286_1 = None
    add_65: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_43, 1e-05);  convert_element_type_43 = None
    sqrt_21: "f32[128]" = torch.ops.aten.sqrt.default(add_65);  add_65 = None
    reciprocal_21: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_21);  sqrt_21 = None
    mul_78: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_21, 1);  reciprocal_21 = None
    sub_25: "f32[392, 128]" = torch.ops.aten.sub.Tensor(view_106, convert_element_type_42);  view_106 = convert_element_type_42 = None
    mul_79: "f32[392, 128]" = torch.ops.aten.mul.Tensor(sub_25, mul_78);  sub_25 = mul_78 = None
    mul_80: "f32[392, 128]" = torch.ops.aten.mul.Tensor(mul_79, arg78_1);  mul_79 = arg78_1 = None
    add_66: "f32[392, 128]" = torch.ops.aten.add.Tensor(mul_80, arg79_1);  mul_80 = arg79_1 = None
    view_107: "f32[8, 49, 128]" = torch.ops.aten.view.default(add_66, [8, 49, 128]);  add_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:334, code: q = self.q(x).view(B, -1, self.num_heads, self.key_dim).permute(0, 2, 1, 3)
    view_108: "f32[8, 49, 8, 16]" = torch.ops.aten.view.default(view_107, [8, -1, 8, 16]);  view_107 = None
    permute_37: "f32[8, 8, 49, 16]" = torch.ops.aten.permute.default(view_108, [0, 2, 1, 3]);  view_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:336, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    expand_16: "f32[8, 8, 49, 16]" = torch.ops.aten.expand.default(permute_37, [8, 8, 49, 16]);  permute_37 = None
    clone_30: "f32[8, 8, 49, 16]" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
    view_109: "f32[64, 49, 16]" = torch.ops.aten.view.default(clone_30, [64, 49, 16]);  clone_30 = None
    expand_17: "f32[8, 8, 16, 196]" = torch.ops.aten.expand.default(permute_34, [8, 8, 16, 196]);  permute_34 = None
    clone_31: "f32[8, 8, 16, 196]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
    view_110: "f32[64, 16, 196]" = torch.ops.aten.view.default(clone_31, [64, 16, 196]);  clone_31 = None
    bmm_8: "f32[64, 49, 196]" = torch.ops.aten.bmm.default(view_109, view_110);  view_109 = view_110 = None
    view_111: "f32[8, 8, 49, 196]" = torch.ops.aten.view.default(bmm_8, [8, 8, 49, 196]);  bmm_8 = None
    mul_81: "f32[8, 8, 49, 196]" = torch.ops.aten.mul.Tensor(view_111, 0.25);  view_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:315, code: self.attention_bias_cache[device_key] = self.attention_biases[:, self.attention_bias_idxs]
    slice_8: "f32[8, 196]" = torch.ops.aten.slice.Tensor(arg4_1, 0, 0, 9223372036854775807);  arg4_1 = None
    index_4: "f32[8, 49, 196]" = torch.ops.aten.index.Tensor(slice_8, [None, arg212_1]);  slice_8 = arg212_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:336, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    add_67: "f32[8, 8, 49, 196]" = torch.ops.aten.add.Tensor(mul_81, index_4);  mul_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:337, code: attn = attn.softmax(dim=-1)
    amax_4: "f32[8, 8, 49, 1]" = torch.ops.aten.amax.default(add_67, [-1], True)
    sub_26: "f32[8, 8, 49, 196]" = torch.ops.aten.sub.Tensor(add_67, amax_4);  add_67 = amax_4 = None
    exp_4: "f32[8, 8, 49, 196]" = torch.ops.aten.exp.default(sub_26);  sub_26 = None
    sum_5: "f32[8, 8, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_15: "f32[8, 8, 49, 196]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:339, code: x = (attn @ v).transpose(1, 2).reshape(B, -1, self.val_attn_dim)
    expand_18: "f32[8, 8, 49, 196]" = torch.ops.aten.expand.default(div_15, [8, 8, 49, 196]);  div_15 = None
    view_112: "f32[64, 49, 196]" = torch.ops.aten.view.default(expand_18, [64, 49, 196]);  expand_18 = None
    expand_19: "f32[8, 8, 196, 64]" = torch.ops.aten.expand.default(permute_35, [8, 8, 196, 64]);  permute_35 = None
    clone_32: "f32[8, 8, 196, 64]" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
    view_113: "f32[64, 196, 64]" = torch.ops.aten.view.default(clone_32, [64, 196, 64]);  clone_32 = None
    bmm_9: "f32[64, 49, 64]" = torch.ops.aten.bmm.default(view_112, view_113);  view_112 = view_113 = None
    view_114: "f32[8, 8, 49, 64]" = torch.ops.aten.view.default(bmm_9, [8, 8, 49, 64]);  bmm_9 = None
    permute_38: "f32[8, 49, 8, 64]" = torch.ops.aten.permute.default(view_114, [0, 2, 1, 3]);  view_114 = None
    clone_33: "f32[8, 49, 8, 64]" = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
    view_115: "f32[8, 49, 512]" = torch.ops.aten.view.default(clone_33, [8, 49, 512]);  clone_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:340, code: x = self.proj(x)
    add_68: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(view_115, 3)
    clamp_min_11: "f32[8, 49, 512]" = torch.ops.aten.clamp_min.default(add_68, 0);  add_68 = None
    clamp_max_11: "f32[8, 49, 512]" = torch.ops.aten.clamp_max.default(clamp_min_11, 6);  clamp_min_11 = None
    mul_82: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(view_115, clamp_max_11);  view_115 = clamp_max_11 = None
    div_16: "f32[8, 49, 512]" = torch.ops.aten.div.Tensor(mul_82, 6);  mul_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_39: "f32[512, 256]" = torch.ops.aten.permute.default(arg80_1, [1, 0]);  arg80_1 = None
    view_116: "f32[392, 512]" = torch.ops.aten.view.default(div_16, [392, 512]);  div_16 = None
    mm_18: "f32[392, 256]" = torch.ops.aten.mm.default(view_116, permute_39);  view_116 = permute_39 = None
    view_117: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_18, [8, 49, 256]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_118: "f32[392, 256]" = torch.ops.aten.view.default(view_117, [392, 256]);  view_117 = None
    convert_element_type_44: "f32[256]" = torch.ops.prims.convert_element_type.default(arg288_1, torch.float32);  arg288_1 = None
    convert_element_type_45: "f32[256]" = torch.ops.prims.convert_element_type.default(arg289_1, torch.float32);  arg289_1 = None
    add_69: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_45, 1e-05);  convert_element_type_45 = None
    sqrt_22: "f32[256]" = torch.ops.aten.sqrt.default(add_69);  add_69 = None
    reciprocal_22: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_22);  sqrt_22 = None
    mul_83: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_22, 1);  reciprocal_22 = None
    sub_27: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_118, convert_element_type_44);  view_118 = convert_element_type_44 = None
    mul_84: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_27, mul_83);  sub_27 = mul_83 = None
    mul_85: "f32[392, 256]" = torch.ops.aten.mul.Tensor(mul_84, arg81_1);  mul_84 = arg81_1 = None
    add_70: "f32[392, 256]" = torch.ops.aten.add.Tensor(mul_85, arg82_1);  mul_85 = arg82_1 = None
    view_119: "f32[8, 49, 256]" = torch.ops.aten.view.default(add_70, [8, 49, 256]);  add_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_40: "f32[256, 512]" = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
    view_120: "f32[392, 256]" = torch.ops.aten.view.default(view_119, [392, 256])
    mm_19: "f32[392, 512]" = torch.ops.aten.mm.default(view_120, permute_40);  view_120 = permute_40 = None
    view_121: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_19, [8, 49, 512]);  mm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_122: "f32[392, 512]" = torch.ops.aten.view.default(view_121, [392, 512]);  view_121 = None
    convert_element_type_46: "f32[512]" = torch.ops.prims.convert_element_type.default(arg291_1, torch.float32);  arg291_1 = None
    convert_element_type_47: "f32[512]" = torch.ops.prims.convert_element_type.default(arg292_1, torch.float32);  arg292_1 = None
    add_71: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_47, 1e-05);  convert_element_type_47 = None
    sqrt_23: "f32[512]" = torch.ops.aten.sqrt.default(add_71);  add_71 = None
    reciprocal_23: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_23);  sqrt_23 = None
    mul_86: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_23, 1);  reciprocal_23 = None
    sub_28: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_122, convert_element_type_46);  view_122 = convert_element_type_46 = None
    mul_87: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_28, mul_86);  sub_28 = mul_86 = None
    mul_88: "f32[392, 512]" = torch.ops.aten.mul.Tensor(mul_87, arg84_1);  mul_87 = arg84_1 = None
    add_72: "f32[392, 512]" = torch.ops.aten.add.Tensor(mul_88, arg85_1);  mul_88 = arg85_1 = None
    view_123: "f32[8, 49, 512]" = torch.ops.aten.view.default(add_72, [8, 49, 512]);  add_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    add_73: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(view_123, 3)
    clamp_min_12: "f32[8, 49, 512]" = torch.ops.aten.clamp_min.default(add_73, 0);  add_73 = None
    clamp_max_12: "f32[8, 49, 512]" = torch.ops.aten.clamp_max.default(clamp_min_12, 6);  clamp_min_12 = None
    mul_89: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(view_123, clamp_max_12);  view_123 = clamp_max_12 = None
    div_17: "f32[8, 49, 512]" = torch.ops.aten.div.Tensor(mul_89, 6);  mul_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:369, code: x = self.drop(x)
    clone_34: "f32[8, 49, 512]" = torch.ops.aten.clone.default(div_17);  div_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_41: "f32[512, 256]" = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
    view_124: "f32[392, 512]" = torch.ops.aten.view.default(clone_34, [392, 512]);  clone_34 = None
    mm_20: "f32[392, 256]" = torch.ops.aten.mm.default(view_124, permute_41);  view_124 = permute_41 = None
    view_125: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_20, [8, 49, 256]);  mm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_126: "f32[392, 256]" = torch.ops.aten.view.default(view_125, [392, 256]);  view_125 = None
    convert_element_type_48: "f32[256]" = torch.ops.prims.convert_element_type.default(arg294_1, torch.float32);  arg294_1 = None
    convert_element_type_49: "f32[256]" = torch.ops.prims.convert_element_type.default(arg295_1, torch.float32);  arg295_1 = None
    add_74: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_49, 1e-05);  convert_element_type_49 = None
    sqrt_24: "f32[256]" = torch.ops.aten.sqrt.default(add_74);  add_74 = None
    reciprocal_24: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_24);  sqrt_24 = None
    mul_90: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_24, 1);  reciprocal_24 = None
    sub_29: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_126, convert_element_type_48);  view_126 = convert_element_type_48 = None
    mul_91: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_29, mul_90);  sub_29 = mul_90 = None
    mul_92: "f32[392, 256]" = torch.ops.aten.mul.Tensor(mul_91, arg87_1);  mul_91 = arg87_1 = None
    add_75: "f32[392, 256]" = torch.ops.aten.add.Tensor(mul_92, arg88_1);  mul_92 = arg88_1 = None
    view_127: "f32[8, 49, 256]" = torch.ops.aten.view.default(add_75, [8, 49, 256]);  add_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:415, code: x = x + self.drop_path(self.mlp(x))
    add_76: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(view_119, view_127);  view_119 = view_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_42: "f32[256, 512]" = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
    view_128: "f32[392, 256]" = torch.ops.aten.view.default(add_76, [392, 256])
    mm_21: "f32[392, 512]" = torch.ops.aten.mm.default(view_128, permute_42);  view_128 = permute_42 = None
    view_129: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_21, [8, 49, 512]);  mm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_130: "f32[392, 512]" = torch.ops.aten.view.default(view_129, [392, 512]);  view_129 = None
    convert_element_type_50: "f32[512]" = torch.ops.prims.convert_element_type.default(arg297_1, torch.float32);  arg297_1 = None
    convert_element_type_51: "f32[512]" = torch.ops.prims.convert_element_type.default(arg298_1, torch.float32);  arg298_1 = None
    add_77: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_51, 1e-05);  convert_element_type_51 = None
    sqrt_25: "f32[512]" = torch.ops.aten.sqrt.default(add_77);  add_77 = None
    reciprocal_25: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_25);  sqrt_25 = None
    mul_93: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_25, 1);  reciprocal_25 = None
    sub_30: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_130, convert_element_type_50);  view_130 = convert_element_type_50 = None
    mul_94: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_30, mul_93);  sub_30 = mul_93 = None
    mul_95: "f32[392, 512]" = torch.ops.aten.mul.Tensor(mul_94, arg90_1);  mul_94 = arg90_1 = None
    add_78: "f32[392, 512]" = torch.ops.aten.add.Tensor(mul_95, arg91_1);  mul_95 = arg91_1 = None
    view_131: "f32[8, 49, 512]" = torch.ops.aten.view.default(add_78, [8, 49, 512]);  add_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    view_132: "f32[8, 49, 8, 64]" = torch.ops.aten.view.default(view_131, [8, 49, 8, -1]);  view_131 = None
    split_with_sizes_5 = torch.ops.aten.split_with_sizes.default(view_132, [16, 16, 32], 3);  view_132 = None
    getitem_14: "f32[8, 49, 8, 16]" = split_with_sizes_5[0]
    getitem_15: "f32[8, 49, 8, 16]" = split_with_sizes_5[1]
    getitem_16: "f32[8, 49, 8, 32]" = split_with_sizes_5[2];  split_with_sizes_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    permute_43: "f32[8, 8, 49, 16]" = torch.ops.aten.permute.default(getitem_14, [0, 2, 1, 3]);  getitem_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    permute_44: "f32[8, 8, 16, 49]" = torch.ops.aten.permute.default(getitem_15, [0, 2, 3, 1]);  getitem_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    permute_45: "f32[8, 8, 49, 32]" = torch.ops.aten.permute.default(getitem_16, [0, 2, 1, 3]);  getitem_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    expand_20: "f32[8, 8, 49, 16]" = torch.ops.aten.expand.default(permute_43, [8, 8, 49, 16]);  permute_43 = None
    clone_35: "f32[8, 8, 49, 16]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
    view_133: "f32[64, 49, 16]" = torch.ops.aten.view.default(clone_35, [64, 49, 16]);  clone_35 = None
    expand_21: "f32[8, 8, 16, 49]" = torch.ops.aten.expand.default(permute_44, [8, 8, 16, 49]);  permute_44 = None
    clone_36: "f32[8, 8, 16, 49]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
    view_134: "f32[64, 16, 49]" = torch.ops.aten.view.default(clone_36, [64, 16, 49]);  clone_36 = None
    bmm_10: "f32[64, 49, 49]" = torch.ops.aten.bmm.default(view_133, view_134);  view_133 = view_134 = None
    view_135: "f32[8, 8, 49, 49]" = torch.ops.aten.view.default(bmm_10, [8, 8, 49, 49]);  bmm_10 = None
    mul_96: "f32[8, 8, 49, 49]" = torch.ops.aten.mul.Tensor(view_135, 0.25);  view_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:215, code: self.attention_bias_cache[device_key] = self.attention_biases[:, self.attention_bias_idxs]
    slice_9: "f32[8, 49]" = torch.ops.aten.slice.Tensor(arg5_1, 0, 0, 9223372036854775807);  arg5_1 = None
    index_5: "f32[8, 49, 49]" = torch.ops.aten.index.Tensor(slice_9, [None, arg213_1]);  slice_9 = arg213_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    add_79: "f32[8, 8, 49, 49]" = torch.ops.aten.add.Tensor(mul_96, index_5);  mul_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    amax_5: "f32[8, 8, 49, 1]" = torch.ops.aten.amax.default(add_79, [-1], True)
    sub_31: "f32[8, 8, 49, 49]" = torch.ops.aten.sub.Tensor(add_79, amax_5);  add_79 = amax_5 = None
    exp_5: "f32[8, 8, 49, 49]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
    sum_6: "f32[8, 8, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_18: "f32[8, 8, 49, 49]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    expand_22: "f32[8, 8, 49, 49]" = torch.ops.aten.expand.default(div_18, [8, 8, 49, 49]);  div_18 = None
    view_136: "f32[64, 49, 49]" = torch.ops.aten.view.default(expand_22, [64, 49, 49]);  expand_22 = None
    expand_23: "f32[8, 8, 49, 32]" = torch.ops.aten.expand.default(permute_45, [8, 8, 49, 32]);  permute_45 = None
    clone_37: "f32[8, 8, 49, 32]" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
    view_137: "f32[64, 49, 32]" = torch.ops.aten.view.default(clone_37, [64, 49, 32]);  clone_37 = None
    bmm_11: "f32[64, 49, 32]" = torch.ops.aten.bmm.default(view_136, view_137);  view_136 = view_137 = None
    view_138: "f32[8, 8, 49, 32]" = torch.ops.aten.view.default(bmm_11, [8, 8, 49, 32]);  bmm_11 = None
    permute_46: "f32[8, 49, 8, 32]" = torch.ops.aten.permute.default(view_138, [0, 2, 1, 3]);  view_138 = None
    clone_38: "f32[8, 49, 8, 32]" = torch.ops.aten.clone.default(permute_46, memory_format = torch.contiguous_format);  permute_46 = None
    view_139: "f32[8, 49, 256]" = torch.ops.aten.view.default(clone_38, [8, 49, 256]);  clone_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    add_80: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(view_139, 3)
    clamp_min_13: "f32[8, 49, 256]" = torch.ops.aten.clamp_min.default(add_80, 0);  add_80 = None
    clamp_max_13: "f32[8, 49, 256]" = torch.ops.aten.clamp_max.default(clamp_min_13, 6);  clamp_min_13 = None
    mul_97: "f32[8, 49, 256]" = torch.ops.aten.mul.Tensor(view_139, clamp_max_13);  view_139 = clamp_max_13 = None
    div_19: "f32[8, 49, 256]" = torch.ops.aten.div.Tensor(mul_97, 6);  mul_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_47: "f32[256, 256]" = torch.ops.aten.permute.default(arg92_1, [1, 0]);  arg92_1 = None
    view_140: "f32[392, 256]" = torch.ops.aten.view.default(div_19, [392, 256]);  div_19 = None
    mm_22: "f32[392, 256]" = torch.ops.aten.mm.default(view_140, permute_47);  view_140 = permute_47 = None
    view_141: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_22, [8, 49, 256]);  mm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_142: "f32[392, 256]" = torch.ops.aten.view.default(view_141, [392, 256]);  view_141 = None
    convert_element_type_52: "f32[256]" = torch.ops.prims.convert_element_type.default(arg300_1, torch.float32);  arg300_1 = None
    convert_element_type_53: "f32[256]" = torch.ops.prims.convert_element_type.default(arg301_1, torch.float32);  arg301_1 = None
    add_81: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_53, 1e-05);  convert_element_type_53 = None
    sqrt_26: "f32[256]" = torch.ops.aten.sqrt.default(add_81);  add_81 = None
    reciprocal_26: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_26);  sqrt_26 = None
    mul_98: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_26, 1);  reciprocal_26 = None
    sub_32: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_142, convert_element_type_52);  view_142 = convert_element_type_52 = None
    mul_99: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_32, mul_98);  sub_32 = mul_98 = None
    mul_100: "f32[392, 256]" = torch.ops.aten.mul.Tensor(mul_99, arg93_1);  mul_99 = arg93_1 = None
    add_82: "f32[392, 256]" = torch.ops.aten.add.Tensor(mul_100, arg94_1);  mul_100 = arg94_1 = None
    view_143: "f32[8, 49, 256]" = torch.ops.aten.view.default(add_82, [8, 49, 256]);  add_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:456, code: x = x + self.drop_path1(self.attn(x))
    add_83: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(add_76, view_143);  add_76 = view_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_48: "f32[256, 512]" = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
    view_144: "f32[392, 256]" = torch.ops.aten.view.default(add_83, [392, 256])
    mm_23: "f32[392, 512]" = torch.ops.aten.mm.default(view_144, permute_48);  view_144 = permute_48 = None
    view_145: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_23, [8, 49, 512]);  mm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_146: "f32[392, 512]" = torch.ops.aten.view.default(view_145, [392, 512]);  view_145 = None
    convert_element_type_54: "f32[512]" = torch.ops.prims.convert_element_type.default(arg303_1, torch.float32);  arg303_1 = None
    convert_element_type_55: "f32[512]" = torch.ops.prims.convert_element_type.default(arg304_1, torch.float32);  arg304_1 = None
    add_84: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_55, 1e-05);  convert_element_type_55 = None
    sqrt_27: "f32[512]" = torch.ops.aten.sqrt.default(add_84);  add_84 = None
    reciprocal_27: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_27);  sqrt_27 = None
    mul_101: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_27, 1);  reciprocal_27 = None
    sub_33: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_146, convert_element_type_54);  view_146 = convert_element_type_54 = None
    mul_102: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_33, mul_101);  sub_33 = mul_101 = None
    mul_103: "f32[392, 512]" = torch.ops.aten.mul.Tensor(mul_102, arg96_1);  mul_102 = arg96_1 = None
    add_85: "f32[392, 512]" = torch.ops.aten.add.Tensor(mul_103, arg97_1);  mul_103 = arg97_1 = None
    view_147: "f32[8, 49, 512]" = torch.ops.aten.view.default(add_85, [8, 49, 512]);  add_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    add_86: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(view_147, 3)
    clamp_min_14: "f32[8, 49, 512]" = torch.ops.aten.clamp_min.default(add_86, 0);  add_86 = None
    clamp_max_14: "f32[8, 49, 512]" = torch.ops.aten.clamp_max.default(clamp_min_14, 6);  clamp_min_14 = None
    mul_104: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(view_147, clamp_max_14);  view_147 = clamp_max_14 = None
    div_20: "f32[8, 49, 512]" = torch.ops.aten.div.Tensor(mul_104, 6);  mul_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:369, code: x = self.drop(x)
    clone_39: "f32[8, 49, 512]" = torch.ops.aten.clone.default(div_20);  div_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_49: "f32[512, 256]" = torch.ops.aten.permute.default(arg98_1, [1, 0]);  arg98_1 = None
    view_148: "f32[392, 512]" = torch.ops.aten.view.default(clone_39, [392, 512]);  clone_39 = None
    mm_24: "f32[392, 256]" = torch.ops.aten.mm.default(view_148, permute_49);  view_148 = permute_49 = None
    view_149: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_24, [8, 49, 256]);  mm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_150: "f32[392, 256]" = torch.ops.aten.view.default(view_149, [392, 256]);  view_149 = None
    convert_element_type_56: "f32[256]" = torch.ops.prims.convert_element_type.default(arg306_1, torch.float32);  arg306_1 = None
    convert_element_type_57: "f32[256]" = torch.ops.prims.convert_element_type.default(arg307_1, torch.float32);  arg307_1 = None
    add_87: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_57, 1e-05);  convert_element_type_57 = None
    sqrt_28: "f32[256]" = torch.ops.aten.sqrt.default(add_87);  add_87 = None
    reciprocal_28: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_28);  sqrt_28 = None
    mul_105: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_28, 1);  reciprocal_28 = None
    sub_34: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_150, convert_element_type_56);  view_150 = convert_element_type_56 = None
    mul_106: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_34, mul_105);  sub_34 = mul_105 = None
    mul_107: "f32[392, 256]" = torch.ops.aten.mul.Tensor(mul_106, arg99_1);  mul_106 = arg99_1 = None
    add_88: "f32[392, 256]" = torch.ops.aten.add.Tensor(mul_107, arg100_1);  mul_107 = arg100_1 = None
    view_151: "f32[8, 49, 256]" = torch.ops.aten.view.default(add_88, [8, 49, 256]);  add_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:457, code: x = x + self.drop_path2(self.mlp(x))
    add_89: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(add_83, view_151);  add_83 = view_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_50: "f32[256, 512]" = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
    view_152: "f32[392, 256]" = torch.ops.aten.view.default(add_89, [392, 256])
    mm_25: "f32[392, 512]" = torch.ops.aten.mm.default(view_152, permute_50);  view_152 = permute_50 = None
    view_153: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_25, [8, 49, 512]);  mm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_154: "f32[392, 512]" = torch.ops.aten.view.default(view_153, [392, 512]);  view_153 = None
    convert_element_type_58: "f32[512]" = torch.ops.prims.convert_element_type.default(arg309_1, torch.float32);  arg309_1 = None
    convert_element_type_59: "f32[512]" = torch.ops.prims.convert_element_type.default(arg310_1, torch.float32);  arg310_1 = None
    add_90: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_59, 1e-05);  convert_element_type_59 = None
    sqrt_29: "f32[512]" = torch.ops.aten.sqrt.default(add_90);  add_90 = None
    reciprocal_29: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_29);  sqrt_29 = None
    mul_108: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_29, 1);  reciprocal_29 = None
    sub_35: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_154, convert_element_type_58);  view_154 = convert_element_type_58 = None
    mul_109: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_35, mul_108);  sub_35 = mul_108 = None
    mul_110: "f32[392, 512]" = torch.ops.aten.mul.Tensor(mul_109, arg102_1);  mul_109 = arg102_1 = None
    add_91: "f32[392, 512]" = torch.ops.aten.add.Tensor(mul_110, arg103_1);  mul_110 = arg103_1 = None
    view_155: "f32[8, 49, 512]" = torch.ops.aten.view.default(add_91, [8, 49, 512]);  add_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    view_156: "f32[8, 49, 8, 64]" = torch.ops.aten.view.default(view_155, [8, 49, 8, -1]);  view_155 = None
    split_with_sizes_6 = torch.ops.aten.split_with_sizes.default(view_156, [16, 16, 32], 3);  view_156 = None
    getitem_17: "f32[8, 49, 8, 16]" = split_with_sizes_6[0]
    getitem_18: "f32[8, 49, 8, 16]" = split_with_sizes_6[1]
    getitem_19: "f32[8, 49, 8, 32]" = split_with_sizes_6[2];  split_with_sizes_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    permute_51: "f32[8, 8, 49, 16]" = torch.ops.aten.permute.default(getitem_17, [0, 2, 1, 3]);  getitem_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    permute_52: "f32[8, 8, 16, 49]" = torch.ops.aten.permute.default(getitem_18, [0, 2, 3, 1]);  getitem_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    permute_53: "f32[8, 8, 49, 32]" = torch.ops.aten.permute.default(getitem_19, [0, 2, 1, 3]);  getitem_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    expand_24: "f32[8, 8, 49, 16]" = torch.ops.aten.expand.default(permute_51, [8, 8, 49, 16]);  permute_51 = None
    clone_40: "f32[8, 8, 49, 16]" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
    view_157: "f32[64, 49, 16]" = torch.ops.aten.view.default(clone_40, [64, 49, 16]);  clone_40 = None
    expand_25: "f32[8, 8, 16, 49]" = torch.ops.aten.expand.default(permute_52, [8, 8, 16, 49]);  permute_52 = None
    clone_41: "f32[8, 8, 16, 49]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
    view_158: "f32[64, 16, 49]" = torch.ops.aten.view.default(clone_41, [64, 16, 49]);  clone_41 = None
    bmm_12: "f32[64, 49, 49]" = torch.ops.aten.bmm.default(view_157, view_158);  view_157 = view_158 = None
    view_159: "f32[8, 8, 49, 49]" = torch.ops.aten.view.default(bmm_12, [8, 8, 49, 49]);  bmm_12 = None
    mul_111: "f32[8, 8, 49, 49]" = torch.ops.aten.mul.Tensor(view_159, 0.25);  view_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:215, code: self.attention_bias_cache[device_key] = self.attention_biases[:, self.attention_bias_idxs]
    slice_10: "f32[8, 49]" = torch.ops.aten.slice.Tensor(arg6_1, 0, 0, 9223372036854775807);  arg6_1 = None
    index_6: "f32[8, 49, 49]" = torch.ops.aten.index.Tensor(slice_10, [None, arg214_1]);  slice_10 = arg214_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    add_92: "f32[8, 8, 49, 49]" = torch.ops.aten.add.Tensor(mul_111, index_6);  mul_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    amax_6: "f32[8, 8, 49, 1]" = torch.ops.aten.amax.default(add_92, [-1], True)
    sub_36: "f32[8, 8, 49, 49]" = torch.ops.aten.sub.Tensor(add_92, amax_6);  add_92 = amax_6 = None
    exp_6: "f32[8, 8, 49, 49]" = torch.ops.aten.exp.default(sub_36);  sub_36 = None
    sum_7: "f32[8, 8, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_21: "f32[8, 8, 49, 49]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    expand_26: "f32[8, 8, 49, 49]" = torch.ops.aten.expand.default(div_21, [8, 8, 49, 49]);  div_21 = None
    view_160: "f32[64, 49, 49]" = torch.ops.aten.view.default(expand_26, [64, 49, 49]);  expand_26 = None
    expand_27: "f32[8, 8, 49, 32]" = torch.ops.aten.expand.default(permute_53, [8, 8, 49, 32]);  permute_53 = None
    clone_42: "f32[8, 8, 49, 32]" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
    view_161: "f32[64, 49, 32]" = torch.ops.aten.view.default(clone_42, [64, 49, 32]);  clone_42 = None
    bmm_13: "f32[64, 49, 32]" = torch.ops.aten.bmm.default(view_160, view_161);  view_160 = view_161 = None
    view_162: "f32[8, 8, 49, 32]" = torch.ops.aten.view.default(bmm_13, [8, 8, 49, 32]);  bmm_13 = None
    permute_54: "f32[8, 49, 8, 32]" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None
    clone_43: "f32[8, 49, 8, 32]" = torch.ops.aten.clone.default(permute_54, memory_format = torch.contiguous_format);  permute_54 = None
    view_163: "f32[8, 49, 256]" = torch.ops.aten.view.default(clone_43, [8, 49, 256]);  clone_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    add_93: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(view_163, 3)
    clamp_min_15: "f32[8, 49, 256]" = torch.ops.aten.clamp_min.default(add_93, 0);  add_93 = None
    clamp_max_15: "f32[8, 49, 256]" = torch.ops.aten.clamp_max.default(clamp_min_15, 6);  clamp_min_15 = None
    mul_112: "f32[8, 49, 256]" = torch.ops.aten.mul.Tensor(view_163, clamp_max_15);  view_163 = clamp_max_15 = None
    div_22: "f32[8, 49, 256]" = torch.ops.aten.div.Tensor(mul_112, 6);  mul_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_55: "f32[256, 256]" = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
    view_164: "f32[392, 256]" = torch.ops.aten.view.default(div_22, [392, 256]);  div_22 = None
    mm_26: "f32[392, 256]" = torch.ops.aten.mm.default(view_164, permute_55);  view_164 = permute_55 = None
    view_165: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_26, [8, 49, 256]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_166: "f32[392, 256]" = torch.ops.aten.view.default(view_165, [392, 256]);  view_165 = None
    convert_element_type_60: "f32[256]" = torch.ops.prims.convert_element_type.default(arg312_1, torch.float32);  arg312_1 = None
    convert_element_type_61: "f32[256]" = torch.ops.prims.convert_element_type.default(arg313_1, torch.float32);  arg313_1 = None
    add_94: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_61, 1e-05);  convert_element_type_61 = None
    sqrt_30: "f32[256]" = torch.ops.aten.sqrt.default(add_94);  add_94 = None
    reciprocal_30: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_30);  sqrt_30 = None
    mul_113: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_30, 1);  reciprocal_30 = None
    sub_37: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_166, convert_element_type_60);  view_166 = convert_element_type_60 = None
    mul_114: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_37, mul_113);  sub_37 = mul_113 = None
    mul_115: "f32[392, 256]" = torch.ops.aten.mul.Tensor(mul_114, arg105_1);  mul_114 = arg105_1 = None
    add_95: "f32[392, 256]" = torch.ops.aten.add.Tensor(mul_115, arg106_1);  mul_115 = arg106_1 = None
    view_167: "f32[8, 49, 256]" = torch.ops.aten.view.default(add_95, [8, 49, 256]);  add_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:456, code: x = x + self.drop_path1(self.attn(x))
    add_96: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(add_89, view_167);  add_89 = view_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_56: "f32[256, 512]" = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
    view_168: "f32[392, 256]" = torch.ops.aten.view.default(add_96, [392, 256])
    mm_27: "f32[392, 512]" = torch.ops.aten.mm.default(view_168, permute_56);  view_168 = permute_56 = None
    view_169: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_27, [8, 49, 512]);  mm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_170: "f32[392, 512]" = torch.ops.aten.view.default(view_169, [392, 512]);  view_169 = None
    convert_element_type_62: "f32[512]" = torch.ops.prims.convert_element_type.default(arg315_1, torch.float32);  arg315_1 = None
    convert_element_type_63: "f32[512]" = torch.ops.prims.convert_element_type.default(arg316_1, torch.float32);  arg316_1 = None
    add_97: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_63, 1e-05);  convert_element_type_63 = None
    sqrt_31: "f32[512]" = torch.ops.aten.sqrt.default(add_97);  add_97 = None
    reciprocal_31: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_31);  sqrt_31 = None
    mul_116: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_31, 1);  reciprocal_31 = None
    sub_38: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_170, convert_element_type_62);  view_170 = convert_element_type_62 = None
    mul_117: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_38, mul_116);  sub_38 = mul_116 = None
    mul_118: "f32[392, 512]" = torch.ops.aten.mul.Tensor(mul_117, arg108_1);  mul_117 = arg108_1 = None
    add_98: "f32[392, 512]" = torch.ops.aten.add.Tensor(mul_118, arg109_1);  mul_118 = arg109_1 = None
    view_171: "f32[8, 49, 512]" = torch.ops.aten.view.default(add_98, [8, 49, 512]);  add_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    add_99: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(view_171, 3)
    clamp_min_16: "f32[8, 49, 512]" = torch.ops.aten.clamp_min.default(add_99, 0);  add_99 = None
    clamp_max_16: "f32[8, 49, 512]" = torch.ops.aten.clamp_max.default(clamp_min_16, 6);  clamp_min_16 = None
    mul_119: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(view_171, clamp_max_16);  view_171 = clamp_max_16 = None
    div_23: "f32[8, 49, 512]" = torch.ops.aten.div.Tensor(mul_119, 6);  mul_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:369, code: x = self.drop(x)
    clone_44: "f32[8, 49, 512]" = torch.ops.aten.clone.default(div_23);  div_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_57: "f32[512, 256]" = torch.ops.aten.permute.default(arg110_1, [1, 0]);  arg110_1 = None
    view_172: "f32[392, 512]" = torch.ops.aten.view.default(clone_44, [392, 512]);  clone_44 = None
    mm_28: "f32[392, 256]" = torch.ops.aten.mm.default(view_172, permute_57);  view_172 = permute_57 = None
    view_173: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_28, [8, 49, 256]);  mm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_174: "f32[392, 256]" = torch.ops.aten.view.default(view_173, [392, 256]);  view_173 = None
    convert_element_type_64: "f32[256]" = torch.ops.prims.convert_element_type.default(arg318_1, torch.float32);  arg318_1 = None
    convert_element_type_65: "f32[256]" = torch.ops.prims.convert_element_type.default(arg319_1, torch.float32);  arg319_1 = None
    add_100: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_65, 1e-05);  convert_element_type_65 = None
    sqrt_32: "f32[256]" = torch.ops.aten.sqrt.default(add_100);  add_100 = None
    reciprocal_32: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_32);  sqrt_32 = None
    mul_120: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_32, 1);  reciprocal_32 = None
    sub_39: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_174, convert_element_type_64);  view_174 = convert_element_type_64 = None
    mul_121: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_39, mul_120);  sub_39 = mul_120 = None
    mul_122: "f32[392, 256]" = torch.ops.aten.mul.Tensor(mul_121, arg111_1);  mul_121 = arg111_1 = None
    add_101: "f32[392, 256]" = torch.ops.aten.add.Tensor(mul_122, arg112_1);  mul_122 = arg112_1 = None
    view_175: "f32[8, 49, 256]" = torch.ops.aten.view.default(add_101, [8, 49, 256]);  add_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:457, code: x = x + self.drop_path2(self.mlp(x))
    add_102: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(add_96, view_175);  add_96 = view_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_58: "f32[256, 512]" = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
    view_176: "f32[392, 256]" = torch.ops.aten.view.default(add_102, [392, 256])
    mm_29: "f32[392, 512]" = torch.ops.aten.mm.default(view_176, permute_58);  view_176 = permute_58 = None
    view_177: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_29, [8, 49, 512]);  mm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_178: "f32[392, 512]" = torch.ops.aten.view.default(view_177, [392, 512]);  view_177 = None
    convert_element_type_66: "f32[512]" = torch.ops.prims.convert_element_type.default(arg321_1, torch.float32);  arg321_1 = None
    convert_element_type_67: "f32[512]" = torch.ops.prims.convert_element_type.default(arg322_1, torch.float32);  arg322_1 = None
    add_103: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_67, 1e-05);  convert_element_type_67 = None
    sqrt_33: "f32[512]" = torch.ops.aten.sqrt.default(add_103);  add_103 = None
    reciprocal_33: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_33);  sqrt_33 = None
    mul_123: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_33, 1);  reciprocal_33 = None
    sub_40: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_178, convert_element_type_66);  view_178 = convert_element_type_66 = None
    mul_124: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_40, mul_123);  sub_40 = mul_123 = None
    mul_125: "f32[392, 512]" = torch.ops.aten.mul.Tensor(mul_124, arg114_1);  mul_124 = arg114_1 = None
    add_104: "f32[392, 512]" = torch.ops.aten.add.Tensor(mul_125, arg115_1);  mul_125 = arg115_1 = None
    view_179: "f32[8, 49, 512]" = torch.ops.aten.view.default(add_104, [8, 49, 512]);  add_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    view_180: "f32[8, 49, 8, 64]" = torch.ops.aten.view.default(view_179, [8, 49, 8, -1]);  view_179 = None
    split_with_sizes_7 = torch.ops.aten.split_with_sizes.default(view_180, [16, 16, 32], 3);  view_180 = None
    getitem_20: "f32[8, 49, 8, 16]" = split_with_sizes_7[0]
    getitem_21: "f32[8, 49, 8, 16]" = split_with_sizes_7[1]
    getitem_22: "f32[8, 49, 8, 32]" = split_with_sizes_7[2];  split_with_sizes_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    permute_59: "f32[8, 8, 49, 16]" = torch.ops.aten.permute.default(getitem_20, [0, 2, 1, 3]);  getitem_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    permute_60: "f32[8, 8, 16, 49]" = torch.ops.aten.permute.default(getitem_21, [0, 2, 3, 1]);  getitem_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    permute_61: "f32[8, 8, 49, 32]" = torch.ops.aten.permute.default(getitem_22, [0, 2, 1, 3]);  getitem_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    expand_28: "f32[8, 8, 49, 16]" = torch.ops.aten.expand.default(permute_59, [8, 8, 49, 16]);  permute_59 = None
    clone_45: "f32[8, 8, 49, 16]" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
    view_181: "f32[64, 49, 16]" = torch.ops.aten.view.default(clone_45, [64, 49, 16]);  clone_45 = None
    expand_29: "f32[8, 8, 16, 49]" = torch.ops.aten.expand.default(permute_60, [8, 8, 16, 49]);  permute_60 = None
    clone_46: "f32[8, 8, 16, 49]" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
    view_182: "f32[64, 16, 49]" = torch.ops.aten.view.default(clone_46, [64, 16, 49]);  clone_46 = None
    bmm_14: "f32[64, 49, 49]" = torch.ops.aten.bmm.default(view_181, view_182);  view_181 = view_182 = None
    view_183: "f32[8, 8, 49, 49]" = torch.ops.aten.view.default(bmm_14, [8, 8, 49, 49]);  bmm_14 = None
    mul_126: "f32[8, 8, 49, 49]" = torch.ops.aten.mul.Tensor(view_183, 0.25);  view_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:215, code: self.attention_bias_cache[device_key] = self.attention_biases[:, self.attention_bias_idxs]
    slice_11: "f32[8, 49]" = torch.ops.aten.slice.Tensor(arg7_1, 0, 0, 9223372036854775807);  arg7_1 = None
    index_7: "f32[8, 49, 49]" = torch.ops.aten.index.Tensor(slice_11, [None, arg215_1]);  slice_11 = arg215_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    add_105: "f32[8, 8, 49, 49]" = torch.ops.aten.add.Tensor(mul_126, index_7);  mul_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    amax_7: "f32[8, 8, 49, 1]" = torch.ops.aten.amax.default(add_105, [-1], True)
    sub_41: "f32[8, 8, 49, 49]" = torch.ops.aten.sub.Tensor(add_105, amax_7);  add_105 = amax_7 = None
    exp_7: "f32[8, 8, 49, 49]" = torch.ops.aten.exp.default(sub_41);  sub_41 = None
    sum_8: "f32[8, 8, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_24: "f32[8, 8, 49, 49]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    expand_30: "f32[8, 8, 49, 49]" = torch.ops.aten.expand.default(div_24, [8, 8, 49, 49]);  div_24 = None
    view_184: "f32[64, 49, 49]" = torch.ops.aten.view.default(expand_30, [64, 49, 49]);  expand_30 = None
    expand_31: "f32[8, 8, 49, 32]" = torch.ops.aten.expand.default(permute_61, [8, 8, 49, 32]);  permute_61 = None
    clone_47: "f32[8, 8, 49, 32]" = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
    view_185: "f32[64, 49, 32]" = torch.ops.aten.view.default(clone_47, [64, 49, 32]);  clone_47 = None
    bmm_15: "f32[64, 49, 32]" = torch.ops.aten.bmm.default(view_184, view_185);  view_184 = view_185 = None
    view_186: "f32[8, 8, 49, 32]" = torch.ops.aten.view.default(bmm_15, [8, 8, 49, 32]);  bmm_15 = None
    permute_62: "f32[8, 49, 8, 32]" = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None
    clone_48: "f32[8, 49, 8, 32]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
    view_187: "f32[8, 49, 256]" = torch.ops.aten.view.default(clone_48, [8, 49, 256]);  clone_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    add_106: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(view_187, 3)
    clamp_min_17: "f32[8, 49, 256]" = torch.ops.aten.clamp_min.default(add_106, 0);  add_106 = None
    clamp_max_17: "f32[8, 49, 256]" = torch.ops.aten.clamp_max.default(clamp_min_17, 6);  clamp_min_17 = None
    mul_127: "f32[8, 49, 256]" = torch.ops.aten.mul.Tensor(view_187, clamp_max_17);  view_187 = clamp_max_17 = None
    div_25: "f32[8, 49, 256]" = torch.ops.aten.div.Tensor(mul_127, 6);  mul_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_63: "f32[256, 256]" = torch.ops.aten.permute.default(arg116_1, [1, 0]);  arg116_1 = None
    view_188: "f32[392, 256]" = torch.ops.aten.view.default(div_25, [392, 256]);  div_25 = None
    mm_30: "f32[392, 256]" = torch.ops.aten.mm.default(view_188, permute_63);  view_188 = permute_63 = None
    view_189: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_30, [8, 49, 256]);  mm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_190: "f32[392, 256]" = torch.ops.aten.view.default(view_189, [392, 256]);  view_189 = None
    convert_element_type_68: "f32[256]" = torch.ops.prims.convert_element_type.default(arg324_1, torch.float32);  arg324_1 = None
    convert_element_type_69: "f32[256]" = torch.ops.prims.convert_element_type.default(arg325_1, torch.float32);  arg325_1 = None
    add_107: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_69, 1e-05);  convert_element_type_69 = None
    sqrt_34: "f32[256]" = torch.ops.aten.sqrt.default(add_107);  add_107 = None
    reciprocal_34: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_34);  sqrt_34 = None
    mul_128: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_34, 1);  reciprocal_34 = None
    sub_42: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_190, convert_element_type_68);  view_190 = convert_element_type_68 = None
    mul_129: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_42, mul_128);  sub_42 = mul_128 = None
    mul_130: "f32[392, 256]" = torch.ops.aten.mul.Tensor(mul_129, arg117_1);  mul_129 = arg117_1 = None
    add_108: "f32[392, 256]" = torch.ops.aten.add.Tensor(mul_130, arg118_1);  mul_130 = arg118_1 = None
    view_191: "f32[8, 49, 256]" = torch.ops.aten.view.default(add_108, [8, 49, 256]);  add_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:456, code: x = x + self.drop_path1(self.attn(x))
    add_109: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(add_102, view_191);  add_102 = view_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_64: "f32[256, 512]" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
    view_192: "f32[392, 256]" = torch.ops.aten.view.default(add_109, [392, 256])
    mm_31: "f32[392, 512]" = torch.ops.aten.mm.default(view_192, permute_64);  view_192 = permute_64 = None
    view_193: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_31, [8, 49, 512]);  mm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_194: "f32[392, 512]" = torch.ops.aten.view.default(view_193, [392, 512]);  view_193 = None
    convert_element_type_70: "f32[512]" = torch.ops.prims.convert_element_type.default(arg327_1, torch.float32);  arg327_1 = None
    convert_element_type_71: "f32[512]" = torch.ops.prims.convert_element_type.default(arg328_1, torch.float32);  arg328_1 = None
    add_110: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_71, 1e-05);  convert_element_type_71 = None
    sqrt_35: "f32[512]" = torch.ops.aten.sqrt.default(add_110);  add_110 = None
    reciprocal_35: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_35);  sqrt_35 = None
    mul_131: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_35, 1);  reciprocal_35 = None
    sub_43: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_194, convert_element_type_70);  view_194 = convert_element_type_70 = None
    mul_132: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_43, mul_131);  sub_43 = mul_131 = None
    mul_133: "f32[392, 512]" = torch.ops.aten.mul.Tensor(mul_132, arg120_1);  mul_132 = arg120_1 = None
    add_111: "f32[392, 512]" = torch.ops.aten.add.Tensor(mul_133, arg121_1);  mul_133 = arg121_1 = None
    view_195: "f32[8, 49, 512]" = torch.ops.aten.view.default(add_111, [8, 49, 512]);  add_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    add_112: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(view_195, 3)
    clamp_min_18: "f32[8, 49, 512]" = torch.ops.aten.clamp_min.default(add_112, 0);  add_112 = None
    clamp_max_18: "f32[8, 49, 512]" = torch.ops.aten.clamp_max.default(clamp_min_18, 6);  clamp_min_18 = None
    mul_134: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(view_195, clamp_max_18);  view_195 = clamp_max_18 = None
    div_26: "f32[8, 49, 512]" = torch.ops.aten.div.Tensor(mul_134, 6);  mul_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:369, code: x = self.drop(x)
    clone_49: "f32[8, 49, 512]" = torch.ops.aten.clone.default(div_26);  div_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_65: "f32[512, 256]" = torch.ops.aten.permute.default(arg122_1, [1, 0]);  arg122_1 = None
    view_196: "f32[392, 512]" = torch.ops.aten.view.default(clone_49, [392, 512]);  clone_49 = None
    mm_32: "f32[392, 256]" = torch.ops.aten.mm.default(view_196, permute_65);  view_196 = permute_65 = None
    view_197: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_32, [8, 49, 256]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_198: "f32[392, 256]" = torch.ops.aten.view.default(view_197, [392, 256]);  view_197 = None
    convert_element_type_72: "f32[256]" = torch.ops.prims.convert_element_type.default(arg330_1, torch.float32);  arg330_1 = None
    convert_element_type_73: "f32[256]" = torch.ops.prims.convert_element_type.default(arg331_1, torch.float32);  arg331_1 = None
    add_113: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_73, 1e-05);  convert_element_type_73 = None
    sqrt_36: "f32[256]" = torch.ops.aten.sqrt.default(add_113);  add_113 = None
    reciprocal_36: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_36);  sqrt_36 = None
    mul_135: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_36, 1);  reciprocal_36 = None
    sub_44: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_198, convert_element_type_72);  view_198 = convert_element_type_72 = None
    mul_136: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_44, mul_135);  sub_44 = mul_135 = None
    mul_137: "f32[392, 256]" = torch.ops.aten.mul.Tensor(mul_136, arg123_1);  mul_136 = arg123_1 = None
    add_114: "f32[392, 256]" = torch.ops.aten.add.Tensor(mul_137, arg124_1);  mul_137 = arg124_1 = None
    view_199: "f32[8, 49, 256]" = torch.ops.aten.view.default(add_114, [8, 49, 256]);  add_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:457, code: x = x + self.drop_path2(self.mlp(x))
    add_115: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(add_109, view_199);  add_109 = view_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_66: "f32[256, 512]" = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
    view_200: "f32[392, 256]" = torch.ops.aten.view.default(add_115, [392, 256])
    mm_33: "f32[392, 512]" = torch.ops.aten.mm.default(view_200, permute_66);  view_200 = permute_66 = None
    view_201: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_33, [8, 49, 512]);  mm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_202: "f32[392, 512]" = torch.ops.aten.view.default(view_201, [392, 512]);  view_201 = None
    convert_element_type_74: "f32[512]" = torch.ops.prims.convert_element_type.default(arg333_1, torch.float32);  arg333_1 = None
    convert_element_type_75: "f32[512]" = torch.ops.prims.convert_element_type.default(arg334_1, torch.float32);  arg334_1 = None
    add_116: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_75, 1e-05);  convert_element_type_75 = None
    sqrt_37: "f32[512]" = torch.ops.aten.sqrt.default(add_116);  add_116 = None
    reciprocal_37: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_37);  sqrt_37 = None
    mul_138: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_37, 1);  reciprocal_37 = None
    sub_45: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_202, convert_element_type_74);  view_202 = convert_element_type_74 = None
    mul_139: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_45, mul_138);  sub_45 = mul_138 = None
    mul_140: "f32[392, 512]" = torch.ops.aten.mul.Tensor(mul_139, arg126_1);  mul_139 = arg126_1 = None
    add_117: "f32[392, 512]" = torch.ops.aten.add.Tensor(mul_140, arg127_1);  mul_140 = arg127_1 = None
    view_203: "f32[8, 49, 512]" = torch.ops.aten.view.default(add_117, [8, 49, 512]);  add_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    view_204: "f32[8, 49, 8, 64]" = torch.ops.aten.view.default(view_203, [8, 49, 8, -1]);  view_203 = None
    split_with_sizes_8 = torch.ops.aten.split_with_sizes.default(view_204, [16, 16, 32], 3);  view_204 = None
    getitem_23: "f32[8, 49, 8, 16]" = split_with_sizes_8[0]
    getitem_24: "f32[8, 49, 8, 16]" = split_with_sizes_8[1]
    getitem_25: "f32[8, 49, 8, 32]" = split_with_sizes_8[2];  split_with_sizes_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    permute_67: "f32[8, 8, 49, 16]" = torch.ops.aten.permute.default(getitem_23, [0, 2, 1, 3]);  getitem_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    permute_68: "f32[8, 8, 16, 49]" = torch.ops.aten.permute.default(getitem_24, [0, 2, 3, 1]);  getitem_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    permute_69: "f32[8, 8, 49, 32]" = torch.ops.aten.permute.default(getitem_25, [0, 2, 1, 3]);  getitem_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    expand_32: "f32[8, 8, 49, 16]" = torch.ops.aten.expand.default(permute_67, [8, 8, 49, 16]);  permute_67 = None
    clone_50: "f32[8, 8, 49, 16]" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
    view_205: "f32[64, 49, 16]" = torch.ops.aten.view.default(clone_50, [64, 49, 16]);  clone_50 = None
    expand_33: "f32[8, 8, 16, 49]" = torch.ops.aten.expand.default(permute_68, [8, 8, 16, 49]);  permute_68 = None
    clone_51: "f32[8, 8, 16, 49]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
    view_206: "f32[64, 16, 49]" = torch.ops.aten.view.default(clone_51, [64, 16, 49]);  clone_51 = None
    bmm_16: "f32[64, 49, 49]" = torch.ops.aten.bmm.default(view_205, view_206);  view_205 = view_206 = None
    view_207: "f32[8, 8, 49, 49]" = torch.ops.aten.view.default(bmm_16, [8, 8, 49, 49]);  bmm_16 = None
    mul_141: "f32[8, 8, 49, 49]" = torch.ops.aten.mul.Tensor(view_207, 0.25);  view_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:215, code: self.attention_bias_cache[device_key] = self.attention_biases[:, self.attention_bias_idxs]
    slice_12: "f32[8, 49]" = torch.ops.aten.slice.Tensor(arg8_1, 0, 0, 9223372036854775807);  arg8_1 = None
    index_8: "f32[8, 49, 49]" = torch.ops.aten.index.Tensor(slice_12, [None, arg216_1]);  slice_12 = arg216_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    add_118: "f32[8, 8, 49, 49]" = torch.ops.aten.add.Tensor(mul_141, index_8);  mul_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    amax_8: "f32[8, 8, 49, 1]" = torch.ops.aten.amax.default(add_118, [-1], True)
    sub_46: "f32[8, 8, 49, 49]" = torch.ops.aten.sub.Tensor(add_118, amax_8);  add_118 = amax_8 = None
    exp_8: "f32[8, 8, 49, 49]" = torch.ops.aten.exp.default(sub_46);  sub_46 = None
    sum_9: "f32[8, 8, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_27: "f32[8, 8, 49, 49]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    expand_34: "f32[8, 8, 49, 49]" = torch.ops.aten.expand.default(div_27, [8, 8, 49, 49]);  div_27 = None
    view_208: "f32[64, 49, 49]" = torch.ops.aten.view.default(expand_34, [64, 49, 49]);  expand_34 = None
    expand_35: "f32[8, 8, 49, 32]" = torch.ops.aten.expand.default(permute_69, [8, 8, 49, 32]);  permute_69 = None
    clone_52: "f32[8, 8, 49, 32]" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
    view_209: "f32[64, 49, 32]" = torch.ops.aten.view.default(clone_52, [64, 49, 32]);  clone_52 = None
    bmm_17: "f32[64, 49, 32]" = torch.ops.aten.bmm.default(view_208, view_209);  view_208 = view_209 = None
    view_210: "f32[8, 8, 49, 32]" = torch.ops.aten.view.default(bmm_17, [8, 8, 49, 32]);  bmm_17 = None
    permute_70: "f32[8, 49, 8, 32]" = torch.ops.aten.permute.default(view_210, [0, 2, 1, 3]);  view_210 = None
    clone_53: "f32[8, 49, 8, 32]" = torch.ops.aten.clone.default(permute_70, memory_format = torch.contiguous_format);  permute_70 = None
    view_211: "f32[8, 49, 256]" = torch.ops.aten.view.default(clone_53, [8, 49, 256]);  clone_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    add_119: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(view_211, 3)
    clamp_min_19: "f32[8, 49, 256]" = torch.ops.aten.clamp_min.default(add_119, 0);  add_119 = None
    clamp_max_19: "f32[8, 49, 256]" = torch.ops.aten.clamp_max.default(clamp_min_19, 6);  clamp_min_19 = None
    mul_142: "f32[8, 49, 256]" = torch.ops.aten.mul.Tensor(view_211, clamp_max_19);  view_211 = clamp_max_19 = None
    div_28: "f32[8, 49, 256]" = torch.ops.aten.div.Tensor(mul_142, 6);  mul_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_71: "f32[256, 256]" = torch.ops.aten.permute.default(arg128_1, [1, 0]);  arg128_1 = None
    view_212: "f32[392, 256]" = torch.ops.aten.view.default(div_28, [392, 256]);  div_28 = None
    mm_34: "f32[392, 256]" = torch.ops.aten.mm.default(view_212, permute_71);  view_212 = permute_71 = None
    view_213: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_34, [8, 49, 256]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_214: "f32[392, 256]" = torch.ops.aten.view.default(view_213, [392, 256]);  view_213 = None
    convert_element_type_76: "f32[256]" = torch.ops.prims.convert_element_type.default(arg336_1, torch.float32);  arg336_1 = None
    convert_element_type_77: "f32[256]" = torch.ops.prims.convert_element_type.default(arg337_1, torch.float32);  arg337_1 = None
    add_120: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_77, 1e-05);  convert_element_type_77 = None
    sqrt_38: "f32[256]" = torch.ops.aten.sqrt.default(add_120);  add_120 = None
    reciprocal_38: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_38);  sqrt_38 = None
    mul_143: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_38, 1);  reciprocal_38 = None
    sub_47: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_214, convert_element_type_76);  view_214 = convert_element_type_76 = None
    mul_144: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_47, mul_143);  sub_47 = mul_143 = None
    mul_145: "f32[392, 256]" = torch.ops.aten.mul.Tensor(mul_144, arg129_1);  mul_144 = arg129_1 = None
    add_121: "f32[392, 256]" = torch.ops.aten.add.Tensor(mul_145, arg130_1);  mul_145 = arg130_1 = None
    view_215: "f32[8, 49, 256]" = torch.ops.aten.view.default(add_121, [8, 49, 256]);  add_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:456, code: x = x + self.drop_path1(self.attn(x))
    add_122: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(add_115, view_215);  add_115 = view_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_72: "f32[256, 512]" = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
    view_216: "f32[392, 256]" = torch.ops.aten.view.default(add_122, [392, 256])
    mm_35: "f32[392, 512]" = torch.ops.aten.mm.default(view_216, permute_72);  view_216 = permute_72 = None
    view_217: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_35, [8, 49, 512]);  mm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_218: "f32[392, 512]" = torch.ops.aten.view.default(view_217, [392, 512]);  view_217 = None
    convert_element_type_78: "f32[512]" = torch.ops.prims.convert_element_type.default(arg339_1, torch.float32);  arg339_1 = None
    convert_element_type_79: "f32[512]" = torch.ops.prims.convert_element_type.default(arg340_1, torch.float32);  arg340_1 = None
    add_123: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_79, 1e-05);  convert_element_type_79 = None
    sqrt_39: "f32[512]" = torch.ops.aten.sqrt.default(add_123);  add_123 = None
    reciprocal_39: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_39);  sqrt_39 = None
    mul_146: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_39, 1);  reciprocal_39 = None
    sub_48: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_218, convert_element_type_78);  view_218 = convert_element_type_78 = None
    mul_147: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_48, mul_146);  sub_48 = mul_146 = None
    mul_148: "f32[392, 512]" = torch.ops.aten.mul.Tensor(mul_147, arg132_1);  mul_147 = arg132_1 = None
    add_124: "f32[392, 512]" = torch.ops.aten.add.Tensor(mul_148, arg133_1);  mul_148 = arg133_1 = None
    view_219: "f32[8, 49, 512]" = torch.ops.aten.view.default(add_124, [8, 49, 512]);  add_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    add_125: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(view_219, 3)
    clamp_min_20: "f32[8, 49, 512]" = torch.ops.aten.clamp_min.default(add_125, 0);  add_125 = None
    clamp_max_20: "f32[8, 49, 512]" = torch.ops.aten.clamp_max.default(clamp_min_20, 6);  clamp_min_20 = None
    mul_149: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(view_219, clamp_max_20);  view_219 = clamp_max_20 = None
    div_29: "f32[8, 49, 512]" = torch.ops.aten.div.Tensor(mul_149, 6);  mul_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:369, code: x = self.drop(x)
    clone_54: "f32[8, 49, 512]" = torch.ops.aten.clone.default(div_29);  div_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_73: "f32[512, 256]" = torch.ops.aten.permute.default(arg134_1, [1, 0]);  arg134_1 = None
    view_220: "f32[392, 512]" = torch.ops.aten.view.default(clone_54, [392, 512]);  clone_54 = None
    mm_36: "f32[392, 256]" = torch.ops.aten.mm.default(view_220, permute_73);  view_220 = permute_73 = None
    view_221: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_36, [8, 49, 256]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_222: "f32[392, 256]" = torch.ops.aten.view.default(view_221, [392, 256]);  view_221 = None
    convert_element_type_80: "f32[256]" = torch.ops.prims.convert_element_type.default(arg342_1, torch.float32);  arg342_1 = None
    convert_element_type_81: "f32[256]" = torch.ops.prims.convert_element_type.default(arg343_1, torch.float32);  arg343_1 = None
    add_126: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_81, 1e-05);  convert_element_type_81 = None
    sqrt_40: "f32[256]" = torch.ops.aten.sqrt.default(add_126);  add_126 = None
    reciprocal_40: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_40);  sqrt_40 = None
    mul_150: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_40, 1);  reciprocal_40 = None
    sub_49: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_222, convert_element_type_80);  view_222 = convert_element_type_80 = None
    mul_151: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_49, mul_150);  sub_49 = mul_150 = None
    mul_152: "f32[392, 256]" = torch.ops.aten.mul.Tensor(mul_151, arg135_1);  mul_151 = arg135_1 = None
    add_127: "f32[392, 256]" = torch.ops.aten.add.Tensor(mul_152, arg136_1);  mul_152 = arg136_1 = None
    view_223: "f32[8, 49, 256]" = torch.ops.aten.view.default(add_127, [8, 49, 256]);  add_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:457, code: x = x + self.drop_path2(self.mlp(x))
    add_128: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(add_122, view_223);  add_122 = view_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_74: "f32[256, 1280]" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
    view_224: "f32[392, 256]" = torch.ops.aten.view.default(add_128, [392, 256])
    mm_37: "f32[392, 1280]" = torch.ops.aten.mm.default(view_224, permute_74);  view_224 = permute_74 = None
    view_225: "f32[8, 49, 1280]" = torch.ops.aten.view.default(mm_37, [8, 49, 1280]);  mm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_226: "f32[392, 1280]" = torch.ops.aten.view.default(view_225, [392, 1280]);  view_225 = None
    convert_element_type_82: "f32[1280]" = torch.ops.prims.convert_element_type.default(arg345_1, torch.float32);  arg345_1 = None
    convert_element_type_83: "f32[1280]" = torch.ops.prims.convert_element_type.default(arg346_1, torch.float32);  arg346_1 = None
    add_129: "f32[1280]" = torch.ops.aten.add.Tensor(convert_element_type_83, 1e-05);  convert_element_type_83 = None
    sqrt_41: "f32[1280]" = torch.ops.aten.sqrt.default(add_129);  add_129 = None
    reciprocal_41: "f32[1280]" = torch.ops.aten.reciprocal.default(sqrt_41);  sqrt_41 = None
    mul_153: "f32[1280]" = torch.ops.aten.mul.Tensor(reciprocal_41, 1);  reciprocal_41 = None
    sub_50: "f32[392, 1280]" = torch.ops.aten.sub.Tensor(view_226, convert_element_type_82);  view_226 = convert_element_type_82 = None
    mul_154: "f32[392, 1280]" = torch.ops.aten.mul.Tensor(sub_50, mul_153);  sub_50 = mul_153 = None
    mul_155: "f32[392, 1280]" = torch.ops.aten.mul.Tensor(mul_154, arg138_1);  mul_154 = arg138_1 = None
    add_130: "f32[392, 1280]" = torch.ops.aten.add.Tensor(mul_155, arg139_1);  mul_155 = arg139_1 = None
    view_227: "f32[8, 49, 1280]" = torch.ops.aten.view.default(add_130, [8, 49, 1280]);  add_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:331, code: k, v = self.kv(x).view(B, N, self.num_heads, -1).split([self.key_dim, self.val_dim], dim=3)
    view_228: "f32[8, 49, 16, 80]" = torch.ops.aten.view.default(view_227, [8, 49, 16, -1]);  view_227 = None
    split_with_sizes_9 = torch.ops.aten.split_with_sizes.default(view_228, [16, 64], 3);  view_228 = None
    getitem_26: "f32[8, 49, 16, 16]" = split_with_sizes_9[0]
    getitem_27: "f32[8, 49, 16, 64]" = split_with_sizes_9[1];  split_with_sizes_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:332, code: k = k.permute(0, 2, 3, 1)  # BHCN
    permute_75: "f32[8, 16, 16, 49]" = torch.ops.aten.permute.default(getitem_26, [0, 2, 3, 1]);  getitem_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:333, code: v = v.permute(0, 2, 1, 3)  # BHNC
    permute_76: "f32[8, 16, 49, 64]" = torch.ops.aten.permute.default(getitem_27, [0, 2, 1, 3]);  getitem_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:157, code: x = x.view(B, self.resolution[0], self.resolution[1], C)
    view_229: "f32[8, 7, 7, 256]" = torch.ops.aten.view.default(add_128, [8, 7, 7, 256]);  add_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:161, code: x = x[:, ::self.stride, ::self.stride]
    slice_13: "f32[8, 7, 7, 256]" = torch.ops.aten.slice.Tensor(view_229, 0, 0, 9223372036854775807);  view_229 = None
    slice_14: "f32[8, 4, 7, 256]" = torch.ops.aten.slice.Tensor(slice_13, 1, 0, 9223372036854775807, 2);  slice_13 = None
    slice_15: "f32[8, 4, 4, 256]" = torch.ops.aten.slice.Tensor(slice_14, 2, 0, 9223372036854775807, 2);  slice_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:162, code: return x.reshape(B, -1, C)
    clone_55: "f32[8, 4, 4, 256]" = torch.ops.aten.clone.default(slice_15, memory_format = torch.contiguous_format);  slice_15 = None
    view_230: "f32[8, 16, 256]" = torch.ops.aten.view.default(clone_55, [8, 16, 256]);  clone_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_77: "f32[256, 256]" = torch.ops.aten.permute.default(arg140_1, [1, 0]);  arg140_1 = None
    view_231: "f32[128, 256]" = torch.ops.aten.view.default(view_230, [128, 256]);  view_230 = None
    mm_38: "f32[128, 256]" = torch.ops.aten.mm.default(view_231, permute_77);  view_231 = permute_77 = None
    view_232: "f32[8, 16, 256]" = torch.ops.aten.view.default(mm_38, [8, 16, 256]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_233: "f32[128, 256]" = torch.ops.aten.view.default(view_232, [128, 256]);  view_232 = None
    convert_element_type_84: "f32[256]" = torch.ops.prims.convert_element_type.default(arg348_1, torch.float32);  arg348_1 = None
    convert_element_type_85: "f32[256]" = torch.ops.prims.convert_element_type.default(arg349_1, torch.float32);  arg349_1 = None
    add_131: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_85, 1e-05);  convert_element_type_85 = None
    sqrt_42: "f32[256]" = torch.ops.aten.sqrt.default(add_131);  add_131 = None
    reciprocal_42: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_42);  sqrt_42 = None
    mul_156: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_42, 1);  reciprocal_42 = None
    sub_51: "f32[128, 256]" = torch.ops.aten.sub.Tensor(view_233, convert_element_type_84);  view_233 = convert_element_type_84 = None
    mul_157: "f32[128, 256]" = torch.ops.aten.mul.Tensor(sub_51, mul_156);  sub_51 = mul_156 = None
    mul_158: "f32[128, 256]" = torch.ops.aten.mul.Tensor(mul_157, arg141_1);  mul_157 = arg141_1 = None
    add_132: "f32[128, 256]" = torch.ops.aten.add.Tensor(mul_158, arg142_1);  mul_158 = arg142_1 = None
    view_234: "f32[8, 16, 256]" = torch.ops.aten.view.default(add_132, [8, 16, 256]);  add_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:334, code: q = self.q(x).view(B, -1, self.num_heads, self.key_dim).permute(0, 2, 1, 3)
    view_235: "f32[8, 16, 16, 16]" = torch.ops.aten.view.default(view_234, [8, -1, 16, 16]);  view_234 = None
    permute_78: "f32[8, 16, 16, 16]" = torch.ops.aten.permute.default(view_235, [0, 2, 1, 3]);  view_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:336, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    expand_36: "f32[8, 16, 16, 16]" = torch.ops.aten.expand.default(permute_78, [8, 16, 16, 16]);  permute_78 = None
    clone_56: "f32[8, 16, 16, 16]" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
    view_236: "f32[128, 16, 16]" = torch.ops.aten.view.default(clone_56, [128, 16, 16]);  clone_56 = None
    expand_37: "f32[8, 16, 16, 49]" = torch.ops.aten.expand.default(permute_75, [8, 16, 16, 49]);  permute_75 = None
    clone_57: "f32[8, 16, 16, 49]" = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
    view_237: "f32[128, 16, 49]" = torch.ops.aten.view.default(clone_57, [128, 16, 49]);  clone_57 = None
    bmm_18: "f32[128, 16, 49]" = torch.ops.aten.bmm.default(view_236, view_237);  view_236 = view_237 = None
    view_238: "f32[8, 16, 16, 49]" = torch.ops.aten.view.default(bmm_18, [8, 16, 16, 49]);  bmm_18 = None
    mul_159: "f32[8, 16, 16, 49]" = torch.ops.aten.mul.Tensor(view_238, 0.25);  view_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:315, code: self.attention_bias_cache[device_key] = self.attention_biases[:, self.attention_bias_idxs]
    slice_16: "f32[16, 49]" = torch.ops.aten.slice.Tensor(arg9_1, 0, 0, 9223372036854775807);  arg9_1 = None
    index_9: "f32[16, 16, 49]" = torch.ops.aten.index.Tensor(slice_16, [None, arg217_1]);  slice_16 = arg217_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:336, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    add_133: "f32[8, 16, 16, 49]" = torch.ops.aten.add.Tensor(mul_159, index_9);  mul_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:337, code: attn = attn.softmax(dim=-1)
    amax_9: "f32[8, 16, 16, 1]" = torch.ops.aten.amax.default(add_133, [-1], True)
    sub_52: "f32[8, 16, 16, 49]" = torch.ops.aten.sub.Tensor(add_133, amax_9);  add_133 = amax_9 = None
    exp_9: "f32[8, 16, 16, 49]" = torch.ops.aten.exp.default(sub_52);  sub_52 = None
    sum_10: "f32[8, 16, 16, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_30: "f32[8, 16, 16, 49]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:339, code: x = (attn @ v).transpose(1, 2).reshape(B, -1, self.val_attn_dim)
    expand_38: "f32[8, 16, 16, 49]" = torch.ops.aten.expand.default(div_30, [8, 16, 16, 49]);  div_30 = None
    view_239: "f32[128, 16, 49]" = torch.ops.aten.view.default(expand_38, [128, 16, 49]);  expand_38 = None
    expand_39: "f32[8, 16, 49, 64]" = torch.ops.aten.expand.default(permute_76, [8, 16, 49, 64]);  permute_76 = None
    clone_58: "f32[8, 16, 49, 64]" = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
    view_240: "f32[128, 49, 64]" = torch.ops.aten.view.default(clone_58, [128, 49, 64]);  clone_58 = None
    bmm_19: "f32[128, 16, 64]" = torch.ops.aten.bmm.default(view_239, view_240);  view_239 = view_240 = None
    view_241: "f32[8, 16, 16, 64]" = torch.ops.aten.view.default(bmm_19, [8, 16, 16, 64]);  bmm_19 = None
    permute_79: "f32[8, 16, 16, 64]" = torch.ops.aten.permute.default(view_241, [0, 2, 1, 3]);  view_241 = None
    clone_59: "f32[8, 16, 16, 64]" = torch.ops.aten.clone.default(permute_79, memory_format = torch.contiguous_format);  permute_79 = None
    view_242: "f32[8, 16, 1024]" = torch.ops.aten.view.default(clone_59, [8, 16, 1024]);  clone_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:340, code: x = self.proj(x)
    add_134: "f32[8, 16, 1024]" = torch.ops.aten.add.Tensor(view_242, 3)
    clamp_min_21: "f32[8, 16, 1024]" = torch.ops.aten.clamp_min.default(add_134, 0);  add_134 = None
    clamp_max_21: "f32[8, 16, 1024]" = torch.ops.aten.clamp_max.default(clamp_min_21, 6);  clamp_min_21 = None
    mul_160: "f32[8, 16, 1024]" = torch.ops.aten.mul.Tensor(view_242, clamp_max_21);  view_242 = clamp_max_21 = None
    div_31: "f32[8, 16, 1024]" = torch.ops.aten.div.Tensor(mul_160, 6);  mul_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_80: "f32[1024, 384]" = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
    view_243: "f32[128, 1024]" = torch.ops.aten.view.default(div_31, [128, 1024]);  div_31 = None
    mm_39: "f32[128, 384]" = torch.ops.aten.mm.default(view_243, permute_80);  view_243 = permute_80 = None
    view_244: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_39, [8, 16, 384]);  mm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_245: "f32[128, 384]" = torch.ops.aten.view.default(view_244, [128, 384]);  view_244 = None
    convert_element_type_86: "f32[384]" = torch.ops.prims.convert_element_type.default(arg351_1, torch.float32);  arg351_1 = None
    convert_element_type_87: "f32[384]" = torch.ops.prims.convert_element_type.default(arg352_1, torch.float32);  arg352_1 = None
    add_135: "f32[384]" = torch.ops.aten.add.Tensor(convert_element_type_87, 1e-05);  convert_element_type_87 = None
    sqrt_43: "f32[384]" = torch.ops.aten.sqrt.default(add_135);  add_135 = None
    reciprocal_43: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_43);  sqrt_43 = None
    mul_161: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_43, 1);  reciprocal_43 = None
    sub_53: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_245, convert_element_type_86);  view_245 = convert_element_type_86 = None
    mul_162: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_53, mul_161);  sub_53 = mul_161 = None
    mul_163: "f32[128, 384]" = torch.ops.aten.mul.Tensor(mul_162, arg144_1);  mul_162 = arg144_1 = None
    add_136: "f32[128, 384]" = torch.ops.aten.add.Tensor(mul_163, arg145_1);  mul_163 = arg145_1 = None
    view_246: "f32[8, 16, 384]" = torch.ops.aten.view.default(add_136, [8, 16, 384]);  add_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_81: "f32[384, 768]" = torch.ops.aten.permute.default(arg146_1, [1, 0]);  arg146_1 = None
    view_247: "f32[128, 384]" = torch.ops.aten.view.default(view_246, [128, 384])
    mm_40: "f32[128, 768]" = torch.ops.aten.mm.default(view_247, permute_81);  view_247 = permute_81 = None
    view_248: "f32[8, 16, 768]" = torch.ops.aten.view.default(mm_40, [8, 16, 768]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_249: "f32[128, 768]" = torch.ops.aten.view.default(view_248, [128, 768]);  view_248 = None
    convert_element_type_88: "f32[768]" = torch.ops.prims.convert_element_type.default(arg354_1, torch.float32);  arg354_1 = None
    convert_element_type_89: "f32[768]" = torch.ops.prims.convert_element_type.default(arg355_1, torch.float32);  arg355_1 = None
    add_137: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_89, 1e-05);  convert_element_type_89 = None
    sqrt_44: "f32[768]" = torch.ops.aten.sqrt.default(add_137);  add_137 = None
    reciprocal_44: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_44);  sqrt_44 = None
    mul_164: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_44, 1);  reciprocal_44 = None
    sub_54: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_249, convert_element_type_88);  view_249 = convert_element_type_88 = None
    mul_165: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_54, mul_164);  sub_54 = mul_164 = None
    mul_166: "f32[128, 768]" = torch.ops.aten.mul.Tensor(mul_165, arg147_1);  mul_165 = arg147_1 = None
    add_138: "f32[128, 768]" = torch.ops.aten.add.Tensor(mul_166, arg148_1);  mul_166 = arg148_1 = None
    view_250: "f32[8, 16, 768]" = torch.ops.aten.view.default(add_138, [8, 16, 768]);  add_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    add_139: "f32[8, 16, 768]" = torch.ops.aten.add.Tensor(view_250, 3)
    clamp_min_22: "f32[8, 16, 768]" = torch.ops.aten.clamp_min.default(add_139, 0);  add_139 = None
    clamp_max_22: "f32[8, 16, 768]" = torch.ops.aten.clamp_max.default(clamp_min_22, 6);  clamp_min_22 = None
    mul_167: "f32[8, 16, 768]" = torch.ops.aten.mul.Tensor(view_250, clamp_max_22);  view_250 = clamp_max_22 = None
    div_32: "f32[8, 16, 768]" = torch.ops.aten.div.Tensor(mul_167, 6);  mul_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:369, code: x = self.drop(x)
    clone_60: "f32[8, 16, 768]" = torch.ops.aten.clone.default(div_32);  div_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_82: "f32[768, 384]" = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
    view_251: "f32[128, 768]" = torch.ops.aten.view.default(clone_60, [128, 768]);  clone_60 = None
    mm_41: "f32[128, 384]" = torch.ops.aten.mm.default(view_251, permute_82);  view_251 = permute_82 = None
    view_252: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_41, [8, 16, 384]);  mm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_253: "f32[128, 384]" = torch.ops.aten.view.default(view_252, [128, 384]);  view_252 = None
    convert_element_type_90: "f32[384]" = torch.ops.prims.convert_element_type.default(arg357_1, torch.float32);  arg357_1 = None
    convert_element_type_91: "f32[384]" = torch.ops.prims.convert_element_type.default(arg358_1, torch.float32);  arg358_1 = None
    add_140: "f32[384]" = torch.ops.aten.add.Tensor(convert_element_type_91, 1e-05);  convert_element_type_91 = None
    sqrt_45: "f32[384]" = torch.ops.aten.sqrt.default(add_140);  add_140 = None
    reciprocal_45: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_45);  sqrt_45 = None
    mul_168: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_45, 1);  reciprocal_45 = None
    sub_55: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_253, convert_element_type_90);  view_253 = convert_element_type_90 = None
    mul_169: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_55, mul_168);  sub_55 = mul_168 = None
    mul_170: "f32[128, 384]" = torch.ops.aten.mul.Tensor(mul_169, arg150_1);  mul_169 = arg150_1 = None
    add_141: "f32[128, 384]" = torch.ops.aten.add.Tensor(mul_170, arg151_1);  mul_170 = arg151_1 = None
    view_254: "f32[8, 16, 384]" = torch.ops.aten.view.default(add_141, [8, 16, 384]);  add_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:415, code: x = x + self.drop_path(self.mlp(x))
    add_142: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(view_246, view_254);  view_246 = view_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_83: "f32[384, 768]" = torch.ops.aten.permute.default(arg152_1, [1, 0]);  arg152_1 = None
    view_255: "f32[128, 384]" = torch.ops.aten.view.default(add_142, [128, 384])
    mm_42: "f32[128, 768]" = torch.ops.aten.mm.default(view_255, permute_83);  view_255 = permute_83 = None
    view_256: "f32[8, 16, 768]" = torch.ops.aten.view.default(mm_42, [8, 16, 768]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_257: "f32[128, 768]" = torch.ops.aten.view.default(view_256, [128, 768]);  view_256 = None
    convert_element_type_92: "f32[768]" = torch.ops.prims.convert_element_type.default(arg360_1, torch.float32);  arg360_1 = None
    convert_element_type_93: "f32[768]" = torch.ops.prims.convert_element_type.default(arg361_1, torch.float32);  arg361_1 = None
    add_143: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_93, 1e-05);  convert_element_type_93 = None
    sqrt_46: "f32[768]" = torch.ops.aten.sqrt.default(add_143);  add_143 = None
    reciprocal_46: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_46);  sqrt_46 = None
    mul_171: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_46, 1);  reciprocal_46 = None
    sub_56: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_257, convert_element_type_92);  view_257 = convert_element_type_92 = None
    mul_172: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_56, mul_171);  sub_56 = mul_171 = None
    mul_173: "f32[128, 768]" = torch.ops.aten.mul.Tensor(mul_172, arg153_1);  mul_172 = arg153_1 = None
    add_144: "f32[128, 768]" = torch.ops.aten.add.Tensor(mul_173, arg154_1);  mul_173 = arg154_1 = None
    view_258: "f32[8, 16, 768]" = torch.ops.aten.view.default(add_144, [8, 16, 768]);  add_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    view_259: "f32[8, 16, 12, 64]" = torch.ops.aten.view.default(view_258, [8, 16, 12, -1]);  view_258 = None
    split_with_sizes_10 = torch.ops.aten.split_with_sizes.default(view_259, [16, 16, 32], 3);  view_259 = None
    getitem_28: "f32[8, 16, 12, 16]" = split_with_sizes_10[0]
    getitem_29: "f32[8, 16, 12, 16]" = split_with_sizes_10[1]
    getitem_30: "f32[8, 16, 12, 32]" = split_with_sizes_10[2];  split_with_sizes_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    permute_84: "f32[8, 12, 16, 16]" = torch.ops.aten.permute.default(getitem_28, [0, 2, 1, 3]);  getitem_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    permute_85: "f32[8, 12, 16, 16]" = torch.ops.aten.permute.default(getitem_29, [0, 2, 3, 1]);  getitem_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    permute_86: "f32[8, 12, 16, 32]" = torch.ops.aten.permute.default(getitem_30, [0, 2, 1, 3]);  getitem_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    expand_40: "f32[8, 12, 16, 16]" = torch.ops.aten.expand.default(permute_84, [8, 12, 16, 16]);  permute_84 = None
    clone_61: "f32[8, 12, 16, 16]" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
    view_260: "f32[96, 16, 16]" = torch.ops.aten.view.default(clone_61, [96, 16, 16]);  clone_61 = None
    expand_41: "f32[8, 12, 16, 16]" = torch.ops.aten.expand.default(permute_85, [8, 12, 16, 16]);  permute_85 = None
    clone_62: "f32[8, 12, 16, 16]" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
    view_261: "f32[96, 16, 16]" = torch.ops.aten.view.default(clone_62, [96, 16, 16]);  clone_62 = None
    bmm_20: "f32[96, 16, 16]" = torch.ops.aten.bmm.default(view_260, view_261);  view_260 = view_261 = None
    view_262: "f32[8, 12, 16, 16]" = torch.ops.aten.view.default(bmm_20, [8, 12, 16, 16]);  bmm_20 = None
    mul_174: "f32[8, 12, 16, 16]" = torch.ops.aten.mul.Tensor(view_262, 0.25);  view_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:215, code: self.attention_bias_cache[device_key] = self.attention_biases[:, self.attention_bias_idxs]
    slice_17: "f32[12, 16]" = torch.ops.aten.slice.Tensor(arg10_1, 0, 0, 9223372036854775807);  arg10_1 = None
    index_10: "f32[12, 16, 16]" = torch.ops.aten.index.Tensor(slice_17, [None, arg218_1]);  slice_17 = arg218_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    add_145: "f32[8, 12, 16, 16]" = torch.ops.aten.add.Tensor(mul_174, index_10);  mul_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    amax_10: "f32[8, 12, 16, 1]" = torch.ops.aten.amax.default(add_145, [-1], True)
    sub_57: "f32[8, 12, 16, 16]" = torch.ops.aten.sub.Tensor(add_145, amax_10);  add_145 = amax_10 = None
    exp_10: "f32[8, 12, 16, 16]" = torch.ops.aten.exp.default(sub_57);  sub_57 = None
    sum_11: "f32[8, 12, 16, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_33: "f32[8, 12, 16, 16]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    expand_42: "f32[8, 12, 16, 16]" = torch.ops.aten.expand.default(div_33, [8, 12, 16, 16]);  div_33 = None
    view_263: "f32[96, 16, 16]" = torch.ops.aten.view.default(expand_42, [96, 16, 16]);  expand_42 = None
    expand_43: "f32[8, 12, 16, 32]" = torch.ops.aten.expand.default(permute_86, [8, 12, 16, 32]);  permute_86 = None
    clone_63: "f32[8, 12, 16, 32]" = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
    view_264: "f32[96, 16, 32]" = torch.ops.aten.view.default(clone_63, [96, 16, 32]);  clone_63 = None
    bmm_21: "f32[96, 16, 32]" = torch.ops.aten.bmm.default(view_263, view_264);  view_263 = view_264 = None
    view_265: "f32[8, 12, 16, 32]" = torch.ops.aten.view.default(bmm_21, [8, 12, 16, 32]);  bmm_21 = None
    permute_87: "f32[8, 16, 12, 32]" = torch.ops.aten.permute.default(view_265, [0, 2, 1, 3]);  view_265 = None
    clone_64: "f32[8, 16, 12, 32]" = torch.ops.aten.clone.default(permute_87, memory_format = torch.contiguous_format);  permute_87 = None
    view_266: "f32[8, 16, 384]" = torch.ops.aten.view.default(clone_64, [8, 16, 384]);  clone_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    add_146: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(view_266, 3)
    clamp_min_23: "f32[8, 16, 384]" = torch.ops.aten.clamp_min.default(add_146, 0);  add_146 = None
    clamp_max_23: "f32[8, 16, 384]" = torch.ops.aten.clamp_max.default(clamp_min_23, 6);  clamp_min_23 = None
    mul_175: "f32[8, 16, 384]" = torch.ops.aten.mul.Tensor(view_266, clamp_max_23);  view_266 = clamp_max_23 = None
    div_34: "f32[8, 16, 384]" = torch.ops.aten.div.Tensor(mul_175, 6);  mul_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_88: "f32[384, 384]" = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
    view_267: "f32[128, 384]" = torch.ops.aten.view.default(div_34, [128, 384]);  div_34 = None
    mm_43: "f32[128, 384]" = torch.ops.aten.mm.default(view_267, permute_88);  view_267 = permute_88 = None
    view_268: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_43, [8, 16, 384]);  mm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_269: "f32[128, 384]" = torch.ops.aten.view.default(view_268, [128, 384]);  view_268 = None
    convert_element_type_94: "f32[384]" = torch.ops.prims.convert_element_type.default(arg363_1, torch.float32);  arg363_1 = None
    convert_element_type_95: "f32[384]" = torch.ops.prims.convert_element_type.default(arg364_1, torch.float32);  arg364_1 = None
    add_147: "f32[384]" = torch.ops.aten.add.Tensor(convert_element_type_95, 1e-05);  convert_element_type_95 = None
    sqrt_47: "f32[384]" = torch.ops.aten.sqrt.default(add_147);  add_147 = None
    reciprocal_47: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_47);  sqrt_47 = None
    mul_176: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_47, 1);  reciprocal_47 = None
    sub_58: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_269, convert_element_type_94);  view_269 = convert_element_type_94 = None
    mul_177: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_58, mul_176);  sub_58 = mul_176 = None
    mul_178: "f32[128, 384]" = torch.ops.aten.mul.Tensor(mul_177, arg156_1);  mul_177 = arg156_1 = None
    add_148: "f32[128, 384]" = torch.ops.aten.add.Tensor(mul_178, arg157_1);  mul_178 = arg157_1 = None
    view_270: "f32[8, 16, 384]" = torch.ops.aten.view.default(add_148, [8, 16, 384]);  add_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:456, code: x = x + self.drop_path1(self.attn(x))
    add_149: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(add_142, view_270);  add_142 = view_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_89: "f32[384, 768]" = torch.ops.aten.permute.default(arg158_1, [1, 0]);  arg158_1 = None
    view_271: "f32[128, 384]" = torch.ops.aten.view.default(add_149, [128, 384])
    mm_44: "f32[128, 768]" = torch.ops.aten.mm.default(view_271, permute_89);  view_271 = permute_89 = None
    view_272: "f32[8, 16, 768]" = torch.ops.aten.view.default(mm_44, [8, 16, 768]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_273: "f32[128, 768]" = torch.ops.aten.view.default(view_272, [128, 768]);  view_272 = None
    convert_element_type_96: "f32[768]" = torch.ops.prims.convert_element_type.default(arg366_1, torch.float32);  arg366_1 = None
    convert_element_type_97: "f32[768]" = torch.ops.prims.convert_element_type.default(arg367_1, torch.float32);  arg367_1 = None
    add_150: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_97, 1e-05);  convert_element_type_97 = None
    sqrt_48: "f32[768]" = torch.ops.aten.sqrt.default(add_150);  add_150 = None
    reciprocal_48: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_48);  sqrt_48 = None
    mul_179: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_48, 1);  reciprocal_48 = None
    sub_59: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_273, convert_element_type_96);  view_273 = convert_element_type_96 = None
    mul_180: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_59, mul_179);  sub_59 = mul_179 = None
    mul_181: "f32[128, 768]" = torch.ops.aten.mul.Tensor(mul_180, arg159_1);  mul_180 = arg159_1 = None
    add_151: "f32[128, 768]" = torch.ops.aten.add.Tensor(mul_181, arg160_1);  mul_181 = arg160_1 = None
    view_274: "f32[8, 16, 768]" = torch.ops.aten.view.default(add_151, [8, 16, 768]);  add_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    add_152: "f32[8, 16, 768]" = torch.ops.aten.add.Tensor(view_274, 3)
    clamp_min_24: "f32[8, 16, 768]" = torch.ops.aten.clamp_min.default(add_152, 0);  add_152 = None
    clamp_max_24: "f32[8, 16, 768]" = torch.ops.aten.clamp_max.default(clamp_min_24, 6);  clamp_min_24 = None
    mul_182: "f32[8, 16, 768]" = torch.ops.aten.mul.Tensor(view_274, clamp_max_24);  view_274 = clamp_max_24 = None
    div_35: "f32[8, 16, 768]" = torch.ops.aten.div.Tensor(mul_182, 6);  mul_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:369, code: x = self.drop(x)
    clone_65: "f32[8, 16, 768]" = torch.ops.aten.clone.default(div_35);  div_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_90: "f32[768, 384]" = torch.ops.aten.permute.default(arg161_1, [1, 0]);  arg161_1 = None
    view_275: "f32[128, 768]" = torch.ops.aten.view.default(clone_65, [128, 768]);  clone_65 = None
    mm_45: "f32[128, 384]" = torch.ops.aten.mm.default(view_275, permute_90);  view_275 = permute_90 = None
    view_276: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_45, [8, 16, 384]);  mm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_277: "f32[128, 384]" = torch.ops.aten.view.default(view_276, [128, 384]);  view_276 = None
    convert_element_type_98: "f32[384]" = torch.ops.prims.convert_element_type.default(arg369_1, torch.float32);  arg369_1 = None
    convert_element_type_99: "f32[384]" = torch.ops.prims.convert_element_type.default(arg370_1, torch.float32);  arg370_1 = None
    add_153: "f32[384]" = torch.ops.aten.add.Tensor(convert_element_type_99, 1e-05);  convert_element_type_99 = None
    sqrt_49: "f32[384]" = torch.ops.aten.sqrt.default(add_153);  add_153 = None
    reciprocal_49: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_49);  sqrt_49 = None
    mul_183: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_49, 1);  reciprocal_49 = None
    sub_60: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_277, convert_element_type_98);  view_277 = convert_element_type_98 = None
    mul_184: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_60, mul_183);  sub_60 = mul_183 = None
    mul_185: "f32[128, 384]" = torch.ops.aten.mul.Tensor(mul_184, arg162_1);  mul_184 = arg162_1 = None
    add_154: "f32[128, 384]" = torch.ops.aten.add.Tensor(mul_185, arg163_1);  mul_185 = arg163_1 = None
    view_278: "f32[8, 16, 384]" = torch.ops.aten.view.default(add_154, [8, 16, 384]);  add_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:457, code: x = x + self.drop_path2(self.mlp(x))
    add_155: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(add_149, view_278);  add_149 = view_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_91: "f32[384, 768]" = torch.ops.aten.permute.default(arg164_1, [1, 0]);  arg164_1 = None
    view_279: "f32[128, 384]" = torch.ops.aten.view.default(add_155, [128, 384])
    mm_46: "f32[128, 768]" = torch.ops.aten.mm.default(view_279, permute_91);  view_279 = permute_91 = None
    view_280: "f32[8, 16, 768]" = torch.ops.aten.view.default(mm_46, [8, 16, 768]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_281: "f32[128, 768]" = torch.ops.aten.view.default(view_280, [128, 768]);  view_280 = None
    convert_element_type_100: "f32[768]" = torch.ops.prims.convert_element_type.default(arg372_1, torch.float32);  arg372_1 = None
    convert_element_type_101: "f32[768]" = torch.ops.prims.convert_element_type.default(arg373_1, torch.float32);  arg373_1 = None
    add_156: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_101, 1e-05);  convert_element_type_101 = None
    sqrt_50: "f32[768]" = torch.ops.aten.sqrt.default(add_156);  add_156 = None
    reciprocal_50: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_50);  sqrt_50 = None
    mul_186: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_50, 1);  reciprocal_50 = None
    sub_61: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_281, convert_element_type_100);  view_281 = convert_element_type_100 = None
    mul_187: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_61, mul_186);  sub_61 = mul_186 = None
    mul_188: "f32[128, 768]" = torch.ops.aten.mul.Tensor(mul_187, arg165_1);  mul_187 = arg165_1 = None
    add_157: "f32[128, 768]" = torch.ops.aten.add.Tensor(mul_188, arg166_1);  mul_188 = arg166_1 = None
    view_282: "f32[8, 16, 768]" = torch.ops.aten.view.default(add_157, [8, 16, 768]);  add_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    view_283: "f32[8, 16, 12, 64]" = torch.ops.aten.view.default(view_282, [8, 16, 12, -1]);  view_282 = None
    split_with_sizes_11 = torch.ops.aten.split_with_sizes.default(view_283, [16, 16, 32], 3);  view_283 = None
    getitem_31: "f32[8, 16, 12, 16]" = split_with_sizes_11[0]
    getitem_32: "f32[8, 16, 12, 16]" = split_with_sizes_11[1]
    getitem_33: "f32[8, 16, 12, 32]" = split_with_sizes_11[2];  split_with_sizes_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    permute_92: "f32[8, 12, 16, 16]" = torch.ops.aten.permute.default(getitem_31, [0, 2, 1, 3]);  getitem_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    permute_93: "f32[8, 12, 16, 16]" = torch.ops.aten.permute.default(getitem_32, [0, 2, 3, 1]);  getitem_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    permute_94: "f32[8, 12, 16, 32]" = torch.ops.aten.permute.default(getitem_33, [0, 2, 1, 3]);  getitem_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    expand_44: "f32[8, 12, 16, 16]" = torch.ops.aten.expand.default(permute_92, [8, 12, 16, 16]);  permute_92 = None
    clone_66: "f32[8, 12, 16, 16]" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
    view_284: "f32[96, 16, 16]" = torch.ops.aten.view.default(clone_66, [96, 16, 16]);  clone_66 = None
    expand_45: "f32[8, 12, 16, 16]" = torch.ops.aten.expand.default(permute_93, [8, 12, 16, 16]);  permute_93 = None
    clone_67: "f32[8, 12, 16, 16]" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
    view_285: "f32[96, 16, 16]" = torch.ops.aten.view.default(clone_67, [96, 16, 16]);  clone_67 = None
    bmm_22: "f32[96, 16, 16]" = torch.ops.aten.bmm.default(view_284, view_285);  view_284 = view_285 = None
    view_286: "f32[8, 12, 16, 16]" = torch.ops.aten.view.default(bmm_22, [8, 12, 16, 16]);  bmm_22 = None
    mul_189: "f32[8, 12, 16, 16]" = torch.ops.aten.mul.Tensor(view_286, 0.25);  view_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:215, code: self.attention_bias_cache[device_key] = self.attention_biases[:, self.attention_bias_idxs]
    slice_18: "f32[12, 16]" = torch.ops.aten.slice.Tensor(arg11_1, 0, 0, 9223372036854775807);  arg11_1 = None
    index_11: "f32[12, 16, 16]" = torch.ops.aten.index.Tensor(slice_18, [None, arg219_1]);  slice_18 = arg219_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    add_158: "f32[8, 12, 16, 16]" = torch.ops.aten.add.Tensor(mul_189, index_11);  mul_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    amax_11: "f32[8, 12, 16, 1]" = torch.ops.aten.amax.default(add_158, [-1], True)
    sub_62: "f32[8, 12, 16, 16]" = torch.ops.aten.sub.Tensor(add_158, amax_11);  add_158 = amax_11 = None
    exp_11: "f32[8, 12, 16, 16]" = torch.ops.aten.exp.default(sub_62);  sub_62 = None
    sum_12: "f32[8, 12, 16, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_36: "f32[8, 12, 16, 16]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    expand_46: "f32[8, 12, 16, 16]" = torch.ops.aten.expand.default(div_36, [8, 12, 16, 16]);  div_36 = None
    view_287: "f32[96, 16, 16]" = torch.ops.aten.view.default(expand_46, [96, 16, 16]);  expand_46 = None
    expand_47: "f32[8, 12, 16, 32]" = torch.ops.aten.expand.default(permute_94, [8, 12, 16, 32]);  permute_94 = None
    clone_68: "f32[8, 12, 16, 32]" = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
    view_288: "f32[96, 16, 32]" = torch.ops.aten.view.default(clone_68, [96, 16, 32]);  clone_68 = None
    bmm_23: "f32[96, 16, 32]" = torch.ops.aten.bmm.default(view_287, view_288);  view_287 = view_288 = None
    view_289: "f32[8, 12, 16, 32]" = torch.ops.aten.view.default(bmm_23, [8, 12, 16, 32]);  bmm_23 = None
    permute_95: "f32[8, 16, 12, 32]" = torch.ops.aten.permute.default(view_289, [0, 2, 1, 3]);  view_289 = None
    clone_69: "f32[8, 16, 12, 32]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
    view_290: "f32[8, 16, 384]" = torch.ops.aten.view.default(clone_69, [8, 16, 384]);  clone_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    add_159: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(view_290, 3)
    clamp_min_25: "f32[8, 16, 384]" = torch.ops.aten.clamp_min.default(add_159, 0);  add_159 = None
    clamp_max_25: "f32[8, 16, 384]" = torch.ops.aten.clamp_max.default(clamp_min_25, 6);  clamp_min_25 = None
    mul_190: "f32[8, 16, 384]" = torch.ops.aten.mul.Tensor(view_290, clamp_max_25);  view_290 = clamp_max_25 = None
    div_37: "f32[8, 16, 384]" = torch.ops.aten.div.Tensor(mul_190, 6);  mul_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_96: "f32[384, 384]" = torch.ops.aten.permute.default(arg167_1, [1, 0]);  arg167_1 = None
    view_291: "f32[128, 384]" = torch.ops.aten.view.default(div_37, [128, 384]);  div_37 = None
    mm_47: "f32[128, 384]" = torch.ops.aten.mm.default(view_291, permute_96);  view_291 = permute_96 = None
    view_292: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_47, [8, 16, 384]);  mm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_293: "f32[128, 384]" = torch.ops.aten.view.default(view_292, [128, 384]);  view_292 = None
    convert_element_type_102: "f32[384]" = torch.ops.prims.convert_element_type.default(arg375_1, torch.float32);  arg375_1 = None
    convert_element_type_103: "f32[384]" = torch.ops.prims.convert_element_type.default(arg376_1, torch.float32);  arg376_1 = None
    add_160: "f32[384]" = torch.ops.aten.add.Tensor(convert_element_type_103, 1e-05);  convert_element_type_103 = None
    sqrt_51: "f32[384]" = torch.ops.aten.sqrt.default(add_160);  add_160 = None
    reciprocal_51: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_51);  sqrt_51 = None
    mul_191: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_51, 1);  reciprocal_51 = None
    sub_63: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_293, convert_element_type_102);  view_293 = convert_element_type_102 = None
    mul_192: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_63, mul_191);  sub_63 = mul_191 = None
    mul_193: "f32[128, 384]" = torch.ops.aten.mul.Tensor(mul_192, arg168_1);  mul_192 = arg168_1 = None
    add_161: "f32[128, 384]" = torch.ops.aten.add.Tensor(mul_193, arg169_1);  mul_193 = arg169_1 = None
    view_294: "f32[8, 16, 384]" = torch.ops.aten.view.default(add_161, [8, 16, 384]);  add_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:456, code: x = x + self.drop_path1(self.attn(x))
    add_162: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(add_155, view_294);  add_155 = view_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_97: "f32[384, 768]" = torch.ops.aten.permute.default(arg170_1, [1, 0]);  arg170_1 = None
    view_295: "f32[128, 384]" = torch.ops.aten.view.default(add_162, [128, 384])
    mm_48: "f32[128, 768]" = torch.ops.aten.mm.default(view_295, permute_97);  view_295 = permute_97 = None
    view_296: "f32[8, 16, 768]" = torch.ops.aten.view.default(mm_48, [8, 16, 768]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_297: "f32[128, 768]" = torch.ops.aten.view.default(view_296, [128, 768]);  view_296 = None
    convert_element_type_104: "f32[768]" = torch.ops.prims.convert_element_type.default(arg378_1, torch.float32);  arg378_1 = None
    convert_element_type_105: "f32[768]" = torch.ops.prims.convert_element_type.default(arg379_1, torch.float32);  arg379_1 = None
    add_163: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_105, 1e-05);  convert_element_type_105 = None
    sqrt_52: "f32[768]" = torch.ops.aten.sqrt.default(add_163);  add_163 = None
    reciprocal_52: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_52);  sqrt_52 = None
    mul_194: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_52, 1);  reciprocal_52 = None
    sub_64: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_297, convert_element_type_104);  view_297 = convert_element_type_104 = None
    mul_195: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_64, mul_194);  sub_64 = mul_194 = None
    mul_196: "f32[128, 768]" = torch.ops.aten.mul.Tensor(mul_195, arg171_1);  mul_195 = arg171_1 = None
    add_164: "f32[128, 768]" = torch.ops.aten.add.Tensor(mul_196, arg172_1);  mul_196 = arg172_1 = None
    view_298: "f32[8, 16, 768]" = torch.ops.aten.view.default(add_164, [8, 16, 768]);  add_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    add_165: "f32[8, 16, 768]" = torch.ops.aten.add.Tensor(view_298, 3)
    clamp_min_26: "f32[8, 16, 768]" = torch.ops.aten.clamp_min.default(add_165, 0);  add_165 = None
    clamp_max_26: "f32[8, 16, 768]" = torch.ops.aten.clamp_max.default(clamp_min_26, 6);  clamp_min_26 = None
    mul_197: "f32[8, 16, 768]" = torch.ops.aten.mul.Tensor(view_298, clamp_max_26);  view_298 = clamp_max_26 = None
    div_38: "f32[8, 16, 768]" = torch.ops.aten.div.Tensor(mul_197, 6);  mul_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:369, code: x = self.drop(x)
    clone_70: "f32[8, 16, 768]" = torch.ops.aten.clone.default(div_38);  div_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_98: "f32[768, 384]" = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
    view_299: "f32[128, 768]" = torch.ops.aten.view.default(clone_70, [128, 768]);  clone_70 = None
    mm_49: "f32[128, 384]" = torch.ops.aten.mm.default(view_299, permute_98);  view_299 = permute_98 = None
    view_300: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_49, [8, 16, 384]);  mm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_301: "f32[128, 384]" = torch.ops.aten.view.default(view_300, [128, 384]);  view_300 = None
    convert_element_type_106: "f32[384]" = torch.ops.prims.convert_element_type.default(arg381_1, torch.float32);  arg381_1 = None
    convert_element_type_107: "f32[384]" = torch.ops.prims.convert_element_type.default(arg382_1, torch.float32);  arg382_1 = None
    add_166: "f32[384]" = torch.ops.aten.add.Tensor(convert_element_type_107, 1e-05);  convert_element_type_107 = None
    sqrt_53: "f32[384]" = torch.ops.aten.sqrt.default(add_166);  add_166 = None
    reciprocal_53: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_53);  sqrt_53 = None
    mul_198: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_53, 1);  reciprocal_53 = None
    sub_65: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_301, convert_element_type_106);  view_301 = convert_element_type_106 = None
    mul_199: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_65, mul_198);  sub_65 = mul_198 = None
    mul_200: "f32[128, 384]" = torch.ops.aten.mul.Tensor(mul_199, arg174_1);  mul_199 = arg174_1 = None
    add_167: "f32[128, 384]" = torch.ops.aten.add.Tensor(mul_200, arg175_1);  mul_200 = arg175_1 = None
    view_302: "f32[8, 16, 384]" = torch.ops.aten.view.default(add_167, [8, 16, 384]);  add_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:457, code: x = x + self.drop_path2(self.mlp(x))
    add_168: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(add_162, view_302);  add_162 = view_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_99: "f32[384, 768]" = torch.ops.aten.permute.default(arg176_1, [1, 0]);  arg176_1 = None
    view_303: "f32[128, 384]" = torch.ops.aten.view.default(add_168, [128, 384])
    mm_50: "f32[128, 768]" = torch.ops.aten.mm.default(view_303, permute_99);  view_303 = permute_99 = None
    view_304: "f32[8, 16, 768]" = torch.ops.aten.view.default(mm_50, [8, 16, 768]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_305: "f32[128, 768]" = torch.ops.aten.view.default(view_304, [128, 768]);  view_304 = None
    convert_element_type_108: "f32[768]" = torch.ops.prims.convert_element_type.default(arg384_1, torch.float32);  arg384_1 = None
    convert_element_type_109: "f32[768]" = torch.ops.prims.convert_element_type.default(arg385_1, torch.float32);  arg385_1 = None
    add_169: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_109, 1e-05);  convert_element_type_109 = None
    sqrt_54: "f32[768]" = torch.ops.aten.sqrt.default(add_169);  add_169 = None
    reciprocal_54: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_54);  sqrt_54 = None
    mul_201: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_54, 1);  reciprocal_54 = None
    sub_66: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_305, convert_element_type_108);  view_305 = convert_element_type_108 = None
    mul_202: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_66, mul_201);  sub_66 = mul_201 = None
    mul_203: "f32[128, 768]" = torch.ops.aten.mul.Tensor(mul_202, arg177_1);  mul_202 = arg177_1 = None
    add_170: "f32[128, 768]" = torch.ops.aten.add.Tensor(mul_203, arg178_1);  mul_203 = arg178_1 = None
    view_306: "f32[8, 16, 768]" = torch.ops.aten.view.default(add_170, [8, 16, 768]);  add_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    view_307: "f32[8, 16, 12, 64]" = torch.ops.aten.view.default(view_306, [8, 16, 12, -1]);  view_306 = None
    split_with_sizes_12 = torch.ops.aten.split_with_sizes.default(view_307, [16, 16, 32], 3);  view_307 = None
    getitem_34: "f32[8, 16, 12, 16]" = split_with_sizes_12[0]
    getitem_35: "f32[8, 16, 12, 16]" = split_with_sizes_12[1]
    getitem_36: "f32[8, 16, 12, 32]" = split_with_sizes_12[2];  split_with_sizes_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    permute_100: "f32[8, 12, 16, 16]" = torch.ops.aten.permute.default(getitem_34, [0, 2, 1, 3]);  getitem_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    permute_101: "f32[8, 12, 16, 16]" = torch.ops.aten.permute.default(getitem_35, [0, 2, 3, 1]);  getitem_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    permute_102: "f32[8, 12, 16, 32]" = torch.ops.aten.permute.default(getitem_36, [0, 2, 1, 3]);  getitem_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    expand_48: "f32[8, 12, 16, 16]" = torch.ops.aten.expand.default(permute_100, [8, 12, 16, 16]);  permute_100 = None
    clone_71: "f32[8, 12, 16, 16]" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
    view_308: "f32[96, 16, 16]" = torch.ops.aten.view.default(clone_71, [96, 16, 16]);  clone_71 = None
    expand_49: "f32[8, 12, 16, 16]" = torch.ops.aten.expand.default(permute_101, [8, 12, 16, 16]);  permute_101 = None
    clone_72: "f32[8, 12, 16, 16]" = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
    view_309: "f32[96, 16, 16]" = torch.ops.aten.view.default(clone_72, [96, 16, 16]);  clone_72 = None
    bmm_24: "f32[96, 16, 16]" = torch.ops.aten.bmm.default(view_308, view_309);  view_308 = view_309 = None
    view_310: "f32[8, 12, 16, 16]" = torch.ops.aten.view.default(bmm_24, [8, 12, 16, 16]);  bmm_24 = None
    mul_204: "f32[8, 12, 16, 16]" = torch.ops.aten.mul.Tensor(view_310, 0.25);  view_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:215, code: self.attention_bias_cache[device_key] = self.attention_biases[:, self.attention_bias_idxs]
    slice_19: "f32[12, 16]" = torch.ops.aten.slice.Tensor(arg12_1, 0, 0, 9223372036854775807);  arg12_1 = None
    index_12: "f32[12, 16, 16]" = torch.ops.aten.index.Tensor(slice_19, [None, arg220_1]);  slice_19 = arg220_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    add_171: "f32[8, 12, 16, 16]" = torch.ops.aten.add.Tensor(mul_204, index_12);  mul_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    amax_12: "f32[8, 12, 16, 1]" = torch.ops.aten.amax.default(add_171, [-1], True)
    sub_67: "f32[8, 12, 16, 16]" = torch.ops.aten.sub.Tensor(add_171, amax_12);  add_171 = amax_12 = None
    exp_12: "f32[8, 12, 16, 16]" = torch.ops.aten.exp.default(sub_67);  sub_67 = None
    sum_13: "f32[8, 12, 16, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
    div_39: "f32[8, 12, 16, 16]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    expand_50: "f32[8, 12, 16, 16]" = torch.ops.aten.expand.default(div_39, [8, 12, 16, 16]);  div_39 = None
    view_311: "f32[96, 16, 16]" = torch.ops.aten.view.default(expand_50, [96, 16, 16]);  expand_50 = None
    expand_51: "f32[8, 12, 16, 32]" = torch.ops.aten.expand.default(permute_102, [8, 12, 16, 32]);  permute_102 = None
    clone_73: "f32[8, 12, 16, 32]" = torch.ops.aten.clone.default(expand_51, memory_format = torch.contiguous_format);  expand_51 = None
    view_312: "f32[96, 16, 32]" = torch.ops.aten.view.default(clone_73, [96, 16, 32]);  clone_73 = None
    bmm_25: "f32[96, 16, 32]" = torch.ops.aten.bmm.default(view_311, view_312);  view_311 = view_312 = None
    view_313: "f32[8, 12, 16, 32]" = torch.ops.aten.view.default(bmm_25, [8, 12, 16, 32]);  bmm_25 = None
    permute_103: "f32[8, 16, 12, 32]" = torch.ops.aten.permute.default(view_313, [0, 2, 1, 3]);  view_313 = None
    clone_74: "f32[8, 16, 12, 32]" = torch.ops.aten.clone.default(permute_103, memory_format = torch.contiguous_format);  permute_103 = None
    view_314: "f32[8, 16, 384]" = torch.ops.aten.view.default(clone_74, [8, 16, 384]);  clone_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    add_172: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(view_314, 3)
    clamp_min_27: "f32[8, 16, 384]" = torch.ops.aten.clamp_min.default(add_172, 0);  add_172 = None
    clamp_max_27: "f32[8, 16, 384]" = torch.ops.aten.clamp_max.default(clamp_min_27, 6);  clamp_min_27 = None
    mul_205: "f32[8, 16, 384]" = torch.ops.aten.mul.Tensor(view_314, clamp_max_27);  view_314 = clamp_max_27 = None
    div_40: "f32[8, 16, 384]" = torch.ops.aten.div.Tensor(mul_205, 6);  mul_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_104: "f32[384, 384]" = torch.ops.aten.permute.default(arg179_1, [1, 0]);  arg179_1 = None
    view_315: "f32[128, 384]" = torch.ops.aten.view.default(div_40, [128, 384]);  div_40 = None
    mm_51: "f32[128, 384]" = torch.ops.aten.mm.default(view_315, permute_104);  view_315 = permute_104 = None
    view_316: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_51, [8, 16, 384]);  mm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_317: "f32[128, 384]" = torch.ops.aten.view.default(view_316, [128, 384]);  view_316 = None
    convert_element_type_110: "f32[384]" = torch.ops.prims.convert_element_type.default(arg387_1, torch.float32);  arg387_1 = None
    convert_element_type_111: "f32[384]" = torch.ops.prims.convert_element_type.default(arg388_1, torch.float32);  arg388_1 = None
    add_173: "f32[384]" = torch.ops.aten.add.Tensor(convert_element_type_111, 1e-05);  convert_element_type_111 = None
    sqrt_55: "f32[384]" = torch.ops.aten.sqrt.default(add_173);  add_173 = None
    reciprocal_55: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_55);  sqrt_55 = None
    mul_206: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_55, 1);  reciprocal_55 = None
    sub_68: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_317, convert_element_type_110);  view_317 = convert_element_type_110 = None
    mul_207: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_68, mul_206);  sub_68 = mul_206 = None
    mul_208: "f32[128, 384]" = torch.ops.aten.mul.Tensor(mul_207, arg180_1);  mul_207 = arg180_1 = None
    add_174: "f32[128, 384]" = torch.ops.aten.add.Tensor(mul_208, arg181_1);  mul_208 = arg181_1 = None
    view_318: "f32[8, 16, 384]" = torch.ops.aten.view.default(add_174, [8, 16, 384]);  add_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:456, code: x = x + self.drop_path1(self.attn(x))
    add_175: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(add_168, view_318);  add_168 = view_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_105: "f32[384, 768]" = torch.ops.aten.permute.default(arg182_1, [1, 0]);  arg182_1 = None
    view_319: "f32[128, 384]" = torch.ops.aten.view.default(add_175, [128, 384])
    mm_52: "f32[128, 768]" = torch.ops.aten.mm.default(view_319, permute_105);  view_319 = permute_105 = None
    view_320: "f32[8, 16, 768]" = torch.ops.aten.view.default(mm_52, [8, 16, 768]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_321: "f32[128, 768]" = torch.ops.aten.view.default(view_320, [128, 768]);  view_320 = None
    convert_element_type_112: "f32[768]" = torch.ops.prims.convert_element_type.default(arg390_1, torch.float32);  arg390_1 = None
    convert_element_type_113: "f32[768]" = torch.ops.prims.convert_element_type.default(arg391_1, torch.float32);  arg391_1 = None
    add_176: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_113, 1e-05);  convert_element_type_113 = None
    sqrt_56: "f32[768]" = torch.ops.aten.sqrt.default(add_176);  add_176 = None
    reciprocal_56: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_56);  sqrt_56 = None
    mul_209: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_56, 1);  reciprocal_56 = None
    sub_69: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_321, convert_element_type_112);  view_321 = convert_element_type_112 = None
    mul_210: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_69, mul_209);  sub_69 = mul_209 = None
    mul_211: "f32[128, 768]" = torch.ops.aten.mul.Tensor(mul_210, arg183_1);  mul_210 = arg183_1 = None
    add_177: "f32[128, 768]" = torch.ops.aten.add.Tensor(mul_211, arg184_1);  mul_211 = arg184_1 = None
    view_322: "f32[8, 16, 768]" = torch.ops.aten.view.default(add_177, [8, 16, 768]);  add_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    add_178: "f32[8, 16, 768]" = torch.ops.aten.add.Tensor(view_322, 3)
    clamp_min_28: "f32[8, 16, 768]" = torch.ops.aten.clamp_min.default(add_178, 0);  add_178 = None
    clamp_max_28: "f32[8, 16, 768]" = torch.ops.aten.clamp_max.default(clamp_min_28, 6);  clamp_min_28 = None
    mul_212: "f32[8, 16, 768]" = torch.ops.aten.mul.Tensor(view_322, clamp_max_28);  view_322 = clamp_max_28 = None
    div_41: "f32[8, 16, 768]" = torch.ops.aten.div.Tensor(mul_212, 6);  mul_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:369, code: x = self.drop(x)
    clone_75: "f32[8, 16, 768]" = torch.ops.aten.clone.default(div_41);  div_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_106: "f32[768, 384]" = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
    view_323: "f32[128, 768]" = torch.ops.aten.view.default(clone_75, [128, 768]);  clone_75 = None
    mm_53: "f32[128, 384]" = torch.ops.aten.mm.default(view_323, permute_106);  view_323 = permute_106 = None
    view_324: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_53, [8, 16, 384]);  mm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_325: "f32[128, 384]" = torch.ops.aten.view.default(view_324, [128, 384]);  view_324 = None
    convert_element_type_114: "f32[384]" = torch.ops.prims.convert_element_type.default(arg393_1, torch.float32);  arg393_1 = None
    convert_element_type_115: "f32[384]" = torch.ops.prims.convert_element_type.default(arg394_1, torch.float32);  arg394_1 = None
    add_179: "f32[384]" = torch.ops.aten.add.Tensor(convert_element_type_115, 1e-05);  convert_element_type_115 = None
    sqrt_57: "f32[384]" = torch.ops.aten.sqrt.default(add_179);  add_179 = None
    reciprocal_57: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_57);  sqrt_57 = None
    mul_213: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_57, 1);  reciprocal_57 = None
    sub_70: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_325, convert_element_type_114);  view_325 = convert_element_type_114 = None
    mul_214: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_70, mul_213);  sub_70 = mul_213 = None
    mul_215: "f32[128, 384]" = torch.ops.aten.mul.Tensor(mul_214, arg186_1);  mul_214 = arg186_1 = None
    add_180: "f32[128, 384]" = torch.ops.aten.add.Tensor(mul_215, arg187_1);  mul_215 = arg187_1 = None
    view_326: "f32[8, 16, 384]" = torch.ops.aten.view.default(add_180, [8, 16, 384]);  add_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:457, code: x = x + self.drop_path2(self.mlp(x))
    add_181: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(add_175, view_326);  add_175 = view_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_107: "f32[384, 768]" = torch.ops.aten.permute.default(arg188_1, [1, 0]);  arg188_1 = None
    view_327: "f32[128, 384]" = torch.ops.aten.view.default(add_181, [128, 384])
    mm_54: "f32[128, 768]" = torch.ops.aten.mm.default(view_327, permute_107);  view_327 = permute_107 = None
    view_328: "f32[8, 16, 768]" = torch.ops.aten.view.default(mm_54, [8, 16, 768]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_329: "f32[128, 768]" = torch.ops.aten.view.default(view_328, [128, 768]);  view_328 = None
    convert_element_type_116: "f32[768]" = torch.ops.prims.convert_element_type.default(arg396_1, torch.float32);  arg396_1 = None
    convert_element_type_117: "f32[768]" = torch.ops.prims.convert_element_type.default(arg397_1, torch.float32);  arg397_1 = None
    add_182: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_117, 1e-05);  convert_element_type_117 = None
    sqrt_58: "f32[768]" = torch.ops.aten.sqrt.default(add_182);  add_182 = None
    reciprocal_58: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_58);  sqrt_58 = None
    mul_216: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_58, 1);  reciprocal_58 = None
    sub_71: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_329, convert_element_type_116);  view_329 = convert_element_type_116 = None
    mul_217: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_71, mul_216);  sub_71 = mul_216 = None
    mul_218: "f32[128, 768]" = torch.ops.aten.mul.Tensor(mul_217, arg189_1);  mul_217 = arg189_1 = None
    add_183: "f32[128, 768]" = torch.ops.aten.add.Tensor(mul_218, arg190_1);  mul_218 = arg190_1 = None
    view_330: "f32[8, 16, 768]" = torch.ops.aten.view.default(add_183, [8, 16, 768]);  add_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    view_331: "f32[8, 16, 12, 64]" = torch.ops.aten.view.default(view_330, [8, 16, 12, -1]);  view_330 = None
    split_with_sizes_13 = torch.ops.aten.split_with_sizes.default(view_331, [16, 16, 32], 3);  view_331 = None
    getitem_37: "f32[8, 16, 12, 16]" = split_with_sizes_13[0]
    getitem_38: "f32[8, 16, 12, 16]" = split_with_sizes_13[1]
    getitem_39: "f32[8, 16, 12, 32]" = split_with_sizes_13[2];  split_with_sizes_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    permute_108: "f32[8, 12, 16, 16]" = torch.ops.aten.permute.default(getitem_37, [0, 2, 1, 3]);  getitem_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    permute_109: "f32[8, 12, 16, 16]" = torch.ops.aten.permute.default(getitem_38, [0, 2, 3, 1]);  getitem_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    permute_110: "f32[8, 12, 16, 32]" = torch.ops.aten.permute.default(getitem_39, [0, 2, 1, 3]);  getitem_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    expand_52: "f32[8, 12, 16, 16]" = torch.ops.aten.expand.default(permute_108, [8, 12, 16, 16]);  permute_108 = None
    clone_76: "f32[8, 12, 16, 16]" = torch.ops.aten.clone.default(expand_52, memory_format = torch.contiguous_format);  expand_52 = None
    view_332: "f32[96, 16, 16]" = torch.ops.aten.view.default(clone_76, [96, 16, 16]);  clone_76 = None
    expand_53: "f32[8, 12, 16, 16]" = torch.ops.aten.expand.default(permute_109, [8, 12, 16, 16]);  permute_109 = None
    clone_77: "f32[8, 12, 16, 16]" = torch.ops.aten.clone.default(expand_53, memory_format = torch.contiguous_format);  expand_53 = None
    view_333: "f32[96, 16, 16]" = torch.ops.aten.view.default(clone_77, [96, 16, 16]);  clone_77 = None
    bmm_26: "f32[96, 16, 16]" = torch.ops.aten.bmm.default(view_332, view_333);  view_332 = view_333 = None
    view_334: "f32[8, 12, 16, 16]" = torch.ops.aten.view.default(bmm_26, [8, 12, 16, 16]);  bmm_26 = None
    mul_219: "f32[8, 12, 16, 16]" = torch.ops.aten.mul.Tensor(view_334, 0.25);  view_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:215, code: self.attention_bias_cache[device_key] = self.attention_biases[:, self.attention_bias_idxs]
    slice_20: "f32[12, 16]" = torch.ops.aten.slice.Tensor(arg13_1, 0, 0, 9223372036854775807);  arg13_1 = None
    index_13: "f32[12, 16, 16]" = torch.ops.aten.index.Tensor(slice_20, [None, arg221_1]);  slice_20 = arg221_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    add_184: "f32[8, 12, 16, 16]" = torch.ops.aten.add.Tensor(mul_219, index_13);  mul_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    amax_13: "f32[8, 12, 16, 1]" = torch.ops.aten.amax.default(add_184, [-1], True)
    sub_72: "f32[8, 12, 16, 16]" = torch.ops.aten.sub.Tensor(add_184, amax_13);  add_184 = amax_13 = None
    exp_13: "f32[8, 12, 16, 16]" = torch.ops.aten.exp.default(sub_72);  sub_72 = None
    sum_14: "f32[8, 12, 16, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
    div_42: "f32[8, 12, 16, 16]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    expand_54: "f32[8, 12, 16, 16]" = torch.ops.aten.expand.default(div_42, [8, 12, 16, 16]);  div_42 = None
    view_335: "f32[96, 16, 16]" = torch.ops.aten.view.default(expand_54, [96, 16, 16]);  expand_54 = None
    expand_55: "f32[8, 12, 16, 32]" = torch.ops.aten.expand.default(permute_110, [8, 12, 16, 32]);  permute_110 = None
    clone_78: "f32[8, 12, 16, 32]" = torch.ops.aten.clone.default(expand_55, memory_format = torch.contiguous_format);  expand_55 = None
    view_336: "f32[96, 16, 32]" = torch.ops.aten.view.default(clone_78, [96, 16, 32]);  clone_78 = None
    bmm_27: "f32[96, 16, 32]" = torch.ops.aten.bmm.default(view_335, view_336);  view_335 = view_336 = None
    view_337: "f32[8, 12, 16, 32]" = torch.ops.aten.view.default(bmm_27, [8, 12, 16, 32]);  bmm_27 = None
    permute_111: "f32[8, 16, 12, 32]" = torch.ops.aten.permute.default(view_337, [0, 2, 1, 3]);  view_337 = None
    clone_79: "f32[8, 16, 12, 32]" = torch.ops.aten.clone.default(permute_111, memory_format = torch.contiguous_format);  permute_111 = None
    view_338: "f32[8, 16, 384]" = torch.ops.aten.view.default(clone_79, [8, 16, 384]);  clone_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    add_185: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(view_338, 3)
    clamp_min_29: "f32[8, 16, 384]" = torch.ops.aten.clamp_min.default(add_185, 0);  add_185 = None
    clamp_max_29: "f32[8, 16, 384]" = torch.ops.aten.clamp_max.default(clamp_min_29, 6);  clamp_min_29 = None
    mul_220: "f32[8, 16, 384]" = torch.ops.aten.mul.Tensor(view_338, clamp_max_29);  view_338 = clamp_max_29 = None
    div_43: "f32[8, 16, 384]" = torch.ops.aten.div.Tensor(mul_220, 6);  mul_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_112: "f32[384, 384]" = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
    view_339: "f32[128, 384]" = torch.ops.aten.view.default(div_43, [128, 384]);  div_43 = None
    mm_55: "f32[128, 384]" = torch.ops.aten.mm.default(view_339, permute_112);  view_339 = permute_112 = None
    view_340: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_55, [8, 16, 384]);  mm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_341: "f32[128, 384]" = torch.ops.aten.view.default(view_340, [128, 384]);  view_340 = None
    convert_element_type_118: "f32[384]" = torch.ops.prims.convert_element_type.default(arg399_1, torch.float32);  arg399_1 = None
    convert_element_type_119: "f32[384]" = torch.ops.prims.convert_element_type.default(arg400_1, torch.float32);  arg400_1 = None
    add_186: "f32[384]" = torch.ops.aten.add.Tensor(convert_element_type_119, 1e-05);  convert_element_type_119 = None
    sqrt_59: "f32[384]" = torch.ops.aten.sqrt.default(add_186);  add_186 = None
    reciprocal_59: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_59);  sqrt_59 = None
    mul_221: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_59, 1);  reciprocal_59 = None
    sub_73: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_341, convert_element_type_118);  view_341 = convert_element_type_118 = None
    mul_222: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_73, mul_221);  sub_73 = mul_221 = None
    mul_223: "f32[128, 384]" = torch.ops.aten.mul.Tensor(mul_222, arg192_1);  mul_222 = arg192_1 = None
    add_187: "f32[128, 384]" = torch.ops.aten.add.Tensor(mul_223, arg193_1);  mul_223 = arg193_1 = None
    view_342: "f32[8, 16, 384]" = torch.ops.aten.view.default(add_187, [8, 16, 384]);  add_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:456, code: x = x + self.drop_path1(self.attn(x))
    add_188: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(add_181, view_342);  add_181 = view_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_113: "f32[384, 768]" = torch.ops.aten.permute.default(arg194_1, [1, 0]);  arg194_1 = None
    view_343: "f32[128, 384]" = torch.ops.aten.view.default(add_188, [128, 384])
    mm_56: "f32[128, 768]" = torch.ops.aten.mm.default(view_343, permute_113);  view_343 = permute_113 = None
    view_344: "f32[8, 16, 768]" = torch.ops.aten.view.default(mm_56, [8, 16, 768]);  mm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_345: "f32[128, 768]" = torch.ops.aten.view.default(view_344, [128, 768]);  view_344 = None
    convert_element_type_120: "f32[768]" = torch.ops.prims.convert_element_type.default(arg402_1, torch.float32);  arg402_1 = None
    convert_element_type_121: "f32[768]" = torch.ops.prims.convert_element_type.default(arg403_1, torch.float32);  arg403_1 = None
    add_189: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_121, 1e-05);  convert_element_type_121 = None
    sqrt_60: "f32[768]" = torch.ops.aten.sqrt.default(add_189);  add_189 = None
    reciprocal_60: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_60);  sqrt_60 = None
    mul_224: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_60, 1);  reciprocal_60 = None
    sub_74: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_345, convert_element_type_120);  view_345 = convert_element_type_120 = None
    mul_225: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_74, mul_224);  sub_74 = mul_224 = None
    mul_226: "f32[128, 768]" = torch.ops.aten.mul.Tensor(mul_225, arg195_1);  mul_225 = arg195_1 = None
    add_190: "f32[128, 768]" = torch.ops.aten.add.Tensor(mul_226, arg196_1);  mul_226 = arg196_1 = None
    view_346: "f32[8, 16, 768]" = torch.ops.aten.view.default(add_190, [8, 16, 768]);  add_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    add_191: "f32[8, 16, 768]" = torch.ops.aten.add.Tensor(view_346, 3)
    clamp_min_30: "f32[8, 16, 768]" = torch.ops.aten.clamp_min.default(add_191, 0);  add_191 = None
    clamp_max_30: "f32[8, 16, 768]" = torch.ops.aten.clamp_max.default(clamp_min_30, 6);  clamp_min_30 = None
    mul_227: "f32[8, 16, 768]" = torch.ops.aten.mul.Tensor(view_346, clamp_max_30);  view_346 = clamp_max_30 = None
    div_44: "f32[8, 16, 768]" = torch.ops.aten.div.Tensor(mul_227, 6);  mul_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:369, code: x = self.drop(x)
    clone_80: "f32[8, 16, 768]" = torch.ops.aten.clone.default(div_44);  div_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_114: "f32[768, 384]" = torch.ops.aten.permute.default(arg197_1, [1, 0]);  arg197_1 = None
    view_347: "f32[128, 768]" = torch.ops.aten.view.default(clone_80, [128, 768]);  clone_80 = None
    mm_57: "f32[128, 384]" = torch.ops.aten.mm.default(view_347, permute_114);  view_347 = permute_114 = None
    view_348: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_57, [8, 16, 384]);  mm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_349: "f32[128, 384]" = torch.ops.aten.view.default(view_348, [128, 384]);  view_348 = None
    convert_element_type_122: "f32[384]" = torch.ops.prims.convert_element_type.default(arg405_1, torch.float32);  arg405_1 = None
    convert_element_type_123: "f32[384]" = torch.ops.prims.convert_element_type.default(arg406_1, torch.float32);  arg406_1 = None
    add_192: "f32[384]" = torch.ops.aten.add.Tensor(convert_element_type_123, 1e-05);  convert_element_type_123 = None
    sqrt_61: "f32[384]" = torch.ops.aten.sqrt.default(add_192);  add_192 = None
    reciprocal_61: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_61);  sqrt_61 = None
    mul_228: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_61, 1);  reciprocal_61 = None
    sub_75: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_349, convert_element_type_122);  view_349 = convert_element_type_122 = None
    mul_229: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_75, mul_228);  sub_75 = mul_228 = None
    mul_230: "f32[128, 384]" = torch.ops.aten.mul.Tensor(mul_229, arg198_1);  mul_229 = arg198_1 = None
    add_193: "f32[128, 384]" = torch.ops.aten.add.Tensor(mul_230, arg199_1);  mul_230 = arg199_1 = None
    view_350: "f32[8, 16, 384]" = torch.ops.aten.view.default(add_193, [8, 16, 384]);  add_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:457, code: x = x + self.drop_path2(self.mlp(x))
    add_194: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(add_188, view_350);  add_188 = view_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:681, code: x = x.mean(dim=(-2, -1)) if self.use_conv else x.mean(dim=1)
    mean: "f32[8, 384]" = torch.ops.aten.mean.dim(add_194, [1]);  add_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:119, code: return self.linear(self.drop(self.bn(x)))
    convert_element_type_124: "f32[384]" = torch.ops.prims.convert_element_type.default(arg408_1, torch.float32);  arg408_1 = None
    convert_element_type_125: "f32[384]" = torch.ops.prims.convert_element_type.default(arg409_1, torch.float32);  arg409_1 = None
    add_195: "f32[384]" = torch.ops.aten.add.Tensor(convert_element_type_125, 1e-05);  convert_element_type_125 = None
    sqrt_62: "f32[384]" = torch.ops.aten.sqrt.default(add_195);  add_195 = None
    reciprocal_62: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_62);  sqrt_62 = None
    mul_231: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_62, 1);  reciprocal_62 = None
    sub_76: "f32[8, 384]" = torch.ops.aten.sub.Tensor(mean, convert_element_type_124);  convert_element_type_124 = None
    mul_232: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sub_76, mul_231);  sub_76 = mul_231 = None
    mul_233: "f32[8, 384]" = torch.ops.aten.mul.Tensor(mul_232, arg200_1);  mul_232 = arg200_1 = None
    add_196: "f32[8, 384]" = torch.ops.aten.add.Tensor(mul_233, arg201_1);  mul_233 = arg201_1 = None
    clone_81: "f32[8, 384]" = torch.ops.aten.clone.default(add_196);  add_196 = None
    permute_115: "f32[384, 1000]" = torch.ops.aten.permute.default(arg202_1, [1, 0]);  arg202_1 = None
    addmm: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg203_1, clone_81, permute_115);  arg203_1 = clone_81 = permute_115 = None
    convert_element_type_126: "f32[384]" = torch.ops.prims.convert_element_type.default(arg411_1, torch.float32);  arg411_1 = None
    convert_element_type_127: "f32[384]" = torch.ops.prims.convert_element_type.default(arg412_1, torch.float32);  arg412_1 = None
    add_197: "f32[384]" = torch.ops.aten.add.Tensor(convert_element_type_127, 1e-05);  convert_element_type_127 = None
    sqrt_63: "f32[384]" = torch.ops.aten.sqrt.default(add_197);  add_197 = None
    reciprocal_63: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_63);  sqrt_63 = None
    mul_234: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_63, 1);  reciprocal_63 = None
    sub_77: "f32[8, 384]" = torch.ops.aten.sub.Tensor(mean, convert_element_type_126);  mean = convert_element_type_126 = None
    mul_235: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sub_77, mul_234);  sub_77 = mul_234 = None
    mul_236: "f32[8, 384]" = torch.ops.aten.mul.Tensor(mul_235, arg204_1);  mul_235 = arg204_1 = None
    add_198: "f32[8, 384]" = torch.ops.aten.add.Tensor(mul_236, arg205_1);  mul_236 = arg205_1 = None
    clone_82: "f32[8, 384]" = torch.ops.aten.clone.default(add_198);  add_198 = None
    permute_116: "f32[384, 1000]" = torch.ops.aten.permute.default(arg206_1, [1, 0]);  arg206_1 = None
    addmm_1: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg207_1, clone_82, permute_116);  arg207_1 = clone_82 = permute_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:690, code: return (x + x_dist) / 2
    add_199: "f32[8, 1000]" = torch.ops.aten.add.Tensor(addmm, addmm_1);  addmm = addmm_1 = None
    div_45: "f32[8, 1000]" = torch.ops.aten.div.Tensor(add_199, 2);  add_199 = None
    return (div_45, index, index_1, index_2, index_3, index_4, index_5, index_6, index_7, index_8, index_9, index_10, index_11, index_12, index_13)
    