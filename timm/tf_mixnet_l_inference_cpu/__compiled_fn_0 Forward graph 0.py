from __future__ import annotations



def forward(self, arg0_1: "f32[32, 3, 3, 3]", arg1_1: "f32[32]", arg2_1: "f32[32]", arg3_1: "f32[32]", arg4_1: "f32[32]", arg5_1: "f32[32]", arg6_1: "f32[32]", arg7_1: "f32[192]", arg8_1: "f32[192]", arg9_1: "f32[64, 1, 3, 3]", arg10_1: "f32[64, 1, 5, 5]", arg11_1: "f32[64, 1, 7, 7]", arg12_1: "f32[192]", arg13_1: "f32[192]", arg14_1: "f32[40]", arg15_1: "f32[40]", arg16_1: "f32[120]", arg17_1: "f32[120]", arg18_1: "f32[120]", arg19_1: "f32[120]", arg20_1: "f32[40]", arg21_1: "f32[40]", arg22_1: "f32[240]", arg23_1: "f32[240]", arg24_1: "f32[60, 1, 3, 3]", arg25_1: "f32[60, 1, 5, 5]", arg26_1: "f32[60, 1, 7, 7]", arg27_1: "f32[60, 1, 9, 9]", arg28_1: "f32[240]", arg29_1: "f32[240]", arg30_1: "f32[56]", arg31_1: "f32[56]", arg32_1: "f32[336]", arg33_1: "f32[336]", arg34_1: "f32[336]", arg35_1: "f32[336]", arg36_1: "f32[56]", arg37_1: "f32[56]", arg38_1: "f32[336]", arg39_1: "f32[336]", arg40_1: "f32[336]", arg41_1: "f32[336]", arg42_1: "f32[56]", arg43_1: "f32[56]", arg44_1: "f32[336]", arg45_1: "f32[336]", arg46_1: "f32[336]", arg47_1: "f32[336]", arg48_1: "f32[56]", arg49_1: "f32[56]", arg50_1: "f32[336]", arg51_1: "f32[336]", arg52_1: "f32[112, 1, 3, 3]", arg53_1: "f32[112, 1, 5, 5]", arg54_1: "f32[112, 1, 7, 7]", arg55_1: "f32[336]", arg56_1: "f32[336]", arg57_1: "f32[104]", arg58_1: "f32[104]", arg59_1: "f32[624]", arg60_1: "f32[624]", arg61_1: "f32[624]", arg62_1: "f32[624]", arg63_1: "f32[104]", arg64_1: "f32[104]", arg65_1: "f32[624]", arg66_1: "f32[624]", arg67_1: "f32[624]", arg68_1: "f32[624]", arg69_1: "f32[104]", arg70_1: "f32[104]", arg71_1: "f32[624]", arg72_1: "f32[624]", arg73_1: "f32[624]", arg74_1: "f32[624]", arg75_1: "f32[104]", arg76_1: "f32[104]", arg77_1: "f32[624]", arg78_1: "f32[624]", arg79_1: "f32[624]", arg80_1: "f32[624]", arg81_1: "f32[160]", arg82_1: "f32[160]", arg83_1: "f32[480]", arg84_1: "f32[480]", arg85_1: "f32[480]", arg86_1: "f32[480]", arg87_1: "f32[160]", arg88_1: "f32[160]", arg89_1: "f32[480]", arg90_1: "f32[480]", arg91_1: "f32[480]", arg92_1: "f32[480]", arg93_1: "f32[160]", arg94_1: "f32[160]", arg95_1: "f32[480]", arg96_1: "f32[480]", arg97_1: "f32[480]", arg98_1: "f32[480]", arg99_1: "f32[160]", arg100_1: "f32[160]", arg101_1: "f32[960]", arg102_1: "f32[960]", arg103_1: "f32[240, 1, 3, 3]", arg104_1: "f32[240, 1, 5, 5]", arg105_1: "f32[240, 1, 7, 7]", arg106_1: "f32[240, 1, 9, 9]", arg107_1: "f32[960]", arg108_1: "f32[960]", arg109_1: "f32[264]", arg110_1: "f32[264]", arg111_1: "f32[1584]", arg112_1: "f32[1584]", arg113_1: "f32[1584]", arg114_1: "f32[1584]", arg115_1: "f32[264]", arg116_1: "f32[264]", arg117_1: "f32[1584]", arg118_1: "f32[1584]", arg119_1: "f32[1584]", arg120_1: "f32[1584]", arg121_1: "f32[264]", arg122_1: "f32[264]", arg123_1: "f32[1584]", arg124_1: "f32[1584]", arg125_1: "f32[1584]", arg126_1: "f32[1584]", arg127_1: "f32[264]", arg128_1: "f32[264]", arg129_1: "f32[1536]", arg130_1: "f32[1536]", arg131_1: "f32[32, 1, 3, 3]", arg132_1: "f32[32, 32, 1, 1]", arg133_1: "f32[96, 16, 1, 1]", arg134_1: "f32[96, 16, 1, 1]", arg135_1: "f32[20, 96, 1, 1]", arg136_1: "f32[20, 96, 1, 1]", arg137_1: "f32[60, 20, 1, 1]", arg138_1: "f32[60, 20, 1, 1]", arg139_1: "f32[120, 1, 3, 3]", arg140_1: "f32[20, 60, 1, 1]", arg141_1: "f32[20, 60, 1, 1]", arg142_1: "f32[240, 40, 1, 1]", arg143_1: "f32[20, 240, 1, 1]", arg144_1: "f32[20]", arg145_1: "f32[240, 20, 1, 1]", arg146_1: "f32[240]", arg147_1: "f32[56, 240, 1, 1]", arg148_1: "f32[168, 28, 1, 1]", arg149_1: "f32[168, 28, 1, 1]", arg150_1: "f32[168, 1, 3, 3]", arg151_1: "f32[168, 1, 5, 5]", arg152_1: "f32[28, 336, 1, 1]", arg153_1: "f32[28]", arg154_1: "f32[336, 28, 1, 1]", arg155_1: "f32[336]", arg156_1: "f32[28, 168, 1, 1]", arg157_1: "f32[28, 168, 1, 1]", arg158_1: "f32[168, 28, 1, 1]", arg159_1: "f32[168, 28, 1, 1]", arg160_1: "f32[168, 1, 3, 3]", arg161_1: "f32[168, 1, 5, 5]", arg162_1: "f32[28, 336, 1, 1]", arg163_1: "f32[28]", arg164_1: "f32[336, 28, 1, 1]", arg165_1: "f32[336]", arg166_1: "f32[28, 168, 1, 1]", arg167_1: "f32[28, 168, 1, 1]", arg168_1: "f32[168, 28, 1, 1]", arg169_1: "f32[168, 28, 1, 1]", arg170_1: "f32[168, 1, 3, 3]", arg171_1: "f32[168, 1, 5, 5]", arg172_1: "f32[28, 336, 1, 1]", arg173_1: "f32[28]", arg174_1: "f32[336, 28, 1, 1]", arg175_1: "f32[336]", arg176_1: "f32[28, 168, 1, 1]", arg177_1: "f32[28, 168, 1, 1]", arg178_1: "f32[336, 56, 1, 1]", arg179_1: "f32[14, 336, 1, 1]", arg180_1: "f32[14]", arg181_1: "f32[336, 14, 1, 1]", arg182_1: "f32[336]", arg183_1: "f32[104, 336, 1, 1]", arg184_1: "f32[312, 52, 1, 1]", arg185_1: "f32[312, 52, 1, 1]", arg186_1: "f32[156, 1, 3, 3]", arg187_1: "f32[156, 1, 5, 5]", arg188_1: "f32[156, 1, 7, 7]", arg189_1: "f32[156, 1, 9, 9]", arg190_1: "f32[26, 624, 1, 1]", arg191_1: "f32[26]", arg192_1: "f32[624, 26, 1, 1]", arg193_1: "f32[624]", arg194_1: "f32[52, 312, 1, 1]", arg195_1: "f32[52, 312, 1, 1]", arg196_1: "f32[312, 52, 1, 1]", arg197_1: "f32[312, 52, 1, 1]", arg198_1: "f32[156, 1, 3, 3]", arg199_1: "f32[156, 1, 5, 5]", arg200_1: "f32[156, 1, 7, 7]", arg201_1: "f32[156, 1, 9, 9]", arg202_1: "f32[26, 624, 1, 1]", arg203_1: "f32[26]", arg204_1: "f32[624, 26, 1, 1]", arg205_1: "f32[624]", arg206_1: "f32[52, 312, 1, 1]", arg207_1: "f32[52, 312, 1, 1]", arg208_1: "f32[312, 52, 1, 1]", arg209_1: "f32[312, 52, 1, 1]", arg210_1: "f32[156, 1, 3, 3]", arg211_1: "f32[156, 1, 5, 5]", arg212_1: "f32[156, 1, 7, 7]", arg213_1: "f32[156, 1, 9, 9]", arg214_1: "f32[26, 624, 1, 1]", arg215_1: "f32[26]", arg216_1: "f32[624, 26, 1, 1]", arg217_1: "f32[624]", arg218_1: "f32[52, 312, 1, 1]", arg219_1: "f32[52, 312, 1, 1]", arg220_1: "f32[624, 104, 1, 1]", arg221_1: "f32[624, 1, 3, 3]", arg222_1: "f32[52, 624, 1, 1]", arg223_1: "f32[52]", arg224_1: "f32[624, 52, 1, 1]", arg225_1: "f32[624]", arg226_1: "f32[160, 624, 1, 1]", arg227_1: "f32[240, 80, 1, 1]", arg228_1: "f32[240, 80, 1, 1]", arg229_1: "f32[120, 1, 3, 3]", arg230_1: "f32[120, 1, 5, 5]", arg231_1: "f32[120, 1, 7, 7]", arg232_1: "f32[120, 1, 9, 9]", arg233_1: "f32[80, 480, 1, 1]", arg234_1: "f32[80]", arg235_1: "f32[480, 80, 1, 1]", arg236_1: "f32[480]", arg237_1: "f32[80, 240, 1, 1]", arg238_1: "f32[80, 240, 1, 1]", arg239_1: "f32[240, 80, 1, 1]", arg240_1: "f32[240, 80, 1, 1]", arg241_1: "f32[120, 1, 3, 3]", arg242_1: "f32[120, 1, 5, 5]", arg243_1: "f32[120, 1, 7, 7]", arg244_1: "f32[120, 1, 9, 9]", arg245_1: "f32[80, 480, 1, 1]", arg246_1: "f32[80]", arg247_1: "f32[480, 80, 1, 1]", arg248_1: "f32[480]", arg249_1: "f32[80, 240, 1, 1]", arg250_1: "f32[80, 240, 1, 1]", arg251_1: "f32[240, 80, 1, 1]", arg252_1: "f32[240, 80, 1, 1]", arg253_1: "f32[120, 1, 3, 3]", arg254_1: "f32[120, 1, 5, 5]", arg255_1: "f32[120, 1, 7, 7]", arg256_1: "f32[120, 1, 9, 9]", arg257_1: "f32[80, 480, 1, 1]", arg258_1: "f32[80]", arg259_1: "f32[480, 80, 1, 1]", arg260_1: "f32[480]", arg261_1: "f32[80, 240, 1, 1]", arg262_1: "f32[80, 240, 1, 1]", arg263_1: "f32[960, 160, 1, 1]", arg264_1: "f32[80, 960, 1, 1]", arg265_1: "f32[80]", arg266_1: "f32[960, 80, 1, 1]", arg267_1: "f32[960]", arg268_1: "f32[264, 960, 1, 1]", arg269_1: "f32[1584, 264, 1, 1]", arg270_1: "f32[396, 1, 3, 3]", arg271_1: "f32[396, 1, 5, 5]", arg272_1: "f32[396, 1, 7, 7]", arg273_1: "f32[396, 1, 9, 9]", arg274_1: "f32[132, 1584, 1, 1]", arg275_1: "f32[132]", arg276_1: "f32[1584, 132, 1, 1]", arg277_1: "f32[1584]", arg278_1: "f32[132, 792, 1, 1]", arg279_1: "f32[132, 792, 1, 1]", arg280_1: "f32[1584, 264, 1, 1]", arg281_1: "f32[396, 1, 3, 3]", arg282_1: "f32[396, 1, 5, 5]", arg283_1: "f32[396, 1, 7, 7]", arg284_1: "f32[396, 1, 9, 9]", arg285_1: "f32[132, 1584, 1, 1]", arg286_1: "f32[132]", arg287_1: "f32[1584, 132, 1, 1]", arg288_1: "f32[1584]", arg289_1: "f32[132, 792, 1, 1]", arg290_1: "f32[132, 792, 1, 1]", arg291_1: "f32[1584, 264, 1, 1]", arg292_1: "f32[396, 1, 3, 3]", arg293_1: "f32[396, 1, 5, 5]", arg294_1: "f32[396, 1, 7, 7]", arg295_1: "f32[396, 1, 9, 9]", arg296_1: "f32[132, 1584, 1, 1]", arg297_1: "f32[132]", arg298_1: "f32[1584, 132, 1, 1]", arg299_1: "f32[1584]", arg300_1: "f32[132, 792, 1, 1]", arg301_1: "f32[132, 792, 1, 1]", arg302_1: "f32[1536, 264, 1, 1]", arg303_1: "f32[1000, 1536]", arg304_1: "f32[1000]", arg305_1: "f32[32]", arg306_1: "f32[32]", arg307_1: "f32[32]", arg308_1: "f32[32]", arg309_1: "f32[32]", arg310_1: "f32[32]", arg311_1: "f32[192]", arg312_1: "f32[192]", arg313_1: "f32[192]", arg314_1: "f32[192]", arg315_1: "f32[40]", arg316_1: "f32[40]", arg317_1: "f32[120]", arg318_1: "f32[120]", arg319_1: "f32[120]", arg320_1: "f32[120]", arg321_1: "f32[40]", arg322_1: "f32[40]", arg323_1: "f32[240]", arg324_1: "f32[240]", arg325_1: "f32[240]", arg326_1: "f32[240]", arg327_1: "f32[56]", arg328_1: "f32[56]", arg329_1: "f32[336]", arg330_1: "f32[336]", arg331_1: "f32[336]", arg332_1: "f32[336]", arg333_1: "f32[56]", arg334_1: "f32[56]", arg335_1: "f32[336]", arg336_1: "f32[336]", arg337_1: "f32[336]", arg338_1: "f32[336]", arg339_1: "f32[56]", arg340_1: "f32[56]", arg341_1: "f32[336]", arg342_1: "f32[336]", arg343_1: "f32[336]", arg344_1: "f32[336]", arg345_1: "f32[56]", arg346_1: "f32[56]", arg347_1: "f32[336]", arg348_1: "f32[336]", arg349_1: "f32[336]", arg350_1: "f32[336]", arg351_1: "f32[104]", arg352_1: "f32[104]", arg353_1: "f32[624]", arg354_1: "f32[624]", arg355_1: "f32[624]", arg356_1: "f32[624]", arg357_1: "f32[104]", arg358_1: "f32[104]", arg359_1: "f32[624]", arg360_1: "f32[624]", arg361_1: "f32[624]", arg362_1: "f32[624]", arg363_1: "f32[104]", arg364_1: "f32[104]", arg365_1: "f32[624]", arg366_1: "f32[624]", arg367_1: "f32[624]", arg368_1: "f32[624]", arg369_1: "f32[104]", arg370_1: "f32[104]", arg371_1: "f32[624]", arg372_1: "f32[624]", arg373_1: "f32[624]", arg374_1: "f32[624]", arg375_1: "f32[160]", arg376_1: "f32[160]", arg377_1: "f32[480]", arg378_1: "f32[480]", arg379_1: "f32[480]", arg380_1: "f32[480]", arg381_1: "f32[160]", arg382_1: "f32[160]", arg383_1: "f32[480]", arg384_1: "f32[480]", arg385_1: "f32[480]", arg386_1: "f32[480]", arg387_1: "f32[160]", arg388_1: "f32[160]", arg389_1: "f32[480]", arg390_1: "f32[480]", arg391_1: "f32[480]", arg392_1: "f32[480]", arg393_1: "f32[160]", arg394_1: "f32[160]", arg395_1: "f32[960]", arg396_1: "f32[960]", arg397_1: "f32[960]", arg398_1: "f32[960]", arg399_1: "f32[264]", arg400_1: "f32[264]", arg401_1: "f32[1584]", arg402_1: "f32[1584]", arg403_1: "f32[1584]", arg404_1: "f32[1584]", arg405_1: "f32[264]", arg406_1: "f32[264]", arg407_1: "f32[1584]", arg408_1: "f32[1584]", arg409_1: "f32[1584]", arg410_1: "f32[1584]", arg411_1: "f32[264]", arg412_1: "f32[264]", arg413_1: "f32[1584]", arg414_1: "f32[1584]", arg415_1: "f32[1584]", arg416_1: "f32[1584]", arg417_1: "f32[264]", arg418_1: "f32[264]", arg419_1: "f32[1536]", arg420_1: "f32[1536]", arg421_1: "f32[8, 3, 224, 224]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd: "f32[8, 3, 225, 225]" = torch.ops.aten.constant_pad_nd.default(arg421_1, [0, 1, 0, 1], 0.0);  arg421_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution: "f32[8, 32, 112, 112]" = torch.ops.aten.convolution.default(constant_pad_nd, arg0_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  constant_pad_nd = arg0_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type: "f32[32]" = torch.ops.prims.convert_element_type.default(arg305_1, torch.float32);  arg305_1 = None
    convert_element_type_1: "f32[32]" = torch.ops.prims.convert_element_type.default(arg306_1, torch.float32);  arg306_1 = None
    add: "f32[32]" = torch.ops.aten.add.Tensor(convert_element_type_1, 0.001);  convert_element_type_1 = None
    sqrt: "f32[32]" = torch.ops.aten.sqrt.default(add);  add = None
    reciprocal: "f32[32]" = torch.ops.aten.reciprocal.default(sqrt);  sqrt = None
    mul: "f32[32]" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
    unsqueeze: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type, -1);  convert_element_type = None
    unsqueeze_1: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    unsqueeze_2: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(mul, -1);  mul = None
    unsqueeze_3: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    sub: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1);  convolution = unsqueeze_1 = None
    mul_1: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub, unsqueeze_3);  sub = unsqueeze_3 = None
    unsqueeze_4: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg1_1, -1);  arg1_1 = None
    unsqueeze_5: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_2: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None
    unsqueeze_6: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
    unsqueeze_7: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_1: "f32[8, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_2, unsqueeze_7);  mul_2 = unsqueeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu: "f32[8, 32, 112, 112]" = torch.ops.aten.relu.default(add_1);  add_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_1: "f32[8, 32, 112, 112]" = torch.ops.aten.convolution.default(relu, arg131_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32);  arg131_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_2: "f32[32]" = torch.ops.prims.convert_element_type.default(arg307_1, torch.float32);  arg307_1 = None
    convert_element_type_3: "f32[32]" = torch.ops.prims.convert_element_type.default(arg308_1, torch.float32);  arg308_1 = None
    add_2: "f32[32]" = torch.ops.aten.add.Tensor(convert_element_type_3, 0.001);  convert_element_type_3 = None
    sqrt_1: "f32[32]" = torch.ops.aten.sqrt.default(add_2);  add_2 = None
    reciprocal_1: "f32[32]" = torch.ops.aten.reciprocal.default(sqrt_1);  sqrt_1 = None
    mul_3: "f32[32]" = torch.ops.aten.mul.Tensor(reciprocal_1, 1);  reciprocal_1 = None
    unsqueeze_8: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, -1);  convert_element_type_2 = None
    unsqueeze_9: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    unsqueeze_10: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(mul_3, -1);  mul_3 = None
    unsqueeze_11: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    sub_1: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_9);  convolution_1 = unsqueeze_9 = None
    mul_4: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_1, unsqueeze_11);  sub_1 = unsqueeze_11 = None
    unsqueeze_12: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, -1);  arg3_1 = None
    unsqueeze_13: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_5: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_4, unsqueeze_13);  mul_4 = unsqueeze_13 = None
    unsqueeze_14: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
    unsqueeze_15: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_3: "f32[8, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_5, unsqueeze_15);  mul_5 = unsqueeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_1: "f32[8, 32, 112, 112]" = torch.ops.aten.relu.default(add_3);  add_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_2: "f32[8, 32, 112, 112]" = torch.ops.aten.convolution.default(relu_1, arg132_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_1 = arg132_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_4: "f32[32]" = torch.ops.prims.convert_element_type.default(arg309_1, torch.float32);  arg309_1 = None
    convert_element_type_5: "f32[32]" = torch.ops.prims.convert_element_type.default(arg310_1, torch.float32);  arg310_1 = None
    add_4: "f32[32]" = torch.ops.aten.add.Tensor(convert_element_type_5, 0.001);  convert_element_type_5 = None
    sqrt_2: "f32[32]" = torch.ops.aten.sqrt.default(add_4);  add_4 = None
    reciprocal_2: "f32[32]" = torch.ops.aten.reciprocal.default(sqrt_2);  sqrt_2 = None
    mul_6: "f32[32]" = torch.ops.aten.mul.Tensor(reciprocal_2, 1);  reciprocal_2 = None
    unsqueeze_16: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_4, -1);  convert_element_type_4 = None
    unsqueeze_17: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    unsqueeze_18: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(mul_6, -1);  mul_6 = None
    unsqueeze_19: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    sub_2: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_17);  convolution_2 = unsqueeze_17 = None
    mul_7: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_2, unsqueeze_19);  sub_2 = unsqueeze_19 = None
    unsqueeze_20: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
    unsqueeze_21: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_8: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_21);  mul_7 = unsqueeze_21 = None
    unsqueeze_22: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg6_1, -1);  arg6_1 = None
    unsqueeze_23: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_5: "f32[8, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_8, unsqueeze_23);  mul_8 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:129, code: x = self.drop_path(x) + shortcut
    add_6: "f32[8, 32, 112, 112]" = torch.ops.aten.add.Tensor(add_5, relu);  add_5 = relu = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes = torch.ops.aten.split_with_sizes.default(add_6, [16, 16], 1);  add_6 = None
    getitem: "f32[8, 16, 112, 112]" = split_with_sizes[0]
    getitem_1: "f32[8, 16, 112, 112]" = split_with_sizes[1];  split_with_sizes = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_3: "f32[8, 96, 112, 112]" = torch.ops.aten.convolution.default(getitem, arg133_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem = arg133_1 = None
    convolution_4: "f32[8, 96, 112, 112]" = torch.ops.aten.convolution.default(getitem_1, arg134_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_1 = arg134_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat: "f32[8, 192, 112, 112]" = torch.ops.aten.cat.default([convolution_3, convolution_4], 1);  convolution_3 = convolution_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_6: "f32[192]" = torch.ops.prims.convert_element_type.default(arg311_1, torch.float32);  arg311_1 = None
    convert_element_type_7: "f32[192]" = torch.ops.prims.convert_element_type.default(arg312_1, torch.float32);  arg312_1 = None
    add_7: "f32[192]" = torch.ops.aten.add.Tensor(convert_element_type_7, 0.001);  convert_element_type_7 = None
    sqrt_3: "f32[192]" = torch.ops.aten.sqrt.default(add_7);  add_7 = None
    reciprocal_3: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_3);  sqrt_3 = None
    mul_9: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_3, 1);  reciprocal_3 = None
    unsqueeze_24: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_6, -1);  convert_element_type_6 = None
    unsqueeze_25: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
    unsqueeze_26: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_9, -1);  mul_9 = None
    unsqueeze_27: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
    sub_3: "f32[8, 192, 112, 112]" = torch.ops.aten.sub.Tensor(cat, unsqueeze_25);  cat = unsqueeze_25 = None
    mul_10: "f32[8, 192, 112, 112]" = torch.ops.aten.mul.Tensor(sub_3, unsqueeze_27);  sub_3 = unsqueeze_27 = None
    unsqueeze_28: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
    unsqueeze_29: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    mul_11: "f32[8, 192, 112, 112]" = torch.ops.aten.mul.Tensor(mul_10, unsqueeze_29);  mul_10 = unsqueeze_29 = None
    unsqueeze_30: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg8_1, -1);  arg8_1 = None
    unsqueeze_31: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
    add_8: "f32[8, 192, 112, 112]" = torch.ops.aten.add.Tensor(mul_11, unsqueeze_31);  mul_11 = unsqueeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_2: "f32[8, 192, 112, 112]" = torch.ops.aten.relu.default(add_8);  add_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    split_with_sizes_2 = torch.ops.aten.split_with_sizes.default(relu_2, [64, 64, 64], 1)
    getitem_5: "f32[8, 64, 112, 112]" = split_with_sizes_2[0];  split_with_sizes_2 = None
    constant_pad_nd_1: "f32[8, 64, 113, 113]" = torch.ops.aten.constant_pad_nd.default(getitem_5, [0, 1, 0, 1], 0.0);  getitem_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_5: "f32[8, 64, 56, 56]" = torch.ops.aten.convolution.default(constant_pad_nd_1, arg9_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 64);  constant_pad_nd_1 = arg9_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    split_with_sizes_3 = torch.ops.aten.split_with_sizes.default(relu_2, [64, 64, 64], 1)
    getitem_9: "f32[8, 64, 112, 112]" = split_with_sizes_3[1];  split_with_sizes_3 = None
    constant_pad_nd_2: "f32[8, 64, 115, 115]" = torch.ops.aten.constant_pad_nd.default(getitem_9, [1, 2, 1, 2], 0.0);  getitem_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_6: "f32[8, 64, 56, 56]" = torch.ops.aten.convolution.default(constant_pad_nd_2, arg10_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 64);  constant_pad_nd_2 = arg10_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    split_with_sizes_4 = torch.ops.aten.split_with_sizes.default(relu_2, [64, 64, 64], 1);  relu_2 = None
    getitem_13: "f32[8, 64, 112, 112]" = split_with_sizes_4[2];  split_with_sizes_4 = None
    constant_pad_nd_3: "f32[8, 64, 117, 117]" = torch.ops.aten.constant_pad_nd.default(getitem_13, [2, 3, 2, 3], 0.0);  getitem_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_7: "f32[8, 64, 56, 56]" = torch.ops.aten.convolution.default(constant_pad_nd_3, arg11_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 64);  constant_pad_nd_3 = arg11_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_1: "f32[8, 192, 56, 56]" = torch.ops.aten.cat.default([convolution_5, convolution_6, convolution_7], 1);  convolution_5 = convolution_6 = convolution_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_8: "f32[192]" = torch.ops.prims.convert_element_type.default(arg313_1, torch.float32);  arg313_1 = None
    convert_element_type_9: "f32[192]" = torch.ops.prims.convert_element_type.default(arg314_1, torch.float32);  arg314_1 = None
    add_9: "f32[192]" = torch.ops.aten.add.Tensor(convert_element_type_9, 0.001);  convert_element_type_9 = None
    sqrt_4: "f32[192]" = torch.ops.aten.sqrt.default(add_9);  add_9 = None
    reciprocal_4: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_4);  sqrt_4 = None
    mul_12: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_4, 1);  reciprocal_4 = None
    unsqueeze_32: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_8, -1);  convert_element_type_8 = None
    unsqueeze_33: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    unsqueeze_34: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_12, -1);  mul_12 = None
    unsqueeze_35: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
    sub_4: "f32[8, 192, 56, 56]" = torch.ops.aten.sub.Tensor(cat_1, unsqueeze_33);  cat_1 = unsqueeze_33 = None
    mul_13: "f32[8, 192, 56, 56]" = torch.ops.aten.mul.Tensor(sub_4, unsqueeze_35);  sub_4 = unsqueeze_35 = None
    unsqueeze_36: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg12_1, -1);  arg12_1 = None
    unsqueeze_37: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
    mul_14: "f32[8, 192, 56, 56]" = torch.ops.aten.mul.Tensor(mul_13, unsqueeze_37);  mul_13 = unsqueeze_37 = None
    unsqueeze_38: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg13_1, -1);  arg13_1 = None
    unsqueeze_39: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
    add_10: "f32[8, 192, 56, 56]" = torch.ops.aten.add.Tensor(mul_14, unsqueeze_39);  mul_14 = unsqueeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_3: "f32[8, 192, 56, 56]" = torch.ops.aten.relu.default(add_10);  add_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    split_with_sizes_6 = torch.ops.aten.split_with_sizes.default(relu_3, [96, 96], 1)
    getitem_16: "f32[8, 96, 56, 56]" = split_with_sizes_6[0];  split_with_sizes_6 = None
    convolution_8: "f32[8, 20, 56, 56]" = torch.ops.aten.convolution.default(getitem_16, arg135_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_16 = arg135_1 = None
    split_with_sizes_7 = torch.ops.aten.split_with_sizes.default(relu_3, [96, 96], 1);  relu_3 = None
    getitem_19: "f32[8, 96, 56, 56]" = split_with_sizes_7[1];  split_with_sizes_7 = None
    convolution_9: "f32[8, 20, 56, 56]" = torch.ops.aten.convolution.default(getitem_19, arg136_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_19 = arg136_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_2: "f32[8, 40, 56, 56]" = torch.ops.aten.cat.default([convolution_8, convolution_9], 1);  convolution_8 = convolution_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_10: "f32[40]" = torch.ops.prims.convert_element_type.default(arg315_1, torch.float32);  arg315_1 = None
    convert_element_type_11: "f32[40]" = torch.ops.prims.convert_element_type.default(arg316_1, torch.float32);  arg316_1 = None
    add_11: "f32[40]" = torch.ops.aten.add.Tensor(convert_element_type_11, 0.001);  convert_element_type_11 = None
    sqrt_5: "f32[40]" = torch.ops.aten.sqrt.default(add_11);  add_11 = None
    reciprocal_5: "f32[40]" = torch.ops.aten.reciprocal.default(sqrt_5);  sqrt_5 = None
    mul_15: "f32[40]" = torch.ops.aten.mul.Tensor(reciprocal_5, 1);  reciprocal_5 = None
    unsqueeze_40: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_10, -1);  convert_element_type_10 = None
    unsqueeze_41: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    unsqueeze_42: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(mul_15, -1);  mul_15 = None
    unsqueeze_43: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    sub_5: "f32[8, 40, 56, 56]" = torch.ops.aten.sub.Tensor(cat_2, unsqueeze_41);  cat_2 = unsqueeze_41 = None
    mul_16: "f32[8, 40, 56, 56]" = torch.ops.aten.mul.Tensor(sub_5, unsqueeze_43);  sub_5 = unsqueeze_43 = None
    unsqueeze_44: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
    unsqueeze_45: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    mul_17: "f32[8, 40, 56, 56]" = torch.ops.aten.mul.Tensor(mul_16, unsqueeze_45);  mul_16 = unsqueeze_45 = None
    unsqueeze_46: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
    unsqueeze_47: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
    add_12: "f32[8, 40, 56, 56]" = torch.ops.aten.add.Tensor(mul_17, unsqueeze_47);  mul_17 = unsqueeze_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_8 = torch.ops.aten.split_with_sizes.default(add_12, [20, 20], 1)
    getitem_20: "f32[8, 20, 56, 56]" = split_with_sizes_8[0]
    getitem_21: "f32[8, 20, 56, 56]" = split_with_sizes_8[1];  split_with_sizes_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_10: "f32[8, 60, 56, 56]" = torch.ops.aten.convolution.default(getitem_20, arg137_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_20 = arg137_1 = None
    convolution_11: "f32[8, 60, 56, 56]" = torch.ops.aten.convolution.default(getitem_21, arg138_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_21 = arg138_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_3: "f32[8, 120, 56, 56]" = torch.ops.aten.cat.default([convolution_10, convolution_11], 1);  convolution_10 = convolution_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_12: "f32[120]" = torch.ops.prims.convert_element_type.default(arg317_1, torch.float32);  arg317_1 = None
    convert_element_type_13: "f32[120]" = torch.ops.prims.convert_element_type.default(arg318_1, torch.float32);  arg318_1 = None
    add_13: "f32[120]" = torch.ops.aten.add.Tensor(convert_element_type_13, 0.001);  convert_element_type_13 = None
    sqrt_6: "f32[120]" = torch.ops.aten.sqrt.default(add_13);  add_13 = None
    reciprocal_6: "f32[120]" = torch.ops.aten.reciprocal.default(sqrt_6);  sqrt_6 = None
    mul_18: "f32[120]" = torch.ops.aten.mul.Tensor(reciprocal_6, 1);  reciprocal_6 = None
    unsqueeze_48: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_12, -1);  convert_element_type_12 = None
    unsqueeze_49: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
    unsqueeze_50: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(mul_18, -1);  mul_18 = None
    unsqueeze_51: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
    sub_6: "f32[8, 120, 56, 56]" = torch.ops.aten.sub.Tensor(cat_3, unsqueeze_49);  cat_3 = unsqueeze_49 = None
    mul_19: "f32[8, 120, 56, 56]" = torch.ops.aten.mul.Tensor(sub_6, unsqueeze_51);  sub_6 = unsqueeze_51 = None
    unsqueeze_52: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg16_1, -1);  arg16_1 = None
    unsqueeze_53: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    mul_20: "f32[8, 120, 56, 56]" = torch.ops.aten.mul.Tensor(mul_19, unsqueeze_53);  mul_19 = unsqueeze_53 = None
    unsqueeze_54: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg17_1, -1);  arg17_1 = None
    unsqueeze_55: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
    add_14: "f32[8, 120, 56, 56]" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_55);  mul_20 = unsqueeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_4: "f32[8, 120, 56, 56]" = torch.ops.aten.relu.default(add_14);  add_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_12: "f32[8, 120, 56, 56]" = torch.ops.aten.convolution.default(relu_4, arg139_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 120);  relu_4 = arg139_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_14: "f32[120]" = torch.ops.prims.convert_element_type.default(arg319_1, torch.float32);  arg319_1 = None
    convert_element_type_15: "f32[120]" = torch.ops.prims.convert_element_type.default(arg320_1, torch.float32);  arg320_1 = None
    add_15: "f32[120]" = torch.ops.aten.add.Tensor(convert_element_type_15, 0.001);  convert_element_type_15 = None
    sqrt_7: "f32[120]" = torch.ops.aten.sqrt.default(add_15);  add_15 = None
    reciprocal_7: "f32[120]" = torch.ops.aten.reciprocal.default(sqrt_7);  sqrt_7 = None
    mul_21: "f32[120]" = torch.ops.aten.mul.Tensor(reciprocal_7, 1);  reciprocal_7 = None
    unsqueeze_56: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_14, -1);  convert_element_type_14 = None
    unsqueeze_57: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    unsqueeze_58: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(mul_21, -1);  mul_21 = None
    unsqueeze_59: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
    sub_7: "f32[8, 120, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_57);  convolution_12 = unsqueeze_57 = None
    mul_22: "f32[8, 120, 56, 56]" = torch.ops.aten.mul.Tensor(sub_7, unsqueeze_59);  sub_7 = unsqueeze_59 = None
    unsqueeze_60: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg18_1, -1);  arg18_1 = None
    unsqueeze_61: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
    mul_23: "f32[8, 120, 56, 56]" = torch.ops.aten.mul.Tensor(mul_22, unsqueeze_61);  mul_22 = unsqueeze_61 = None
    unsqueeze_62: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg19_1, -1);  arg19_1 = None
    unsqueeze_63: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
    add_16: "f32[8, 120, 56, 56]" = torch.ops.aten.add.Tensor(mul_23, unsqueeze_63);  mul_23 = unsqueeze_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_5: "f32[8, 120, 56, 56]" = torch.ops.aten.relu.default(add_16);  add_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    split_with_sizes_10 = torch.ops.aten.split_with_sizes.default(relu_5, [60, 60], 1)
    getitem_24: "f32[8, 60, 56, 56]" = split_with_sizes_10[0];  split_with_sizes_10 = None
    convolution_13: "f32[8, 20, 56, 56]" = torch.ops.aten.convolution.default(getitem_24, arg140_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_24 = arg140_1 = None
    split_with_sizes_11 = torch.ops.aten.split_with_sizes.default(relu_5, [60, 60], 1);  relu_5 = None
    getitem_27: "f32[8, 60, 56, 56]" = split_with_sizes_11[1];  split_with_sizes_11 = None
    convolution_14: "f32[8, 20, 56, 56]" = torch.ops.aten.convolution.default(getitem_27, arg141_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_27 = arg141_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_4: "f32[8, 40, 56, 56]" = torch.ops.aten.cat.default([convolution_13, convolution_14], 1);  convolution_13 = convolution_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_16: "f32[40]" = torch.ops.prims.convert_element_type.default(arg321_1, torch.float32);  arg321_1 = None
    convert_element_type_17: "f32[40]" = torch.ops.prims.convert_element_type.default(arg322_1, torch.float32);  arg322_1 = None
    add_17: "f32[40]" = torch.ops.aten.add.Tensor(convert_element_type_17, 0.001);  convert_element_type_17 = None
    sqrt_8: "f32[40]" = torch.ops.aten.sqrt.default(add_17);  add_17 = None
    reciprocal_8: "f32[40]" = torch.ops.aten.reciprocal.default(sqrt_8);  sqrt_8 = None
    mul_24: "f32[40]" = torch.ops.aten.mul.Tensor(reciprocal_8, 1);  reciprocal_8 = None
    unsqueeze_64: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_16, -1);  convert_element_type_16 = None
    unsqueeze_65: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
    unsqueeze_66: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(mul_24, -1);  mul_24 = None
    unsqueeze_67: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
    sub_8: "f32[8, 40, 56, 56]" = torch.ops.aten.sub.Tensor(cat_4, unsqueeze_65);  cat_4 = unsqueeze_65 = None
    mul_25: "f32[8, 40, 56, 56]" = torch.ops.aten.mul.Tensor(sub_8, unsqueeze_67);  sub_8 = unsqueeze_67 = None
    unsqueeze_68: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg20_1, -1);  arg20_1 = None
    unsqueeze_69: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    mul_26: "f32[8, 40, 56, 56]" = torch.ops.aten.mul.Tensor(mul_25, unsqueeze_69);  mul_25 = unsqueeze_69 = None
    unsqueeze_70: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg21_1, -1);  arg21_1 = None
    unsqueeze_71: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    add_18: "f32[8, 40, 56, 56]" = torch.ops.aten.add.Tensor(mul_26, unsqueeze_71);  mul_26 = unsqueeze_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_19: "f32[8, 40, 56, 56]" = torch.ops.aten.add.Tensor(add_18, add_12);  add_18 = add_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_15: "f32[8, 240, 56, 56]" = torch.ops.aten.convolution.default(add_19, arg142_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_19 = arg142_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_18: "f32[240]" = torch.ops.prims.convert_element_type.default(arg323_1, torch.float32);  arg323_1 = None
    convert_element_type_19: "f32[240]" = torch.ops.prims.convert_element_type.default(arg324_1, torch.float32);  arg324_1 = None
    add_20: "f32[240]" = torch.ops.aten.add.Tensor(convert_element_type_19, 0.001);  convert_element_type_19 = None
    sqrt_9: "f32[240]" = torch.ops.aten.sqrt.default(add_20);  add_20 = None
    reciprocal_9: "f32[240]" = torch.ops.aten.reciprocal.default(sqrt_9);  sqrt_9 = None
    mul_27: "f32[240]" = torch.ops.aten.mul.Tensor(reciprocal_9, 1);  reciprocal_9 = None
    unsqueeze_72: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_18, -1);  convert_element_type_18 = None
    unsqueeze_73: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
    unsqueeze_74: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(mul_27, -1);  mul_27 = None
    unsqueeze_75: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
    sub_9: "f32[8, 240, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_73);  convolution_15 = unsqueeze_73 = None
    mul_28: "f32[8, 240, 56, 56]" = torch.ops.aten.mul.Tensor(sub_9, unsqueeze_75);  sub_9 = unsqueeze_75 = None
    unsqueeze_76: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(arg22_1, -1);  arg22_1 = None
    unsqueeze_77: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
    mul_29: "f32[8, 240, 56, 56]" = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_77);  mul_28 = unsqueeze_77 = None
    unsqueeze_78: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(arg23_1, -1);  arg23_1 = None
    unsqueeze_79: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    add_21: "f32[8, 240, 56, 56]" = torch.ops.aten.add.Tensor(mul_29, unsqueeze_79);  mul_29 = unsqueeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid: "f32[8, 240, 56, 56]" = torch.ops.aten.sigmoid.default(add_21)
    mul_30: "f32[8, 240, 56, 56]" = torch.ops.aten.mul.Tensor(add_21, sigmoid);  add_21 = sigmoid = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    split_with_sizes_13 = torch.ops.aten.split_with_sizes.default(mul_30, [60, 60, 60, 60], 1)
    getitem_32: "f32[8, 60, 56, 56]" = split_with_sizes_13[0];  split_with_sizes_13 = None
    constant_pad_nd_4: "f32[8, 60, 57, 57]" = torch.ops.aten.constant_pad_nd.default(getitem_32, [0, 1, 0, 1], 0.0);  getitem_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_16: "f32[8, 60, 28, 28]" = torch.ops.aten.convolution.default(constant_pad_nd_4, arg24_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 60);  constant_pad_nd_4 = arg24_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    split_with_sizes_14 = torch.ops.aten.split_with_sizes.default(mul_30, [60, 60, 60, 60], 1)
    getitem_37: "f32[8, 60, 56, 56]" = split_with_sizes_14[1];  split_with_sizes_14 = None
    constant_pad_nd_5: "f32[8, 60, 59, 59]" = torch.ops.aten.constant_pad_nd.default(getitem_37, [1, 2, 1, 2], 0.0);  getitem_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_17: "f32[8, 60, 28, 28]" = torch.ops.aten.convolution.default(constant_pad_nd_5, arg25_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 60);  constant_pad_nd_5 = arg25_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    split_with_sizes_15 = torch.ops.aten.split_with_sizes.default(mul_30, [60, 60, 60, 60], 1)
    getitem_42: "f32[8, 60, 56, 56]" = split_with_sizes_15[2];  split_with_sizes_15 = None
    constant_pad_nd_6: "f32[8, 60, 61, 61]" = torch.ops.aten.constant_pad_nd.default(getitem_42, [2, 3, 2, 3], 0.0);  getitem_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_18: "f32[8, 60, 28, 28]" = torch.ops.aten.convolution.default(constant_pad_nd_6, arg26_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 60);  constant_pad_nd_6 = arg26_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    split_with_sizes_16 = torch.ops.aten.split_with_sizes.default(mul_30, [60, 60, 60, 60], 1);  mul_30 = None
    getitem_47: "f32[8, 60, 56, 56]" = split_with_sizes_16[3];  split_with_sizes_16 = None
    constant_pad_nd_7: "f32[8, 60, 63, 63]" = torch.ops.aten.constant_pad_nd.default(getitem_47, [3, 4, 3, 4], 0.0);  getitem_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_19: "f32[8, 60, 28, 28]" = torch.ops.aten.convolution.default(constant_pad_nd_7, arg27_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 60);  constant_pad_nd_7 = arg27_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_5: "f32[8, 240, 28, 28]" = torch.ops.aten.cat.default([convolution_16, convolution_17, convolution_18, convolution_19], 1);  convolution_16 = convolution_17 = convolution_18 = convolution_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_20: "f32[240]" = torch.ops.prims.convert_element_type.default(arg325_1, torch.float32);  arg325_1 = None
    convert_element_type_21: "f32[240]" = torch.ops.prims.convert_element_type.default(arg326_1, torch.float32);  arg326_1 = None
    add_22: "f32[240]" = torch.ops.aten.add.Tensor(convert_element_type_21, 0.001);  convert_element_type_21 = None
    sqrt_10: "f32[240]" = torch.ops.aten.sqrt.default(add_22);  add_22 = None
    reciprocal_10: "f32[240]" = torch.ops.aten.reciprocal.default(sqrt_10);  sqrt_10 = None
    mul_31: "f32[240]" = torch.ops.aten.mul.Tensor(reciprocal_10, 1);  reciprocal_10 = None
    unsqueeze_80: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_20, -1);  convert_element_type_20 = None
    unsqueeze_81: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
    unsqueeze_82: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(mul_31, -1);  mul_31 = None
    unsqueeze_83: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    sub_10: "f32[8, 240, 28, 28]" = torch.ops.aten.sub.Tensor(cat_5, unsqueeze_81);  cat_5 = unsqueeze_81 = None
    mul_32: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sub_10, unsqueeze_83);  sub_10 = unsqueeze_83 = None
    unsqueeze_84: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(arg28_1, -1);  arg28_1 = None
    unsqueeze_85: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    mul_33: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(mul_32, unsqueeze_85);  mul_32 = unsqueeze_85 = None
    unsqueeze_86: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(arg29_1, -1);  arg29_1 = None
    unsqueeze_87: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
    add_23: "f32[8, 240, 28, 28]" = torch.ops.aten.add.Tensor(mul_33, unsqueeze_87);  mul_33 = unsqueeze_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_1: "f32[8, 240, 28, 28]" = torch.ops.aten.sigmoid.default(add_23)
    mul_34: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(add_23, sigmoid_1);  add_23 = sigmoid_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean: "f32[8, 240, 1, 1]" = torch.ops.aten.mean.dim(mul_34, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_20: "f32[8, 20, 1, 1]" = torch.ops.aten.convolution.default(mean, arg143_1, arg144_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean = arg143_1 = arg144_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_2: "f32[8, 20, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_20)
    mul_35: "f32[8, 20, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_20, sigmoid_2);  convolution_20 = sigmoid_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_21: "f32[8, 240, 1, 1]" = torch.ops.aten.convolution.default(mul_35, arg145_1, arg146_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_35 = arg145_1 = arg146_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_3: "f32[8, 240, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_21);  convolution_21 = None
    mul_36: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(mul_34, sigmoid_3);  mul_34 = sigmoid_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_22: "f32[8, 56, 28, 28]" = torch.ops.aten.convolution.default(mul_36, arg147_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_36 = arg147_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_22: "f32[56]" = torch.ops.prims.convert_element_type.default(arg327_1, torch.float32);  arg327_1 = None
    convert_element_type_23: "f32[56]" = torch.ops.prims.convert_element_type.default(arg328_1, torch.float32);  arg328_1 = None
    add_24: "f32[56]" = torch.ops.aten.add.Tensor(convert_element_type_23, 0.001);  convert_element_type_23 = None
    sqrt_11: "f32[56]" = torch.ops.aten.sqrt.default(add_24);  add_24 = None
    reciprocal_11: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_11);  sqrt_11 = None
    mul_37: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_11, 1);  reciprocal_11 = None
    unsqueeze_88: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_22, -1);  convert_element_type_22 = None
    unsqueeze_89: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
    unsqueeze_90: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_37, -1);  mul_37 = None
    unsqueeze_91: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    sub_11: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_89);  convolution_22 = unsqueeze_89 = None
    mul_38: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(sub_11, unsqueeze_91);  sub_11 = unsqueeze_91 = None
    unsqueeze_92: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg30_1, -1);  arg30_1 = None
    unsqueeze_93: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
    mul_39: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(mul_38, unsqueeze_93);  mul_38 = unsqueeze_93 = None
    unsqueeze_94: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg31_1, -1);  arg31_1 = None
    unsqueeze_95: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
    add_25: "f32[8, 56, 28, 28]" = torch.ops.aten.add.Tensor(mul_39, unsqueeze_95);  mul_39 = unsqueeze_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_17 = torch.ops.aten.split_with_sizes.default(add_25, [28, 28], 1)
    getitem_48: "f32[8, 28, 28, 28]" = split_with_sizes_17[0]
    getitem_49: "f32[8, 28, 28, 28]" = split_with_sizes_17[1];  split_with_sizes_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_23: "f32[8, 168, 28, 28]" = torch.ops.aten.convolution.default(getitem_48, arg148_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_48 = arg148_1 = None
    convolution_24: "f32[8, 168, 28, 28]" = torch.ops.aten.convolution.default(getitem_49, arg149_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_49 = arg149_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_6: "f32[8, 336, 28, 28]" = torch.ops.aten.cat.default([convolution_23, convolution_24], 1);  convolution_23 = convolution_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_24: "f32[336]" = torch.ops.prims.convert_element_type.default(arg329_1, torch.float32);  arg329_1 = None
    convert_element_type_25: "f32[336]" = torch.ops.prims.convert_element_type.default(arg330_1, torch.float32);  arg330_1 = None
    add_26: "f32[336]" = torch.ops.aten.add.Tensor(convert_element_type_25, 0.001);  convert_element_type_25 = None
    sqrt_12: "f32[336]" = torch.ops.aten.sqrt.default(add_26);  add_26 = None
    reciprocal_12: "f32[336]" = torch.ops.aten.reciprocal.default(sqrt_12);  sqrt_12 = None
    mul_40: "f32[336]" = torch.ops.aten.mul.Tensor(reciprocal_12, 1);  reciprocal_12 = None
    unsqueeze_96: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_24, -1);  convert_element_type_24 = None
    unsqueeze_97: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
    unsqueeze_98: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(mul_40, -1);  mul_40 = None
    unsqueeze_99: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
    sub_12: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(cat_6, unsqueeze_97);  cat_6 = unsqueeze_97 = None
    mul_41: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sub_12, unsqueeze_99);  sub_12 = unsqueeze_99 = None
    unsqueeze_100: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg32_1, -1);  arg32_1 = None
    unsqueeze_101: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
    mul_42: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(mul_41, unsqueeze_101);  mul_41 = unsqueeze_101 = None
    unsqueeze_102: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg33_1, -1);  arg33_1 = None
    unsqueeze_103: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    add_27: "f32[8, 336, 28, 28]" = torch.ops.aten.add.Tensor(mul_42, unsqueeze_103);  mul_42 = unsqueeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_4: "f32[8, 336, 28, 28]" = torch.ops.aten.sigmoid.default(add_27)
    mul_43: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(add_27, sigmoid_4);  add_27 = sigmoid_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    split_with_sizes_19 = torch.ops.aten.split_with_sizes.default(mul_43, [168, 168], 1)
    getitem_52: "f32[8, 168, 28, 28]" = split_with_sizes_19[0];  split_with_sizes_19 = None
    convolution_25: "f32[8, 168, 28, 28]" = torch.ops.aten.convolution.default(getitem_52, arg150_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 168);  getitem_52 = arg150_1 = None
    split_with_sizes_20 = torch.ops.aten.split_with_sizes.default(mul_43, [168, 168], 1);  mul_43 = None
    getitem_55: "f32[8, 168, 28, 28]" = split_with_sizes_20[1];  split_with_sizes_20 = None
    convolution_26: "f32[8, 168, 28, 28]" = torch.ops.aten.convolution.default(getitem_55, arg151_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 168);  getitem_55 = arg151_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_7: "f32[8, 336, 28, 28]" = torch.ops.aten.cat.default([convolution_25, convolution_26], 1);  convolution_25 = convolution_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_26: "f32[336]" = torch.ops.prims.convert_element_type.default(arg331_1, torch.float32);  arg331_1 = None
    convert_element_type_27: "f32[336]" = torch.ops.prims.convert_element_type.default(arg332_1, torch.float32);  arg332_1 = None
    add_28: "f32[336]" = torch.ops.aten.add.Tensor(convert_element_type_27, 0.001);  convert_element_type_27 = None
    sqrt_13: "f32[336]" = torch.ops.aten.sqrt.default(add_28);  add_28 = None
    reciprocal_13: "f32[336]" = torch.ops.aten.reciprocal.default(sqrt_13);  sqrt_13 = None
    mul_44: "f32[336]" = torch.ops.aten.mul.Tensor(reciprocal_13, 1);  reciprocal_13 = None
    unsqueeze_104: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_26, -1);  convert_element_type_26 = None
    unsqueeze_105: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
    unsqueeze_106: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(mul_44, -1);  mul_44 = None
    unsqueeze_107: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
    sub_13: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(cat_7, unsqueeze_105);  cat_7 = unsqueeze_105 = None
    mul_45: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sub_13, unsqueeze_107);  sub_13 = unsqueeze_107 = None
    unsqueeze_108: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg34_1, -1);  arg34_1 = None
    unsqueeze_109: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
    mul_46: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(mul_45, unsqueeze_109);  mul_45 = unsqueeze_109 = None
    unsqueeze_110: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg35_1, -1);  arg35_1 = None
    unsqueeze_111: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
    add_29: "f32[8, 336, 28, 28]" = torch.ops.aten.add.Tensor(mul_46, unsqueeze_111);  mul_46 = unsqueeze_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_5: "f32[8, 336, 28, 28]" = torch.ops.aten.sigmoid.default(add_29)
    mul_47: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(add_29, sigmoid_5);  add_29 = sigmoid_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_1: "f32[8, 336, 1, 1]" = torch.ops.aten.mean.dim(mul_47, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_27: "f32[8, 28, 1, 1]" = torch.ops.aten.convolution.default(mean_1, arg152_1, arg153_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_1 = arg152_1 = arg153_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_6: "f32[8, 28, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_27)
    mul_48: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_27, sigmoid_6);  convolution_27 = sigmoid_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_28: "f32[8, 336, 1, 1]" = torch.ops.aten.convolution.default(mul_48, arg154_1, arg155_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_48 = arg154_1 = arg155_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_7: "f32[8, 336, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_28);  convolution_28 = None
    mul_49: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(mul_47, sigmoid_7);  mul_47 = sigmoid_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_21 = torch.ops.aten.split_with_sizes.default(mul_49, [168, 168], 1);  mul_49 = None
    getitem_56: "f32[8, 168, 28, 28]" = split_with_sizes_21[0]
    getitem_57: "f32[8, 168, 28, 28]" = split_with_sizes_21[1];  split_with_sizes_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_29: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(getitem_56, arg156_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_56 = arg156_1 = None
    convolution_30: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(getitem_57, arg157_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_57 = arg157_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_8: "f32[8, 56, 28, 28]" = torch.ops.aten.cat.default([convolution_29, convolution_30], 1);  convolution_29 = convolution_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_28: "f32[56]" = torch.ops.prims.convert_element_type.default(arg333_1, torch.float32);  arg333_1 = None
    convert_element_type_29: "f32[56]" = torch.ops.prims.convert_element_type.default(arg334_1, torch.float32);  arg334_1 = None
    add_30: "f32[56]" = torch.ops.aten.add.Tensor(convert_element_type_29, 0.001);  convert_element_type_29 = None
    sqrt_14: "f32[56]" = torch.ops.aten.sqrt.default(add_30);  add_30 = None
    reciprocal_14: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_14);  sqrt_14 = None
    mul_50: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_14, 1);  reciprocal_14 = None
    unsqueeze_112: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_28, -1);  convert_element_type_28 = None
    unsqueeze_113: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
    unsqueeze_114: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_50, -1);  mul_50 = None
    unsqueeze_115: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
    sub_14: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(cat_8, unsqueeze_113);  cat_8 = unsqueeze_113 = None
    mul_51: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(sub_14, unsqueeze_115);  sub_14 = unsqueeze_115 = None
    unsqueeze_116: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg36_1, -1);  arg36_1 = None
    unsqueeze_117: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
    mul_52: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(mul_51, unsqueeze_117);  mul_51 = unsqueeze_117 = None
    unsqueeze_118: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg37_1, -1);  arg37_1 = None
    unsqueeze_119: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
    add_31: "f32[8, 56, 28, 28]" = torch.ops.aten.add.Tensor(mul_52, unsqueeze_119);  mul_52 = unsqueeze_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_32: "f32[8, 56, 28, 28]" = torch.ops.aten.add.Tensor(add_31, add_25);  add_31 = add_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_22 = torch.ops.aten.split_with_sizes.default(add_32, [28, 28], 1)
    getitem_58: "f32[8, 28, 28, 28]" = split_with_sizes_22[0]
    getitem_59: "f32[8, 28, 28, 28]" = split_with_sizes_22[1];  split_with_sizes_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_31: "f32[8, 168, 28, 28]" = torch.ops.aten.convolution.default(getitem_58, arg158_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_58 = arg158_1 = None
    convolution_32: "f32[8, 168, 28, 28]" = torch.ops.aten.convolution.default(getitem_59, arg159_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_59 = arg159_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_9: "f32[8, 336, 28, 28]" = torch.ops.aten.cat.default([convolution_31, convolution_32], 1);  convolution_31 = convolution_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_30: "f32[336]" = torch.ops.prims.convert_element_type.default(arg335_1, torch.float32);  arg335_1 = None
    convert_element_type_31: "f32[336]" = torch.ops.prims.convert_element_type.default(arg336_1, torch.float32);  arg336_1 = None
    add_33: "f32[336]" = torch.ops.aten.add.Tensor(convert_element_type_31, 0.001);  convert_element_type_31 = None
    sqrt_15: "f32[336]" = torch.ops.aten.sqrt.default(add_33);  add_33 = None
    reciprocal_15: "f32[336]" = torch.ops.aten.reciprocal.default(sqrt_15);  sqrt_15 = None
    mul_53: "f32[336]" = torch.ops.aten.mul.Tensor(reciprocal_15, 1);  reciprocal_15 = None
    unsqueeze_120: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_30, -1);  convert_element_type_30 = None
    unsqueeze_121: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
    unsqueeze_122: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(mul_53, -1);  mul_53 = None
    unsqueeze_123: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
    sub_15: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(cat_9, unsqueeze_121);  cat_9 = unsqueeze_121 = None
    mul_54: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sub_15, unsqueeze_123);  sub_15 = unsqueeze_123 = None
    unsqueeze_124: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg38_1, -1);  arg38_1 = None
    unsqueeze_125: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
    mul_55: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(mul_54, unsqueeze_125);  mul_54 = unsqueeze_125 = None
    unsqueeze_126: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg39_1, -1);  arg39_1 = None
    unsqueeze_127: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
    add_34: "f32[8, 336, 28, 28]" = torch.ops.aten.add.Tensor(mul_55, unsqueeze_127);  mul_55 = unsqueeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_8: "f32[8, 336, 28, 28]" = torch.ops.aten.sigmoid.default(add_34)
    mul_56: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(add_34, sigmoid_8);  add_34 = sigmoid_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    split_with_sizes_24 = torch.ops.aten.split_with_sizes.default(mul_56, [168, 168], 1)
    getitem_62: "f32[8, 168, 28, 28]" = split_with_sizes_24[0];  split_with_sizes_24 = None
    convolution_33: "f32[8, 168, 28, 28]" = torch.ops.aten.convolution.default(getitem_62, arg160_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 168);  getitem_62 = arg160_1 = None
    split_with_sizes_25 = torch.ops.aten.split_with_sizes.default(mul_56, [168, 168], 1);  mul_56 = None
    getitem_65: "f32[8, 168, 28, 28]" = split_with_sizes_25[1];  split_with_sizes_25 = None
    convolution_34: "f32[8, 168, 28, 28]" = torch.ops.aten.convolution.default(getitem_65, arg161_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 168);  getitem_65 = arg161_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_10: "f32[8, 336, 28, 28]" = torch.ops.aten.cat.default([convolution_33, convolution_34], 1);  convolution_33 = convolution_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_32: "f32[336]" = torch.ops.prims.convert_element_type.default(arg337_1, torch.float32);  arg337_1 = None
    convert_element_type_33: "f32[336]" = torch.ops.prims.convert_element_type.default(arg338_1, torch.float32);  arg338_1 = None
    add_35: "f32[336]" = torch.ops.aten.add.Tensor(convert_element_type_33, 0.001);  convert_element_type_33 = None
    sqrt_16: "f32[336]" = torch.ops.aten.sqrt.default(add_35);  add_35 = None
    reciprocal_16: "f32[336]" = torch.ops.aten.reciprocal.default(sqrt_16);  sqrt_16 = None
    mul_57: "f32[336]" = torch.ops.aten.mul.Tensor(reciprocal_16, 1);  reciprocal_16 = None
    unsqueeze_128: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_32, -1);  convert_element_type_32 = None
    unsqueeze_129: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
    unsqueeze_130: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(mul_57, -1);  mul_57 = None
    unsqueeze_131: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
    sub_16: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(cat_10, unsqueeze_129);  cat_10 = unsqueeze_129 = None
    mul_58: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sub_16, unsqueeze_131);  sub_16 = unsqueeze_131 = None
    unsqueeze_132: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg40_1, -1);  arg40_1 = None
    unsqueeze_133: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
    mul_59: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(mul_58, unsqueeze_133);  mul_58 = unsqueeze_133 = None
    unsqueeze_134: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg41_1, -1);  arg41_1 = None
    unsqueeze_135: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
    add_36: "f32[8, 336, 28, 28]" = torch.ops.aten.add.Tensor(mul_59, unsqueeze_135);  mul_59 = unsqueeze_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_9: "f32[8, 336, 28, 28]" = torch.ops.aten.sigmoid.default(add_36)
    mul_60: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(add_36, sigmoid_9);  add_36 = sigmoid_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_2: "f32[8, 336, 1, 1]" = torch.ops.aten.mean.dim(mul_60, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_35: "f32[8, 28, 1, 1]" = torch.ops.aten.convolution.default(mean_2, arg162_1, arg163_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_2 = arg162_1 = arg163_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_10: "f32[8, 28, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_35)
    mul_61: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_35, sigmoid_10);  convolution_35 = sigmoid_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_36: "f32[8, 336, 1, 1]" = torch.ops.aten.convolution.default(mul_61, arg164_1, arg165_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_61 = arg164_1 = arg165_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_11: "f32[8, 336, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_36);  convolution_36 = None
    mul_62: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(mul_60, sigmoid_11);  mul_60 = sigmoid_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_26 = torch.ops.aten.split_with_sizes.default(mul_62, [168, 168], 1);  mul_62 = None
    getitem_66: "f32[8, 168, 28, 28]" = split_with_sizes_26[0]
    getitem_67: "f32[8, 168, 28, 28]" = split_with_sizes_26[1];  split_with_sizes_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_37: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(getitem_66, arg166_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_66 = arg166_1 = None
    convolution_38: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(getitem_67, arg167_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_67 = arg167_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_11: "f32[8, 56, 28, 28]" = torch.ops.aten.cat.default([convolution_37, convolution_38], 1);  convolution_37 = convolution_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_34: "f32[56]" = torch.ops.prims.convert_element_type.default(arg339_1, torch.float32);  arg339_1 = None
    convert_element_type_35: "f32[56]" = torch.ops.prims.convert_element_type.default(arg340_1, torch.float32);  arg340_1 = None
    add_37: "f32[56]" = torch.ops.aten.add.Tensor(convert_element_type_35, 0.001);  convert_element_type_35 = None
    sqrt_17: "f32[56]" = torch.ops.aten.sqrt.default(add_37);  add_37 = None
    reciprocal_17: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_17);  sqrt_17 = None
    mul_63: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_17, 1);  reciprocal_17 = None
    unsqueeze_136: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_34, -1);  convert_element_type_34 = None
    unsqueeze_137: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
    unsqueeze_138: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_63, -1);  mul_63 = None
    unsqueeze_139: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
    sub_17: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(cat_11, unsqueeze_137);  cat_11 = unsqueeze_137 = None
    mul_64: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(sub_17, unsqueeze_139);  sub_17 = unsqueeze_139 = None
    unsqueeze_140: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg42_1, -1);  arg42_1 = None
    unsqueeze_141: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
    mul_65: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(mul_64, unsqueeze_141);  mul_64 = unsqueeze_141 = None
    unsqueeze_142: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg43_1, -1);  arg43_1 = None
    unsqueeze_143: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
    add_38: "f32[8, 56, 28, 28]" = torch.ops.aten.add.Tensor(mul_65, unsqueeze_143);  mul_65 = unsqueeze_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_39: "f32[8, 56, 28, 28]" = torch.ops.aten.add.Tensor(add_38, add_32);  add_38 = add_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_27 = torch.ops.aten.split_with_sizes.default(add_39, [28, 28], 1)
    getitem_68: "f32[8, 28, 28, 28]" = split_with_sizes_27[0]
    getitem_69: "f32[8, 28, 28, 28]" = split_with_sizes_27[1];  split_with_sizes_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_39: "f32[8, 168, 28, 28]" = torch.ops.aten.convolution.default(getitem_68, arg168_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_68 = arg168_1 = None
    convolution_40: "f32[8, 168, 28, 28]" = torch.ops.aten.convolution.default(getitem_69, arg169_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_69 = arg169_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_12: "f32[8, 336, 28, 28]" = torch.ops.aten.cat.default([convolution_39, convolution_40], 1);  convolution_39 = convolution_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_36: "f32[336]" = torch.ops.prims.convert_element_type.default(arg341_1, torch.float32);  arg341_1 = None
    convert_element_type_37: "f32[336]" = torch.ops.prims.convert_element_type.default(arg342_1, torch.float32);  arg342_1 = None
    add_40: "f32[336]" = torch.ops.aten.add.Tensor(convert_element_type_37, 0.001);  convert_element_type_37 = None
    sqrt_18: "f32[336]" = torch.ops.aten.sqrt.default(add_40);  add_40 = None
    reciprocal_18: "f32[336]" = torch.ops.aten.reciprocal.default(sqrt_18);  sqrt_18 = None
    mul_66: "f32[336]" = torch.ops.aten.mul.Tensor(reciprocal_18, 1);  reciprocal_18 = None
    unsqueeze_144: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_36, -1);  convert_element_type_36 = None
    unsqueeze_145: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
    unsqueeze_146: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(mul_66, -1);  mul_66 = None
    unsqueeze_147: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
    sub_18: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(cat_12, unsqueeze_145);  cat_12 = unsqueeze_145 = None
    mul_67: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sub_18, unsqueeze_147);  sub_18 = unsqueeze_147 = None
    unsqueeze_148: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg44_1, -1);  arg44_1 = None
    unsqueeze_149: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
    mul_68: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(mul_67, unsqueeze_149);  mul_67 = unsqueeze_149 = None
    unsqueeze_150: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg45_1, -1);  arg45_1 = None
    unsqueeze_151: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
    add_41: "f32[8, 336, 28, 28]" = torch.ops.aten.add.Tensor(mul_68, unsqueeze_151);  mul_68 = unsqueeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_12: "f32[8, 336, 28, 28]" = torch.ops.aten.sigmoid.default(add_41)
    mul_69: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(add_41, sigmoid_12);  add_41 = sigmoid_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    split_with_sizes_29 = torch.ops.aten.split_with_sizes.default(mul_69, [168, 168], 1)
    getitem_72: "f32[8, 168, 28, 28]" = split_with_sizes_29[0];  split_with_sizes_29 = None
    convolution_41: "f32[8, 168, 28, 28]" = torch.ops.aten.convolution.default(getitem_72, arg170_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 168);  getitem_72 = arg170_1 = None
    split_with_sizes_30 = torch.ops.aten.split_with_sizes.default(mul_69, [168, 168], 1);  mul_69 = None
    getitem_75: "f32[8, 168, 28, 28]" = split_with_sizes_30[1];  split_with_sizes_30 = None
    convolution_42: "f32[8, 168, 28, 28]" = torch.ops.aten.convolution.default(getitem_75, arg171_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 168);  getitem_75 = arg171_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_13: "f32[8, 336, 28, 28]" = torch.ops.aten.cat.default([convolution_41, convolution_42], 1);  convolution_41 = convolution_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_38: "f32[336]" = torch.ops.prims.convert_element_type.default(arg343_1, torch.float32);  arg343_1 = None
    convert_element_type_39: "f32[336]" = torch.ops.prims.convert_element_type.default(arg344_1, torch.float32);  arg344_1 = None
    add_42: "f32[336]" = torch.ops.aten.add.Tensor(convert_element_type_39, 0.001);  convert_element_type_39 = None
    sqrt_19: "f32[336]" = torch.ops.aten.sqrt.default(add_42);  add_42 = None
    reciprocal_19: "f32[336]" = torch.ops.aten.reciprocal.default(sqrt_19);  sqrt_19 = None
    mul_70: "f32[336]" = torch.ops.aten.mul.Tensor(reciprocal_19, 1);  reciprocal_19 = None
    unsqueeze_152: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_38, -1);  convert_element_type_38 = None
    unsqueeze_153: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
    unsqueeze_154: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(mul_70, -1);  mul_70 = None
    unsqueeze_155: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
    sub_19: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(cat_13, unsqueeze_153);  cat_13 = unsqueeze_153 = None
    mul_71: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sub_19, unsqueeze_155);  sub_19 = unsqueeze_155 = None
    unsqueeze_156: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg46_1, -1);  arg46_1 = None
    unsqueeze_157: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
    mul_72: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(mul_71, unsqueeze_157);  mul_71 = unsqueeze_157 = None
    unsqueeze_158: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg47_1, -1);  arg47_1 = None
    unsqueeze_159: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
    add_43: "f32[8, 336, 28, 28]" = torch.ops.aten.add.Tensor(mul_72, unsqueeze_159);  mul_72 = unsqueeze_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_13: "f32[8, 336, 28, 28]" = torch.ops.aten.sigmoid.default(add_43)
    mul_73: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(add_43, sigmoid_13);  add_43 = sigmoid_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_3: "f32[8, 336, 1, 1]" = torch.ops.aten.mean.dim(mul_73, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_43: "f32[8, 28, 1, 1]" = torch.ops.aten.convolution.default(mean_3, arg172_1, arg173_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_3 = arg172_1 = arg173_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_14: "f32[8, 28, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_43)
    mul_74: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_43, sigmoid_14);  convolution_43 = sigmoid_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_44: "f32[8, 336, 1, 1]" = torch.ops.aten.convolution.default(mul_74, arg174_1, arg175_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_74 = arg174_1 = arg175_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_15: "f32[8, 336, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_44);  convolution_44 = None
    mul_75: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(mul_73, sigmoid_15);  mul_73 = sigmoid_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_31 = torch.ops.aten.split_with_sizes.default(mul_75, [168, 168], 1);  mul_75 = None
    getitem_76: "f32[8, 168, 28, 28]" = split_with_sizes_31[0]
    getitem_77: "f32[8, 168, 28, 28]" = split_with_sizes_31[1];  split_with_sizes_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_45: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(getitem_76, arg176_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_76 = arg176_1 = None
    convolution_46: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(getitem_77, arg177_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_77 = arg177_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_14: "f32[8, 56, 28, 28]" = torch.ops.aten.cat.default([convolution_45, convolution_46], 1);  convolution_45 = convolution_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_40: "f32[56]" = torch.ops.prims.convert_element_type.default(arg345_1, torch.float32);  arg345_1 = None
    convert_element_type_41: "f32[56]" = torch.ops.prims.convert_element_type.default(arg346_1, torch.float32);  arg346_1 = None
    add_44: "f32[56]" = torch.ops.aten.add.Tensor(convert_element_type_41, 0.001);  convert_element_type_41 = None
    sqrt_20: "f32[56]" = torch.ops.aten.sqrt.default(add_44);  add_44 = None
    reciprocal_20: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_20);  sqrt_20 = None
    mul_76: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_20, 1);  reciprocal_20 = None
    unsqueeze_160: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_40, -1);  convert_element_type_40 = None
    unsqueeze_161: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
    unsqueeze_162: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_76, -1);  mul_76 = None
    unsqueeze_163: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
    sub_20: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(cat_14, unsqueeze_161);  cat_14 = unsqueeze_161 = None
    mul_77: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(sub_20, unsqueeze_163);  sub_20 = unsqueeze_163 = None
    unsqueeze_164: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg48_1, -1);  arg48_1 = None
    unsqueeze_165: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
    mul_78: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(mul_77, unsqueeze_165);  mul_77 = unsqueeze_165 = None
    unsqueeze_166: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg49_1, -1);  arg49_1 = None
    unsqueeze_167: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
    add_45: "f32[8, 56, 28, 28]" = torch.ops.aten.add.Tensor(mul_78, unsqueeze_167);  mul_78 = unsqueeze_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_46: "f32[8, 56, 28, 28]" = torch.ops.aten.add.Tensor(add_45, add_39);  add_45 = add_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_47: "f32[8, 336, 28, 28]" = torch.ops.aten.convolution.default(add_46, arg178_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_46 = arg178_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_42: "f32[336]" = torch.ops.prims.convert_element_type.default(arg347_1, torch.float32);  arg347_1 = None
    convert_element_type_43: "f32[336]" = torch.ops.prims.convert_element_type.default(arg348_1, torch.float32);  arg348_1 = None
    add_47: "f32[336]" = torch.ops.aten.add.Tensor(convert_element_type_43, 0.001);  convert_element_type_43 = None
    sqrt_21: "f32[336]" = torch.ops.aten.sqrt.default(add_47);  add_47 = None
    reciprocal_21: "f32[336]" = torch.ops.aten.reciprocal.default(sqrt_21);  sqrt_21 = None
    mul_79: "f32[336]" = torch.ops.aten.mul.Tensor(reciprocal_21, 1);  reciprocal_21 = None
    unsqueeze_168: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_42, -1);  convert_element_type_42 = None
    unsqueeze_169: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
    unsqueeze_170: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(mul_79, -1);  mul_79 = None
    unsqueeze_171: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
    sub_21: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_169);  convolution_47 = unsqueeze_169 = None
    mul_80: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sub_21, unsqueeze_171);  sub_21 = unsqueeze_171 = None
    unsqueeze_172: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg50_1, -1);  arg50_1 = None
    unsqueeze_173: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
    mul_81: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(mul_80, unsqueeze_173);  mul_80 = unsqueeze_173 = None
    unsqueeze_174: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg51_1, -1);  arg51_1 = None
    unsqueeze_175: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
    add_48: "f32[8, 336, 28, 28]" = torch.ops.aten.add.Tensor(mul_81, unsqueeze_175);  mul_81 = unsqueeze_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_16: "f32[8, 336, 28, 28]" = torch.ops.aten.sigmoid.default(add_48)
    mul_82: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(add_48, sigmoid_16);  add_48 = sigmoid_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    split_with_sizes_33 = torch.ops.aten.split_with_sizes.default(mul_82, [112, 112, 112], 1)
    getitem_81: "f32[8, 112, 28, 28]" = split_with_sizes_33[0];  split_with_sizes_33 = None
    constant_pad_nd_8: "f32[8, 112, 29, 29]" = torch.ops.aten.constant_pad_nd.default(getitem_81, [0, 1, 0, 1], 0.0);  getitem_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_48: "f32[8, 112, 14, 14]" = torch.ops.aten.convolution.default(constant_pad_nd_8, arg52_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 112);  constant_pad_nd_8 = arg52_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    split_with_sizes_34 = torch.ops.aten.split_with_sizes.default(mul_82, [112, 112, 112], 1)
    getitem_85: "f32[8, 112, 28, 28]" = split_with_sizes_34[1];  split_with_sizes_34 = None
    constant_pad_nd_9: "f32[8, 112, 31, 31]" = torch.ops.aten.constant_pad_nd.default(getitem_85, [1, 2, 1, 2], 0.0);  getitem_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_49: "f32[8, 112, 14, 14]" = torch.ops.aten.convolution.default(constant_pad_nd_9, arg53_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 112);  constant_pad_nd_9 = arg53_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    split_with_sizes_35 = torch.ops.aten.split_with_sizes.default(mul_82, [112, 112, 112], 1);  mul_82 = None
    getitem_89: "f32[8, 112, 28, 28]" = split_with_sizes_35[2];  split_with_sizes_35 = None
    constant_pad_nd_10: "f32[8, 112, 33, 33]" = torch.ops.aten.constant_pad_nd.default(getitem_89, [2, 3, 2, 3], 0.0);  getitem_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_50: "f32[8, 112, 14, 14]" = torch.ops.aten.convolution.default(constant_pad_nd_10, arg54_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 112);  constant_pad_nd_10 = arg54_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_15: "f32[8, 336, 14, 14]" = torch.ops.aten.cat.default([convolution_48, convolution_49, convolution_50], 1);  convolution_48 = convolution_49 = convolution_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_44: "f32[336]" = torch.ops.prims.convert_element_type.default(arg349_1, torch.float32);  arg349_1 = None
    convert_element_type_45: "f32[336]" = torch.ops.prims.convert_element_type.default(arg350_1, torch.float32);  arg350_1 = None
    add_49: "f32[336]" = torch.ops.aten.add.Tensor(convert_element_type_45, 0.001);  convert_element_type_45 = None
    sqrt_22: "f32[336]" = torch.ops.aten.sqrt.default(add_49);  add_49 = None
    reciprocal_22: "f32[336]" = torch.ops.aten.reciprocal.default(sqrt_22);  sqrt_22 = None
    mul_83: "f32[336]" = torch.ops.aten.mul.Tensor(reciprocal_22, 1);  reciprocal_22 = None
    unsqueeze_176: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_44, -1);  convert_element_type_44 = None
    unsqueeze_177: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
    unsqueeze_178: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(mul_83, -1);  mul_83 = None
    unsqueeze_179: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
    sub_22: "f32[8, 336, 14, 14]" = torch.ops.aten.sub.Tensor(cat_15, unsqueeze_177);  cat_15 = unsqueeze_177 = None
    mul_84: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(sub_22, unsqueeze_179);  sub_22 = unsqueeze_179 = None
    unsqueeze_180: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg55_1, -1);  arg55_1 = None
    unsqueeze_181: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
    mul_85: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(mul_84, unsqueeze_181);  mul_84 = unsqueeze_181 = None
    unsqueeze_182: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg56_1, -1);  arg56_1 = None
    unsqueeze_183: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
    add_50: "f32[8, 336, 14, 14]" = torch.ops.aten.add.Tensor(mul_85, unsqueeze_183);  mul_85 = unsqueeze_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_17: "f32[8, 336, 14, 14]" = torch.ops.aten.sigmoid.default(add_50)
    mul_86: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(add_50, sigmoid_17);  add_50 = sigmoid_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_4: "f32[8, 336, 1, 1]" = torch.ops.aten.mean.dim(mul_86, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_51: "f32[8, 14, 1, 1]" = torch.ops.aten.convolution.default(mean_4, arg179_1, arg180_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_4 = arg179_1 = arg180_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_18: "f32[8, 14, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_51)
    mul_87: "f32[8, 14, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_51, sigmoid_18);  convolution_51 = sigmoid_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_52: "f32[8, 336, 1, 1]" = torch.ops.aten.convolution.default(mul_87, arg181_1, arg182_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_87 = arg181_1 = arg182_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_19: "f32[8, 336, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_52);  convolution_52 = None
    mul_88: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(mul_86, sigmoid_19);  mul_86 = sigmoid_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_53: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(mul_88, arg183_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_88 = arg183_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_46: "f32[104]" = torch.ops.prims.convert_element_type.default(arg351_1, torch.float32);  arg351_1 = None
    convert_element_type_47: "f32[104]" = torch.ops.prims.convert_element_type.default(arg352_1, torch.float32);  arg352_1 = None
    add_51: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_47, 0.001);  convert_element_type_47 = None
    sqrt_23: "f32[104]" = torch.ops.aten.sqrt.default(add_51);  add_51 = None
    reciprocal_23: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_23);  sqrt_23 = None
    mul_89: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_23, 1);  reciprocal_23 = None
    unsqueeze_184: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_46, -1);  convert_element_type_46 = None
    unsqueeze_185: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
    unsqueeze_186: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_89, -1);  mul_89 = None
    unsqueeze_187: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
    sub_23: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_185);  convolution_53 = unsqueeze_185 = None
    mul_90: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_23, unsqueeze_187);  sub_23 = unsqueeze_187 = None
    unsqueeze_188: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg57_1, -1);  arg57_1 = None
    unsqueeze_189: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
    mul_91: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_90, unsqueeze_189);  mul_90 = unsqueeze_189 = None
    unsqueeze_190: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg58_1, -1);  arg58_1 = None
    unsqueeze_191: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
    add_52: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_91, unsqueeze_191);  mul_91 = unsqueeze_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_36 = torch.ops.aten.split_with_sizes.default(add_52, [52, 52], 1)
    getitem_90: "f32[8, 52, 14, 14]" = split_with_sizes_36[0]
    getitem_91: "f32[8, 52, 14, 14]" = split_with_sizes_36[1];  split_with_sizes_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_54: "f32[8, 312, 14, 14]" = torch.ops.aten.convolution.default(getitem_90, arg184_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_90 = arg184_1 = None
    convolution_55: "f32[8, 312, 14, 14]" = torch.ops.aten.convolution.default(getitem_91, arg185_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_91 = arg185_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_16: "f32[8, 624, 14, 14]" = torch.ops.aten.cat.default([convolution_54, convolution_55], 1);  convolution_54 = convolution_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_48: "f32[624]" = torch.ops.prims.convert_element_type.default(arg353_1, torch.float32);  arg353_1 = None
    convert_element_type_49: "f32[624]" = torch.ops.prims.convert_element_type.default(arg354_1, torch.float32);  arg354_1 = None
    add_53: "f32[624]" = torch.ops.aten.add.Tensor(convert_element_type_49, 0.001);  convert_element_type_49 = None
    sqrt_24: "f32[624]" = torch.ops.aten.sqrt.default(add_53);  add_53 = None
    reciprocal_24: "f32[624]" = torch.ops.aten.reciprocal.default(sqrt_24);  sqrt_24 = None
    mul_92: "f32[624]" = torch.ops.aten.mul.Tensor(reciprocal_24, 1);  reciprocal_24 = None
    unsqueeze_192: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_48, -1);  convert_element_type_48 = None
    unsqueeze_193: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
    unsqueeze_194: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(mul_92, -1);  mul_92 = None
    unsqueeze_195: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
    sub_24: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(cat_16, unsqueeze_193);  cat_16 = unsqueeze_193 = None
    mul_93: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sub_24, unsqueeze_195);  sub_24 = unsqueeze_195 = None
    unsqueeze_196: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(arg59_1, -1);  arg59_1 = None
    unsqueeze_197: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
    mul_94: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_93, unsqueeze_197);  mul_93 = unsqueeze_197 = None
    unsqueeze_198: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(arg60_1, -1);  arg60_1 = None
    unsqueeze_199: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
    add_54: "f32[8, 624, 14, 14]" = torch.ops.aten.add.Tensor(mul_94, unsqueeze_199);  mul_94 = unsqueeze_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_20: "f32[8, 624, 14, 14]" = torch.ops.aten.sigmoid.default(add_54)
    mul_95: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(add_54, sigmoid_20);  add_54 = sigmoid_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    split_with_sizes_38 = torch.ops.aten.split_with_sizes.default(mul_95, [156, 156, 156, 156], 1)
    getitem_96: "f32[8, 156, 14, 14]" = split_with_sizes_38[0];  split_with_sizes_38 = None
    convolution_56: "f32[8, 156, 14, 14]" = torch.ops.aten.convolution.default(getitem_96, arg186_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 156);  getitem_96 = arg186_1 = None
    split_with_sizes_39 = torch.ops.aten.split_with_sizes.default(mul_95, [156, 156, 156, 156], 1)
    getitem_101: "f32[8, 156, 14, 14]" = split_with_sizes_39[1];  split_with_sizes_39 = None
    convolution_57: "f32[8, 156, 14, 14]" = torch.ops.aten.convolution.default(getitem_101, arg187_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 156);  getitem_101 = arg187_1 = None
    split_with_sizes_40 = torch.ops.aten.split_with_sizes.default(mul_95, [156, 156, 156, 156], 1)
    getitem_106: "f32[8, 156, 14, 14]" = split_with_sizes_40[2];  split_with_sizes_40 = None
    convolution_58: "f32[8, 156, 14, 14]" = torch.ops.aten.convolution.default(getitem_106, arg188_1, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 156);  getitem_106 = arg188_1 = None
    split_with_sizes_41 = torch.ops.aten.split_with_sizes.default(mul_95, [156, 156, 156, 156], 1);  mul_95 = None
    getitem_111: "f32[8, 156, 14, 14]" = split_with_sizes_41[3];  split_with_sizes_41 = None
    convolution_59: "f32[8, 156, 14, 14]" = torch.ops.aten.convolution.default(getitem_111, arg189_1, None, [1, 1], [4, 4], [1, 1], False, [0, 0], 156);  getitem_111 = arg189_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_17: "f32[8, 624, 14, 14]" = torch.ops.aten.cat.default([convolution_56, convolution_57, convolution_58, convolution_59], 1);  convolution_56 = convolution_57 = convolution_58 = convolution_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_50: "f32[624]" = torch.ops.prims.convert_element_type.default(arg355_1, torch.float32);  arg355_1 = None
    convert_element_type_51: "f32[624]" = torch.ops.prims.convert_element_type.default(arg356_1, torch.float32);  arg356_1 = None
    add_55: "f32[624]" = torch.ops.aten.add.Tensor(convert_element_type_51, 0.001);  convert_element_type_51 = None
    sqrt_25: "f32[624]" = torch.ops.aten.sqrt.default(add_55);  add_55 = None
    reciprocal_25: "f32[624]" = torch.ops.aten.reciprocal.default(sqrt_25);  sqrt_25 = None
    mul_96: "f32[624]" = torch.ops.aten.mul.Tensor(reciprocal_25, 1);  reciprocal_25 = None
    unsqueeze_200: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_50, -1);  convert_element_type_50 = None
    unsqueeze_201: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
    unsqueeze_202: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(mul_96, -1);  mul_96 = None
    unsqueeze_203: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
    sub_25: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(cat_17, unsqueeze_201);  cat_17 = unsqueeze_201 = None
    mul_97: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sub_25, unsqueeze_203);  sub_25 = unsqueeze_203 = None
    unsqueeze_204: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(arg61_1, -1);  arg61_1 = None
    unsqueeze_205: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
    mul_98: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_97, unsqueeze_205);  mul_97 = unsqueeze_205 = None
    unsqueeze_206: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(arg62_1, -1);  arg62_1 = None
    unsqueeze_207: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
    add_56: "f32[8, 624, 14, 14]" = torch.ops.aten.add.Tensor(mul_98, unsqueeze_207);  mul_98 = unsqueeze_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_21: "f32[8, 624, 14, 14]" = torch.ops.aten.sigmoid.default(add_56)
    mul_99: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(add_56, sigmoid_21);  add_56 = sigmoid_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_5: "f32[8, 624, 1, 1]" = torch.ops.aten.mean.dim(mul_99, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_60: "f32[8, 26, 1, 1]" = torch.ops.aten.convolution.default(mean_5, arg190_1, arg191_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_5 = arg190_1 = arg191_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_22: "f32[8, 26, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_60)
    mul_100: "f32[8, 26, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_60, sigmoid_22);  convolution_60 = sigmoid_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_61: "f32[8, 624, 1, 1]" = torch.ops.aten.convolution.default(mul_100, arg192_1, arg193_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_100 = arg192_1 = arg193_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_23: "f32[8, 624, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_61);  convolution_61 = None
    mul_101: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_99, sigmoid_23);  mul_99 = sigmoid_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_42 = torch.ops.aten.split_with_sizes.default(mul_101, [312, 312], 1);  mul_101 = None
    getitem_112: "f32[8, 312, 14, 14]" = split_with_sizes_42[0]
    getitem_113: "f32[8, 312, 14, 14]" = split_with_sizes_42[1];  split_with_sizes_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_62: "f32[8, 52, 14, 14]" = torch.ops.aten.convolution.default(getitem_112, arg194_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_112 = arg194_1 = None
    convolution_63: "f32[8, 52, 14, 14]" = torch.ops.aten.convolution.default(getitem_113, arg195_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_113 = arg195_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_18: "f32[8, 104, 14, 14]" = torch.ops.aten.cat.default([convolution_62, convolution_63], 1);  convolution_62 = convolution_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_52: "f32[104]" = torch.ops.prims.convert_element_type.default(arg357_1, torch.float32);  arg357_1 = None
    convert_element_type_53: "f32[104]" = torch.ops.prims.convert_element_type.default(arg358_1, torch.float32);  arg358_1 = None
    add_57: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_53, 0.001);  convert_element_type_53 = None
    sqrt_26: "f32[104]" = torch.ops.aten.sqrt.default(add_57);  add_57 = None
    reciprocal_26: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_26);  sqrt_26 = None
    mul_102: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_26, 1);  reciprocal_26 = None
    unsqueeze_208: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_52, -1);  convert_element_type_52 = None
    unsqueeze_209: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
    unsqueeze_210: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_102, -1);  mul_102 = None
    unsqueeze_211: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
    sub_26: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(cat_18, unsqueeze_209);  cat_18 = unsqueeze_209 = None
    mul_103: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_26, unsqueeze_211);  sub_26 = unsqueeze_211 = None
    unsqueeze_212: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg63_1, -1);  arg63_1 = None
    unsqueeze_213: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
    mul_104: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_103, unsqueeze_213);  mul_103 = unsqueeze_213 = None
    unsqueeze_214: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg64_1, -1);  arg64_1 = None
    unsqueeze_215: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
    add_58: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_104, unsqueeze_215);  mul_104 = unsqueeze_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_59: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(add_58, add_52);  add_58 = add_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_43 = torch.ops.aten.split_with_sizes.default(add_59, [52, 52], 1)
    getitem_114: "f32[8, 52, 14, 14]" = split_with_sizes_43[0]
    getitem_115: "f32[8, 52, 14, 14]" = split_with_sizes_43[1];  split_with_sizes_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_64: "f32[8, 312, 14, 14]" = torch.ops.aten.convolution.default(getitem_114, arg196_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_114 = arg196_1 = None
    convolution_65: "f32[8, 312, 14, 14]" = torch.ops.aten.convolution.default(getitem_115, arg197_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_115 = arg197_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_19: "f32[8, 624, 14, 14]" = torch.ops.aten.cat.default([convolution_64, convolution_65], 1);  convolution_64 = convolution_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_54: "f32[624]" = torch.ops.prims.convert_element_type.default(arg359_1, torch.float32);  arg359_1 = None
    convert_element_type_55: "f32[624]" = torch.ops.prims.convert_element_type.default(arg360_1, torch.float32);  arg360_1 = None
    add_60: "f32[624]" = torch.ops.aten.add.Tensor(convert_element_type_55, 0.001);  convert_element_type_55 = None
    sqrt_27: "f32[624]" = torch.ops.aten.sqrt.default(add_60);  add_60 = None
    reciprocal_27: "f32[624]" = torch.ops.aten.reciprocal.default(sqrt_27);  sqrt_27 = None
    mul_105: "f32[624]" = torch.ops.aten.mul.Tensor(reciprocal_27, 1);  reciprocal_27 = None
    unsqueeze_216: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_54, -1);  convert_element_type_54 = None
    unsqueeze_217: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
    unsqueeze_218: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(mul_105, -1);  mul_105 = None
    unsqueeze_219: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, -1);  unsqueeze_218 = None
    sub_27: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(cat_19, unsqueeze_217);  cat_19 = unsqueeze_217 = None
    mul_106: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sub_27, unsqueeze_219);  sub_27 = unsqueeze_219 = None
    unsqueeze_220: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(arg65_1, -1);  arg65_1 = None
    unsqueeze_221: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, -1);  unsqueeze_220 = None
    mul_107: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_106, unsqueeze_221);  mul_106 = unsqueeze_221 = None
    unsqueeze_222: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(arg66_1, -1);  arg66_1 = None
    unsqueeze_223: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, -1);  unsqueeze_222 = None
    add_61: "f32[8, 624, 14, 14]" = torch.ops.aten.add.Tensor(mul_107, unsqueeze_223);  mul_107 = unsqueeze_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_24: "f32[8, 624, 14, 14]" = torch.ops.aten.sigmoid.default(add_61)
    mul_108: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(add_61, sigmoid_24);  add_61 = sigmoid_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    split_with_sizes_45 = torch.ops.aten.split_with_sizes.default(mul_108, [156, 156, 156, 156], 1)
    getitem_120: "f32[8, 156, 14, 14]" = split_with_sizes_45[0];  split_with_sizes_45 = None
    convolution_66: "f32[8, 156, 14, 14]" = torch.ops.aten.convolution.default(getitem_120, arg198_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 156);  getitem_120 = arg198_1 = None
    split_with_sizes_46 = torch.ops.aten.split_with_sizes.default(mul_108, [156, 156, 156, 156], 1)
    getitem_125: "f32[8, 156, 14, 14]" = split_with_sizes_46[1];  split_with_sizes_46 = None
    convolution_67: "f32[8, 156, 14, 14]" = torch.ops.aten.convolution.default(getitem_125, arg199_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 156);  getitem_125 = arg199_1 = None
    split_with_sizes_47 = torch.ops.aten.split_with_sizes.default(mul_108, [156, 156, 156, 156], 1)
    getitem_130: "f32[8, 156, 14, 14]" = split_with_sizes_47[2];  split_with_sizes_47 = None
    convolution_68: "f32[8, 156, 14, 14]" = torch.ops.aten.convolution.default(getitem_130, arg200_1, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 156);  getitem_130 = arg200_1 = None
    split_with_sizes_48 = torch.ops.aten.split_with_sizes.default(mul_108, [156, 156, 156, 156], 1);  mul_108 = None
    getitem_135: "f32[8, 156, 14, 14]" = split_with_sizes_48[3];  split_with_sizes_48 = None
    convolution_69: "f32[8, 156, 14, 14]" = torch.ops.aten.convolution.default(getitem_135, arg201_1, None, [1, 1], [4, 4], [1, 1], False, [0, 0], 156);  getitem_135 = arg201_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_20: "f32[8, 624, 14, 14]" = torch.ops.aten.cat.default([convolution_66, convolution_67, convolution_68, convolution_69], 1);  convolution_66 = convolution_67 = convolution_68 = convolution_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_56: "f32[624]" = torch.ops.prims.convert_element_type.default(arg361_1, torch.float32);  arg361_1 = None
    convert_element_type_57: "f32[624]" = torch.ops.prims.convert_element_type.default(arg362_1, torch.float32);  arg362_1 = None
    add_62: "f32[624]" = torch.ops.aten.add.Tensor(convert_element_type_57, 0.001);  convert_element_type_57 = None
    sqrt_28: "f32[624]" = torch.ops.aten.sqrt.default(add_62);  add_62 = None
    reciprocal_28: "f32[624]" = torch.ops.aten.reciprocal.default(sqrt_28);  sqrt_28 = None
    mul_109: "f32[624]" = torch.ops.aten.mul.Tensor(reciprocal_28, 1);  reciprocal_28 = None
    unsqueeze_224: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_56, -1);  convert_element_type_56 = None
    unsqueeze_225: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, -1);  unsqueeze_224 = None
    unsqueeze_226: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(mul_109, -1);  mul_109 = None
    unsqueeze_227: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, -1);  unsqueeze_226 = None
    sub_28: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(cat_20, unsqueeze_225);  cat_20 = unsqueeze_225 = None
    mul_110: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sub_28, unsqueeze_227);  sub_28 = unsqueeze_227 = None
    unsqueeze_228: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(arg67_1, -1);  arg67_1 = None
    unsqueeze_229: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, -1);  unsqueeze_228 = None
    mul_111: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_110, unsqueeze_229);  mul_110 = unsqueeze_229 = None
    unsqueeze_230: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(arg68_1, -1);  arg68_1 = None
    unsqueeze_231: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, -1);  unsqueeze_230 = None
    add_63: "f32[8, 624, 14, 14]" = torch.ops.aten.add.Tensor(mul_111, unsqueeze_231);  mul_111 = unsqueeze_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_25: "f32[8, 624, 14, 14]" = torch.ops.aten.sigmoid.default(add_63)
    mul_112: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(add_63, sigmoid_25);  add_63 = sigmoid_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_6: "f32[8, 624, 1, 1]" = torch.ops.aten.mean.dim(mul_112, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_70: "f32[8, 26, 1, 1]" = torch.ops.aten.convolution.default(mean_6, arg202_1, arg203_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_6 = arg202_1 = arg203_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_26: "f32[8, 26, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_70)
    mul_113: "f32[8, 26, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_70, sigmoid_26);  convolution_70 = sigmoid_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_71: "f32[8, 624, 1, 1]" = torch.ops.aten.convolution.default(mul_113, arg204_1, arg205_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_113 = arg204_1 = arg205_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_27: "f32[8, 624, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_71);  convolution_71 = None
    mul_114: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_112, sigmoid_27);  mul_112 = sigmoid_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_49 = torch.ops.aten.split_with_sizes.default(mul_114, [312, 312], 1);  mul_114 = None
    getitem_136: "f32[8, 312, 14, 14]" = split_with_sizes_49[0]
    getitem_137: "f32[8, 312, 14, 14]" = split_with_sizes_49[1];  split_with_sizes_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_72: "f32[8, 52, 14, 14]" = torch.ops.aten.convolution.default(getitem_136, arg206_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_136 = arg206_1 = None
    convolution_73: "f32[8, 52, 14, 14]" = torch.ops.aten.convolution.default(getitem_137, arg207_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_137 = arg207_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_21: "f32[8, 104, 14, 14]" = torch.ops.aten.cat.default([convolution_72, convolution_73], 1);  convolution_72 = convolution_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_58: "f32[104]" = torch.ops.prims.convert_element_type.default(arg363_1, torch.float32);  arg363_1 = None
    convert_element_type_59: "f32[104]" = torch.ops.prims.convert_element_type.default(arg364_1, torch.float32);  arg364_1 = None
    add_64: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_59, 0.001);  convert_element_type_59 = None
    sqrt_29: "f32[104]" = torch.ops.aten.sqrt.default(add_64);  add_64 = None
    reciprocal_29: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_29);  sqrt_29 = None
    mul_115: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_29, 1);  reciprocal_29 = None
    unsqueeze_232: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_58, -1);  convert_element_type_58 = None
    unsqueeze_233: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, -1);  unsqueeze_232 = None
    unsqueeze_234: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_115, -1);  mul_115 = None
    unsqueeze_235: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, -1);  unsqueeze_234 = None
    sub_29: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(cat_21, unsqueeze_233);  cat_21 = unsqueeze_233 = None
    mul_116: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_29, unsqueeze_235);  sub_29 = unsqueeze_235 = None
    unsqueeze_236: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg69_1, -1);  arg69_1 = None
    unsqueeze_237: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, -1);  unsqueeze_236 = None
    mul_117: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_116, unsqueeze_237);  mul_116 = unsqueeze_237 = None
    unsqueeze_238: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg70_1, -1);  arg70_1 = None
    unsqueeze_239: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, -1);  unsqueeze_238 = None
    add_65: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_117, unsqueeze_239);  mul_117 = unsqueeze_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_66: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(add_65, add_59);  add_65 = add_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_50 = torch.ops.aten.split_with_sizes.default(add_66, [52, 52], 1)
    getitem_138: "f32[8, 52, 14, 14]" = split_with_sizes_50[0]
    getitem_139: "f32[8, 52, 14, 14]" = split_with_sizes_50[1];  split_with_sizes_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_74: "f32[8, 312, 14, 14]" = torch.ops.aten.convolution.default(getitem_138, arg208_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_138 = arg208_1 = None
    convolution_75: "f32[8, 312, 14, 14]" = torch.ops.aten.convolution.default(getitem_139, arg209_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_139 = arg209_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_22: "f32[8, 624, 14, 14]" = torch.ops.aten.cat.default([convolution_74, convolution_75], 1);  convolution_74 = convolution_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_60: "f32[624]" = torch.ops.prims.convert_element_type.default(arg365_1, torch.float32);  arg365_1 = None
    convert_element_type_61: "f32[624]" = torch.ops.prims.convert_element_type.default(arg366_1, torch.float32);  arg366_1 = None
    add_67: "f32[624]" = torch.ops.aten.add.Tensor(convert_element_type_61, 0.001);  convert_element_type_61 = None
    sqrt_30: "f32[624]" = torch.ops.aten.sqrt.default(add_67);  add_67 = None
    reciprocal_30: "f32[624]" = torch.ops.aten.reciprocal.default(sqrt_30);  sqrt_30 = None
    mul_118: "f32[624]" = torch.ops.aten.mul.Tensor(reciprocal_30, 1);  reciprocal_30 = None
    unsqueeze_240: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_60, -1);  convert_element_type_60 = None
    unsqueeze_241: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, -1);  unsqueeze_240 = None
    unsqueeze_242: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(mul_118, -1);  mul_118 = None
    unsqueeze_243: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, -1);  unsqueeze_242 = None
    sub_30: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(cat_22, unsqueeze_241);  cat_22 = unsqueeze_241 = None
    mul_119: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sub_30, unsqueeze_243);  sub_30 = unsqueeze_243 = None
    unsqueeze_244: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(arg71_1, -1);  arg71_1 = None
    unsqueeze_245: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, -1);  unsqueeze_244 = None
    mul_120: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_119, unsqueeze_245);  mul_119 = unsqueeze_245 = None
    unsqueeze_246: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(arg72_1, -1);  arg72_1 = None
    unsqueeze_247: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, -1);  unsqueeze_246 = None
    add_68: "f32[8, 624, 14, 14]" = torch.ops.aten.add.Tensor(mul_120, unsqueeze_247);  mul_120 = unsqueeze_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_28: "f32[8, 624, 14, 14]" = torch.ops.aten.sigmoid.default(add_68)
    mul_121: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(add_68, sigmoid_28);  add_68 = sigmoid_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    split_with_sizes_52 = torch.ops.aten.split_with_sizes.default(mul_121, [156, 156, 156, 156], 1)
    getitem_144: "f32[8, 156, 14, 14]" = split_with_sizes_52[0];  split_with_sizes_52 = None
    convolution_76: "f32[8, 156, 14, 14]" = torch.ops.aten.convolution.default(getitem_144, arg210_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 156);  getitem_144 = arg210_1 = None
    split_with_sizes_53 = torch.ops.aten.split_with_sizes.default(mul_121, [156, 156, 156, 156], 1)
    getitem_149: "f32[8, 156, 14, 14]" = split_with_sizes_53[1];  split_with_sizes_53 = None
    convolution_77: "f32[8, 156, 14, 14]" = torch.ops.aten.convolution.default(getitem_149, arg211_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 156);  getitem_149 = arg211_1 = None
    split_with_sizes_54 = torch.ops.aten.split_with_sizes.default(mul_121, [156, 156, 156, 156], 1)
    getitem_154: "f32[8, 156, 14, 14]" = split_with_sizes_54[2];  split_with_sizes_54 = None
    convolution_78: "f32[8, 156, 14, 14]" = torch.ops.aten.convolution.default(getitem_154, arg212_1, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 156);  getitem_154 = arg212_1 = None
    split_with_sizes_55 = torch.ops.aten.split_with_sizes.default(mul_121, [156, 156, 156, 156], 1);  mul_121 = None
    getitem_159: "f32[8, 156, 14, 14]" = split_with_sizes_55[3];  split_with_sizes_55 = None
    convolution_79: "f32[8, 156, 14, 14]" = torch.ops.aten.convolution.default(getitem_159, arg213_1, None, [1, 1], [4, 4], [1, 1], False, [0, 0], 156);  getitem_159 = arg213_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_23: "f32[8, 624, 14, 14]" = torch.ops.aten.cat.default([convolution_76, convolution_77, convolution_78, convolution_79], 1);  convolution_76 = convolution_77 = convolution_78 = convolution_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_62: "f32[624]" = torch.ops.prims.convert_element_type.default(arg367_1, torch.float32);  arg367_1 = None
    convert_element_type_63: "f32[624]" = torch.ops.prims.convert_element_type.default(arg368_1, torch.float32);  arg368_1 = None
    add_69: "f32[624]" = torch.ops.aten.add.Tensor(convert_element_type_63, 0.001);  convert_element_type_63 = None
    sqrt_31: "f32[624]" = torch.ops.aten.sqrt.default(add_69);  add_69 = None
    reciprocal_31: "f32[624]" = torch.ops.aten.reciprocal.default(sqrt_31);  sqrt_31 = None
    mul_122: "f32[624]" = torch.ops.aten.mul.Tensor(reciprocal_31, 1);  reciprocal_31 = None
    unsqueeze_248: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_62, -1);  convert_element_type_62 = None
    unsqueeze_249: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, -1);  unsqueeze_248 = None
    unsqueeze_250: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(mul_122, -1);  mul_122 = None
    unsqueeze_251: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, -1);  unsqueeze_250 = None
    sub_31: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(cat_23, unsqueeze_249);  cat_23 = unsqueeze_249 = None
    mul_123: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sub_31, unsqueeze_251);  sub_31 = unsqueeze_251 = None
    unsqueeze_252: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(arg73_1, -1);  arg73_1 = None
    unsqueeze_253: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, -1);  unsqueeze_252 = None
    mul_124: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_123, unsqueeze_253);  mul_123 = unsqueeze_253 = None
    unsqueeze_254: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(arg74_1, -1);  arg74_1 = None
    unsqueeze_255: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, -1);  unsqueeze_254 = None
    add_70: "f32[8, 624, 14, 14]" = torch.ops.aten.add.Tensor(mul_124, unsqueeze_255);  mul_124 = unsqueeze_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_29: "f32[8, 624, 14, 14]" = torch.ops.aten.sigmoid.default(add_70)
    mul_125: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(add_70, sigmoid_29);  add_70 = sigmoid_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_7: "f32[8, 624, 1, 1]" = torch.ops.aten.mean.dim(mul_125, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_80: "f32[8, 26, 1, 1]" = torch.ops.aten.convolution.default(mean_7, arg214_1, arg215_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_7 = arg214_1 = arg215_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_30: "f32[8, 26, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_80)
    mul_126: "f32[8, 26, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_80, sigmoid_30);  convolution_80 = sigmoid_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_81: "f32[8, 624, 1, 1]" = torch.ops.aten.convolution.default(mul_126, arg216_1, arg217_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_126 = arg216_1 = arg217_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_31: "f32[8, 624, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_81);  convolution_81 = None
    mul_127: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_125, sigmoid_31);  mul_125 = sigmoid_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_56 = torch.ops.aten.split_with_sizes.default(mul_127, [312, 312], 1);  mul_127 = None
    getitem_160: "f32[8, 312, 14, 14]" = split_with_sizes_56[0]
    getitem_161: "f32[8, 312, 14, 14]" = split_with_sizes_56[1];  split_with_sizes_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_82: "f32[8, 52, 14, 14]" = torch.ops.aten.convolution.default(getitem_160, arg218_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_160 = arg218_1 = None
    convolution_83: "f32[8, 52, 14, 14]" = torch.ops.aten.convolution.default(getitem_161, arg219_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_161 = arg219_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_24: "f32[8, 104, 14, 14]" = torch.ops.aten.cat.default([convolution_82, convolution_83], 1);  convolution_82 = convolution_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_64: "f32[104]" = torch.ops.prims.convert_element_type.default(arg369_1, torch.float32);  arg369_1 = None
    convert_element_type_65: "f32[104]" = torch.ops.prims.convert_element_type.default(arg370_1, torch.float32);  arg370_1 = None
    add_71: "f32[104]" = torch.ops.aten.add.Tensor(convert_element_type_65, 0.001);  convert_element_type_65 = None
    sqrt_32: "f32[104]" = torch.ops.aten.sqrt.default(add_71);  add_71 = None
    reciprocal_32: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_32);  sqrt_32 = None
    mul_128: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_32, 1);  reciprocal_32 = None
    unsqueeze_256: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_64, -1);  convert_element_type_64 = None
    unsqueeze_257: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, -1);  unsqueeze_256 = None
    unsqueeze_258: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_128, -1);  mul_128 = None
    unsqueeze_259: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, -1);  unsqueeze_258 = None
    sub_32: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(cat_24, unsqueeze_257);  cat_24 = unsqueeze_257 = None
    mul_129: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_32, unsqueeze_259);  sub_32 = unsqueeze_259 = None
    unsqueeze_260: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg75_1, -1);  arg75_1 = None
    unsqueeze_261: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, -1);  unsqueeze_260 = None
    mul_130: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_129, unsqueeze_261);  mul_129 = unsqueeze_261 = None
    unsqueeze_262: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg76_1, -1);  arg76_1 = None
    unsqueeze_263: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, -1);  unsqueeze_262 = None
    add_72: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_130, unsqueeze_263);  mul_130 = unsqueeze_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_73: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(add_72, add_66);  add_72 = add_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_84: "f32[8, 624, 14, 14]" = torch.ops.aten.convolution.default(add_73, arg220_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_73 = arg220_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_66: "f32[624]" = torch.ops.prims.convert_element_type.default(arg371_1, torch.float32);  arg371_1 = None
    convert_element_type_67: "f32[624]" = torch.ops.prims.convert_element_type.default(arg372_1, torch.float32);  arg372_1 = None
    add_74: "f32[624]" = torch.ops.aten.add.Tensor(convert_element_type_67, 0.001);  convert_element_type_67 = None
    sqrt_33: "f32[624]" = torch.ops.aten.sqrt.default(add_74);  add_74 = None
    reciprocal_33: "f32[624]" = torch.ops.aten.reciprocal.default(sqrt_33);  sqrt_33 = None
    mul_131: "f32[624]" = torch.ops.aten.mul.Tensor(reciprocal_33, 1);  reciprocal_33 = None
    unsqueeze_264: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_66, -1);  convert_element_type_66 = None
    unsqueeze_265: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, -1);  unsqueeze_264 = None
    unsqueeze_266: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(mul_131, -1);  mul_131 = None
    unsqueeze_267: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, -1);  unsqueeze_266 = None
    sub_33: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_84, unsqueeze_265);  convolution_84 = unsqueeze_265 = None
    mul_132: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sub_33, unsqueeze_267);  sub_33 = unsqueeze_267 = None
    unsqueeze_268: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(arg77_1, -1);  arg77_1 = None
    unsqueeze_269: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, -1);  unsqueeze_268 = None
    mul_133: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_132, unsqueeze_269);  mul_132 = unsqueeze_269 = None
    unsqueeze_270: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(arg78_1, -1);  arg78_1 = None
    unsqueeze_271: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, -1);  unsqueeze_270 = None
    add_75: "f32[8, 624, 14, 14]" = torch.ops.aten.add.Tensor(mul_133, unsqueeze_271);  mul_133 = unsqueeze_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_32: "f32[8, 624, 14, 14]" = torch.ops.aten.sigmoid.default(add_75)
    mul_134: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(add_75, sigmoid_32);  add_75 = sigmoid_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_85: "f32[8, 624, 14, 14]" = torch.ops.aten.convolution.default(mul_134, arg221_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 624);  mul_134 = arg221_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_68: "f32[624]" = torch.ops.prims.convert_element_type.default(arg373_1, torch.float32);  arg373_1 = None
    convert_element_type_69: "f32[624]" = torch.ops.prims.convert_element_type.default(arg374_1, torch.float32);  arg374_1 = None
    add_76: "f32[624]" = torch.ops.aten.add.Tensor(convert_element_type_69, 0.001);  convert_element_type_69 = None
    sqrt_34: "f32[624]" = torch.ops.aten.sqrt.default(add_76);  add_76 = None
    reciprocal_34: "f32[624]" = torch.ops.aten.reciprocal.default(sqrt_34);  sqrt_34 = None
    mul_135: "f32[624]" = torch.ops.aten.mul.Tensor(reciprocal_34, 1);  reciprocal_34 = None
    unsqueeze_272: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_68, -1);  convert_element_type_68 = None
    unsqueeze_273: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, -1);  unsqueeze_272 = None
    unsqueeze_274: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(mul_135, -1);  mul_135 = None
    unsqueeze_275: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, -1);  unsqueeze_274 = None
    sub_34: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_85, unsqueeze_273);  convolution_85 = unsqueeze_273 = None
    mul_136: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sub_34, unsqueeze_275);  sub_34 = unsqueeze_275 = None
    unsqueeze_276: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(arg79_1, -1);  arg79_1 = None
    unsqueeze_277: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, -1);  unsqueeze_276 = None
    mul_137: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_136, unsqueeze_277);  mul_136 = unsqueeze_277 = None
    unsqueeze_278: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(arg80_1, -1);  arg80_1 = None
    unsqueeze_279: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, -1);  unsqueeze_278 = None
    add_77: "f32[8, 624, 14, 14]" = torch.ops.aten.add.Tensor(mul_137, unsqueeze_279);  mul_137 = unsqueeze_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_33: "f32[8, 624, 14, 14]" = torch.ops.aten.sigmoid.default(add_77)
    mul_138: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(add_77, sigmoid_33);  add_77 = sigmoid_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_8: "f32[8, 624, 1, 1]" = torch.ops.aten.mean.dim(mul_138, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_86: "f32[8, 52, 1, 1]" = torch.ops.aten.convolution.default(mean_8, arg222_1, arg223_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_8 = arg222_1 = arg223_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_34: "f32[8, 52, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_86)
    mul_139: "f32[8, 52, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_86, sigmoid_34);  convolution_86 = sigmoid_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_87: "f32[8, 624, 1, 1]" = torch.ops.aten.convolution.default(mul_139, arg224_1, arg225_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_139 = arg224_1 = arg225_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_35: "f32[8, 624, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_87);  convolution_87 = None
    mul_140: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_138, sigmoid_35);  mul_138 = sigmoid_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_88: "f32[8, 160, 14, 14]" = torch.ops.aten.convolution.default(mul_140, arg226_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_140 = arg226_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_70: "f32[160]" = torch.ops.prims.convert_element_type.default(arg375_1, torch.float32);  arg375_1 = None
    convert_element_type_71: "f32[160]" = torch.ops.prims.convert_element_type.default(arg376_1, torch.float32);  arg376_1 = None
    add_78: "f32[160]" = torch.ops.aten.add.Tensor(convert_element_type_71, 0.001);  convert_element_type_71 = None
    sqrt_35: "f32[160]" = torch.ops.aten.sqrt.default(add_78);  add_78 = None
    reciprocal_35: "f32[160]" = torch.ops.aten.reciprocal.default(sqrt_35);  sqrt_35 = None
    mul_141: "f32[160]" = torch.ops.aten.mul.Tensor(reciprocal_35, 1);  reciprocal_35 = None
    unsqueeze_280: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_70, -1);  convert_element_type_70 = None
    unsqueeze_281: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, -1);  unsqueeze_280 = None
    unsqueeze_282: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(mul_141, -1);  mul_141 = None
    unsqueeze_283: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, -1);  unsqueeze_282 = None
    sub_35: "f32[8, 160, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_88, unsqueeze_281);  convolution_88 = unsqueeze_281 = None
    mul_142: "f32[8, 160, 14, 14]" = torch.ops.aten.mul.Tensor(sub_35, unsqueeze_283);  sub_35 = unsqueeze_283 = None
    unsqueeze_284: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg81_1, -1);  arg81_1 = None
    unsqueeze_285: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, -1);  unsqueeze_284 = None
    mul_143: "f32[8, 160, 14, 14]" = torch.ops.aten.mul.Tensor(mul_142, unsqueeze_285);  mul_142 = unsqueeze_285 = None
    unsqueeze_286: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg82_1, -1);  arg82_1 = None
    unsqueeze_287: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, -1);  unsqueeze_286 = None
    add_79: "f32[8, 160, 14, 14]" = torch.ops.aten.add.Tensor(mul_143, unsqueeze_287);  mul_143 = unsqueeze_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_57 = torch.ops.aten.split_with_sizes.default(add_79, [80, 80], 1)
    getitem_162: "f32[8, 80, 14, 14]" = split_with_sizes_57[0]
    getitem_163: "f32[8, 80, 14, 14]" = split_with_sizes_57[1];  split_with_sizes_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_89: "f32[8, 240, 14, 14]" = torch.ops.aten.convolution.default(getitem_162, arg227_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_162 = arg227_1 = None
    convolution_90: "f32[8, 240, 14, 14]" = torch.ops.aten.convolution.default(getitem_163, arg228_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_163 = arg228_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_25: "f32[8, 480, 14, 14]" = torch.ops.aten.cat.default([convolution_89, convolution_90], 1);  convolution_89 = convolution_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_72: "f32[480]" = torch.ops.prims.convert_element_type.default(arg377_1, torch.float32);  arg377_1 = None
    convert_element_type_73: "f32[480]" = torch.ops.prims.convert_element_type.default(arg378_1, torch.float32);  arg378_1 = None
    add_80: "f32[480]" = torch.ops.aten.add.Tensor(convert_element_type_73, 0.001);  convert_element_type_73 = None
    sqrt_36: "f32[480]" = torch.ops.aten.sqrt.default(add_80);  add_80 = None
    reciprocal_36: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_36);  sqrt_36 = None
    mul_144: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_36, 1);  reciprocal_36 = None
    unsqueeze_288: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_72, -1);  convert_element_type_72 = None
    unsqueeze_289: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, -1);  unsqueeze_288 = None
    unsqueeze_290: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_144, -1);  mul_144 = None
    unsqueeze_291: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, -1);  unsqueeze_290 = None
    sub_36: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(cat_25, unsqueeze_289);  cat_25 = unsqueeze_289 = None
    mul_145: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_36, unsqueeze_291);  sub_36 = unsqueeze_291 = None
    unsqueeze_292: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg83_1, -1);  arg83_1 = None
    unsqueeze_293: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, -1);  unsqueeze_292 = None
    mul_146: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_145, unsqueeze_293);  mul_145 = unsqueeze_293 = None
    unsqueeze_294: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg84_1, -1);  arg84_1 = None
    unsqueeze_295: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, -1);  unsqueeze_294 = None
    add_81: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_146, unsqueeze_295);  mul_146 = unsqueeze_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_36: "f32[8, 480, 14, 14]" = torch.ops.aten.sigmoid.default(add_81)
    mul_147: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_81, sigmoid_36);  add_81 = sigmoid_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    split_with_sizes_59 = torch.ops.aten.split_with_sizes.default(mul_147, [120, 120, 120, 120], 1)
    getitem_168: "f32[8, 120, 14, 14]" = split_with_sizes_59[0];  split_with_sizes_59 = None
    convolution_91: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(getitem_168, arg229_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 120);  getitem_168 = arg229_1 = None
    split_with_sizes_60 = torch.ops.aten.split_with_sizes.default(mul_147, [120, 120, 120, 120], 1)
    getitem_173: "f32[8, 120, 14, 14]" = split_with_sizes_60[1];  split_with_sizes_60 = None
    convolution_92: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(getitem_173, arg230_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 120);  getitem_173 = arg230_1 = None
    split_with_sizes_61 = torch.ops.aten.split_with_sizes.default(mul_147, [120, 120, 120, 120], 1)
    getitem_178: "f32[8, 120, 14, 14]" = split_with_sizes_61[2];  split_with_sizes_61 = None
    convolution_93: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(getitem_178, arg231_1, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 120);  getitem_178 = arg231_1 = None
    split_with_sizes_62 = torch.ops.aten.split_with_sizes.default(mul_147, [120, 120, 120, 120], 1);  mul_147 = None
    getitem_183: "f32[8, 120, 14, 14]" = split_with_sizes_62[3];  split_with_sizes_62 = None
    convolution_94: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(getitem_183, arg232_1, None, [1, 1], [4, 4], [1, 1], False, [0, 0], 120);  getitem_183 = arg232_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_26: "f32[8, 480, 14, 14]" = torch.ops.aten.cat.default([convolution_91, convolution_92, convolution_93, convolution_94], 1);  convolution_91 = convolution_92 = convolution_93 = convolution_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_74: "f32[480]" = torch.ops.prims.convert_element_type.default(arg379_1, torch.float32);  arg379_1 = None
    convert_element_type_75: "f32[480]" = torch.ops.prims.convert_element_type.default(arg380_1, torch.float32);  arg380_1 = None
    add_82: "f32[480]" = torch.ops.aten.add.Tensor(convert_element_type_75, 0.001);  convert_element_type_75 = None
    sqrt_37: "f32[480]" = torch.ops.aten.sqrt.default(add_82);  add_82 = None
    reciprocal_37: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_37);  sqrt_37 = None
    mul_148: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_37, 1);  reciprocal_37 = None
    unsqueeze_296: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_74, -1);  convert_element_type_74 = None
    unsqueeze_297: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, -1);  unsqueeze_296 = None
    unsqueeze_298: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_148, -1);  mul_148 = None
    unsqueeze_299: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, -1);  unsqueeze_298 = None
    sub_37: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(cat_26, unsqueeze_297);  cat_26 = unsqueeze_297 = None
    mul_149: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_37, unsqueeze_299);  sub_37 = unsqueeze_299 = None
    unsqueeze_300: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg85_1, -1);  arg85_1 = None
    unsqueeze_301: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, -1);  unsqueeze_300 = None
    mul_150: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_149, unsqueeze_301);  mul_149 = unsqueeze_301 = None
    unsqueeze_302: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg86_1, -1);  arg86_1 = None
    unsqueeze_303: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, -1);  unsqueeze_302 = None
    add_83: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_150, unsqueeze_303);  mul_150 = unsqueeze_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_37: "f32[8, 480, 14, 14]" = torch.ops.aten.sigmoid.default(add_83)
    mul_151: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_83, sigmoid_37);  add_83 = sigmoid_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_9: "f32[8, 480, 1, 1]" = torch.ops.aten.mean.dim(mul_151, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_95: "f32[8, 80, 1, 1]" = torch.ops.aten.convolution.default(mean_9, arg233_1, arg234_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_9 = arg233_1 = arg234_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_38: "f32[8, 80, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_95)
    mul_152: "f32[8, 80, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_95, sigmoid_38);  convolution_95 = sigmoid_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_96: "f32[8, 480, 1, 1]" = torch.ops.aten.convolution.default(mul_152, arg235_1, arg236_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_152 = arg235_1 = arg236_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_39: "f32[8, 480, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_96);  convolution_96 = None
    mul_153: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_151, sigmoid_39);  mul_151 = sigmoid_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_63 = torch.ops.aten.split_with_sizes.default(mul_153, [240, 240], 1);  mul_153 = None
    getitem_184: "f32[8, 240, 14, 14]" = split_with_sizes_63[0]
    getitem_185: "f32[8, 240, 14, 14]" = split_with_sizes_63[1];  split_with_sizes_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_97: "f32[8, 80, 14, 14]" = torch.ops.aten.convolution.default(getitem_184, arg237_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_184 = arg237_1 = None
    convolution_98: "f32[8, 80, 14, 14]" = torch.ops.aten.convolution.default(getitem_185, arg238_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_185 = arg238_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_27: "f32[8, 160, 14, 14]" = torch.ops.aten.cat.default([convolution_97, convolution_98], 1);  convolution_97 = convolution_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_76: "f32[160]" = torch.ops.prims.convert_element_type.default(arg381_1, torch.float32);  arg381_1 = None
    convert_element_type_77: "f32[160]" = torch.ops.prims.convert_element_type.default(arg382_1, torch.float32);  arg382_1 = None
    add_84: "f32[160]" = torch.ops.aten.add.Tensor(convert_element_type_77, 0.001);  convert_element_type_77 = None
    sqrt_38: "f32[160]" = torch.ops.aten.sqrt.default(add_84);  add_84 = None
    reciprocal_38: "f32[160]" = torch.ops.aten.reciprocal.default(sqrt_38);  sqrt_38 = None
    mul_154: "f32[160]" = torch.ops.aten.mul.Tensor(reciprocal_38, 1);  reciprocal_38 = None
    unsqueeze_304: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_76, -1);  convert_element_type_76 = None
    unsqueeze_305: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, -1);  unsqueeze_304 = None
    unsqueeze_306: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(mul_154, -1);  mul_154 = None
    unsqueeze_307: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, -1);  unsqueeze_306 = None
    sub_38: "f32[8, 160, 14, 14]" = torch.ops.aten.sub.Tensor(cat_27, unsqueeze_305);  cat_27 = unsqueeze_305 = None
    mul_155: "f32[8, 160, 14, 14]" = torch.ops.aten.mul.Tensor(sub_38, unsqueeze_307);  sub_38 = unsqueeze_307 = None
    unsqueeze_308: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg87_1, -1);  arg87_1 = None
    unsqueeze_309: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, -1);  unsqueeze_308 = None
    mul_156: "f32[8, 160, 14, 14]" = torch.ops.aten.mul.Tensor(mul_155, unsqueeze_309);  mul_155 = unsqueeze_309 = None
    unsqueeze_310: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg88_1, -1);  arg88_1 = None
    unsqueeze_311: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, -1);  unsqueeze_310 = None
    add_85: "f32[8, 160, 14, 14]" = torch.ops.aten.add.Tensor(mul_156, unsqueeze_311);  mul_156 = unsqueeze_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_86: "f32[8, 160, 14, 14]" = torch.ops.aten.add.Tensor(add_85, add_79);  add_85 = add_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_64 = torch.ops.aten.split_with_sizes.default(add_86, [80, 80], 1)
    getitem_186: "f32[8, 80, 14, 14]" = split_with_sizes_64[0]
    getitem_187: "f32[8, 80, 14, 14]" = split_with_sizes_64[1];  split_with_sizes_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_99: "f32[8, 240, 14, 14]" = torch.ops.aten.convolution.default(getitem_186, arg239_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_186 = arg239_1 = None
    convolution_100: "f32[8, 240, 14, 14]" = torch.ops.aten.convolution.default(getitem_187, arg240_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_187 = arg240_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_28: "f32[8, 480, 14, 14]" = torch.ops.aten.cat.default([convolution_99, convolution_100], 1);  convolution_99 = convolution_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_78: "f32[480]" = torch.ops.prims.convert_element_type.default(arg383_1, torch.float32);  arg383_1 = None
    convert_element_type_79: "f32[480]" = torch.ops.prims.convert_element_type.default(arg384_1, torch.float32);  arg384_1 = None
    add_87: "f32[480]" = torch.ops.aten.add.Tensor(convert_element_type_79, 0.001);  convert_element_type_79 = None
    sqrt_39: "f32[480]" = torch.ops.aten.sqrt.default(add_87);  add_87 = None
    reciprocal_39: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_39);  sqrt_39 = None
    mul_157: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_39, 1);  reciprocal_39 = None
    unsqueeze_312: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_78, -1);  convert_element_type_78 = None
    unsqueeze_313: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, -1);  unsqueeze_312 = None
    unsqueeze_314: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_157, -1);  mul_157 = None
    unsqueeze_315: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, -1);  unsqueeze_314 = None
    sub_39: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(cat_28, unsqueeze_313);  cat_28 = unsqueeze_313 = None
    mul_158: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_39, unsqueeze_315);  sub_39 = unsqueeze_315 = None
    unsqueeze_316: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg89_1, -1);  arg89_1 = None
    unsqueeze_317: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, -1);  unsqueeze_316 = None
    mul_159: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_158, unsqueeze_317);  mul_158 = unsqueeze_317 = None
    unsqueeze_318: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg90_1, -1);  arg90_1 = None
    unsqueeze_319: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, -1);  unsqueeze_318 = None
    add_88: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_159, unsqueeze_319);  mul_159 = unsqueeze_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_40: "f32[8, 480, 14, 14]" = torch.ops.aten.sigmoid.default(add_88)
    mul_160: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_88, sigmoid_40);  add_88 = sigmoid_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    split_with_sizes_66 = torch.ops.aten.split_with_sizes.default(mul_160, [120, 120, 120, 120], 1)
    getitem_192: "f32[8, 120, 14, 14]" = split_with_sizes_66[0];  split_with_sizes_66 = None
    convolution_101: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(getitem_192, arg241_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 120);  getitem_192 = arg241_1 = None
    split_with_sizes_67 = torch.ops.aten.split_with_sizes.default(mul_160, [120, 120, 120, 120], 1)
    getitem_197: "f32[8, 120, 14, 14]" = split_with_sizes_67[1];  split_with_sizes_67 = None
    convolution_102: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(getitem_197, arg242_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 120);  getitem_197 = arg242_1 = None
    split_with_sizes_68 = torch.ops.aten.split_with_sizes.default(mul_160, [120, 120, 120, 120], 1)
    getitem_202: "f32[8, 120, 14, 14]" = split_with_sizes_68[2];  split_with_sizes_68 = None
    convolution_103: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(getitem_202, arg243_1, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 120);  getitem_202 = arg243_1 = None
    split_with_sizes_69 = torch.ops.aten.split_with_sizes.default(mul_160, [120, 120, 120, 120], 1);  mul_160 = None
    getitem_207: "f32[8, 120, 14, 14]" = split_with_sizes_69[3];  split_with_sizes_69 = None
    convolution_104: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(getitem_207, arg244_1, None, [1, 1], [4, 4], [1, 1], False, [0, 0], 120);  getitem_207 = arg244_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_29: "f32[8, 480, 14, 14]" = torch.ops.aten.cat.default([convolution_101, convolution_102, convolution_103, convolution_104], 1);  convolution_101 = convolution_102 = convolution_103 = convolution_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_80: "f32[480]" = torch.ops.prims.convert_element_type.default(arg385_1, torch.float32);  arg385_1 = None
    convert_element_type_81: "f32[480]" = torch.ops.prims.convert_element_type.default(arg386_1, torch.float32);  arg386_1 = None
    add_89: "f32[480]" = torch.ops.aten.add.Tensor(convert_element_type_81, 0.001);  convert_element_type_81 = None
    sqrt_40: "f32[480]" = torch.ops.aten.sqrt.default(add_89);  add_89 = None
    reciprocal_40: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_40);  sqrt_40 = None
    mul_161: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_40, 1);  reciprocal_40 = None
    unsqueeze_320: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_80, -1);  convert_element_type_80 = None
    unsqueeze_321: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, -1);  unsqueeze_320 = None
    unsqueeze_322: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_161, -1);  mul_161 = None
    unsqueeze_323: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, -1);  unsqueeze_322 = None
    sub_40: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(cat_29, unsqueeze_321);  cat_29 = unsqueeze_321 = None
    mul_162: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_40, unsqueeze_323);  sub_40 = unsqueeze_323 = None
    unsqueeze_324: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg91_1, -1);  arg91_1 = None
    unsqueeze_325: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, -1);  unsqueeze_324 = None
    mul_163: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_162, unsqueeze_325);  mul_162 = unsqueeze_325 = None
    unsqueeze_326: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg92_1, -1);  arg92_1 = None
    unsqueeze_327: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, -1);  unsqueeze_326 = None
    add_90: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_163, unsqueeze_327);  mul_163 = unsqueeze_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_41: "f32[8, 480, 14, 14]" = torch.ops.aten.sigmoid.default(add_90)
    mul_164: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_90, sigmoid_41);  add_90 = sigmoid_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_10: "f32[8, 480, 1, 1]" = torch.ops.aten.mean.dim(mul_164, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_105: "f32[8, 80, 1, 1]" = torch.ops.aten.convolution.default(mean_10, arg245_1, arg246_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_10 = arg245_1 = arg246_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_42: "f32[8, 80, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_105)
    mul_165: "f32[8, 80, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_105, sigmoid_42);  convolution_105 = sigmoid_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_106: "f32[8, 480, 1, 1]" = torch.ops.aten.convolution.default(mul_165, arg247_1, arg248_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_165 = arg247_1 = arg248_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_43: "f32[8, 480, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_106);  convolution_106 = None
    mul_166: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_164, sigmoid_43);  mul_164 = sigmoid_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_70 = torch.ops.aten.split_with_sizes.default(mul_166, [240, 240], 1);  mul_166 = None
    getitem_208: "f32[8, 240, 14, 14]" = split_with_sizes_70[0]
    getitem_209: "f32[8, 240, 14, 14]" = split_with_sizes_70[1];  split_with_sizes_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_107: "f32[8, 80, 14, 14]" = torch.ops.aten.convolution.default(getitem_208, arg249_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_208 = arg249_1 = None
    convolution_108: "f32[8, 80, 14, 14]" = torch.ops.aten.convolution.default(getitem_209, arg250_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_209 = arg250_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_30: "f32[8, 160, 14, 14]" = torch.ops.aten.cat.default([convolution_107, convolution_108], 1);  convolution_107 = convolution_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_82: "f32[160]" = torch.ops.prims.convert_element_type.default(arg387_1, torch.float32);  arg387_1 = None
    convert_element_type_83: "f32[160]" = torch.ops.prims.convert_element_type.default(arg388_1, torch.float32);  arg388_1 = None
    add_91: "f32[160]" = torch.ops.aten.add.Tensor(convert_element_type_83, 0.001);  convert_element_type_83 = None
    sqrt_41: "f32[160]" = torch.ops.aten.sqrt.default(add_91);  add_91 = None
    reciprocal_41: "f32[160]" = torch.ops.aten.reciprocal.default(sqrt_41);  sqrt_41 = None
    mul_167: "f32[160]" = torch.ops.aten.mul.Tensor(reciprocal_41, 1);  reciprocal_41 = None
    unsqueeze_328: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_82, -1);  convert_element_type_82 = None
    unsqueeze_329: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, -1);  unsqueeze_328 = None
    unsqueeze_330: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(mul_167, -1);  mul_167 = None
    unsqueeze_331: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, -1);  unsqueeze_330 = None
    sub_41: "f32[8, 160, 14, 14]" = torch.ops.aten.sub.Tensor(cat_30, unsqueeze_329);  cat_30 = unsqueeze_329 = None
    mul_168: "f32[8, 160, 14, 14]" = torch.ops.aten.mul.Tensor(sub_41, unsqueeze_331);  sub_41 = unsqueeze_331 = None
    unsqueeze_332: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg93_1, -1);  arg93_1 = None
    unsqueeze_333: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, -1);  unsqueeze_332 = None
    mul_169: "f32[8, 160, 14, 14]" = torch.ops.aten.mul.Tensor(mul_168, unsqueeze_333);  mul_168 = unsqueeze_333 = None
    unsqueeze_334: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg94_1, -1);  arg94_1 = None
    unsqueeze_335: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, -1);  unsqueeze_334 = None
    add_92: "f32[8, 160, 14, 14]" = torch.ops.aten.add.Tensor(mul_169, unsqueeze_335);  mul_169 = unsqueeze_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_93: "f32[8, 160, 14, 14]" = torch.ops.aten.add.Tensor(add_92, add_86);  add_92 = add_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_71 = torch.ops.aten.split_with_sizes.default(add_93, [80, 80], 1)
    getitem_210: "f32[8, 80, 14, 14]" = split_with_sizes_71[0]
    getitem_211: "f32[8, 80, 14, 14]" = split_with_sizes_71[1];  split_with_sizes_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_109: "f32[8, 240, 14, 14]" = torch.ops.aten.convolution.default(getitem_210, arg251_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_210 = arg251_1 = None
    convolution_110: "f32[8, 240, 14, 14]" = torch.ops.aten.convolution.default(getitem_211, arg252_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_211 = arg252_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_31: "f32[8, 480, 14, 14]" = torch.ops.aten.cat.default([convolution_109, convolution_110], 1);  convolution_109 = convolution_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_84: "f32[480]" = torch.ops.prims.convert_element_type.default(arg389_1, torch.float32);  arg389_1 = None
    convert_element_type_85: "f32[480]" = torch.ops.prims.convert_element_type.default(arg390_1, torch.float32);  arg390_1 = None
    add_94: "f32[480]" = torch.ops.aten.add.Tensor(convert_element_type_85, 0.001);  convert_element_type_85 = None
    sqrt_42: "f32[480]" = torch.ops.aten.sqrt.default(add_94);  add_94 = None
    reciprocal_42: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_42);  sqrt_42 = None
    mul_170: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_42, 1);  reciprocal_42 = None
    unsqueeze_336: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_84, -1);  convert_element_type_84 = None
    unsqueeze_337: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, -1);  unsqueeze_336 = None
    unsqueeze_338: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_170, -1);  mul_170 = None
    unsqueeze_339: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, -1);  unsqueeze_338 = None
    sub_42: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(cat_31, unsqueeze_337);  cat_31 = unsqueeze_337 = None
    mul_171: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_42, unsqueeze_339);  sub_42 = unsqueeze_339 = None
    unsqueeze_340: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg95_1, -1);  arg95_1 = None
    unsqueeze_341: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, -1);  unsqueeze_340 = None
    mul_172: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_171, unsqueeze_341);  mul_171 = unsqueeze_341 = None
    unsqueeze_342: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg96_1, -1);  arg96_1 = None
    unsqueeze_343: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, -1);  unsqueeze_342 = None
    add_95: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_172, unsqueeze_343);  mul_172 = unsqueeze_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_44: "f32[8, 480, 14, 14]" = torch.ops.aten.sigmoid.default(add_95)
    mul_173: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_95, sigmoid_44);  add_95 = sigmoid_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    split_with_sizes_73 = torch.ops.aten.split_with_sizes.default(mul_173, [120, 120, 120, 120], 1)
    getitem_216: "f32[8, 120, 14, 14]" = split_with_sizes_73[0];  split_with_sizes_73 = None
    convolution_111: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(getitem_216, arg253_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 120);  getitem_216 = arg253_1 = None
    split_with_sizes_74 = torch.ops.aten.split_with_sizes.default(mul_173, [120, 120, 120, 120], 1)
    getitem_221: "f32[8, 120, 14, 14]" = split_with_sizes_74[1];  split_with_sizes_74 = None
    convolution_112: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(getitem_221, arg254_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 120);  getitem_221 = arg254_1 = None
    split_with_sizes_75 = torch.ops.aten.split_with_sizes.default(mul_173, [120, 120, 120, 120], 1)
    getitem_226: "f32[8, 120, 14, 14]" = split_with_sizes_75[2];  split_with_sizes_75 = None
    convolution_113: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(getitem_226, arg255_1, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 120);  getitem_226 = arg255_1 = None
    split_with_sizes_76 = torch.ops.aten.split_with_sizes.default(mul_173, [120, 120, 120, 120], 1);  mul_173 = None
    getitem_231: "f32[8, 120, 14, 14]" = split_with_sizes_76[3];  split_with_sizes_76 = None
    convolution_114: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(getitem_231, arg256_1, None, [1, 1], [4, 4], [1, 1], False, [0, 0], 120);  getitem_231 = arg256_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_32: "f32[8, 480, 14, 14]" = torch.ops.aten.cat.default([convolution_111, convolution_112, convolution_113, convolution_114], 1);  convolution_111 = convolution_112 = convolution_113 = convolution_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_86: "f32[480]" = torch.ops.prims.convert_element_type.default(arg391_1, torch.float32);  arg391_1 = None
    convert_element_type_87: "f32[480]" = torch.ops.prims.convert_element_type.default(arg392_1, torch.float32);  arg392_1 = None
    add_96: "f32[480]" = torch.ops.aten.add.Tensor(convert_element_type_87, 0.001);  convert_element_type_87 = None
    sqrt_43: "f32[480]" = torch.ops.aten.sqrt.default(add_96);  add_96 = None
    reciprocal_43: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_43);  sqrt_43 = None
    mul_174: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_43, 1);  reciprocal_43 = None
    unsqueeze_344: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_86, -1);  convert_element_type_86 = None
    unsqueeze_345: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, -1);  unsqueeze_344 = None
    unsqueeze_346: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_174, -1);  mul_174 = None
    unsqueeze_347: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, -1);  unsqueeze_346 = None
    sub_43: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(cat_32, unsqueeze_345);  cat_32 = unsqueeze_345 = None
    mul_175: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_43, unsqueeze_347);  sub_43 = unsqueeze_347 = None
    unsqueeze_348: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg97_1, -1);  arg97_1 = None
    unsqueeze_349: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, -1);  unsqueeze_348 = None
    mul_176: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_175, unsqueeze_349);  mul_175 = unsqueeze_349 = None
    unsqueeze_350: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg98_1, -1);  arg98_1 = None
    unsqueeze_351: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, -1);  unsqueeze_350 = None
    add_97: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_176, unsqueeze_351);  mul_176 = unsqueeze_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_45: "f32[8, 480, 14, 14]" = torch.ops.aten.sigmoid.default(add_97)
    mul_177: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_97, sigmoid_45);  add_97 = sigmoid_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_11: "f32[8, 480, 1, 1]" = torch.ops.aten.mean.dim(mul_177, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_115: "f32[8, 80, 1, 1]" = torch.ops.aten.convolution.default(mean_11, arg257_1, arg258_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_11 = arg257_1 = arg258_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_46: "f32[8, 80, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_115)
    mul_178: "f32[8, 80, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_115, sigmoid_46);  convolution_115 = sigmoid_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_116: "f32[8, 480, 1, 1]" = torch.ops.aten.convolution.default(mul_178, arg259_1, arg260_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_178 = arg259_1 = arg260_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_47: "f32[8, 480, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_116);  convolution_116 = None
    mul_179: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_177, sigmoid_47);  mul_177 = sigmoid_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_77 = torch.ops.aten.split_with_sizes.default(mul_179, [240, 240], 1);  mul_179 = None
    getitem_232: "f32[8, 240, 14, 14]" = split_with_sizes_77[0]
    getitem_233: "f32[8, 240, 14, 14]" = split_with_sizes_77[1];  split_with_sizes_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_117: "f32[8, 80, 14, 14]" = torch.ops.aten.convolution.default(getitem_232, arg261_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_232 = arg261_1 = None
    convolution_118: "f32[8, 80, 14, 14]" = torch.ops.aten.convolution.default(getitem_233, arg262_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_233 = arg262_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_33: "f32[8, 160, 14, 14]" = torch.ops.aten.cat.default([convolution_117, convolution_118], 1);  convolution_117 = convolution_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_88: "f32[160]" = torch.ops.prims.convert_element_type.default(arg393_1, torch.float32);  arg393_1 = None
    convert_element_type_89: "f32[160]" = torch.ops.prims.convert_element_type.default(arg394_1, torch.float32);  arg394_1 = None
    add_98: "f32[160]" = torch.ops.aten.add.Tensor(convert_element_type_89, 0.001);  convert_element_type_89 = None
    sqrt_44: "f32[160]" = torch.ops.aten.sqrt.default(add_98);  add_98 = None
    reciprocal_44: "f32[160]" = torch.ops.aten.reciprocal.default(sqrt_44);  sqrt_44 = None
    mul_180: "f32[160]" = torch.ops.aten.mul.Tensor(reciprocal_44, 1);  reciprocal_44 = None
    unsqueeze_352: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_88, -1);  convert_element_type_88 = None
    unsqueeze_353: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, -1);  unsqueeze_352 = None
    unsqueeze_354: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(mul_180, -1);  mul_180 = None
    unsqueeze_355: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, -1);  unsqueeze_354 = None
    sub_44: "f32[8, 160, 14, 14]" = torch.ops.aten.sub.Tensor(cat_33, unsqueeze_353);  cat_33 = unsqueeze_353 = None
    mul_181: "f32[8, 160, 14, 14]" = torch.ops.aten.mul.Tensor(sub_44, unsqueeze_355);  sub_44 = unsqueeze_355 = None
    unsqueeze_356: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg99_1, -1);  arg99_1 = None
    unsqueeze_357: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, -1);  unsqueeze_356 = None
    mul_182: "f32[8, 160, 14, 14]" = torch.ops.aten.mul.Tensor(mul_181, unsqueeze_357);  mul_181 = unsqueeze_357 = None
    unsqueeze_358: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg100_1, -1);  arg100_1 = None
    unsqueeze_359: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, -1);  unsqueeze_358 = None
    add_99: "f32[8, 160, 14, 14]" = torch.ops.aten.add.Tensor(mul_182, unsqueeze_359);  mul_182 = unsqueeze_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_100: "f32[8, 160, 14, 14]" = torch.ops.aten.add.Tensor(add_99, add_93);  add_99 = add_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_119: "f32[8, 960, 14, 14]" = torch.ops.aten.convolution.default(add_100, arg263_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_100 = arg263_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_90: "f32[960]" = torch.ops.prims.convert_element_type.default(arg395_1, torch.float32);  arg395_1 = None
    convert_element_type_91: "f32[960]" = torch.ops.prims.convert_element_type.default(arg396_1, torch.float32);  arg396_1 = None
    add_101: "f32[960]" = torch.ops.aten.add.Tensor(convert_element_type_91, 0.001);  convert_element_type_91 = None
    sqrt_45: "f32[960]" = torch.ops.aten.sqrt.default(add_101);  add_101 = None
    reciprocal_45: "f32[960]" = torch.ops.aten.reciprocal.default(sqrt_45);  sqrt_45 = None
    mul_183: "f32[960]" = torch.ops.aten.mul.Tensor(reciprocal_45, 1);  reciprocal_45 = None
    unsqueeze_360: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_90, -1);  convert_element_type_90 = None
    unsqueeze_361: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, -1);  unsqueeze_360 = None
    unsqueeze_362: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(mul_183, -1);  mul_183 = None
    unsqueeze_363: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, -1);  unsqueeze_362 = None
    sub_45: "f32[8, 960, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_119, unsqueeze_361);  convolution_119 = unsqueeze_361 = None
    mul_184: "f32[8, 960, 14, 14]" = torch.ops.aten.mul.Tensor(sub_45, unsqueeze_363);  sub_45 = unsqueeze_363 = None
    unsqueeze_364: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(arg101_1, -1);  arg101_1 = None
    unsqueeze_365: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, -1);  unsqueeze_364 = None
    mul_185: "f32[8, 960, 14, 14]" = torch.ops.aten.mul.Tensor(mul_184, unsqueeze_365);  mul_184 = unsqueeze_365 = None
    unsqueeze_366: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(arg102_1, -1);  arg102_1 = None
    unsqueeze_367: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, -1);  unsqueeze_366 = None
    add_102: "f32[8, 960, 14, 14]" = torch.ops.aten.add.Tensor(mul_185, unsqueeze_367);  mul_185 = unsqueeze_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_48: "f32[8, 960, 14, 14]" = torch.ops.aten.sigmoid.default(add_102)
    mul_186: "f32[8, 960, 14, 14]" = torch.ops.aten.mul.Tensor(add_102, sigmoid_48);  add_102 = sigmoid_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    split_with_sizes_79 = torch.ops.aten.split_with_sizes.default(mul_186, [240, 240, 240, 240], 1)
    getitem_238: "f32[8, 240, 14, 14]" = split_with_sizes_79[0];  split_with_sizes_79 = None
    constant_pad_nd_11: "f32[8, 240, 15, 15]" = torch.ops.aten.constant_pad_nd.default(getitem_238, [0, 1, 0, 1], 0.0);  getitem_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_120: "f32[8, 240, 7, 7]" = torch.ops.aten.convolution.default(constant_pad_nd_11, arg103_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 240);  constant_pad_nd_11 = arg103_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    split_with_sizes_80 = torch.ops.aten.split_with_sizes.default(mul_186, [240, 240, 240, 240], 1)
    getitem_243: "f32[8, 240, 14, 14]" = split_with_sizes_80[1];  split_with_sizes_80 = None
    constant_pad_nd_12: "f32[8, 240, 17, 17]" = torch.ops.aten.constant_pad_nd.default(getitem_243, [1, 2, 1, 2], 0.0);  getitem_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_121: "f32[8, 240, 7, 7]" = torch.ops.aten.convolution.default(constant_pad_nd_12, arg104_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 240);  constant_pad_nd_12 = arg104_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    split_with_sizes_81 = torch.ops.aten.split_with_sizes.default(mul_186, [240, 240, 240, 240], 1)
    getitem_248: "f32[8, 240, 14, 14]" = split_with_sizes_81[2];  split_with_sizes_81 = None
    constant_pad_nd_13: "f32[8, 240, 19, 19]" = torch.ops.aten.constant_pad_nd.default(getitem_248, [2, 3, 2, 3], 0.0);  getitem_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_122: "f32[8, 240, 7, 7]" = torch.ops.aten.convolution.default(constant_pad_nd_13, arg105_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 240);  constant_pad_nd_13 = arg105_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    split_with_sizes_82 = torch.ops.aten.split_with_sizes.default(mul_186, [240, 240, 240, 240], 1);  mul_186 = None
    getitem_253: "f32[8, 240, 14, 14]" = split_with_sizes_82[3];  split_with_sizes_82 = None
    constant_pad_nd_14: "f32[8, 240, 21, 21]" = torch.ops.aten.constant_pad_nd.default(getitem_253, [3, 4, 3, 4], 0.0);  getitem_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_123: "f32[8, 240, 7, 7]" = torch.ops.aten.convolution.default(constant_pad_nd_14, arg106_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 240);  constant_pad_nd_14 = arg106_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_34: "f32[8, 960, 7, 7]" = torch.ops.aten.cat.default([convolution_120, convolution_121, convolution_122, convolution_123], 1);  convolution_120 = convolution_121 = convolution_122 = convolution_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_92: "f32[960]" = torch.ops.prims.convert_element_type.default(arg397_1, torch.float32);  arg397_1 = None
    convert_element_type_93: "f32[960]" = torch.ops.prims.convert_element_type.default(arg398_1, torch.float32);  arg398_1 = None
    add_103: "f32[960]" = torch.ops.aten.add.Tensor(convert_element_type_93, 0.001);  convert_element_type_93 = None
    sqrt_46: "f32[960]" = torch.ops.aten.sqrt.default(add_103);  add_103 = None
    reciprocal_46: "f32[960]" = torch.ops.aten.reciprocal.default(sqrt_46);  sqrt_46 = None
    mul_187: "f32[960]" = torch.ops.aten.mul.Tensor(reciprocal_46, 1);  reciprocal_46 = None
    unsqueeze_368: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_92, -1);  convert_element_type_92 = None
    unsqueeze_369: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, -1);  unsqueeze_368 = None
    unsqueeze_370: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(mul_187, -1);  mul_187 = None
    unsqueeze_371: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, -1);  unsqueeze_370 = None
    sub_46: "f32[8, 960, 7, 7]" = torch.ops.aten.sub.Tensor(cat_34, unsqueeze_369);  cat_34 = unsqueeze_369 = None
    mul_188: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_46, unsqueeze_371);  sub_46 = unsqueeze_371 = None
    unsqueeze_372: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(arg107_1, -1);  arg107_1 = None
    unsqueeze_373: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_372, -1);  unsqueeze_372 = None
    mul_189: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(mul_188, unsqueeze_373);  mul_188 = unsqueeze_373 = None
    unsqueeze_374: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(arg108_1, -1);  arg108_1 = None
    unsqueeze_375: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, -1);  unsqueeze_374 = None
    add_104: "f32[8, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_189, unsqueeze_375);  mul_189 = unsqueeze_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_49: "f32[8, 960, 7, 7]" = torch.ops.aten.sigmoid.default(add_104)
    mul_190: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(add_104, sigmoid_49);  add_104 = sigmoid_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_12: "f32[8, 960, 1, 1]" = torch.ops.aten.mean.dim(mul_190, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_124: "f32[8, 80, 1, 1]" = torch.ops.aten.convolution.default(mean_12, arg264_1, arg265_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_12 = arg264_1 = arg265_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_50: "f32[8, 80, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_124)
    mul_191: "f32[8, 80, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_124, sigmoid_50);  convolution_124 = sigmoid_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_125: "f32[8, 960, 1, 1]" = torch.ops.aten.convolution.default(mul_191, arg266_1, arg267_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_191 = arg266_1 = arg267_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_51: "f32[8, 960, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_125);  convolution_125 = None
    mul_192: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(mul_190, sigmoid_51);  mul_190 = sigmoid_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_126: "f32[8, 264, 7, 7]" = torch.ops.aten.convolution.default(mul_192, arg268_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_192 = arg268_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_94: "f32[264]" = torch.ops.prims.convert_element_type.default(arg399_1, torch.float32);  arg399_1 = None
    convert_element_type_95: "f32[264]" = torch.ops.prims.convert_element_type.default(arg400_1, torch.float32);  arg400_1 = None
    add_105: "f32[264]" = torch.ops.aten.add.Tensor(convert_element_type_95, 0.001);  convert_element_type_95 = None
    sqrt_47: "f32[264]" = torch.ops.aten.sqrt.default(add_105);  add_105 = None
    reciprocal_47: "f32[264]" = torch.ops.aten.reciprocal.default(sqrt_47);  sqrt_47 = None
    mul_193: "f32[264]" = torch.ops.aten.mul.Tensor(reciprocal_47, 1);  reciprocal_47 = None
    unsqueeze_376: "f32[264, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_94, -1);  convert_element_type_94 = None
    unsqueeze_377: "f32[264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, -1);  unsqueeze_376 = None
    unsqueeze_378: "f32[264, 1]" = torch.ops.aten.unsqueeze.default(mul_193, -1);  mul_193 = None
    unsqueeze_379: "f32[264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, -1);  unsqueeze_378 = None
    sub_47: "f32[8, 264, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_126, unsqueeze_377);  convolution_126 = unsqueeze_377 = None
    mul_194: "f32[8, 264, 7, 7]" = torch.ops.aten.mul.Tensor(sub_47, unsqueeze_379);  sub_47 = unsqueeze_379 = None
    unsqueeze_380: "f32[264, 1]" = torch.ops.aten.unsqueeze.default(arg109_1, -1);  arg109_1 = None
    unsqueeze_381: "f32[264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, -1);  unsqueeze_380 = None
    mul_195: "f32[8, 264, 7, 7]" = torch.ops.aten.mul.Tensor(mul_194, unsqueeze_381);  mul_194 = unsqueeze_381 = None
    unsqueeze_382: "f32[264, 1]" = torch.ops.aten.unsqueeze.default(arg110_1, -1);  arg110_1 = None
    unsqueeze_383: "f32[264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, -1);  unsqueeze_382 = None
    add_106: "f32[8, 264, 7, 7]" = torch.ops.aten.add.Tensor(mul_195, unsqueeze_383);  mul_195 = unsqueeze_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_127: "f32[8, 1584, 7, 7]" = torch.ops.aten.convolution.default(add_106, arg269_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg269_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_96: "f32[1584]" = torch.ops.prims.convert_element_type.default(arg401_1, torch.float32);  arg401_1 = None
    convert_element_type_97: "f32[1584]" = torch.ops.prims.convert_element_type.default(arg402_1, torch.float32);  arg402_1 = None
    add_107: "f32[1584]" = torch.ops.aten.add.Tensor(convert_element_type_97, 0.001);  convert_element_type_97 = None
    sqrt_48: "f32[1584]" = torch.ops.aten.sqrt.default(add_107);  add_107 = None
    reciprocal_48: "f32[1584]" = torch.ops.aten.reciprocal.default(sqrt_48);  sqrt_48 = None
    mul_196: "f32[1584]" = torch.ops.aten.mul.Tensor(reciprocal_48, 1);  reciprocal_48 = None
    unsqueeze_384: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_96, -1);  convert_element_type_96 = None
    unsqueeze_385: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_384, -1);  unsqueeze_384 = None
    unsqueeze_386: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(mul_196, -1);  mul_196 = None
    unsqueeze_387: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, -1);  unsqueeze_386 = None
    sub_48: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_127, unsqueeze_385);  convolution_127 = unsqueeze_385 = None
    mul_197: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(sub_48, unsqueeze_387);  sub_48 = unsqueeze_387 = None
    unsqueeze_388: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(arg111_1, -1);  arg111_1 = None
    unsqueeze_389: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, -1);  unsqueeze_388 = None
    mul_198: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(mul_197, unsqueeze_389);  mul_197 = unsqueeze_389 = None
    unsqueeze_390: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(arg112_1, -1);  arg112_1 = None
    unsqueeze_391: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_390, -1);  unsqueeze_390 = None
    add_108: "f32[8, 1584, 7, 7]" = torch.ops.aten.add.Tensor(mul_198, unsqueeze_391);  mul_198 = unsqueeze_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_52: "f32[8, 1584, 7, 7]" = torch.ops.aten.sigmoid.default(add_108)
    mul_199: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(add_108, sigmoid_52);  add_108 = sigmoid_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    split_with_sizes_84 = torch.ops.aten.split_with_sizes.default(mul_199, [396, 396, 396, 396], 1)
    getitem_258: "f32[8, 396, 7, 7]" = split_with_sizes_84[0];  split_with_sizes_84 = None
    convolution_128: "f32[8, 396, 7, 7]" = torch.ops.aten.convolution.default(getitem_258, arg270_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 396);  getitem_258 = arg270_1 = None
    split_with_sizes_85 = torch.ops.aten.split_with_sizes.default(mul_199, [396, 396, 396, 396], 1)
    getitem_263: "f32[8, 396, 7, 7]" = split_with_sizes_85[1];  split_with_sizes_85 = None
    convolution_129: "f32[8, 396, 7, 7]" = torch.ops.aten.convolution.default(getitem_263, arg271_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 396);  getitem_263 = arg271_1 = None
    split_with_sizes_86 = torch.ops.aten.split_with_sizes.default(mul_199, [396, 396, 396, 396], 1)
    getitem_268: "f32[8, 396, 7, 7]" = split_with_sizes_86[2];  split_with_sizes_86 = None
    convolution_130: "f32[8, 396, 7, 7]" = torch.ops.aten.convolution.default(getitem_268, arg272_1, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 396);  getitem_268 = arg272_1 = None
    split_with_sizes_87 = torch.ops.aten.split_with_sizes.default(mul_199, [396, 396, 396, 396], 1);  mul_199 = None
    getitem_273: "f32[8, 396, 7, 7]" = split_with_sizes_87[3];  split_with_sizes_87 = None
    convolution_131: "f32[8, 396, 7, 7]" = torch.ops.aten.convolution.default(getitem_273, arg273_1, None, [1, 1], [4, 4], [1, 1], False, [0, 0], 396);  getitem_273 = arg273_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_35: "f32[8, 1584, 7, 7]" = torch.ops.aten.cat.default([convolution_128, convolution_129, convolution_130, convolution_131], 1);  convolution_128 = convolution_129 = convolution_130 = convolution_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_98: "f32[1584]" = torch.ops.prims.convert_element_type.default(arg403_1, torch.float32);  arg403_1 = None
    convert_element_type_99: "f32[1584]" = torch.ops.prims.convert_element_type.default(arg404_1, torch.float32);  arg404_1 = None
    add_109: "f32[1584]" = torch.ops.aten.add.Tensor(convert_element_type_99, 0.001);  convert_element_type_99 = None
    sqrt_49: "f32[1584]" = torch.ops.aten.sqrt.default(add_109);  add_109 = None
    reciprocal_49: "f32[1584]" = torch.ops.aten.reciprocal.default(sqrt_49);  sqrt_49 = None
    mul_200: "f32[1584]" = torch.ops.aten.mul.Tensor(reciprocal_49, 1);  reciprocal_49 = None
    unsqueeze_392: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_98, -1);  convert_element_type_98 = None
    unsqueeze_393: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, -1);  unsqueeze_392 = None
    unsqueeze_394: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(mul_200, -1);  mul_200 = None
    unsqueeze_395: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, -1);  unsqueeze_394 = None
    sub_49: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(cat_35, unsqueeze_393);  cat_35 = unsqueeze_393 = None
    mul_201: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(sub_49, unsqueeze_395);  sub_49 = unsqueeze_395 = None
    unsqueeze_396: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(arg113_1, -1);  arg113_1 = None
    unsqueeze_397: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_396, -1);  unsqueeze_396 = None
    mul_202: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(mul_201, unsqueeze_397);  mul_201 = unsqueeze_397 = None
    unsqueeze_398: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(arg114_1, -1);  arg114_1 = None
    unsqueeze_399: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, -1);  unsqueeze_398 = None
    add_110: "f32[8, 1584, 7, 7]" = torch.ops.aten.add.Tensor(mul_202, unsqueeze_399);  mul_202 = unsqueeze_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_53: "f32[8, 1584, 7, 7]" = torch.ops.aten.sigmoid.default(add_110)
    mul_203: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(add_110, sigmoid_53);  add_110 = sigmoid_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_13: "f32[8, 1584, 1, 1]" = torch.ops.aten.mean.dim(mul_203, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_132: "f32[8, 132, 1, 1]" = torch.ops.aten.convolution.default(mean_13, arg274_1, arg275_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_13 = arg274_1 = arg275_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_54: "f32[8, 132, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_132)
    mul_204: "f32[8, 132, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_132, sigmoid_54);  convolution_132 = sigmoid_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_133: "f32[8, 1584, 1, 1]" = torch.ops.aten.convolution.default(mul_204, arg276_1, arg277_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_204 = arg276_1 = arg277_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_55: "f32[8, 1584, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_133);  convolution_133 = None
    mul_205: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(mul_203, sigmoid_55);  mul_203 = sigmoid_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_88 = torch.ops.aten.split_with_sizes.default(mul_205, [792, 792], 1);  mul_205 = None
    getitem_274: "f32[8, 792, 7, 7]" = split_with_sizes_88[0]
    getitem_275: "f32[8, 792, 7, 7]" = split_with_sizes_88[1];  split_with_sizes_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_134: "f32[8, 132, 7, 7]" = torch.ops.aten.convolution.default(getitem_274, arg278_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_274 = arg278_1 = None
    convolution_135: "f32[8, 132, 7, 7]" = torch.ops.aten.convolution.default(getitem_275, arg279_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_275 = arg279_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_36: "f32[8, 264, 7, 7]" = torch.ops.aten.cat.default([convolution_134, convolution_135], 1);  convolution_134 = convolution_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_100: "f32[264]" = torch.ops.prims.convert_element_type.default(arg405_1, torch.float32);  arg405_1 = None
    convert_element_type_101: "f32[264]" = torch.ops.prims.convert_element_type.default(arg406_1, torch.float32);  arg406_1 = None
    add_111: "f32[264]" = torch.ops.aten.add.Tensor(convert_element_type_101, 0.001);  convert_element_type_101 = None
    sqrt_50: "f32[264]" = torch.ops.aten.sqrt.default(add_111);  add_111 = None
    reciprocal_50: "f32[264]" = torch.ops.aten.reciprocal.default(sqrt_50);  sqrt_50 = None
    mul_206: "f32[264]" = torch.ops.aten.mul.Tensor(reciprocal_50, 1);  reciprocal_50 = None
    unsqueeze_400: "f32[264, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_100, -1);  convert_element_type_100 = None
    unsqueeze_401: "f32[264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, -1);  unsqueeze_400 = None
    unsqueeze_402: "f32[264, 1]" = torch.ops.aten.unsqueeze.default(mul_206, -1);  mul_206 = None
    unsqueeze_403: "f32[264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, -1);  unsqueeze_402 = None
    sub_50: "f32[8, 264, 7, 7]" = torch.ops.aten.sub.Tensor(cat_36, unsqueeze_401);  cat_36 = unsqueeze_401 = None
    mul_207: "f32[8, 264, 7, 7]" = torch.ops.aten.mul.Tensor(sub_50, unsqueeze_403);  sub_50 = unsqueeze_403 = None
    unsqueeze_404: "f32[264, 1]" = torch.ops.aten.unsqueeze.default(arg115_1, -1);  arg115_1 = None
    unsqueeze_405: "f32[264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, -1);  unsqueeze_404 = None
    mul_208: "f32[8, 264, 7, 7]" = torch.ops.aten.mul.Tensor(mul_207, unsqueeze_405);  mul_207 = unsqueeze_405 = None
    unsqueeze_406: "f32[264, 1]" = torch.ops.aten.unsqueeze.default(arg116_1, -1);  arg116_1 = None
    unsqueeze_407: "f32[264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, -1);  unsqueeze_406 = None
    add_112: "f32[8, 264, 7, 7]" = torch.ops.aten.add.Tensor(mul_208, unsqueeze_407);  mul_208 = unsqueeze_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_113: "f32[8, 264, 7, 7]" = torch.ops.aten.add.Tensor(add_112, add_106);  add_112 = add_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_136: "f32[8, 1584, 7, 7]" = torch.ops.aten.convolution.default(add_113, arg280_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg280_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_102: "f32[1584]" = torch.ops.prims.convert_element_type.default(arg407_1, torch.float32);  arg407_1 = None
    convert_element_type_103: "f32[1584]" = torch.ops.prims.convert_element_type.default(arg408_1, torch.float32);  arg408_1 = None
    add_114: "f32[1584]" = torch.ops.aten.add.Tensor(convert_element_type_103, 0.001);  convert_element_type_103 = None
    sqrt_51: "f32[1584]" = torch.ops.aten.sqrt.default(add_114);  add_114 = None
    reciprocal_51: "f32[1584]" = torch.ops.aten.reciprocal.default(sqrt_51);  sqrt_51 = None
    mul_209: "f32[1584]" = torch.ops.aten.mul.Tensor(reciprocal_51, 1);  reciprocal_51 = None
    unsqueeze_408: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_102, -1);  convert_element_type_102 = None
    unsqueeze_409: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_408, -1);  unsqueeze_408 = None
    unsqueeze_410: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(mul_209, -1);  mul_209 = None
    unsqueeze_411: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, -1);  unsqueeze_410 = None
    sub_51: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_136, unsqueeze_409);  convolution_136 = unsqueeze_409 = None
    mul_210: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(sub_51, unsqueeze_411);  sub_51 = unsqueeze_411 = None
    unsqueeze_412: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(arg117_1, -1);  arg117_1 = None
    unsqueeze_413: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, -1);  unsqueeze_412 = None
    mul_211: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(mul_210, unsqueeze_413);  mul_210 = unsqueeze_413 = None
    unsqueeze_414: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(arg118_1, -1);  arg118_1 = None
    unsqueeze_415: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, -1);  unsqueeze_414 = None
    add_115: "f32[8, 1584, 7, 7]" = torch.ops.aten.add.Tensor(mul_211, unsqueeze_415);  mul_211 = unsqueeze_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_56: "f32[8, 1584, 7, 7]" = torch.ops.aten.sigmoid.default(add_115)
    mul_212: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(add_115, sigmoid_56);  add_115 = sigmoid_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    split_with_sizes_90 = torch.ops.aten.split_with_sizes.default(mul_212, [396, 396, 396, 396], 1)
    getitem_280: "f32[8, 396, 7, 7]" = split_with_sizes_90[0];  split_with_sizes_90 = None
    convolution_137: "f32[8, 396, 7, 7]" = torch.ops.aten.convolution.default(getitem_280, arg281_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 396);  getitem_280 = arg281_1 = None
    split_with_sizes_91 = torch.ops.aten.split_with_sizes.default(mul_212, [396, 396, 396, 396], 1)
    getitem_285: "f32[8, 396, 7, 7]" = split_with_sizes_91[1];  split_with_sizes_91 = None
    convolution_138: "f32[8, 396, 7, 7]" = torch.ops.aten.convolution.default(getitem_285, arg282_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 396);  getitem_285 = arg282_1 = None
    split_with_sizes_92 = torch.ops.aten.split_with_sizes.default(mul_212, [396, 396, 396, 396], 1)
    getitem_290: "f32[8, 396, 7, 7]" = split_with_sizes_92[2];  split_with_sizes_92 = None
    convolution_139: "f32[8, 396, 7, 7]" = torch.ops.aten.convolution.default(getitem_290, arg283_1, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 396);  getitem_290 = arg283_1 = None
    split_with_sizes_93 = torch.ops.aten.split_with_sizes.default(mul_212, [396, 396, 396, 396], 1);  mul_212 = None
    getitem_295: "f32[8, 396, 7, 7]" = split_with_sizes_93[3];  split_with_sizes_93 = None
    convolution_140: "f32[8, 396, 7, 7]" = torch.ops.aten.convolution.default(getitem_295, arg284_1, None, [1, 1], [4, 4], [1, 1], False, [0, 0], 396);  getitem_295 = arg284_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_37: "f32[8, 1584, 7, 7]" = torch.ops.aten.cat.default([convolution_137, convolution_138, convolution_139, convolution_140], 1);  convolution_137 = convolution_138 = convolution_139 = convolution_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_104: "f32[1584]" = torch.ops.prims.convert_element_type.default(arg409_1, torch.float32);  arg409_1 = None
    convert_element_type_105: "f32[1584]" = torch.ops.prims.convert_element_type.default(arg410_1, torch.float32);  arg410_1 = None
    add_116: "f32[1584]" = torch.ops.aten.add.Tensor(convert_element_type_105, 0.001);  convert_element_type_105 = None
    sqrt_52: "f32[1584]" = torch.ops.aten.sqrt.default(add_116);  add_116 = None
    reciprocal_52: "f32[1584]" = torch.ops.aten.reciprocal.default(sqrt_52);  sqrt_52 = None
    mul_213: "f32[1584]" = torch.ops.aten.mul.Tensor(reciprocal_52, 1);  reciprocal_52 = None
    unsqueeze_416: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_104, -1);  convert_element_type_104 = None
    unsqueeze_417: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, -1);  unsqueeze_416 = None
    unsqueeze_418: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(mul_213, -1);  mul_213 = None
    unsqueeze_419: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, -1);  unsqueeze_418 = None
    sub_52: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(cat_37, unsqueeze_417);  cat_37 = unsqueeze_417 = None
    mul_214: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(sub_52, unsqueeze_419);  sub_52 = unsqueeze_419 = None
    unsqueeze_420: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(arg119_1, -1);  arg119_1 = None
    unsqueeze_421: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_420, -1);  unsqueeze_420 = None
    mul_215: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(mul_214, unsqueeze_421);  mul_214 = unsqueeze_421 = None
    unsqueeze_422: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(arg120_1, -1);  arg120_1 = None
    unsqueeze_423: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, -1);  unsqueeze_422 = None
    add_117: "f32[8, 1584, 7, 7]" = torch.ops.aten.add.Tensor(mul_215, unsqueeze_423);  mul_215 = unsqueeze_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_57: "f32[8, 1584, 7, 7]" = torch.ops.aten.sigmoid.default(add_117)
    mul_216: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(add_117, sigmoid_57);  add_117 = sigmoid_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_14: "f32[8, 1584, 1, 1]" = torch.ops.aten.mean.dim(mul_216, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_141: "f32[8, 132, 1, 1]" = torch.ops.aten.convolution.default(mean_14, arg285_1, arg286_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_14 = arg285_1 = arg286_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_58: "f32[8, 132, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_141)
    mul_217: "f32[8, 132, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_141, sigmoid_58);  convolution_141 = sigmoid_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_142: "f32[8, 1584, 1, 1]" = torch.ops.aten.convolution.default(mul_217, arg287_1, arg288_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_217 = arg287_1 = arg288_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_59: "f32[8, 1584, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_142);  convolution_142 = None
    mul_218: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(mul_216, sigmoid_59);  mul_216 = sigmoid_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_94 = torch.ops.aten.split_with_sizes.default(mul_218, [792, 792], 1);  mul_218 = None
    getitem_296: "f32[8, 792, 7, 7]" = split_with_sizes_94[0]
    getitem_297: "f32[8, 792, 7, 7]" = split_with_sizes_94[1];  split_with_sizes_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_143: "f32[8, 132, 7, 7]" = torch.ops.aten.convolution.default(getitem_296, arg289_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_296 = arg289_1 = None
    convolution_144: "f32[8, 132, 7, 7]" = torch.ops.aten.convolution.default(getitem_297, arg290_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_297 = arg290_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_38: "f32[8, 264, 7, 7]" = torch.ops.aten.cat.default([convolution_143, convolution_144], 1);  convolution_143 = convolution_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_106: "f32[264]" = torch.ops.prims.convert_element_type.default(arg411_1, torch.float32);  arg411_1 = None
    convert_element_type_107: "f32[264]" = torch.ops.prims.convert_element_type.default(arg412_1, torch.float32);  arg412_1 = None
    add_118: "f32[264]" = torch.ops.aten.add.Tensor(convert_element_type_107, 0.001);  convert_element_type_107 = None
    sqrt_53: "f32[264]" = torch.ops.aten.sqrt.default(add_118);  add_118 = None
    reciprocal_53: "f32[264]" = torch.ops.aten.reciprocal.default(sqrt_53);  sqrt_53 = None
    mul_219: "f32[264]" = torch.ops.aten.mul.Tensor(reciprocal_53, 1);  reciprocal_53 = None
    unsqueeze_424: "f32[264, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_106, -1);  convert_element_type_106 = None
    unsqueeze_425: "f32[264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_424, -1);  unsqueeze_424 = None
    unsqueeze_426: "f32[264, 1]" = torch.ops.aten.unsqueeze.default(mul_219, -1);  mul_219 = None
    unsqueeze_427: "f32[264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, -1);  unsqueeze_426 = None
    sub_53: "f32[8, 264, 7, 7]" = torch.ops.aten.sub.Tensor(cat_38, unsqueeze_425);  cat_38 = unsqueeze_425 = None
    mul_220: "f32[8, 264, 7, 7]" = torch.ops.aten.mul.Tensor(sub_53, unsqueeze_427);  sub_53 = unsqueeze_427 = None
    unsqueeze_428: "f32[264, 1]" = torch.ops.aten.unsqueeze.default(arg121_1, -1);  arg121_1 = None
    unsqueeze_429: "f32[264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, -1);  unsqueeze_428 = None
    mul_221: "f32[8, 264, 7, 7]" = torch.ops.aten.mul.Tensor(mul_220, unsqueeze_429);  mul_220 = unsqueeze_429 = None
    unsqueeze_430: "f32[264, 1]" = torch.ops.aten.unsqueeze.default(arg122_1, -1);  arg122_1 = None
    unsqueeze_431: "f32[264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, -1);  unsqueeze_430 = None
    add_119: "f32[8, 264, 7, 7]" = torch.ops.aten.add.Tensor(mul_221, unsqueeze_431);  mul_221 = unsqueeze_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_120: "f32[8, 264, 7, 7]" = torch.ops.aten.add.Tensor(add_119, add_113);  add_119 = add_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_145: "f32[8, 1584, 7, 7]" = torch.ops.aten.convolution.default(add_120, arg291_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg291_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_108: "f32[1584]" = torch.ops.prims.convert_element_type.default(arg413_1, torch.float32);  arg413_1 = None
    convert_element_type_109: "f32[1584]" = torch.ops.prims.convert_element_type.default(arg414_1, torch.float32);  arg414_1 = None
    add_121: "f32[1584]" = torch.ops.aten.add.Tensor(convert_element_type_109, 0.001);  convert_element_type_109 = None
    sqrt_54: "f32[1584]" = torch.ops.aten.sqrt.default(add_121);  add_121 = None
    reciprocal_54: "f32[1584]" = torch.ops.aten.reciprocal.default(sqrt_54);  sqrt_54 = None
    mul_222: "f32[1584]" = torch.ops.aten.mul.Tensor(reciprocal_54, 1);  reciprocal_54 = None
    unsqueeze_432: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_108, -1);  convert_element_type_108 = None
    unsqueeze_433: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_432, -1);  unsqueeze_432 = None
    unsqueeze_434: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(mul_222, -1);  mul_222 = None
    unsqueeze_435: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, -1);  unsqueeze_434 = None
    sub_54: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_145, unsqueeze_433);  convolution_145 = unsqueeze_433 = None
    mul_223: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(sub_54, unsqueeze_435);  sub_54 = unsqueeze_435 = None
    unsqueeze_436: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(arg123_1, -1);  arg123_1 = None
    unsqueeze_437: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_436, -1);  unsqueeze_436 = None
    mul_224: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(mul_223, unsqueeze_437);  mul_223 = unsqueeze_437 = None
    unsqueeze_438: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(arg124_1, -1);  arg124_1 = None
    unsqueeze_439: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_438, -1);  unsqueeze_438 = None
    add_122: "f32[8, 1584, 7, 7]" = torch.ops.aten.add.Tensor(mul_224, unsqueeze_439);  mul_224 = unsqueeze_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_60: "f32[8, 1584, 7, 7]" = torch.ops.aten.sigmoid.default(add_122)
    mul_225: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(add_122, sigmoid_60);  add_122 = sigmoid_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    split_with_sizes_96 = torch.ops.aten.split_with_sizes.default(mul_225, [396, 396, 396, 396], 1)
    getitem_302: "f32[8, 396, 7, 7]" = split_with_sizes_96[0];  split_with_sizes_96 = None
    convolution_146: "f32[8, 396, 7, 7]" = torch.ops.aten.convolution.default(getitem_302, arg292_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 396);  getitem_302 = arg292_1 = None
    split_with_sizes_97 = torch.ops.aten.split_with_sizes.default(mul_225, [396, 396, 396, 396], 1)
    getitem_307: "f32[8, 396, 7, 7]" = split_with_sizes_97[1];  split_with_sizes_97 = None
    convolution_147: "f32[8, 396, 7, 7]" = torch.ops.aten.convolution.default(getitem_307, arg293_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 396);  getitem_307 = arg293_1 = None
    split_with_sizes_98 = torch.ops.aten.split_with_sizes.default(mul_225, [396, 396, 396, 396], 1)
    getitem_312: "f32[8, 396, 7, 7]" = split_with_sizes_98[2];  split_with_sizes_98 = None
    convolution_148: "f32[8, 396, 7, 7]" = torch.ops.aten.convolution.default(getitem_312, arg294_1, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 396);  getitem_312 = arg294_1 = None
    split_with_sizes_99 = torch.ops.aten.split_with_sizes.default(mul_225, [396, 396, 396, 396], 1);  mul_225 = None
    getitem_317: "f32[8, 396, 7, 7]" = split_with_sizes_99[3];  split_with_sizes_99 = None
    convolution_149: "f32[8, 396, 7, 7]" = torch.ops.aten.convolution.default(getitem_317, arg295_1, None, [1, 1], [4, 4], [1, 1], False, [0, 0], 396);  getitem_317 = arg295_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_39: "f32[8, 1584, 7, 7]" = torch.ops.aten.cat.default([convolution_146, convolution_147, convolution_148, convolution_149], 1);  convolution_146 = convolution_147 = convolution_148 = convolution_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_110: "f32[1584]" = torch.ops.prims.convert_element_type.default(arg415_1, torch.float32);  arg415_1 = None
    convert_element_type_111: "f32[1584]" = torch.ops.prims.convert_element_type.default(arg416_1, torch.float32);  arg416_1 = None
    add_123: "f32[1584]" = torch.ops.aten.add.Tensor(convert_element_type_111, 0.001);  convert_element_type_111 = None
    sqrt_55: "f32[1584]" = torch.ops.aten.sqrt.default(add_123);  add_123 = None
    reciprocal_55: "f32[1584]" = torch.ops.aten.reciprocal.default(sqrt_55);  sqrt_55 = None
    mul_226: "f32[1584]" = torch.ops.aten.mul.Tensor(reciprocal_55, 1);  reciprocal_55 = None
    unsqueeze_440: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_110, -1);  convert_element_type_110 = None
    unsqueeze_441: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, -1);  unsqueeze_440 = None
    unsqueeze_442: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(mul_226, -1);  mul_226 = None
    unsqueeze_443: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_442, -1);  unsqueeze_442 = None
    sub_55: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(cat_39, unsqueeze_441);  cat_39 = unsqueeze_441 = None
    mul_227: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(sub_55, unsqueeze_443);  sub_55 = unsqueeze_443 = None
    unsqueeze_444: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(arg125_1, -1);  arg125_1 = None
    unsqueeze_445: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_444, -1);  unsqueeze_444 = None
    mul_228: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(mul_227, unsqueeze_445);  mul_227 = unsqueeze_445 = None
    unsqueeze_446: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(arg126_1, -1);  arg126_1 = None
    unsqueeze_447: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, -1);  unsqueeze_446 = None
    add_124: "f32[8, 1584, 7, 7]" = torch.ops.aten.add.Tensor(mul_228, unsqueeze_447);  mul_228 = unsqueeze_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_61: "f32[8, 1584, 7, 7]" = torch.ops.aten.sigmoid.default(add_124)
    mul_229: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(add_124, sigmoid_61);  add_124 = sigmoid_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_15: "f32[8, 1584, 1, 1]" = torch.ops.aten.mean.dim(mul_229, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_150: "f32[8, 132, 1, 1]" = torch.ops.aten.convolution.default(mean_15, arg296_1, arg297_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_15 = arg296_1 = arg297_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_62: "f32[8, 132, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_150)
    mul_230: "f32[8, 132, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_150, sigmoid_62);  convolution_150 = sigmoid_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_151: "f32[8, 1584, 1, 1]" = torch.ops.aten.convolution.default(mul_230, arg298_1, arg299_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_230 = arg298_1 = arg299_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_63: "f32[8, 1584, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_151);  convolution_151 = None
    mul_231: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(mul_229, sigmoid_63);  mul_229 = sigmoid_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_100 = torch.ops.aten.split_with_sizes.default(mul_231, [792, 792], 1);  mul_231 = None
    getitem_318: "f32[8, 792, 7, 7]" = split_with_sizes_100[0]
    getitem_319: "f32[8, 792, 7, 7]" = split_with_sizes_100[1];  split_with_sizes_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_152: "f32[8, 132, 7, 7]" = torch.ops.aten.convolution.default(getitem_318, arg300_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_318 = arg300_1 = None
    convolution_153: "f32[8, 132, 7, 7]" = torch.ops.aten.convolution.default(getitem_319, arg301_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_319 = arg301_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_40: "f32[8, 264, 7, 7]" = torch.ops.aten.cat.default([convolution_152, convolution_153], 1);  convolution_152 = convolution_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_112: "f32[264]" = torch.ops.prims.convert_element_type.default(arg417_1, torch.float32);  arg417_1 = None
    convert_element_type_113: "f32[264]" = torch.ops.prims.convert_element_type.default(arg418_1, torch.float32);  arg418_1 = None
    add_125: "f32[264]" = torch.ops.aten.add.Tensor(convert_element_type_113, 0.001);  convert_element_type_113 = None
    sqrt_56: "f32[264]" = torch.ops.aten.sqrt.default(add_125);  add_125 = None
    reciprocal_56: "f32[264]" = torch.ops.aten.reciprocal.default(sqrt_56);  sqrt_56 = None
    mul_232: "f32[264]" = torch.ops.aten.mul.Tensor(reciprocal_56, 1);  reciprocal_56 = None
    unsqueeze_448: "f32[264, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_112, -1);  convert_element_type_112 = None
    unsqueeze_449: "f32[264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_448, -1);  unsqueeze_448 = None
    unsqueeze_450: "f32[264, 1]" = torch.ops.aten.unsqueeze.default(mul_232, -1);  mul_232 = None
    unsqueeze_451: "f32[264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_450, -1);  unsqueeze_450 = None
    sub_56: "f32[8, 264, 7, 7]" = torch.ops.aten.sub.Tensor(cat_40, unsqueeze_449);  cat_40 = unsqueeze_449 = None
    mul_233: "f32[8, 264, 7, 7]" = torch.ops.aten.mul.Tensor(sub_56, unsqueeze_451);  sub_56 = unsqueeze_451 = None
    unsqueeze_452: "f32[264, 1]" = torch.ops.aten.unsqueeze.default(arg127_1, -1);  arg127_1 = None
    unsqueeze_453: "f32[264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, -1);  unsqueeze_452 = None
    mul_234: "f32[8, 264, 7, 7]" = torch.ops.aten.mul.Tensor(mul_233, unsqueeze_453);  mul_233 = unsqueeze_453 = None
    unsqueeze_454: "f32[264, 1]" = torch.ops.aten.unsqueeze.default(arg128_1, -1);  arg128_1 = None
    unsqueeze_455: "f32[264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_454, -1);  unsqueeze_454 = None
    add_126: "f32[8, 264, 7, 7]" = torch.ops.aten.add.Tensor(mul_234, unsqueeze_455);  mul_234 = unsqueeze_455 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_127: "f32[8, 264, 7, 7]" = torch.ops.aten.add.Tensor(add_126, add_120);  add_126 = add_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/efficientnet.py:168, code: x = self.conv_head(x)
    convolution_154: "f32[8, 1536, 7, 7]" = torch.ops.aten.convolution.default(add_127, arg302_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_127 = arg302_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_114: "f32[1536]" = torch.ops.prims.convert_element_type.default(arg419_1, torch.float32);  arg419_1 = None
    convert_element_type_115: "f32[1536]" = torch.ops.prims.convert_element_type.default(arg420_1, torch.float32);  arg420_1 = None
    add_128: "f32[1536]" = torch.ops.aten.add.Tensor(convert_element_type_115, 0.001);  convert_element_type_115 = None
    sqrt_57: "f32[1536]" = torch.ops.aten.sqrt.default(add_128);  add_128 = None
    reciprocal_57: "f32[1536]" = torch.ops.aten.reciprocal.default(sqrt_57);  sqrt_57 = None
    mul_235: "f32[1536]" = torch.ops.aten.mul.Tensor(reciprocal_57, 1);  reciprocal_57 = None
    unsqueeze_456: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_114, -1);  convert_element_type_114 = None
    unsqueeze_457: "f32[1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_456, -1);  unsqueeze_456 = None
    unsqueeze_458: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(mul_235, -1);  mul_235 = None
    unsqueeze_459: "f32[1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, -1);  unsqueeze_458 = None
    sub_57: "f32[8, 1536, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_154, unsqueeze_457);  convolution_154 = unsqueeze_457 = None
    mul_236: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(sub_57, unsqueeze_459);  sub_57 = unsqueeze_459 = None
    unsqueeze_460: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(arg129_1, -1);  arg129_1 = None
    unsqueeze_461: "f32[1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_460, -1);  unsqueeze_460 = None
    mul_237: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(mul_236, unsqueeze_461);  mul_236 = unsqueeze_461 = None
    unsqueeze_462: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(arg130_1, -1);  arg130_1 = None
    unsqueeze_463: "f32[1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_462, -1);  unsqueeze_462 = None
    add_129: "f32[8, 1536, 7, 7]" = torch.ops.aten.add.Tensor(mul_237, unsqueeze_463);  mul_237 = unsqueeze_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_6: "f32[8, 1536, 7, 7]" = torch.ops.aten.relu.default(add_129);  add_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean_16: "f32[8, 1536, 1, 1]" = torch.ops.aten.mean.dim(relu_6, [-1, -2], True);  relu_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view: "f32[8, 1536]" = torch.ops.aten.view.default(mean_16, [8, 1536]);  mean_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/efficientnet.py:176, code: return x if pre_logits else self.classifier(x)
    permute: "f32[1536, 1000]" = torch.ops.aten.permute.default(arg303_1, [1, 0]);  arg303_1 = None
    addmm: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg304_1, view, permute);  arg304_1 = view = permute = None
    return (addmm,)
    