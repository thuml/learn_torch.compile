from __future__ import annotations



def forward(self, arg0_1: "f32[960]", arg1_1: "f32[960]", arg2_1: "f32[1000, 1280]", arg3_1: "f32[1000]", arg4_1: "f32[16, 3, 3, 3]", arg5_1: "f32[16]", arg6_1: "f32[16]", arg7_1: "f32[8, 16, 1, 1]", arg8_1: "f32[8]", arg9_1: "f32[8]", arg10_1: "f32[8, 1, 3, 3]", arg11_1: "f32[8]", arg12_1: "f32[8]", arg13_1: "f32[8, 16, 1, 1]", arg14_1: "f32[8]", arg15_1: "f32[8]", arg16_1: "f32[8, 1, 3, 3]", arg17_1: "f32[8]", arg18_1: "f32[8]", arg19_1: "f32[24, 16, 1, 1]", arg20_1: "f32[24]", arg21_1: "f32[24]", arg22_1: "f32[24, 1, 3, 3]", arg23_1: "f32[24]", arg24_1: "f32[24]", arg25_1: "f32[48, 1, 3, 3]", arg26_1: "f32[48]", arg27_1: "f32[48]", arg28_1: "f32[12, 48, 1, 1]", arg29_1: "f32[12]", arg30_1: "f32[12]", arg31_1: "f32[12, 1, 3, 3]", arg32_1: "f32[12]", arg33_1: "f32[12]", arg34_1: "f32[16, 1, 3, 3]", arg35_1: "f32[16]", arg36_1: "f32[16]", arg37_1: "f32[24, 16, 1, 1]", arg38_1: "f32[24]", arg39_1: "f32[24]", arg40_1: "f32[36, 24, 1, 1]", arg41_1: "f32[36]", arg42_1: "f32[36]", arg43_1: "f32[36, 1, 3, 3]", arg44_1: "f32[36]", arg45_1: "f32[36]", arg46_1: "f32[12, 72, 1, 1]", arg47_1: "f32[12]", arg48_1: "f32[12]", arg49_1: "f32[12, 1, 3, 3]", arg50_1: "f32[12]", arg51_1: "f32[12]", arg52_1: "f32[36, 24, 1, 1]", arg53_1: "f32[36]", arg54_1: "f32[36]", arg55_1: "f32[36, 1, 3, 3]", arg56_1: "f32[36]", arg57_1: "f32[36]", arg58_1: "f32[72, 1, 5, 5]", arg59_1: "f32[72]", arg60_1: "f32[72]", arg61_1: "f32[20, 72, 1, 1]", arg62_1: "f32[20]", arg63_1: "f32[72, 20, 1, 1]", arg64_1: "f32[72]", arg65_1: "f32[20, 72, 1, 1]", arg66_1: "f32[20]", arg67_1: "f32[20]", arg68_1: "f32[20, 1, 3, 3]", arg69_1: "f32[20]", arg70_1: "f32[20]", arg71_1: "f32[24, 1, 5, 5]", arg72_1: "f32[24]", arg73_1: "f32[24]", arg74_1: "f32[40, 24, 1, 1]", arg75_1: "f32[40]", arg76_1: "f32[40]", arg77_1: "f32[60, 40, 1, 1]", arg78_1: "f32[60]", arg79_1: "f32[60]", arg80_1: "f32[60, 1, 3, 3]", arg81_1: "f32[60]", arg82_1: "f32[60]", arg83_1: "f32[32, 120, 1, 1]", arg84_1: "f32[32]", arg85_1: "f32[120, 32, 1, 1]", arg86_1: "f32[120]", arg87_1: "f32[20, 120, 1, 1]", arg88_1: "f32[20]", arg89_1: "f32[20]", arg90_1: "f32[20, 1, 3, 3]", arg91_1: "f32[20]", arg92_1: "f32[20]", arg93_1: "f32[120, 40, 1, 1]", arg94_1: "f32[120]", arg95_1: "f32[120]", arg96_1: "f32[120, 1, 3, 3]", arg97_1: "f32[120]", arg98_1: "f32[120]", arg99_1: "f32[240, 1, 3, 3]", arg100_1: "f32[240]", arg101_1: "f32[240]", arg102_1: "f32[40, 240, 1, 1]", arg103_1: "f32[40]", arg104_1: "f32[40]", arg105_1: "f32[40, 1, 3, 3]", arg106_1: "f32[40]", arg107_1: "f32[40]", arg108_1: "f32[40, 1, 3, 3]", arg109_1: "f32[40]", arg110_1: "f32[40]", arg111_1: "f32[80, 40, 1, 1]", arg112_1: "f32[80]", arg113_1: "f32[80]", arg114_1: "f32[100, 80, 1, 1]", arg115_1: "f32[100]", arg116_1: "f32[100]", arg117_1: "f32[100, 1, 3, 3]", arg118_1: "f32[100]", arg119_1: "f32[100]", arg120_1: "f32[40, 200, 1, 1]", arg121_1: "f32[40]", arg122_1: "f32[40]", arg123_1: "f32[40, 1, 3, 3]", arg124_1: "f32[40]", arg125_1: "f32[40]", arg126_1: "f32[92, 80, 1, 1]", arg127_1: "f32[92]", arg128_1: "f32[92]", arg129_1: "f32[92, 1, 3, 3]", arg130_1: "f32[92]", arg131_1: "f32[92]", arg132_1: "f32[40, 184, 1, 1]", arg133_1: "f32[40]", arg134_1: "f32[40]", arg135_1: "f32[40, 1, 3, 3]", arg136_1: "f32[40]", arg137_1: "f32[40]", arg138_1: "f32[92, 80, 1, 1]", arg139_1: "f32[92]", arg140_1: "f32[92]", arg141_1: "f32[92, 1, 3, 3]", arg142_1: "f32[92]", arg143_1: "f32[92]", arg144_1: "f32[40, 184, 1, 1]", arg145_1: "f32[40]", arg146_1: "f32[40]", arg147_1: "f32[40, 1, 3, 3]", arg148_1: "f32[40]", arg149_1: "f32[40]", arg150_1: "f32[240, 80, 1, 1]", arg151_1: "f32[240]", arg152_1: "f32[240]", arg153_1: "f32[240, 1, 3, 3]", arg154_1: "f32[240]", arg155_1: "f32[240]", arg156_1: "f32[120, 480, 1, 1]", arg157_1: "f32[120]", arg158_1: "f32[480, 120, 1, 1]", arg159_1: "f32[480]", arg160_1: "f32[56, 480, 1, 1]", arg161_1: "f32[56]", arg162_1: "f32[56]", arg163_1: "f32[56, 1, 3, 3]", arg164_1: "f32[56]", arg165_1: "f32[56]", arg166_1: "f32[80, 1, 3, 3]", arg167_1: "f32[80]", arg168_1: "f32[80]", arg169_1: "f32[112, 80, 1, 1]", arg170_1: "f32[112]", arg171_1: "f32[112]", arg172_1: "f32[336, 112, 1, 1]", arg173_1: "f32[336]", arg174_1: "f32[336]", arg175_1: "f32[336, 1, 3, 3]", arg176_1: "f32[336]", arg177_1: "f32[336]", arg178_1: "f32[168, 672, 1, 1]", arg179_1: "f32[168]", arg180_1: "f32[672, 168, 1, 1]", arg181_1: "f32[672]", arg182_1: "f32[56, 672, 1, 1]", arg183_1: "f32[56]", arg184_1: "f32[56]", arg185_1: "f32[56, 1, 3, 3]", arg186_1: "f32[56]", arg187_1: "f32[56]", arg188_1: "f32[336, 112, 1, 1]", arg189_1: "f32[336]", arg190_1: "f32[336]", arg191_1: "f32[336, 1, 3, 3]", arg192_1: "f32[336]", arg193_1: "f32[336]", arg194_1: "f32[672, 1, 5, 5]", arg195_1: "f32[672]", arg196_1: "f32[672]", arg197_1: "f32[168, 672, 1, 1]", arg198_1: "f32[168]", arg199_1: "f32[672, 168, 1, 1]", arg200_1: "f32[672]", arg201_1: "f32[80, 672, 1, 1]", arg202_1: "f32[80]", arg203_1: "f32[80]", arg204_1: "f32[80, 1, 3, 3]", arg205_1: "f32[80]", arg206_1: "f32[80]", arg207_1: "f32[112, 1, 5, 5]", arg208_1: "f32[112]", arg209_1: "f32[112]", arg210_1: "f32[160, 112, 1, 1]", arg211_1: "f32[160]", arg212_1: "f32[160]", arg213_1: "f32[480, 160, 1, 1]", arg214_1: "f32[480]", arg215_1: "f32[480]", arg216_1: "f32[480, 1, 3, 3]", arg217_1: "f32[480]", arg218_1: "f32[480]", arg219_1: "f32[80, 960, 1, 1]", arg220_1: "f32[80]", arg221_1: "f32[80]", arg222_1: "f32[80, 1, 3, 3]", arg223_1: "f32[80]", arg224_1: "f32[80]", arg225_1: "f32[480, 160, 1, 1]", arg226_1: "f32[480]", arg227_1: "f32[480]", arg228_1: "f32[480, 1, 3, 3]", arg229_1: "f32[480]", arg230_1: "f32[480]", arg231_1: "f32[240, 960, 1, 1]", arg232_1: "f32[240]", arg233_1: "f32[960, 240, 1, 1]", arg234_1: "f32[960]", arg235_1: "f32[80, 960, 1, 1]", arg236_1: "f32[80]", arg237_1: "f32[80]", arg238_1: "f32[80, 1, 3, 3]", arg239_1: "f32[80]", arg240_1: "f32[80]", arg241_1: "f32[480, 160, 1, 1]", arg242_1: "f32[480]", arg243_1: "f32[480]", arg244_1: "f32[480, 1, 3, 3]", arg245_1: "f32[480]", arg246_1: "f32[480]", arg247_1: "f32[80, 960, 1, 1]", arg248_1: "f32[80]", arg249_1: "f32[80]", arg250_1: "f32[80, 1, 3, 3]", arg251_1: "f32[80]", arg252_1: "f32[80]", arg253_1: "f32[480, 160, 1, 1]", arg254_1: "f32[480]", arg255_1: "f32[480]", arg256_1: "f32[480, 1, 3, 3]", arg257_1: "f32[480]", arg258_1: "f32[480]", arg259_1: "f32[240, 960, 1, 1]", arg260_1: "f32[240]", arg261_1: "f32[960, 240, 1, 1]", arg262_1: "f32[960]", arg263_1: "f32[80, 960, 1, 1]", arg264_1: "f32[80]", arg265_1: "f32[80]", arg266_1: "f32[80, 1, 3, 3]", arg267_1: "f32[80]", arg268_1: "f32[80]", arg269_1: "f32[960, 160, 1, 1]", arg270_1: "f32[1280, 960, 1, 1]", arg271_1: "f32[1280]", arg272_1: "f32[960]", arg273_1: "f32[960]", arg274_1: "f32[16]", arg275_1: "f32[16]", arg276_1: "i64[]", arg277_1: "f32[8]", arg278_1: "f32[8]", arg279_1: "i64[]", arg280_1: "f32[8]", arg281_1: "f32[8]", arg282_1: "i64[]", arg283_1: "f32[8]", arg284_1: "f32[8]", arg285_1: "i64[]", arg286_1: "f32[8]", arg287_1: "f32[8]", arg288_1: "i64[]", arg289_1: "f32[24]", arg290_1: "f32[24]", arg291_1: "i64[]", arg292_1: "f32[24]", arg293_1: "f32[24]", arg294_1: "i64[]", arg295_1: "f32[48]", arg296_1: "f32[48]", arg297_1: "i64[]", arg298_1: "f32[12]", arg299_1: "f32[12]", arg300_1: "i64[]", arg301_1: "f32[12]", arg302_1: "f32[12]", arg303_1: "i64[]", arg304_1: "f32[16]", arg305_1: "f32[16]", arg306_1: "i64[]", arg307_1: "f32[24]", arg308_1: "f32[24]", arg309_1: "i64[]", arg310_1: "f32[36]", arg311_1: "f32[36]", arg312_1: "i64[]", arg313_1: "f32[36]", arg314_1: "f32[36]", arg315_1: "i64[]", arg316_1: "f32[12]", arg317_1: "f32[12]", arg318_1: "i64[]", arg319_1: "f32[12]", arg320_1: "f32[12]", arg321_1: "i64[]", arg322_1: "f32[36]", arg323_1: "f32[36]", arg324_1: "i64[]", arg325_1: "f32[36]", arg326_1: "f32[36]", arg327_1: "i64[]", arg328_1: "f32[72]", arg329_1: "f32[72]", arg330_1: "i64[]", arg331_1: "f32[20]", arg332_1: "f32[20]", arg333_1: "i64[]", arg334_1: "f32[20]", arg335_1: "f32[20]", arg336_1: "i64[]", arg337_1: "f32[24]", arg338_1: "f32[24]", arg339_1: "i64[]", arg340_1: "f32[40]", arg341_1: "f32[40]", arg342_1: "i64[]", arg343_1: "f32[60]", arg344_1: "f32[60]", arg345_1: "i64[]", arg346_1: "f32[60]", arg347_1: "f32[60]", arg348_1: "i64[]", arg349_1: "f32[20]", arg350_1: "f32[20]", arg351_1: "i64[]", arg352_1: "f32[20]", arg353_1: "f32[20]", arg354_1: "i64[]", arg355_1: "f32[120]", arg356_1: "f32[120]", arg357_1: "i64[]", arg358_1: "f32[120]", arg359_1: "f32[120]", arg360_1: "i64[]", arg361_1: "f32[240]", arg362_1: "f32[240]", arg363_1: "i64[]", arg364_1: "f32[40]", arg365_1: "f32[40]", arg366_1: "i64[]", arg367_1: "f32[40]", arg368_1: "f32[40]", arg369_1: "i64[]", arg370_1: "f32[40]", arg371_1: "f32[40]", arg372_1: "i64[]", arg373_1: "f32[80]", arg374_1: "f32[80]", arg375_1: "i64[]", arg376_1: "f32[100]", arg377_1: "f32[100]", arg378_1: "i64[]", arg379_1: "f32[100]", arg380_1: "f32[100]", arg381_1: "i64[]", arg382_1: "f32[40]", arg383_1: "f32[40]", arg384_1: "i64[]", arg385_1: "f32[40]", arg386_1: "f32[40]", arg387_1: "i64[]", arg388_1: "f32[92]", arg389_1: "f32[92]", arg390_1: "i64[]", arg391_1: "f32[92]", arg392_1: "f32[92]", arg393_1: "i64[]", arg394_1: "f32[40]", arg395_1: "f32[40]", arg396_1: "i64[]", arg397_1: "f32[40]", arg398_1: "f32[40]", arg399_1: "i64[]", arg400_1: "f32[92]", arg401_1: "f32[92]", arg402_1: "i64[]", arg403_1: "f32[92]", arg404_1: "f32[92]", arg405_1: "i64[]", arg406_1: "f32[40]", arg407_1: "f32[40]", arg408_1: "i64[]", arg409_1: "f32[40]", arg410_1: "f32[40]", arg411_1: "i64[]", arg412_1: "f32[240]", arg413_1: "f32[240]", arg414_1: "i64[]", arg415_1: "f32[240]", arg416_1: "f32[240]", arg417_1: "i64[]", arg418_1: "f32[56]", arg419_1: "f32[56]", arg420_1: "i64[]", arg421_1: "f32[56]", arg422_1: "f32[56]", arg423_1: "i64[]", arg424_1: "f32[80]", arg425_1: "f32[80]", arg426_1: "i64[]", arg427_1: "f32[112]", arg428_1: "f32[112]", arg429_1: "i64[]", arg430_1: "f32[336]", arg431_1: "f32[336]", arg432_1: "i64[]", arg433_1: "f32[336]", arg434_1: "f32[336]", arg435_1: "i64[]", arg436_1: "f32[56]", arg437_1: "f32[56]", arg438_1: "i64[]", arg439_1: "f32[56]", arg440_1: "f32[56]", arg441_1: "i64[]", arg442_1: "f32[336]", arg443_1: "f32[336]", arg444_1: "i64[]", arg445_1: "f32[336]", arg446_1: "f32[336]", arg447_1: "i64[]", arg448_1: "f32[672]", arg449_1: "f32[672]", arg450_1: "i64[]", arg451_1: "f32[80]", arg452_1: "f32[80]", arg453_1: "i64[]", arg454_1: "f32[80]", arg455_1: "f32[80]", arg456_1: "i64[]", arg457_1: "f32[112]", arg458_1: "f32[112]", arg459_1: "i64[]", arg460_1: "f32[160]", arg461_1: "f32[160]", arg462_1: "i64[]", arg463_1: "f32[480]", arg464_1: "f32[480]", arg465_1: "i64[]", arg466_1: "f32[480]", arg467_1: "f32[480]", arg468_1: "i64[]", arg469_1: "f32[80]", arg470_1: "f32[80]", arg471_1: "i64[]", arg472_1: "f32[80]", arg473_1: "f32[80]", arg474_1: "i64[]", arg475_1: "f32[480]", arg476_1: "f32[480]", arg477_1: "i64[]", arg478_1: "f32[480]", arg479_1: "f32[480]", arg480_1: "i64[]", arg481_1: "f32[80]", arg482_1: "f32[80]", arg483_1: "i64[]", arg484_1: "f32[80]", arg485_1: "f32[80]", arg486_1: "i64[]", arg487_1: "f32[480]", arg488_1: "f32[480]", arg489_1: "i64[]", arg490_1: "f32[480]", arg491_1: "f32[480]", arg492_1: "i64[]", arg493_1: "f32[80]", arg494_1: "f32[80]", arg495_1: "i64[]", arg496_1: "f32[80]", arg497_1: "f32[80]", arg498_1: "i64[]", arg499_1: "f32[480]", arg500_1: "f32[480]", arg501_1: "i64[]", arg502_1: "f32[480]", arg503_1: "f32[480]", arg504_1: "i64[]", arg505_1: "f32[80]", arg506_1: "f32[80]", arg507_1: "i64[]", arg508_1: "f32[80]", arg509_1: "f32[80]", arg510_1: "i64[]", arg511_1: "f32[8, 3, 224, 224]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:282, code: x = self.conv_stem(x)
    convolution: "f32[8, 16, 112, 112]" = torch.ops.aten.convolution.default(arg511_1, arg4_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg511_1 = arg4_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:283, code: x = self.bn1(x)
    unsqueeze: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg274_1, -1);  arg274_1 = None
    unsqueeze_1: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    sub: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1);  convolution = unsqueeze_1 = None
    add: "f32[16]" = torch.ops.aten.add.Tensor(arg275_1, 1e-05);  arg275_1 = None
    sqrt: "f32[16]" = torch.ops.aten.sqrt.default(add);  add = None
    reciprocal: "f32[16]" = torch.ops.aten.reciprocal.default(sqrt);  sqrt = None
    mul: "f32[16]" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
    unsqueeze_2: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(mul, -1);  mul = None
    unsqueeze_3: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    mul_1: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub, unsqueeze_3);  sub = unsqueeze_3 = None
    unsqueeze_4: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
    unsqueeze_5: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_2: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None
    unsqueeze_6: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg6_1, -1);  arg6_1 = None
    unsqueeze_7: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_1: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(mul_2, unsqueeze_7);  mul_2 = unsqueeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:284, code: x = self.act1(x)
    relu: "f32[8, 16, 112, 112]" = torch.ops.aten.relu.default(add_1);  add_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_1: "f32[8, 8, 112, 112]" = torch.ops.aten.convolution.default(relu, arg7_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg7_1 = None
    unsqueeze_8: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(arg277_1, -1);  arg277_1 = None
    unsqueeze_9: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    sub_1: "f32[8, 8, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_9);  convolution_1 = unsqueeze_9 = None
    add_2: "f32[8]" = torch.ops.aten.add.Tensor(arg278_1, 1e-05);  arg278_1 = None
    sqrt_1: "f32[8]" = torch.ops.aten.sqrt.default(add_2);  add_2 = None
    reciprocal_1: "f32[8]" = torch.ops.aten.reciprocal.default(sqrt_1);  sqrt_1 = None
    mul_3: "f32[8]" = torch.ops.aten.mul.Tensor(reciprocal_1, 1);  reciprocal_1 = None
    unsqueeze_10: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(mul_3, -1);  mul_3 = None
    unsqueeze_11: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    mul_4: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(sub_1, unsqueeze_11);  sub_1 = unsqueeze_11 = None
    unsqueeze_12: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(arg8_1, -1);  arg8_1 = None
    unsqueeze_13: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_5: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(mul_4, unsqueeze_13);  mul_4 = unsqueeze_13 = None
    unsqueeze_14: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
    unsqueeze_15: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_3: "f32[8, 8, 112, 112]" = torch.ops.aten.add.Tensor(mul_5, unsqueeze_15);  mul_5 = unsqueeze_15 = None
    relu_1: "f32[8, 8, 112, 112]" = torch.ops.aten.relu.default(add_3);  add_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_2: "f32[8, 8, 112, 112]" = torch.ops.aten.convolution.default(relu_1, arg10_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  arg10_1 = None
    unsqueeze_16: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(arg280_1, -1);  arg280_1 = None
    unsqueeze_17: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    sub_2: "f32[8, 8, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_17);  convolution_2 = unsqueeze_17 = None
    add_4: "f32[8]" = torch.ops.aten.add.Tensor(arg281_1, 1e-05);  arg281_1 = None
    sqrt_2: "f32[8]" = torch.ops.aten.sqrt.default(add_4);  add_4 = None
    reciprocal_2: "f32[8]" = torch.ops.aten.reciprocal.default(sqrt_2);  sqrt_2 = None
    mul_6: "f32[8]" = torch.ops.aten.mul.Tensor(reciprocal_2, 1);  reciprocal_2 = None
    unsqueeze_18: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(mul_6, -1);  mul_6 = None
    unsqueeze_19: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    mul_7: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(sub_2, unsqueeze_19);  sub_2 = unsqueeze_19 = None
    unsqueeze_20: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(arg11_1, -1);  arg11_1 = None
    unsqueeze_21: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_8: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_21);  mul_7 = unsqueeze_21 = None
    unsqueeze_22: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(arg12_1, -1);  arg12_1 = None
    unsqueeze_23: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_5: "f32[8, 8, 112, 112]" = torch.ops.aten.add.Tensor(mul_8, unsqueeze_23);  mul_8 = unsqueeze_23 = None
    relu_2: "f32[8, 8, 112, 112]" = torch.ops.aten.relu.default(add_5);  add_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat: "f32[8, 16, 112, 112]" = torch.ops.aten.cat.default([relu_1, relu_2], 1);  relu_1 = relu_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_3: "f32[8, 8, 112, 112]" = torch.ops.aten.convolution.default(cat, arg13_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat = arg13_1 = None
    unsqueeze_24: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(arg283_1, -1);  arg283_1 = None
    unsqueeze_25: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
    sub_3: "f32[8, 8, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_25);  convolution_3 = unsqueeze_25 = None
    add_6: "f32[8]" = torch.ops.aten.add.Tensor(arg284_1, 1e-05);  arg284_1 = None
    sqrt_3: "f32[8]" = torch.ops.aten.sqrt.default(add_6);  add_6 = None
    reciprocal_3: "f32[8]" = torch.ops.aten.reciprocal.default(sqrt_3);  sqrt_3 = None
    mul_9: "f32[8]" = torch.ops.aten.mul.Tensor(reciprocal_3, 1);  reciprocal_3 = None
    unsqueeze_26: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(mul_9, -1);  mul_9 = None
    unsqueeze_27: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
    mul_10: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(sub_3, unsqueeze_27);  sub_3 = unsqueeze_27 = None
    unsqueeze_28: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
    unsqueeze_29: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    mul_11: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(mul_10, unsqueeze_29);  mul_10 = unsqueeze_29 = None
    unsqueeze_30: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
    unsqueeze_31: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
    add_7: "f32[8, 8, 112, 112]" = torch.ops.aten.add.Tensor(mul_11, unsqueeze_31);  mul_11 = unsqueeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_4: "f32[8, 8, 112, 112]" = torch.ops.aten.convolution.default(add_7, arg16_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  arg16_1 = None
    unsqueeze_32: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(arg286_1, -1);  arg286_1 = None
    unsqueeze_33: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    sub_4: "f32[8, 8, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_33);  convolution_4 = unsqueeze_33 = None
    add_8: "f32[8]" = torch.ops.aten.add.Tensor(arg287_1, 1e-05);  arg287_1 = None
    sqrt_4: "f32[8]" = torch.ops.aten.sqrt.default(add_8);  add_8 = None
    reciprocal_4: "f32[8]" = torch.ops.aten.reciprocal.default(sqrt_4);  sqrt_4 = None
    mul_12: "f32[8]" = torch.ops.aten.mul.Tensor(reciprocal_4, 1);  reciprocal_4 = None
    unsqueeze_34: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(mul_12, -1);  mul_12 = None
    unsqueeze_35: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
    mul_13: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(sub_4, unsqueeze_35);  sub_4 = unsqueeze_35 = None
    unsqueeze_36: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(arg17_1, -1);  arg17_1 = None
    unsqueeze_37: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
    mul_14: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(mul_13, unsqueeze_37);  mul_13 = unsqueeze_37 = None
    unsqueeze_38: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(arg18_1, -1);  arg18_1 = None
    unsqueeze_39: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
    add_9: "f32[8, 8, 112, 112]" = torch.ops.aten.add.Tensor(mul_14, unsqueeze_39);  mul_14 = unsqueeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat_1: "f32[8, 16, 112, 112]" = torch.ops.aten.cat.default([add_7, add_9], 1);  add_7 = add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    add_10: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(cat_1, relu);  cat_1 = relu = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_5: "f32[8, 24, 112, 112]" = torch.ops.aten.convolution.default(add_10, arg19_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg19_1 = None
    unsqueeze_40: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg289_1, -1);  arg289_1 = None
    unsqueeze_41: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    sub_5: "f32[8, 24, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_41);  convolution_5 = unsqueeze_41 = None
    add_11: "f32[24]" = torch.ops.aten.add.Tensor(arg290_1, 1e-05);  arg290_1 = None
    sqrt_5: "f32[24]" = torch.ops.aten.sqrt.default(add_11);  add_11 = None
    reciprocal_5: "f32[24]" = torch.ops.aten.reciprocal.default(sqrt_5);  sqrt_5 = None
    mul_15: "f32[24]" = torch.ops.aten.mul.Tensor(reciprocal_5, 1);  reciprocal_5 = None
    unsqueeze_42: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(mul_15, -1);  mul_15 = None
    unsqueeze_43: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    mul_16: "f32[8, 24, 112, 112]" = torch.ops.aten.mul.Tensor(sub_5, unsqueeze_43);  sub_5 = unsqueeze_43 = None
    unsqueeze_44: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg20_1, -1);  arg20_1 = None
    unsqueeze_45: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    mul_17: "f32[8, 24, 112, 112]" = torch.ops.aten.mul.Tensor(mul_16, unsqueeze_45);  mul_16 = unsqueeze_45 = None
    unsqueeze_46: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg21_1, -1);  arg21_1 = None
    unsqueeze_47: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
    add_12: "f32[8, 24, 112, 112]" = torch.ops.aten.add.Tensor(mul_17, unsqueeze_47);  mul_17 = unsqueeze_47 = None
    relu_3: "f32[8, 24, 112, 112]" = torch.ops.aten.relu.default(add_12);  add_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_6: "f32[8, 24, 112, 112]" = torch.ops.aten.convolution.default(relu_3, arg22_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 24);  arg22_1 = None
    unsqueeze_48: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg292_1, -1);  arg292_1 = None
    unsqueeze_49: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
    sub_6: "f32[8, 24, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_49);  convolution_6 = unsqueeze_49 = None
    add_13: "f32[24]" = torch.ops.aten.add.Tensor(arg293_1, 1e-05);  arg293_1 = None
    sqrt_6: "f32[24]" = torch.ops.aten.sqrt.default(add_13);  add_13 = None
    reciprocal_6: "f32[24]" = torch.ops.aten.reciprocal.default(sqrt_6);  sqrt_6 = None
    mul_18: "f32[24]" = torch.ops.aten.mul.Tensor(reciprocal_6, 1);  reciprocal_6 = None
    unsqueeze_50: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(mul_18, -1);  mul_18 = None
    unsqueeze_51: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
    mul_19: "f32[8, 24, 112, 112]" = torch.ops.aten.mul.Tensor(sub_6, unsqueeze_51);  sub_6 = unsqueeze_51 = None
    unsqueeze_52: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg23_1, -1);  arg23_1 = None
    unsqueeze_53: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    mul_20: "f32[8, 24, 112, 112]" = torch.ops.aten.mul.Tensor(mul_19, unsqueeze_53);  mul_19 = unsqueeze_53 = None
    unsqueeze_54: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg24_1, -1);  arg24_1 = None
    unsqueeze_55: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
    add_14: "f32[8, 24, 112, 112]" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_55);  mul_20 = unsqueeze_55 = None
    relu_4: "f32[8, 24, 112, 112]" = torch.ops.aten.relu.default(add_14);  add_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat_2: "f32[8, 48, 112, 112]" = torch.ops.aten.cat.default([relu_3, relu_4], 1);  relu_3 = relu_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:172, code: x = self.conv_dw(x)
    convolution_7: "f32[8, 48, 56, 56]" = torch.ops.aten.convolution.default(cat_2, arg25_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 48);  cat_2 = arg25_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:173, code: x = self.bn_dw(x)
    unsqueeze_56: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(arg295_1, -1);  arg295_1 = None
    unsqueeze_57: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    sub_7: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_57);  convolution_7 = unsqueeze_57 = None
    add_15: "f32[48]" = torch.ops.aten.add.Tensor(arg296_1, 1e-05);  arg296_1 = None
    sqrt_7: "f32[48]" = torch.ops.aten.sqrt.default(add_15);  add_15 = None
    reciprocal_7: "f32[48]" = torch.ops.aten.reciprocal.default(sqrt_7);  sqrt_7 = None
    mul_21: "f32[48]" = torch.ops.aten.mul.Tensor(reciprocal_7, 1);  reciprocal_7 = None
    unsqueeze_58: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(mul_21, -1);  mul_21 = None
    unsqueeze_59: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
    mul_22: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(sub_7, unsqueeze_59);  sub_7 = unsqueeze_59 = None
    unsqueeze_60: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(arg26_1, -1);  arg26_1 = None
    unsqueeze_61: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
    mul_23: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(mul_22, unsqueeze_61);  mul_22 = unsqueeze_61 = None
    unsqueeze_62: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(arg27_1, -1);  arg27_1 = None
    unsqueeze_63: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
    add_16: "f32[8, 48, 56, 56]" = torch.ops.aten.add.Tensor(mul_23, unsqueeze_63);  mul_23 = unsqueeze_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_8: "f32[8, 12, 56, 56]" = torch.ops.aten.convolution.default(add_16, arg28_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_16 = arg28_1 = None
    unsqueeze_64: "f32[12, 1]" = torch.ops.aten.unsqueeze.default(arg298_1, -1);  arg298_1 = None
    unsqueeze_65: "f32[12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
    sub_8: "f32[8, 12, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_65);  convolution_8 = unsqueeze_65 = None
    add_17: "f32[12]" = torch.ops.aten.add.Tensor(arg299_1, 1e-05);  arg299_1 = None
    sqrt_8: "f32[12]" = torch.ops.aten.sqrt.default(add_17);  add_17 = None
    reciprocal_8: "f32[12]" = torch.ops.aten.reciprocal.default(sqrt_8);  sqrt_8 = None
    mul_24: "f32[12]" = torch.ops.aten.mul.Tensor(reciprocal_8, 1);  reciprocal_8 = None
    unsqueeze_66: "f32[12, 1]" = torch.ops.aten.unsqueeze.default(mul_24, -1);  mul_24 = None
    unsqueeze_67: "f32[12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
    mul_25: "f32[8, 12, 56, 56]" = torch.ops.aten.mul.Tensor(sub_8, unsqueeze_67);  sub_8 = unsqueeze_67 = None
    unsqueeze_68: "f32[12, 1]" = torch.ops.aten.unsqueeze.default(arg29_1, -1);  arg29_1 = None
    unsqueeze_69: "f32[12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    mul_26: "f32[8, 12, 56, 56]" = torch.ops.aten.mul.Tensor(mul_25, unsqueeze_69);  mul_25 = unsqueeze_69 = None
    unsqueeze_70: "f32[12, 1]" = torch.ops.aten.unsqueeze.default(arg30_1, -1);  arg30_1 = None
    unsqueeze_71: "f32[12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    add_18: "f32[8, 12, 56, 56]" = torch.ops.aten.add.Tensor(mul_26, unsqueeze_71);  mul_26 = unsqueeze_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_9: "f32[8, 12, 56, 56]" = torch.ops.aten.convolution.default(add_18, arg31_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 12);  arg31_1 = None
    unsqueeze_72: "f32[12, 1]" = torch.ops.aten.unsqueeze.default(arg301_1, -1);  arg301_1 = None
    unsqueeze_73: "f32[12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
    sub_9: "f32[8, 12, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_73);  convolution_9 = unsqueeze_73 = None
    add_19: "f32[12]" = torch.ops.aten.add.Tensor(arg302_1, 1e-05);  arg302_1 = None
    sqrt_9: "f32[12]" = torch.ops.aten.sqrt.default(add_19);  add_19 = None
    reciprocal_9: "f32[12]" = torch.ops.aten.reciprocal.default(sqrt_9);  sqrt_9 = None
    mul_27: "f32[12]" = torch.ops.aten.mul.Tensor(reciprocal_9, 1);  reciprocal_9 = None
    unsqueeze_74: "f32[12, 1]" = torch.ops.aten.unsqueeze.default(mul_27, -1);  mul_27 = None
    unsqueeze_75: "f32[12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
    mul_28: "f32[8, 12, 56, 56]" = torch.ops.aten.mul.Tensor(sub_9, unsqueeze_75);  sub_9 = unsqueeze_75 = None
    unsqueeze_76: "f32[12, 1]" = torch.ops.aten.unsqueeze.default(arg32_1, -1);  arg32_1 = None
    unsqueeze_77: "f32[12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
    mul_29: "f32[8, 12, 56, 56]" = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_77);  mul_28 = unsqueeze_77 = None
    unsqueeze_78: "f32[12, 1]" = torch.ops.aten.unsqueeze.default(arg33_1, -1);  arg33_1 = None
    unsqueeze_79: "f32[12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    add_20: "f32[8, 12, 56, 56]" = torch.ops.aten.add.Tensor(mul_29, unsqueeze_79);  mul_29 = unsqueeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat_3: "f32[8, 24, 56, 56]" = torch.ops.aten.cat.default([add_18, add_20], 1);  add_18 = add_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    convolution_10: "f32[8, 16, 56, 56]" = torch.ops.aten.convolution.default(add_10, arg34_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 16);  add_10 = arg34_1 = None
    unsqueeze_80: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg304_1, -1);  arg304_1 = None
    unsqueeze_81: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
    sub_10: "f32[8, 16, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_81);  convolution_10 = unsqueeze_81 = None
    add_21: "f32[16]" = torch.ops.aten.add.Tensor(arg305_1, 1e-05);  arg305_1 = None
    sqrt_10: "f32[16]" = torch.ops.aten.sqrt.default(add_21);  add_21 = None
    reciprocal_10: "f32[16]" = torch.ops.aten.reciprocal.default(sqrt_10);  sqrt_10 = None
    mul_30: "f32[16]" = torch.ops.aten.mul.Tensor(reciprocal_10, 1);  reciprocal_10 = None
    unsqueeze_82: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(mul_30, -1);  mul_30 = None
    unsqueeze_83: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    mul_31: "f32[8, 16, 56, 56]" = torch.ops.aten.mul.Tensor(sub_10, unsqueeze_83);  sub_10 = unsqueeze_83 = None
    unsqueeze_84: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg35_1, -1);  arg35_1 = None
    unsqueeze_85: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    mul_32: "f32[8, 16, 56, 56]" = torch.ops.aten.mul.Tensor(mul_31, unsqueeze_85);  mul_31 = unsqueeze_85 = None
    unsqueeze_86: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg36_1, -1);  arg36_1 = None
    unsqueeze_87: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
    add_22: "f32[8, 16, 56, 56]" = torch.ops.aten.add.Tensor(mul_32, unsqueeze_87);  mul_32 = unsqueeze_87 = None
    convolution_11: "f32[8, 24, 56, 56]" = torch.ops.aten.convolution.default(add_22, arg37_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_22 = arg37_1 = None
    unsqueeze_88: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg307_1, -1);  arg307_1 = None
    unsqueeze_89: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
    sub_11: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_89);  convolution_11 = unsqueeze_89 = None
    add_23: "f32[24]" = torch.ops.aten.add.Tensor(arg308_1, 1e-05);  arg308_1 = None
    sqrt_11: "f32[24]" = torch.ops.aten.sqrt.default(add_23);  add_23 = None
    reciprocal_11: "f32[24]" = torch.ops.aten.reciprocal.default(sqrt_11);  sqrt_11 = None
    mul_33: "f32[24]" = torch.ops.aten.mul.Tensor(reciprocal_11, 1);  reciprocal_11 = None
    unsqueeze_90: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(mul_33, -1);  mul_33 = None
    unsqueeze_91: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    mul_34: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_11, unsqueeze_91);  sub_11 = unsqueeze_91 = None
    unsqueeze_92: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg38_1, -1);  arg38_1 = None
    unsqueeze_93: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
    mul_35: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(mul_34, unsqueeze_93);  mul_34 = unsqueeze_93 = None
    unsqueeze_94: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg39_1, -1);  arg39_1 = None
    unsqueeze_95: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
    add_24: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(mul_35, unsqueeze_95);  mul_35 = unsqueeze_95 = None
    add_25: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(cat_3, add_24);  cat_3 = add_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_12: "f32[8, 36, 56, 56]" = torch.ops.aten.convolution.default(add_25, arg40_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg40_1 = None
    unsqueeze_96: "f32[36, 1]" = torch.ops.aten.unsqueeze.default(arg310_1, -1);  arg310_1 = None
    unsqueeze_97: "f32[36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
    sub_12: "f32[8, 36, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_97);  convolution_12 = unsqueeze_97 = None
    add_26: "f32[36]" = torch.ops.aten.add.Tensor(arg311_1, 1e-05);  arg311_1 = None
    sqrt_12: "f32[36]" = torch.ops.aten.sqrt.default(add_26);  add_26 = None
    reciprocal_12: "f32[36]" = torch.ops.aten.reciprocal.default(sqrt_12);  sqrt_12 = None
    mul_36: "f32[36]" = torch.ops.aten.mul.Tensor(reciprocal_12, 1);  reciprocal_12 = None
    unsqueeze_98: "f32[36, 1]" = torch.ops.aten.unsqueeze.default(mul_36, -1);  mul_36 = None
    unsqueeze_99: "f32[36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
    mul_37: "f32[8, 36, 56, 56]" = torch.ops.aten.mul.Tensor(sub_12, unsqueeze_99);  sub_12 = unsqueeze_99 = None
    unsqueeze_100: "f32[36, 1]" = torch.ops.aten.unsqueeze.default(arg41_1, -1);  arg41_1 = None
    unsqueeze_101: "f32[36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
    mul_38: "f32[8, 36, 56, 56]" = torch.ops.aten.mul.Tensor(mul_37, unsqueeze_101);  mul_37 = unsqueeze_101 = None
    unsqueeze_102: "f32[36, 1]" = torch.ops.aten.unsqueeze.default(arg42_1, -1);  arg42_1 = None
    unsqueeze_103: "f32[36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    add_27: "f32[8, 36, 56, 56]" = torch.ops.aten.add.Tensor(mul_38, unsqueeze_103);  mul_38 = unsqueeze_103 = None
    relu_5: "f32[8, 36, 56, 56]" = torch.ops.aten.relu.default(add_27);  add_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_13: "f32[8, 36, 56, 56]" = torch.ops.aten.convolution.default(relu_5, arg43_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 36);  arg43_1 = None
    unsqueeze_104: "f32[36, 1]" = torch.ops.aten.unsqueeze.default(arg313_1, -1);  arg313_1 = None
    unsqueeze_105: "f32[36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
    sub_13: "f32[8, 36, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_105);  convolution_13 = unsqueeze_105 = None
    add_28: "f32[36]" = torch.ops.aten.add.Tensor(arg314_1, 1e-05);  arg314_1 = None
    sqrt_13: "f32[36]" = torch.ops.aten.sqrt.default(add_28);  add_28 = None
    reciprocal_13: "f32[36]" = torch.ops.aten.reciprocal.default(sqrt_13);  sqrt_13 = None
    mul_39: "f32[36]" = torch.ops.aten.mul.Tensor(reciprocal_13, 1);  reciprocal_13 = None
    unsqueeze_106: "f32[36, 1]" = torch.ops.aten.unsqueeze.default(mul_39, -1);  mul_39 = None
    unsqueeze_107: "f32[36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
    mul_40: "f32[8, 36, 56, 56]" = torch.ops.aten.mul.Tensor(sub_13, unsqueeze_107);  sub_13 = unsqueeze_107 = None
    unsqueeze_108: "f32[36, 1]" = torch.ops.aten.unsqueeze.default(arg44_1, -1);  arg44_1 = None
    unsqueeze_109: "f32[36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
    mul_41: "f32[8, 36, 56, 56]" = torch.ops.aten.mul.Tensor(mul_40, unsqueeze_109);  mul_40 = unsqueeze_109 = None
    unsqueeze_110: "f32[36, 1]" = torch.ops.aten.unsqueeze.default(arg45_1, -1);  arg45_1 = None
    unsqueeze_111: "f32[36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
    add_29: "f32[8, 36, 56, 56]" = torch.ops.aten.add.Tensor(mul_41, unsqueeze_111);  mul_41 = unsqueeze_111 = None
    relu_6: "f32[8, 36, 56, 56]" = torch.ops.aten.relu.default(add_29);  add_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat_4: "f32[8, 72, 56, 56]" = torch.ops.aten.cat.default([relu_5, relu_6], 1);  relu_5 = relu_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_14: "f32[8, 12, 56, 56]" = torch.ops.aten.convolution.default(cat_4, arg46_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_4 = arg46_1 = None
    unsqueeze_112: "f32[12, 1]" = torch.ops.aten.unsqueeze.default(arg316_1, -1);  arg316_1 = None
    unsqueeze_113: "f32[12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
    sub_14: "f32[8, 12, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_113);  convolution_14 = unsqueeze_113 = None
    add_30: "f32[12]" = torch.ops.aten.add.Tensor(arg317_1, 1e-05);  arg317_1 = None
    sqrt_14: "f32[12]" = torch.ops.aten.sqrt.default(add_30);  add_30 = None
    reciprocal_14: "f32[12]" = torch.ops.aten.reciprocal.default(sqrt_14);  sqrt_14 = None
    mul_42: "f32[12]" = torch.ops.aten.mul.Tensor(reciprocal_14, 1);  reciprocal_14 = None
    unsqueeze_114: "f32[12, 1]" = torch.ops.aten.unsqueeze.default(mul_42, -1);  mul_42 = None
    unsqueeze_115: "f32[12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
    mul_43: "f32[8, 12, 56, 56]" = torch.ops.aten.mul.Tensor(sub_14, unsqueeze_115);  sub_14 = unsqueeze_115 = None
    unsqueeze_116: "f32[12, 1]" = torch.ops.aten.unsqueeze.default(arg47_1, -1);  arg47_1 = None
    unsqueeze_117: "f32[12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
    mul_44: "f32[8, 12, 56, 56]" = torch.ops.aten.mul.Tensor(mul_43, unsqueeze_117);  mul_43 = unsqueeze_117 = None
    unsqueeze_118: "f32[12, 1]" = torch.ops.aten.unsqueeze.default(arg48_1, -1);  arg48_1 = None
    unsqueeze_119: "f32[12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
    add_31: "f32[8, 12, 56, 56]" = torch.ops.aten.add.Tensor(mul_44, unsqueeze_119);  mul_44 = unsqueeze_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_15: "f32[8, 12, 56, 56]" = torch.ops.aten.convolution.default(add_31, arg49_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 12);  arg49_1 = None
    unsqueeze_120: "f32[12, 1]" = torch.ops.aten.unsqueeze.default(arg319_1, -1);  arg319_1 = None
    unsqueeze_121: "f32[12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
    sub_15: "f32[8, 12, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_121);  convolution_15 = unsqueeze_121 = None
    add_32: "f32[12]" = torch.ops.aten.add.Tensor(arg320_1, 1e-05);  arg320_1 = None
    sqrt_15: "f32[12]" = torch.ops.aten.sqrt.default(add_32);  add_32 = None
    reciprocal_15: "f32[12]" = torch.ops.aten.reciprocal.default(sqrt_15);  sqrt_15 = None
    mul_45: "f32[12]" = torch.ops.aten.mul.Tensor(reciprocal_15, 1);  reciprocal_15 = None
    unsqueeze_122: "f32[12, 1]" = torch.ops.aten.unsqueeze.default(mul_45, -1);  mul_45 = None
    unsqueeze_123: "f32[12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
    mul_46: "f32[8, 12, 56, 56]" = torch.ops.aten.mul.Tensor(sub_15, unsqueeze_123);  sub_15 = unsqueeze_123 = None
    unsqueeze_124: "f32[12, 1]" = torch.ops.aten.unsqueeze.default(arg50_1, -1);  arg50_1 = None
    unsqueeze_125: "f32[12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
    mul_47: "f32[8, 12, 56, 56]" = torch.ops.aten.mul.Tensor(mul_46, unsqueeze_125);  mul_46 = unsqueeze_125 = None
    unsqueeze_126: "f32[12, 1]" = torch.ops.aten.unsqueeze.default(arg51_1, -1);  arg51_1 = None
    unsqueeze_127: "f32[12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
    add_33: "f32[8, 12, 56, 56]" = torch.ops.aten.add.Tensor(mul_47, unsqueeze_127);  mul_47 = unsqueeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat_5: "f32[8, 24, 56, 56]" = torch.ops.aten.cat.default([add_31, add_33], 1);  add_31 = add_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    add_34: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(cat_5, add_25);  cat_5 = add_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_16: "f32[8, 36, 56, 56]" = torch.ops.aten.convolution.default(add_34, arg52_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg52_1 = None
    unsqueeze_128: "f32[36, 1]" = torch.ops.aten.unsqueeze.default(arg322_1, -1);  arg322_1 = None
    unsqueeze_129: "f32[36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
    sub_16: "f32[8, 36, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_129);  convolution_16 = unsqueeze_129 = None
    add_35: "f32[36]" = torch.ops.aten.add.Tensor(arg323_1, 1e-05);  arg323_1 = None
    sqrt_16: "f32[36]" = torch.ops.aten.sqrt.default(add_35);  add_35 = None
    reciprocal_16: "f32[36]" = torch.ops.aten.reciprocal.default(sqrt_16);  sqrt_16 = None
    mul_48: "f32[36]" = torch.ops.aten.mul.Tensor(reciprocal_16, 1);  reciprocal_16 = None
    unsqueeze_130: "f32[36, 1]" = torch.ops.aten.unsqueeze.default(mul_48, -1);  mul_48 = None
    unsqueeze_131: "f32[36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
    mul_49: "f32[8, 36, 56, 56]" = torch.ops.aten.mul.Tensor(sub_16, unsqueeze_131);  sub_16 = unsqueeze_131 = None
    unsqueeze_132: "f32[36, 1]" = torch.ops.aten.unsqueeze.default(arg53_1, -1);  arg53_1 = None
    unsqueeze_133: "f32[36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
    mul_50: "f32[8, 36, 56, 56]" = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_133);  mul_49 = unsqueeze_133 = None
    unsqueeze_134: "f32[36, 1]" = torch.ops.aten.unsqueeze.default(arg54_1, -1);  arg54_1 = None
    unsqueeze_135: "f32[36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
    add_36: "f32[8, 36, 56, 56]" = torch.ops.aten.add.Tensor(mul_50, unsqueeze_135);  mul_50 = unsqueeze_135 = None
    relu_7: "f32[8, 36, 56, 56]" = torch.ops.aten.relu.default(add_36);  add_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_17: "f32[8, 36, 56, 56]" = torch.ops.aten.convolution.default(relu_7, arg55_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 36);  arg55_1 = None
    unsqueeze_136: "f32[36, 1]" = torch.ops.aten.unsqueeze.default(arg325_1, -1);  arg325_1 = None
    unsqueeze_137: "f32[36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
    sub_17: "f32[8, 36, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_137);  convolution_17 = unsqueeze_137 = None
    add_37: "f32[36]" = torch.ops.aten.add.Tensor(arg326_1, 1e-05);  arg326_1 = None
    sqrt_17: "f32[36]" = torch.ops.aten.sqrt.default(add_37);  add_37 = None
    reciprocal_17: "f32[36]" = torch.ops.aten.reciprocal.default(sqrt_17);  sqrt_17 = None
    mul_51: "f32[36]" = torch.ops.aten.mul.Tensor(reciprocal_17, 1);  reciprocal_17 = None
    unsqueeze_138: "f32[36, 1]" = torch.ops.aten.unsqueeze.default(mul_51, -1);  mul_51 = None
    unsqueeze_139: "f32[36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
    mul_52: "f32[8, 36, 56, 56]" = torch.ops.aten.mul.Tensor(sub_17, unsqueeze_139);  sub_17 = unsqueeze_139 = None
    unsqueeze_140: "f32[36, 1]" = torch.ops.aten.unsqueeze.default(arg56_1, -1);  arg56_1 = None
    unsqueeze_141: "f32[36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
    mul_53: "f32[8, 36, 56, 56]" = torch.ops.aten.mul.Tensor(mul_52, unsqueeze_141);  mul_52 = unsqueeze_141 = None
    unsqueeze_142: "f32[36, 1]" = torch.ops.aten.unsqueeze.default(arg57_1, -1);  arg57_1 = None
    unsqueeze_143: "f32[36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
    add_38: "f32[8, 36, 56, 56]" = torch.ops.aten.add.Tensor(mul_53, unsqueeze_143);  mul_53 = unsqueeze_143 = None
    relu_8: "f32[8, 36, 56, 56]" = torch.ops.aten.relu.default(add_38);  add_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat_6: "f32[8, 72, 56, 56]" = torch.ops.aten.cat.default([relu_7, relu_8], 1);  relu_7 = relu_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:172, code: x = self.conv_dw(x)
    convolution_18: "f32[8, 72, 28, 28]" = torch.ops.aten.convolution.default(cat_6, arg58_1, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 72);  cat_6 = arg58_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:173, code: x = self.bn_dw(x)
    unsqueeze_144: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(arg328_1, -1);  arg328_1 = None
    unsqueeze_145: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
    sub_18: "f32[8, 72, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_145);  convolution_18 = unsqueeze_145 = None
    add_39: "f32[72]" = torch.ops.aten.add.Tensor(arg329_1, 1e-05);  arg329_1 = None
    sqrt_18: "f32[72]" = torch.ops.aten.sqrt.default(add_39);  add_39 = None
    reciprocal_18: "f32[72]" = torch.ops.aten.reciprocal.default(sqrt_18);  sqrt_18 = None
    mul_54: "f32[72]" = torch.ops.aten.mul.Tensor(reciprocal_18, 1);  reciprocal_18 = None
    unsqueeze_146: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(mul_54, -1);  mul_54 = None
    unsqueeze_147: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
    mul_55: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(sub_18, unsqueeze_147);  sub_18 = unsqueeze_147 = None
    unsqueeze_148: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(arg59_1, -1);  arg59_1 = None
    unsqueeze_149: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
    mul_56: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(mul_55, unsqueeze_149);  mul_55 = unsqueeze_149 = None
    unsqueeze_150: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(arg60_1, -1);  arg60_1 = None
    unsqueeze_151: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
    add_40: "f32[8, 72, 28, 28]" = torch.ops.aten.add.Tensor(mul_56, unsqueeze_151);  mul_56 = unsqueeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean: "f32[8, 72, 1, 1]" = torch.ops.aten.mean.dim(add_40, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_19: "f32[8, 20, 1, 1]" = torch.ops.aten.convolution.default(mean, arg61_1, arg62_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean = arg61_1 = arg62_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    relu_9: "f32[8, 20, 1, 1]" = torch.ops.aten.relu.default(convolution_19);  convolution_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_20: "f32[8, 72, 1, 1]" = torch.ops.aten.convolution.default(relu_9, arg63_1, arg64_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_9 = arg63_1 = arg64_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    add_41: "f32[8, 72, 1, 1]" = torch.ops.aten.add.Tensor(convolution_20, 3);  convolution_20 = None
    clamp_min: "f32[8, 72, 1, 1]" = torch.ops.aten.clamp_min.default(add_41, 0);  add_41 = None
    clamp_max: "f32[8, 72, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min, 6);  clamp_min = None
    div: "f32[8, 72, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max, 6);  clamp_max = None
    mul_57: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(add_40, div);  add_40 = div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_21: "f32[8, 20, 28, 28]" = torch.ops.aten.convolution.default(mul_57, arg65_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_57 = arg65_1 = None
    unsqueeze_152: "f32[20, 1]" = torch.ops.aten.unsqueeze.default(arg331_1, -1);  arg331_1 = None
    unsqueeze_153: "f32[20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
    sub_19: "f32[8, 20, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_153);  convolution_21 = unsqueeze_153 = None
    add_42: "f32[20]" = torch.ops.aten.add.Tensor(arg332_1, 1e-05);  arg332_1 = None
    sqrt_19: "f32[20]" = torch.ops.aten.sqrt.default(add_42);  add_42 = None
    reciprocal_19: "f32[20]" = torch.ops.aten.reciprocal.default(sqrt_19);  sqrt_19 = None
    mul_58: "f32[20]" = torch.ops.aten.mul.Tensor(reciprocal_19, 1);  reciprocal_19 = None
    unsqueeze_154: "f32[20, 1]" = torch.ops.aten.unsqueeze.default(mul_58, -1);  mul_58 = None
    unsqueeze_155: "f32[20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
    mul_59: "f32[8, 20, 28, 28]" = torch.ops.aten.mul.Tensor(sub_19, unsqueeze_155);  sub_19 = unsqueeze_155 = None
    unsqueeze_156: "f32[20, 1]" = torch.ops.aten.unsqueeze.default(arg66_1, -1);  arg66_1 = None
    unsqueeze_157: "f32[20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
    mul_60: "f32[8, 20, 28, 28]" = torch.ops.aten.mul.Tensor(mul_59, unsqueeze_157);  mul_59 = unsqueeze_157 = None
    unsqueeze_158: "f32[20, 1]" = torch.ops.aten.unsqueeze.default(arg67_1, -1);  arg67_1 = None
    unsqueeze_159: "f32[20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
    add_43: "f32[8, 20, 28, 28]" = torch.ops.aten.add.Tensor(mul_60, unsqueeze_159);  mul_60 = unsqueeze_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_22: "f32[8, 20, 28, 28]" = torch.ops.aten.convolution.default(add_43, arg68_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 20);  arg68_1 = None
    unsqueeze_160: "f32[20, 1]" = torch.ops.aten.unsqueeze.default(arg334_1, -1);  arg334_1 = None
    unsqueeze_161: "f32[20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
    sub_20: "f32[8, 20, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_161);  convolution_22 = unsqueeze_161 = None
    add_44: "f32[20]" = torch.ops.aten.add.Tensor(arg335_1, 1e-05);  arg335_1 = None
    sqrt_20: "f32[20]" = torch.ops.aten.sqrt.default(add_44);  add_44 = None
    reciprocal_20: "f32[20]" = torch.ops.aten.reciprocal.default(sqrt_20);  sqrt_20 = None
    mul_61: "f32[20]" = torch.ops.aten.mul.Tensor(reciprocal_20, 1);  reciprocal_20 = None
    unsqueeze_162: "f32[20, 1]" = torch.ops.aten.unsqueeze.default(mul_61, -1);  mul_61 = None
    unsqueeze_163: "f32[20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
    mul_62: "f32[8, 20, 28, 28]" = torch.ops.aten.mul.Tensor(sub_20, unsqueeze_163);  sub_20 = unsqueeze_163 = None
    unsqueeze_164: "f32[20, 1]" = torch.ops.aten.unsqueeze.default(arg69_1, -1);  arg69_1 = None
    unsqueeze_165: "f32[20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
    mul_63: "f32[8, 20, 28, 28]" = torch.ops.aten.mul.Tensor(mul_62, unsqueeze_165);  mul_62 = unsqueeze_165 = None
    unsqueeze_166: "f32[20, 1]" = torch.ops.aten.unsqueeze.default(arg70_1, -1);  arg70_1 = None
    unsqueeze_167: "f32[20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
    add_45: "f32[8, 20, 28, 28]" = torch.ops.aten.add.Tensor(mul_63, unsqueeze_167);  mul_63 = unsqueeze_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat_7: "f32[8, 40, 28, 28]" = torch.ops.aten.cat.default([add_43, add_45], 1);  add_43 = add_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    convolution_23: "f32[8, 24, 28, 28]" = torch.ops.aten.convolution.default(add_34, arg71_1, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 24);  add_34 = arg71_1 = None
    unsqueeze_168: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg337_1, -1);  arg337_1 = None
    unsqueeze_169: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
    sub_21: "f32[8, 24, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_169);  convolution_23 = unsqueeze_169 = None
    add_46: "f32[24]" = torch.ops.aten.add.Tensor(arg338_1, 1e-05);  arg338_1 = None
    sqrt_21: "f32[24]" = torch.ops.aten.sqrt.default(add_46);  add_46 = None
    reciprocal_21: "f32[24]" = torch.ops.aten.reciprocal.default(sqrt_21);  sqrt_21 = None
    mul_64: "f32[24]" = torch.ops.aten.mul.Tensor(reciprocal_21, 1);  reciprocal_21 = None
    unsqueeze_170: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(mul_64, -1);  mul_64 = None
    unsqueeze_171: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
    mul_65: "f32[8, 24, 28, 28]" = torch.ops.aten.mul.Tensor(sub_21, unsqueeze_171);  sub_21 = unsqueeze_171 = None
    unsqueeze_172: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg72_1, -1);  arg72_1 = None
    unsqueeze_173: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
    mul_66: "f32[8, 24, 28, 28]" = torch.ops.aten.mul.Tensor(mul_65, unsqueeze_173);  mul_65 = unsqueeze_173 = None
    unsqueeze_174: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg73_1, -1);  arg73_1 = None
    unsqueeze_175: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
    add_47: "f32[8, 24, 28, 28]" = torch.ops.aten.add.Tensor(mul_66, unsqueeze_175);  mul_66 = unsqueeze_175 = None
    convolution_24: "f32[8, 40, 28, 28]" = torch.ops.aten.convolution.default(add_47, arg74_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_47 = arg74_1 = None
    unsqueeze_176: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg340_1, -1);  arg340_1 = None
    unsqueeze_177: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
    sub_22: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_177);  convolution_24 = unsqueeze_177 = None
    add_48: "f32[40]" = torch.ops.aten.add.Tensor(arg341_1, 1e-05);  arg341_1 = None
    sqrt_22: "f32[40]" = torch.ops.aten.sqrt.default(add_48);  add_48 = None
    reciprocal_22: "f32[40]" = torch.ops.aten.reciprocal.default(sqrt_22);  sqrt_22 = None
    mul_67: "f32[40]" = torch.ops.aten.mul.Tensor(reciprocal_22, 1);  reciprocal_22 = None
    unsqueeze_178: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(mul_67, -1);  mul_67 = None
    unsqueeze_179: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
    mul_68: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_22, unsqueeze_179);  sub_22 = unsqueeze_179 = None
    unsqueeze_180: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg75_1, -1);  arg75_1 = None
    unsqueeze_181: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
    mul_69: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(mul_68, unsqueeze_181);  mul_68 = unsqueeze_181 = None
    unsqueeze_182: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg76_1, -1);  arg76_1 = None
    unsqueeze_183: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
    add_49: "f32[8, 40, 28, 28]" = torch.ops.aten.add.Tensor(mul_69, unsqueeze_183);  mul_69 = unsqueeze_183 = None
    add_50: "f32[8, 40, 28, 28]" = torch.ops.aten.add.Tensor(cat_7, add_49);  cat_7 = add_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_25: "f32[8, 60, 28, 28]" = torch.ops.aten.convolution.default(add_50, arg77_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg77_1 = None
    unsqueeze_184: "f32[60, 1]" = torch.ops.aten.unsqueeze.default(arg343_1, -1);  arg343_1 = None
    unsqueeze_185: "f32[60, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
    sub_23: "f32[8, 60, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_185);  convolution_25 = unsqueeze_185 = None
    add_51: "f32[60]" = torch.ops.aten.add.Tensor(arg344_1, 1e-05);  arg344_1 = None
    sqrt_23: "f32[60]" = torch.ops.aten.sqrt.default(add_51);  add_51 = None
    reciprocal_23: "f32[60]" = torch.ops.aten.reciprocal.default(sqrt_23);  sqrt_23 = None
    mul_70: "f32[60]" = torch.ops.aten.mul.Tensor(reciprocal_23, 1);  reciprocal_23 = None
    unsqueeze_186: "f32[60, 1]" = torch.ops.aten.unsqueeze.default(mul_70, -1);  mul_70 = None
    unsqueeze_187: "f32[60, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
    mul_71: "f32[8, 60, 28, 28]" = torch.ops.aten.mul.Tensor(sub_23, unsqueeze_187);  sub_23 = unsqueeze_187 = None
    unsqueeze_188: "f32[60, 1]" = torch.ops.aten.unsqueeze.default(arg78_1, -1);  arg78_1 = None
    unsqueeze_189: "f32[60, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
    mul_72: "f32[8, 60, 28, 28]" = torch.ops.aten.mul.Tensor(mul_71, unsqueeze_189);  mul_71 = unsqueeze_189 = None
    unsqueeze_190: "f32[60, 1]" = torch.ops.aten.unsqueeze.default(arg79_1, -1);  arg79_1 = None
    unsqueeze_191: "f32[60, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
    add_52: "f32[8, 60, 28, 28]" = torch.ops.aten.add.Tensor(mul_72, unsqueeze_191);  mul_72 = unsqueeze_191 = None
    relu_10: "f32[8, 60, 28, 28]" = torch.ops.aten.relu.default(add_52);  add_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_26: "f32[8, 60, 28, 28]" = torch.ops.aten.convolution.default(relu_10, arg80_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 60);  arg80_1 = None
    unsqueeze_192: "f32[60, 1]" = torch.ops.aten.unsqueeze.default(arg346_1, -1);  arg346_1 = None
    unsqueeze_193: "f32[60, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
    sub_24: "f32[8, 60, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_193);  convolution_26 = unsqueeze_193 = None
    add_53: "f32[60]" = torch.ops.aten.add.Tensor(arg347_1, 1e-05);  arg347_1 = None
    sqrt_24: "f32[60]" = torch.ops.aten.sqrt.default(add_53);  add_53 = None
    reciprocal_24: "f32[60]" = torch.ops.aten.reciprocal.default(sqrt_24);  sqrt_24 = None
    mul_73: "f32[60]" = torch.ops.aten.mul.Tensor(reciprocal_24, 1);  reciprocal_24 = None
    unsqueeze_194: "f32[60, 1]" = torch.ops.aten.unsqueeze.default(mul_73, -1);  mul_73 = None
    unsqueeze_195: "f32[60, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
    mul_74: "f32[8, 60, 28, 28]" = torch.ops.aten.mul.Tensor(sub_24, unsqueeze_195);  sub_24 = unsqueeze_195 = None
    unsqueeze_196: "f32[60, 1]" = torch.ops.aten.unsqueeze.default(arg81_1, -1);  arg81_1 = None
    unsqueeze_197: "f32[60, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
    mul_75: "f32[8, 60, 28, 28]" = torch.ops.aten.mul.Tensor(mul_74, unsqueeze_197);  mul_74 = unsqueeze_197 = None
    unsqueeze_198: "f32[60, 1]" = torch.ops.aten.unsqueeze.default(arg82_1, -1);  arg82_1 = None
    unsqueeze_199: "f32[60, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
    add_54: "f32[8, 60, 28, 28]" = torch.ops.aten.add.Tensor(mul_75, unsqueeze_199);  mul_75 = unsqueeze_199 = None
    relu_11: "f32[8, 60, 28, 28]" = torch.ops.aten.relu.default(add_54);  add_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat_8: "f32[8, 120, 28, 28]" = torch.ops.aten.cat.default([relu_10, relu_11], 1);  relu_10 = relu_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_1: "f32[8, 120, 1, 1]" = torch.ops.aten.mean.dim(cat_8, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_27: "f32[8, 32, 1, 1]" = torch.ops.aten.convolution.default(mean_1, arg83_1, arg84_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_1 = arg83_1 = arg84_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    relu_12: "f32[8, 32, 1, 1]" = torch.ops.aten.relu.default(convolution_27);  convolution_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_28: "f32[8, 120, 1, 1]" = torch.ops.aten.convolution.default(relu_12, arg85_1, arg86_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_12 = arg85_1 = arg86_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    add_55: "f32[8, 120, 1, 1]" = torch.ops.aten.add.Tensor(convolution_28, 3);  convolution_28 = None
    clamp_min_1: "f32[8, 120, 1, 1]" = torch.ops.aten.clamp_min.default(add_55, 0);  add_55 = None
    clamp_max_1: "f32[8, 120, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_1, 6);  clamp_min_1 = None
    div_1: "f32[8, 120, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_1, 6);  clamp_max_1 = None
    mul_76: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(cat_8, div_1);  cat_8 = div_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_29: "f32[8, 20, 28, 28]" = torch.ops.aten.convolution.default(mul_76, arg87_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_76 = arg87_1 = None
    unsqueeze_200: "f32[20, 1]" = torch.ops.aten.unsqueeze.default(arg349_1, -1);  arg349_1 = None
    unsqueeze_201: "f32[20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
    sub_25: "f32[8, 20, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_201);  convolution_29 = unsqueeze_201 = None
    add_56: "f32[20]" = torch.ops.aten.add.Tensor(arg350_1, 1e-05);  arg350_1 = None
    sqrt_25: "f32[20]" = torch.ops.aten.sqrt.default(add_56);  add_56 = None
    reciprocal_25: "f32[20]" = torch.ops.aten.reciprocal.default(sqrt_25);  sqrt_25 = None
    mul_77: "f32[20]" = torch.ops.aten.mul.Tensor(reciprocal_25, 1);  reciprocal_25 = None
    unsqueeze_202: "f32[20, 1]" = torch.ops.aten.unsqueeze.default(mul_77, -1);  mul_77 = None
    unsqueeze_203: "f32[20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
    mul_78: "f32[8, 20, 28, 28]" = torch.ops.aten.mul.Tensor(sub_25, unsqueeze_203);  sub_25 = unsqueeze_203 = None
    unsqueeze_204: "f32[20, 1]" = torch.ops.aten.unsqueeze.default(arg88_1, -1);  arg88_1 = None
    unsqueeze_205: "f32[20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
    mul_79: "f32[8, 20, 28, 28]" = torch.ops.aten.mul.Tensor(mul_78, unsqueeze_205);  mul_78 = unsqueeze_205 = None
    unsqueeze_206: "f32[20, 1]" = torch.ops.aten.unsqueeze.default(arg89_1, -1);  arg89_1 = None
    unsqueeze_207: "f32[20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
    add_57: "f32[8, 20, 28, 28]" = torch.ops.aten.add.Tensor(mul_79, unsqueeze_207);  mul_79 = unsqueeze_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_30: "f32[8, 20, 28, 28]" = torch.ops.aten.convolution.default(add_57, arg90_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 20);  arg90_1 = None
    unsqueeze_208: "f32[20, 1]" = torch.ops.aten.unsqueeze.default(arg352_1, -1);  arg352_1 = None
    unsqueeze_209: "f32[20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
    sub_26: "f32[8, 20, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_209);  convolution_30 = unsqueeze_209 = None
    add_58: "f32[20]" = torch.ops.aten.add.Tensor(arg353_1, 1e-05);  arg353_1 = None
    sqrt_26: "f32[20]" = torch.ops.aten.sqrt.default(add_58);  add_58 = None
    reciprocal_26: "f32[20]" = torch.ops.aten.reciprocal.default(sqrt_26);  sqrt_26 = None
    mul_80: "f32[20]" = torch.ops.aten.mul.Tensor(reciprocal_26, 1);  reciprocal_26 = None
    unsqueeze_210: "f32[20, 1]" = torch.ops.aten.unsqueeze.default(mul_80, -1);  mul_80 = None
    unsqueeze_211: "f32[20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
    mul_81: "f32[8, 20, 28, 28]" = torch.ops.aten.mul.Tensor(sub_26, unsqueeze_211);  sub_26 = unsqueeze_211 = None
    unsqueeze_212: "f32[20, 1]" = torch.ops.aten.unsqueeze.default(arg91_1, -1);  arg91_1 = None
    unsqueeze_213: "f32[20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
    mul_82: "f32[8, 20, 28, 28]" = torch.ops.aten.mul.Tensor(mul_81, unsqueeze_213);  mul_81 = unsqueeze_213 = None
    unsqueeze_214: "f32[20, 1]" = torch.ops.aten.unsqueeze.default(arg92_1, -1);  arg92_1 = None
    unsqueeze_215: "f32[20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
    add_59: "f32[8, 20, 28, 28]" = torch.ops.aten.add.Tensor(mul_82, unsqueeze_215);  mul_82 = unsqueeze_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat_9: "f32[8, 40, 28, 28]" = torch.ops.aten.cat.default([add_57, add_59], 1);  add_57 = add_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    add_60: "f32[8, 40, 28, 28]" = torch.ops.aten.add.Tensor(cat_9, add_50);  cat_9 = add_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_31: "f32[8, 120, 28, 28]" = torch.ops.aten.convolution.default(add_60, arg93_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg93_1 = None
    unsqueeze_216: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg355_1, -1);  arg355_1 = None
    unsqueeze_217: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
    sub_27: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_217);  convolution_31 = unsqueeze_217 = None
    add_61: "f32[120]" = torch.ops.aten.add.Tensor(arg356_1, 1e-05);  arg356_1 = None
    sqrt_27: "f32[120]" = torch.ops.aten.sqrt.default(add_61);  add_61 = None
    reciprocal_27: "f32[120]" = torch.ops.aten.reciprocal.default(sqrt_27);  sqrt_27 = None
    mul_83: "f32[120]" = torch.ops.aten.mul.Tensor(reciprocal_27, 1);  reciprocal_27 = None
    unsqueeze_218: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(mul_83, -1);  mul_83 = None
    unsqueeze_219: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, -1);  unsqueeze_218 = None
    mul_84: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_27, unsqueeze_219);  sub_27 = unsqueeze_219 = None
    unsqueeze_220: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg94_1, -1);  arg94_1 = None
    unsqueeze_221: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, -1);  unsqueeze_220 = None
    mul_85: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(mul_84, unsqueeze_221);  mul_84 = unsqueeze_221 = None
    unsqueeze_222: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg95_1, -1);  arg95_1 = None
    unsqueeze_223: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, -1);  unsqueeze_222 = None
    add_62: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(mul_85, unsqueeze_223);  mul_85 = unsqueeze_223 = None
    relu_13: "f32[8, 120, 28, 28]" = torch.ops.aten.relu.default(add_62);  add_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_32: "f32[8, 120, 28, 28]" = torch.ops.aten.convolution.default(relu_13, arg96_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 120);  arg96_1 = None
    unsqueeze_224: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg358_1, -1);  arg358_1 = None
    unsqueeze_225: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, -1);  unsqueeze_224 = None
    sub_28: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_225);  convolution_32 = unsqueeze_225 = None
    add_63: "f32[120]" = torch.ops.aten.add.Tensor(arg359_1, 1e-05);  arg359_1 = None
    sqrt_28: "f32[120]" = torch.ops.aten.sqrt.default(add_63);  add_63 = None
    reciprocal_28: "f32[120]" = torch.ops.aten.reciprocal.default(sqrt_28);  sqrt_28 = None
    mul_86: "f32[120]" = torch.ops.aten.mul.Tensor(reciprocal_28, 1);  reciprocal_28 = None
    unsqueeze_226: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(mul_86, -1);  mul_86 = None
    unsqueeze_227: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, -1);  unsqueeze_226 = None
    mul_87: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_28, unsqueeze_227);  sub_28 = unsqueeze_227 = None
    unsqueeze_228: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg97_1, -1);  arg97_1 = None
    unsqueeze_229: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, -1);  unsqueeze_228 = None
    mul_88: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(mul_87, unsqueeze_229);  mul_87 = unsqueeze_229 = None
    unsqueeze_230: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg98_1, -1);  arg98_1 = None
    unsqueeze_231: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, -1);  unsqueeze_230 = None
    add_64: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(mul_88, unsqueeze_231);  mul_88 = unsqueeze_231 = None
    relu_14: "f32[8, 120, 28, 28]" = torch.ops.aten.relu.default(add_64);  add_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat_10: "f32[8, 240, 28, 28]" = torch.ops.aten.cat.default([relu_13, relu_14], 1);  relu_13 = relu_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:172, code: x = self.conv_dw(x)
    convolution_33: "f32[8, 240, 14, 14]" = torch.ops.aten.convolution.default(cat_10, arg99_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 240);  cat_10 = arg99_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:173, code: x = self.bn_dw(x)
    unsqueeze_232: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(arg361_1, -1);  arg361_1 = None
    unsqueeze_233: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, -1);  unsqueeze_232 = None
    sub_29: "f32[8, 240, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_233);  convolution_33 = unsqueeze_233 = None
    add_65: "f32[240]" = torch.ops.aten.add.Tensor(arg362_1, 1e-05);  arg362_1 = None
    sqrt_29: "f32[240]" = torch.ops.aten.sqrt.default(add_65);  add_65 = None
    reciprocal_29: "f32[240]" = torch.ops.aten.reciprocal.default(sqrt_29);  sqrt_29 = None
    mul_89: "f32[240]" = torch.ops.aten.mul.Tensor(reciprocal_29, 1);  reciprocal_29 = None
    unsqueeze_234: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(mul_89, -1);  mul_89 = None
    unsqueeze_235: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, -1);  unsqueeze_234 = None
    mul_90: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_29, unsqueeze_235);  sub_29 = unsqueeze_235 = None
    unsqueeze_236: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(arg100_1, -1);  arg100_1 = None
    unsqueeze_237: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, -1);  unsqueeze_236 = None
    mul_91: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(mul_90, unsqueeze_237);  mul_90 = unsqueeze_237 = None
    unsqueeze_238: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(arg101_1, -1);  arg101_1 = None
    unsqueeze_239: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, -1);  unsqueeze_238 = None
    add_66: "f32[8, 240, 14, 14]" = torch.ops.aten.add.Tensor(mul_91, unsqueeze_239);  mul_91 = unsqueeze_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_34: "f32[8, 40, 14, 14]" = torch.ops.aten.convolution.default(add_66, arg102_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_66 = arg102_1 = None
    unsqueeze_240: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg364_1, -1);  arg364_1 = None
    unsqueeze_241: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, -1);  unsqueeze_240 = None
    sub_30: "f32[8, 40, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_241);  convolution_34 = unsqueeze_241 = None
    add_67: "f32[40]" = torch.ops.aten.add.Tensor(arg365_1, 1e-05);  arg365_1 = None
    sqrt_30: "f32[40]" = torch.ops.aten.sqrt.default(add_67);  add_67 = None
    reciprocal_30: "f32[40]" = torch.ops.aten.reciprocal.default(sqrt_30);  sqrt_30 = None
    mul_92: "f32[40]" = torch.ops.aten.mul.Tensor(reciprocal_30, 1);  reciprocal_30 = None
    unsqueeze_242: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(mul_92, -1);  mul_92 = None
    unsqueeze_243: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, -1);  unsqueeze_242 = None
    mul_93: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(sub_30, unsqueeze_243);  sub_30 = unsqueeze_243 = None
    unsqueeze_244: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg103_1, -1);  arg103_1 = None
    unsqueeze_245: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, -1);  unsqueeze_244 = None
    mul_94: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(mul_93, unsqueeze_245);  mul_93 = unsqueeze_245 = None
    unsqueeze_246: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg104_1, -1);  arg104_1 = None
    unsqueeze_247: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, -1);  unsqueeze_246 = None
    add_68: "f32[8, 40, 14, 14]" = torch.ops.aten.add.Tensor(mul_94, unsqueeze_247);  mul_94 = unsqueeze_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_35: "f32[8, 40, 14, 14]" = torch.ops.aten.convolution.default(add_68, arg105_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 40);  arg105_1 = None
    unsqueeze_248: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg367_1, -1);  arg367_1 = None
    unsqueeze_249: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, -1);  unsqueeze_248 = None
    sub_31: "f32[8, 40, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_249);  convolution_35 = unsqueeze_249 = None
    add_69: "f32[40]" = torch.ops.aten.add.Tensor(arg368_1, 1e-05);  arg368_1 = None
    sqrt_31: "f32[40]" = torch.ops.aten.sqrt.default(add_69);  add_69 = None
    reciprocal_31: "f32[40]" = torch.ops.aten.reciprocal.default(sqrt_31);  sqrt_31 = None
    mul_95: "f32[40]" = torch.ops.aten.mul.Tensor(reciprocal_31, 1);  reciprocal_31 = None
    unsqueeze_250: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(mul_95, -1);  mul_95 = None
    unsqueeze_251: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, -1);  unsqueeze_250 = None
    mul_96: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(sub_31, unsqueeze_251);  sub_31 = unsqueeze_251 = None
    unsqueeze_252: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg106_1, -1);  arg106_1 = None
    unsqueeze_253: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, -1);  unsqueeze_252 = None
    mul_97: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(mul_96, unsqueeze_253);  mul_96 = unsqueeze_253 = None
    unsqueeze_254: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg107_1, -1);  arg107_1 = None
    unsqueeze_255: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, -1);  unsqueeze_254 = None
    add_70: "f32[8, 40, 14, 14]" = torch.ops.aten.add.Tensor(mul_97, unsqueeze_255);  mul_97 = unsqueeze_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat_11: "f32[8, 80, 14, 14]" = torch.ops.aten.cat.default([add_68, add_70], 1);  add_68 = add_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    convolution_36: "f32[8, 40, 14, 14]" = torch.ops.aten.convolution.default(add_60, arg108_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 40);  add_60 = arg108_1 = None
    unsqueeze_256: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg370_1, -1);  arg370_1 = None
    unsqueeze_257: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, -1);  unsqueeze_256 = None
    sub_32: "f32[8, 40, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_257);  convolution_36 = unsqueeze_257 = None
    add_71: "f32[40]" = torch.ops.aten.add.Tensor(arg371_1, 1e-05);  arg371_1 = None
    sqrt_32: "f32[40]" = torch.ops.aten.sqrt.default(add_71);  add_71 = None
    reciprocal_32: "f32[40]" = torch.ops.aten.reciprocal.default(sqrt_32);  sqrt_32 = None
    mul_98: "f32[40]" = torch.ops.aten.mul.Tensor(reciprocal_32, 1);  reciprocal_32 = None
    unsqueeze_258: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(mul_98, -1);  mul_98 = None
    unsqueeze_259: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, -1);  unsqueeze_258 = None
    mul_99: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(sub_32, unsqueeze_259);  sub_32 = unsqueeze_259 = None
    unsqueeze_260: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg109_1, -1);  arg109_1 = None
    unsqueeze_261: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, -1);  unsqueeze_260 = None
    mul_100: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(mul_99, unsqueeze_261);  mul_99 = unsqueeze_261 = None
    unsqueeze_262: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg110_1, -1);  arg110_1 = None
    unsqueeze_263: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, -1);  unsqueeze_262 = None
    add_72: "f32[8, 40, 14, 14]" = torch.ops.aten.add.Tensor(mul_100, unsqueeze_263);  mul_100 = unsqueeze_263 = None
    convolution_37: "f32[8, 80, 14, 14]" = torch.ops.aten.convolution.default(add_72, arg111_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_72 = arg111_1 = None
    unsqueeze_264: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg373_1, -1);  arg373_1 = None
    unsqueeze_265: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, -1);  unsqueeze_264 = None
    sub_33: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_265);  convolution_37 = unsqueeze_265 = None
    add_73: "f32[80]" = torch.ops.aten.add.Tensor(arg374_1, 1e-05);  arg374_1 = None
    sqrt_33: "f32[80]" = torch.ops.aten.sqrt.default(add_73);  add_73 = None
    reciprocal_33: "f32[80]" = torch.ops.aten.reciprocal.default(sqrt_33);  sqrt_33 = None
    mul_101: "f32[80]" = torch.ops.aten.mul.Tensor(reciprocal_33, 1);  reciprocal_33 = None
    unsqueeze_266: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(mul_101, -1);  mul_101 = None
    unsqueeze_267: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, -1);  unsqueeze_266 = None
    mul_102: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_33, unsqueeze_267);  sub_33 = unsqueeze_267 = None
    unsqueeze_268: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg112_1, -1);  arg112_1 = None
    unsqueeze_269: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, -1);  unsqueeze_268 = None
    mul_103: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(mul_102, unsqueeze_269);  mul_102 = unsqueeze_269 = None
    unsqueeze_270: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg113_1, -1);  arg113_1 = None
    unsqueeze_271: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, -1);  unsqueeze_270 = None
    add_74: "f32[8, 80, 14, 14]" = torch.ops.aten.add.Tensor(mul_103, unsqueeze_271);  mul_103 = unsqueeze_271 = None
    add_75: "f32[8, 80, 14, 14]" = torch.ops.aten.add.Tensor(cat_11, add_74);  cat_11 = add_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_38: "f32[8, 100, 14, 14]" = torch.ops.aten.convolution.default(add_75, arg114_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg114_1 = None
    unsqueeze_272: "f32[100, 1]" = torch.ops.aten.unsqueeze.default(arg376_1, -1);  arg376_1 = None
    unsqueeze_273: "f32[100, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, -1);  unsqueeze_272 = None
    sub_34: "f32[8, 100, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_273);  convolution_38 = unsqueeze_273 = None
    add_76: "f32[100]" = torch.ops.aten.add.Tensor(arg377_1, 1e-05);  arg377_1 = None
    sqrt_34: "f32[100]" = torch.ops.aten.sqrt.default(add_76);  add_76 = None
    reciprocal_34: "f32[100]" = torch.ops.aten.reciprocal.default(sqrt_34);  sqrt_34 = None
    mul_104: "f32[100]" = torch.ops.aten.mul.Tensor(reciprocal_34, 1);  reciprocal_34 = None
    unsqueeze_274: "f32[100, 1]" = torch.ops.aten.unsqueeze.default(mul_104, -1);  mul_104 = None
    unsqueeze_275: "f32[100, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, -1);  unsqueeze_274 = None
    mul_105: "f32[8, 100, 14, 14]" = torch.ops.aten.mul.Tensor(sub_34, unsqueeze_275);  sub_34 = unsqueeze_275 = None
    unsqueeze_276: "f32[100, 1]" = torch.ops.aten.unsqueeze.default(arg115_1, -1);  arg115_1 = None
    unsqueeze_277: "f32[100, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, -1);  unsqueeze_276 = None
    mul_106: "f32[8, 100, 14, 14]" = torch.ops.aten.mul.Tensor(mul_105, unsqueeze_277);  mul_105 = unsqueeze_277 = None
    unsqueeze_278: "f32[100, 1]" = torch.ops.aten.unsqueeze.default(arg116_1, -1);  arg116_1 = None
    unsqueeze_279: "f32[100, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, -1);  unsqueeze_278 = None
    add_77: "f32[8, 100, 14, 14]" = torch.ops.aten.add.Tensor(mul_106, unsqueeze_279);  mul_106 = unsqueeze_279 = None
    relu_15: "f32[8, 100, 14, 14]" = torch.ops.aten.relu.default(add_77);  add_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_39: "f32[8, 100, 14, 14]" = torch.ops.aten.convolution.default(relu_15, arg117_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 100);  arg117_1 = None
    unsqueeze_280: "f32[100, 1]" = torch.ops.aten.unsqueeze.default(arg379_1, -1);  arg379_1 = None
    unsqueeze_281: "f32[100, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, -1);  unsqueeze_280 = None
    sub_35: "f32[8, 100, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_281);  convolution_39 = unsqueeze_281 = None
    add_78: "f32[100]" = torch.ops.aten.add.Tensor(arg380_1, 1e-05);  arg380_1 = None
    sqrt_35: "f32[100]" = torch.ops.aten.sqrt.default(add_78);  add_78 = None
    reciprocal_35: "f32[100]" = torch.ops.aten.reciprocal.default(sqrt_35);  sqrt_35 = None
    mul_107: "f32[100]" = torch.ops.aten.mul.Tensor(reciprocal_35, 1);  reciprocal_35 = None
    unsqueeze_282: "f32[100, 1]" = torch.ops.aten.unsqueeze.default(mul_107, -1);  mul_107 = None
    unsqueeze_283: "f32[100, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, -1);  unsqueeze_282 = None
    mul_108: "f32[8, 100, 14, 14]" = torch.ops.aten.mul.Tensor(sub_35, unsqueeze_283);  sub_35 = unsqueeze_283 = None
    unsqueeze_284: "f32[100, 1]" = torch.ops.aten.unsqueeze.default(arg118_1, -1);  arg118_1 = None
    unsqueeze_285: "f32[100, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, -1);  unsqueeze_284 = None
    mul_109: "f32[8, 100, 14, 14]" = torch.ops.aten.mul.Tensor(mul_108, unsqueeze_285);  mul_108 = unsqueeze_285 = None
    unsqueeze_286: "f32[100, 1]" = torch.ops.aten.unsqueeze.default(arg119_1, -1);  arg119_1 = None
    unsqueeze_287: "f32[100, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, -1);  unsqueeze_286 = None
    add_79: "f32[8, 100, 14, 14]" = torch.ops.aten.add.Tensor(mul_109, unsqueeze_287);  mul_109 = unsqueeze_287 = None
    relu_16: "f32[8, 100, 14, 14]" = torch.ops.aten.relu.default(add_79);  add_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat_12: "f32[8, 200, 14, 14]" = torch.ops.aten.cat.default([relu_15, relu_16], 1);  relu_15 = relu_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_40: "f32[8, 40, 14, 14]" = torch.ops.aten.convolution.default(cat_12, arg120_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_12 = arg120_1 = None
    unsqueeze_288: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg382_1, -1);  arg382_1 = None
    unsqueeze_289: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, -1);  unsqueeze_288 = None
    sub_36: "f32[8, 40, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_289);  convolution_40 = unsqueeze_289 = None
    add_80: "f32[40]" = torch.ops.aten.add.Tensor(arg383_1, 1e-05);  arg383_1 = None
    sqrt_36: "f32[40]" = torch.ops.aten.sqrt.default(add_80);  add_80 = None
    reciprocal_36: "f32[40]" = torch.ops.aten.reciprocal.default(sqrt_36);  sqrt_36 = None
    mul_110: "f32[40]" = torch.ops.aten.mul.Tensor(reciprocal_36, 1);  reciprocal_36 = None
    unsqueeze_290: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(mul_110, -1);  mul_110 = None
    unsqueeze_291: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, -1);  unsqueeze_290 = None
    mul_111: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(sub_36, unsqueeze_291);  sub_36 = unsqueeze_291 = None
    unsqueeze_292: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg121_1, -1);  arg121_1 = None
    unsqueeze_293: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, -1);  unsqueeze_292 = None
    mul_112: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(mul_111, unsqueeze_293);  mul_111 = unsqueeze_293 = None
    unsqueeze_294: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg122_1, -1);  arg122_1 = None
    unsqueeze_295: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, -1);  unsqueeze_294 = None
    add_81: "f32[8, 40, 14, 14]" = torch.ops.aten.add.Tensor(mul_112, unsqueeze_295);  mul_112 = unsqueeze_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_41: "f32[8, 40, 14, 14]" = torch.ops.aten.convolution.default(add_81, arg123_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 40);  arg123_1 = None
    unsqueeze_296: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg385_1, -1);  arg385_1 = None
    unsqueeze_297: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, -1);  unsqueeze_296 = None
    sub_37: "f32[8, 40, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_297);  convolution_41 = unsqueeze_297 = None
    add_82: "f32[40]" = torch.ops.aten.add.Tensor(arg386_1, 1e-05);  arg386_1 = None
    sqrt_37: "f32[40]" = torch.ops.aten.sqrt.default(add_82);  add_82 = None
    reciprocal_37: "f32[40]" = torch.ops.aten.reciprocal.default(sqrt_37);  sqrt_37 = None
    mul_113: "f32[40]" = torch.ops.aten.mul.Tensor(reciprocal_37, 1);  reciprocal_37 = None
    unsqueeze_298: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(mul_113, -1);  mul_113 = None
    unsqueeze_299: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, -1);  unsqueeze_298 = None
    mul_114: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(sub_37, unsqueeze_299);  sub_37 = unsqueeze_299 = None
    unsqueeze_300: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg124_1, -1);  arg124_1 = None
    unsqueeze_301: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, -1);  unsqueeze_300 = None
    mul_115: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(mul_114, unsqueeze_301);  mul_114 = unsqueeze_301 = None
    unsqueeze_302: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg125_1, -1);  arg125_1 = None
    unsqueeze_303: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, -1);  unsqueeze_302 = None
    add_83: "f32[8, 40, 14, 14]" = torch.ops.aten.add.Tensor(mul_115, unsqueeze_303);  mul_115 = unsqueeze_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat_13: "f32[8, 80, 14, 14]" = torch.ops.aten.cat.default([add_81, add_83], 1);  add_81 = add_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    add_84: "f32[8, 80, 14, 14]" = torch.ops.aten.add.Tensor(cat_13, add_75);  cat_13 = add_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_42: "f32[8, 92, 14, 14]" = torch.ops.aten.convolution.default(add_84, arg126_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg126_1 = None
    unsqueeze_304: "f32[92, 1]" = torch.ops.aten.unsqueeze.default(arg388_1, -1);  arg388_1 = None
    unsqueeze_305: "f32[92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, -1);  unsqueeze_304 = None
    sub_38: "f32[8, 92, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_305);  convolution_42 = unsqueeze_305 = None
    add_85: "f32[92]" = torch.ops.aten.add.Tensor(arg389_1, 1e-05);  arg389_1 = None
    sqrt_38: "f32[92]" = torch.ops.aten.sqrt.default(add_85);  add_85 = None
    reciprocal_38: "f32[92]" = torch.ops.aten.reciprocal.default(sqrt_38);  sqrt_38 = None
    mul_116: "f32[92]" = torch.ops.aten.mul.Tensor(reciprocal_38, 1);  reciprocal_38 = None
    unsqueeze_306: "f32[92, 1]" = torch.ops.aten.unsqueeze.default(mul_116, -1);  mul_116 = None
    unsqueeze_307: "f32[92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, -1);  unsqueeze_306 = None
    mul_117: "f32[8, 92, 14, 14]" = torch.ops.aten.mul.Tensor(sub_38, unsqueeze_307);  sub_38 = unsqueeze_307 = None
    unsqueeze_308: "f32[92, 1]" = torch.ops.aten.unsqueeze.default(arg127_1, -1);  arg127_1 = None
    unsqueeze_309: "f32[92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, -1);  unsqueeze_308 = None
    mul_118: "f32[8, 92, 14, 14]" = torch.ops.aten.mul.Tensor(mul_117, unsqueeze_309);  mul_117 = unsqueeze_309 = None
    unsqueeze_310: "f32[92, 1]" = torch.ops.aten.unsqueeze.default(arg128_1, -1);  arg128_1 = None
    unsqueeze_311: "f32[92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, -1);  unsqueeze_310 = None
    add_86: "f32[8, 92, 14, 14]" = torch.ops.aten.add.Tensor(mul_118, unsqueeze_311);  mul_118 = unsqueeze_311 = None
    relu_17: "f32[8, 92, 14, 14]" = torch.ops.aten.relu.default(add_86);  add_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_43: "f32[8, 92, 14, 14]" = torch.ops.aten.convolution.default(relu_17, arg129_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 92);  arg129_1 = None
    unsqueeze_312: "f32[92, 1]" = torch.ops.aten.unsqueeze.default(arg391_1, -1);  arg391_1 = None
    unsqueeze_313: "f32[92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, -1);  unsqueeze_312 = None
    sub_39: "f32[8, 92, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_313);  convolution_43 = unsqueeze_313 = None
    add_87: "f32[92]" = torch.ops.aten.add.Tensor(arg392_1, 1e-05);  arg392_1 = None
    sqrt_39: "f32[92]" = torch.ops.aten.sqrt.default(add_87);  add_87 = None
    reciprocal_39: "f32[92]" = torch.ops.aten.reciprocal.default(sqrt_39);  sqrt_39 = None
    mul_119: "f32[92]" = torch.ops.aten.mul.Tensor(reciprocal_39, 1);  reciprocal_39 = None
    unsqueeze_314: "f32[92, 1]" = torch.ops.aten.unsqueeze.default(mul_119, -1);  mul_119 = None
    unsqueeze_315: "f32[92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, -1);  unsqueeze_314 = None
    mul_120: "f32[8, 92, 14, 14]" = torch.ops.aten.mul.Tensor(sub_39, unsqueeze_315);  sub_39 = unsqueeze_315 = None
    unsqueeze_316: "f32[92, 1]" = torch.ops.aten.unsqueeze.default(arg130_1, -1);  arg130_1 = None
    unsqueeze_317: "f32[92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, -1);  unsqueeze_316 = None
    mul_121: "f32[8, 92, 14, 14]" = torch.ops.aten.mul.Tensor(mul_120, unsqueeze_317);  mul_120 = unsqueeze_317 = None
    unsqueeze_318: "f32[92, 1]" = torch.ops.aten.unsqueeze.default(arg131_1, -1);  arg131_1 = None
    unsqueeze_319: "f32[92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, -1);  unsqueeze_318 = None
    add_88: "f32[8, 92, 14, 14]" = torch.ops.aten.add.Tensor(mul_121, unsqueeze_319);  mul_121 = unsqueeze_319 = None
    relu_18: "f32[8, 92, 14, 14]" = torch.ops.aten.relu.default(add_88);  add_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat_14: "f32[8, 184, 14, 14]" = torch.ops.aten.cat.default([relu_17, relu_18], 1);  relu_17 = relu_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_44: "f32[8, 40, 14, 14]" = torch.ops.aten.convolution.default(cat_14, arg132_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_14 = arg132_1 = None
    unsqueeze_320: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg394_1, -1);  arg394_1 = None
    unsqueeze_321: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, -1);  unsqueeze_320 = None
    sub_40: "f32[8, 40, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_321);  convolution_44 = unsqueeze_321 = None
    add_89: "f32[40]" = torch.ops.aten.add.Tensor(arg395_1, 1e-05);  arg395_1 = None
    sqrt_40: "f32[40]" = torch.ops.aten.sqrt.default(add_89);  add_89 = None
    reciprocal_40: "f32[40]" = torch.ops.aten.reciprocal.default(sqrt_40);  sqrt_40 = None
    mul_122: "f32[40]" = torch.ops.aten.mul.Tensor(reciprocal_40, 1);  reciprocal_40 = None
    unsqueeze_322: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(mul_122, -1);  mul_122 = None
    unsqueeze_323: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, -1);  unsqueeze_322 = None
    mul_123: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(sub_40, unsqueeze_323);  sub_40 = unsqueeze_323 = None
    unsqueeze_324: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg133_1, -1);  arg133_1 = None
    unsqueeze_325: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, -1);  unsqueeze_324 = None
    mul_124: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(mul_123, unsqueeze_325);  mul_123 = unsqueeze_325 = None
    unsqueeze_326: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg134_1, -1);  arg134_1 = None
    unsqueeze_327: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, -1);  unsqueeze_326 = None
    add_90: "f32[8, 40, 14, 14]" = torch.ops.aten.add.Tensor(mul_124, unsqueeze_327);  mul_124 = unsqueeze_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_45: "f32[8, 40, 14, 14]" = torch.ops.aten.convolution.default(add_90, arg135_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 40);  arg135_1 = None
    unsqueeze_328: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg397_1, -1);  arg397_1 = None
    unsqueeze_329: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, -1);  unsqueeze_328 = None
    sub_41: "f32[8, 40, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_329);  convolution_45 = unsqueeze_329 = None
    add_91: "f32[40]" = torch.ops.aten.add.Tensor(arg398_1, 1e-05);  arg398_1 = None
    sqrt_41: "f32[40]" = torch.ops.aten.sqrt.default(add_91);  add_91 = None
    reciprocal_41: "f32[40]" = torch.ops.aten.reciprocal.default(sqrt_41);  sqrt_41 = None
    mul_125: "f32[40]" = torch.ops.aten.mul.Tensor(reciprocal_41, 1);  reciprocal_41 = None
    unsqueeze_330: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(mul_125, -1);  mul_125 = None
    unsqueeze_331: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, -1);  unsqueeze_330 = None
    mul_126: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(sub_41, unsqueeze_331);  sub_41 = unsqueeze_331 = None
    unsqueeze_332: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg136_1, -1);  arg136_1 = None
    unsqueeze_333: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, -1);  unsqueeze_332 = None
    mul_127: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(mul_126, unsqueeze_333);  mul_126 = unsqueeze_333 = None
    unsqueeze_334: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg137_1, -1);  arg137_1 = None
    unsqueeze_335: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, -1);  unsqueeze_334 = None
    add_92: "f32[8, 40, 14, 14]" = torch.ops.aten.add.Tensor(mul_127, unsqueeze_335);  mul_127 = unsqueeze_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat_15: "f32[8, 80, 14, 14]" = torch.ops.aten.cat.default([add_90, add_92], 1);  add_90 = add_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    add_93: "f32[8, 80, 14, 14]" = torch.ops.aten.add.Tensor(cat_15, add_84);  cat_15 = add_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_46: "f32[8, 92, 14, 14]" = torch.ops.aten.convolution.default(add_93, arg138_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg138_1 = None
    unsqueeze_336: "f32[92, 1]" = torch.ops.aten.unsqueeze.default(arg400_1, -1);  arg400_1 = None
    unsqueeze_337: "f32[92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, -1);  unsqueeze_336 = None
    sub_42: "f32[8, 92, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_337);  convolution_46 = unsqueeze_337 = None
    add_94: "f32[92]" = torch.ops.aten.add.Tensor(arg401_1, 1e-05);  arg401_1 = None
    sqrt_42: "f32[92]" = torch.ops.aten.sqrt.default(add_94);  add_94 = None
    reciprocal_42: "f32[92]" = torch.ops.aten.reciprocal.default(sqrt_42);  sqrt_42 = None
    mul_128: "f32[92]" = torch.ops.aten.mul.Tensor(reciprocal_42, 1);  reciprocal_42 = None
    unsqueeze_338: "f32[92, 1]" = torch.ops.aten.unsqueeze.default(mul_128, -1);  mul_128 = None
    unsqueeze_339: "f32[92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, -1);  unsqueeze_338 = None
    mul_129: "f32[8, 92, 14, 14]" = torch.ops.aten.mul.Tensor(sub_42, unsqueeze_339);  sub_42 = unsqueeze_339 = None
    unsqueeze_340: "f32[92, 1]" = torch.ops.aten.unsqueeze.default(arg139_1, -1);  arg139_1 = None
    unsqueeze_341: "f32[92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, -1);  unsqueeze_340 = None
    mul_130: "f32[8, 92, 14, 14]" = torch.ops.aten.mul.Tensor(mul_129, unsqueeze_341);  mul_129 = unsqueeze_341 = None
    unsqueeze_342: "f32[92, 1]" = torch.ops.aten.unsqueeze.default(arg140_1, -1);  arg140_1 = None
    unsqueeze_343: "f32[92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, -1);  unsqueeze_342 = None
    add_95: "f32[8, 92, 14, 14]" = torch.ops.aten.add.Tensor(mul_130, unsqueeze_343);  mul_130 = unsqueeze_343 = None
    relu_19: "f32[8, 92, 14, 14]" = torch.ops.aten.relu.default(add_95);  add_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_47: "f32[8, 92, 14, 14]" = torch.ops.aten.convolution.default(relu_19, arg141_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 92);  arg141_1 = None
    unsqueeze_344: "f32[92, 1]" = torch.ops.aten.unsqueeze.default(arg403_1, -1);  arg403_1 = None
    unsqueeze_345: "f32[92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, -1);  unsqueeze_344 = None
    sub_43: "f32[8, 92, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_345);  convolution_47 = unsqueeze_345 = None
    add_96: "f32[92]" = torch.ops.aten.add.Tensor(arg404_1, 1e-05);  arg404_1 = None
    sqrt_43: "f32[92]" = torch.ops.aten.sqrt.default(add_96);  add_96 = None
    reciprocal_43: "f32[92]" = torch.ops.aten.reciprocal.default(sqrt_43);  sqrt_43 = None
    mul_131: "f32[92]" = torch.ops.aten.mul.Tensor(reciprocal_43, 1);  reciprocal_43 = None
    unsqueeze_346: "f32[92, 1]" = torch.ops.aten.unsqueeze.default(mul_131, -1);  mul_131 = None
    unsqueeze_347: "f32[92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, -1);  unsqueeze_346 = None
    mul_132: "f32[8, 92, 14, 14]" = torch.ops.aten.mul.Tensor(sub_43, unsqueeze_347);  sub_43 = unsqueeze_347 = None
    unsqueeze_348: "f32[92, 1]" = torch.ops.aten.unsqueeze.default(arg142_1, -1);  arg142_1 = None
    unsqueeze_349: "f32[92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, -1);  unsqueeze_348 = None
    mul_133: "f32[8, 92, 14, 14]" = torch.ops.aten.mul.Tensor(mul_132, unsqueeze_349);  mul_132 = unsqueeze_349 = None
    unsqueeze_350: "f32[92, 1]" = torch.ops.aten.unsqueeze.default(arg143_1, -1);  arg143_1 = None
    unsqueeze_351: "f32[92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, -1);  unsqueeze_350 = None
    add_97: "f32[8, 92, 14, 14]" = torch.ops.aten.add.Tensor(mul_133, unsqueeze_351);  mul_133 = unsqueeze_351 = None
    relu_20: "f32[8, 92, 14, 14]" = torch.ops.aten.relu.default(add_97);  add_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat_16: "f32[8, 184, 14, 14]" = torch.ops.aten.cat.default([relu_19, relu_20], 1);  relu_19 = relu_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_48: "f32[8, 40, 14, 14]" = torch.ops.aten.convolution.default(cat_16, arg144_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_16 = arg144_1 = None
    unsqueeze_352: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg406_1, -1);  arg406_1 = None
    unsqueeze_353: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, -1);  unsqueeze_352 = None
    sub_44: "f32[8, 40, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_353);  convolution_48 = unsqueeze_353 = None
    add_98: "f32[40]" = torch.ops.aten.add.Tensor(arg407_1, 1e-05);  arg407_1 = None
    sqrt_44: "f32[40]" = torch.ops.aten.sqrt.default(add_98);  add_98 = None
    reciprocal_44: "f32[40]" = torch.ops.aten.reciprocal.default(sqrt_44);  sqrt_44 = None
    mul_134: "f32[40]" = torch.ops.aten.mul.Tensor(reciprocal_44, 1);  reciprocal_44 = None
    unsqueeze_354: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(mul_134, -1);  mul_134 = None
    unsqueeze_355: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, -1);  unsqueeze_354 = None
    mul_135: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(sub_44, unsqueeze_355);  sub_44 = unsqueeze_355 = None
    unsqueeze_356: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg145_1, -1);  arg145_1 = None
    unsqueeze_357: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, -1);  unsqueeze_356 = None
    mul_136: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(mul_135, unsqueeze_357);  mul_135 = unsqueeze_357 = None
    unsqueeze_358: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg146_1, -1);  arg146_1 = None
    unsqueeze_359: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, -1);  unsqueeze_358 = None
    add_99: "f32[8, 40, 14, 14]" = torch.ops.aten.add.Tensor(mul_136, unsqueeze_359);  mul_136 = unsqueeze_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_49: "f32[8, 40, 14, 14]" = torch.ops.aten.convolution.default(add_99, arg147_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 40);  arg147_1 = None
    unsqueeze_360: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg409_1, -1);  arg409_1 = None
    unsqueeze_361: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, -1);  unsqueeze_360 = None
    sub_45: "f32[8, 40, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_361);  convolution_49 = unsqueeze_361 = None
    add_100: "f32[40]" = torch.ops.aten.add.Tensor(arg410_1, 1e-05);  arg410_1 = None
    sqrt_45: "f32[40]" = torch.ops.aten.sqrt.default(add_100);  add_100 = None
    reciprocal_45: "f32[40]" = torch.ops.aten.reciprocal.default(sqrt_45);  sqrt_45 = None
    mul_137: "f32[40]" = torch.ops.aten.mul.Tensor(reciprocal_45, 1);  reciprocal_45 = None
    unsqueeze_362: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(mul_137, -1);  mul_137 = None
    unsqueeze_363: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, -1);  unsqueeze_362 = None
    mul_138: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(sub_45, unsqueeze_363);  sub_45 = unsqueeze_363 = None
    unsqueeze_364: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg148_1, -1);  arg148_1 = None
    unsqueeze_365: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, -1);  unsqueeze_364 = None
    mul_139: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(mul_138, unsqueeze_365);  mul_138 = unsqueeze_365 = None
    unsqueeze_366: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg149_1, -1);  arg149_1 = None
    unsqueeze_367: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, -1);  unsqueeze_366 = None
    add_101: "f32[8, 40, 14, 14]" = torch.ops.aten.add.Tensor(mul_139, unsqueeze_367);  mul_139 = unsqueeze_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat_17: "f32[8, 80, 14, 14]" = torch.ops.aten.cat.default([add_99, add_101], 1);  add_99 = add_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    add_102: "f32[8, 80, 14, 14]" = torch.ops.aten.add.Tensor(cat_17, add_93);  cat_17 = add_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_50: "f32[8, 240, 14, 14]" = torch.ops.aten.convolution.default(add_102, arg150_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg150_1 = None
    unsqueeze_368: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(arg412_1, -1);  arg412_1 = None
    unsqueeze_369: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, -1);  unsqueeze_368 = None
    sub_46: "f32[8, 240, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_369);  convolution_50 = unsqueeze_369 = None
    add_103: "f32[240]" = torch.ops.aten.add.Tensor(arg413_1, 1e-05);  arg413_1 = None
    sqrt_46: "f32[240]" = torch.ops.aten.sqrt.default(add_103);  add_103 = None
    reciprocal_46: "f32[240]" = torch.ops.aten.reciprocal.default(sqrt_46);  sqrt_46 = None
    mul_140: "f32[240]" = torch.ops.aten.mul.Tensor(reciprocal_46, 1);  reciprocal_46 = None
    unsqueeze_370: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(mul_140, -1);  mul_140 = None
    unsqueeze_371: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, -1);  unsqueeze_370 = None
    mul_141: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_46, unsqueeze_371);  sub_46 = unsqueeze_371 = None
    unsqueeze_372: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(arg151_1, -1);  arg151_1 = None
    unsqueeze_373: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_372, -1);  unsqueeze_372 = None
    mul_142: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(mul_141, unsqueeze_373);  mul_141 = unsqueeze_373 = None
    unsqueeze_374: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(arg152_1, -1);  arg152_1 = None
    unsqueeze_375: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, -1);  unsqueeze_374 = None
    add_104: "f32[8, 240, 14, 14]" = torch.ops.aten.add.Tensor(mul_142, unsqueeze_375);  mul_142 = unsqueeze_375 = None
    relu_21: "f32[8, 240, 14, 14]" = torch.ops.aten.relu.default(add_104);  add_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_51: "f32[8, 240, 14, 14]" = torch.ops.aten.convolution.default(relu_21, arg153_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 240);  arg153_1 = None
    unsqueeze_376: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(arg415_1, -1);  arg415_1 = None
    unsqueeze_377: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, -1);  unsqueeze_376 = None
    sub_47: "f32[8, 240, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_377);  convolution_51 = unsqueeze_377 = None
    add_105: "f32[240]" = torch.ops.aten.add.Tensor(arg416_1, 1e-05);  arg416_1 = None
    sqrt_47: "f32[240]" = torch.ops.aten.sqrt.default(add_105);  add_105 = None
    reciprocal_47: "f32[240]" = torch.ops.aten.reciprocal.default(sqrt_47);  sqrt_47 = None
    mul_143: "f32[240]" = torch.ops.aten.mul.Tensor(reciprocal_47, 1);  reciprocal_47 = None
    unsqueeze_378: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(mul_143, -1);  mul_143 = None
    unsqueeze_379: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, -1);  unsqueeze_378 = None
    mul_144: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_47, unsqueeze_379);  sub_47 = unsqueeze_379 = None
    unsqueeze_380: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(arg154_1, -1);  arg154_1 = None
    unsqueeze_381: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, -1);  unsqueeze_380 = None
    mul_145: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(mul_144, unsqueeze_381);  mul_144 = unsqueeze_381 = None
    unsqueeze_382: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(arg155_1, -1);  arg155_1 = None
    unsqueeze_383: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, -1);  unsqueeze_382 = None
    add_106: "f32[8, 240, 14, 14]" = torch.ops.aten.add.Tensor(mul_145, unsqueeze_383);  mul_145 = unsqueeze_383 = None
    relu_22: "f32[8, 240, 14, 14]" = torch.ops.aten.relu.default(add_106);  add_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat_18: "f32[8, 480, 14, 14]" = torch.ops.aten.cat.default([relu_21, relu_22], 1);  relu_21 = relu_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_2: "f32[8, 480, 1, 1]" = torch.ops.aten.mean.dim(cat_18, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_52: "f32[8, 120, 1, 1]" = torch.ops.aten.convolution.default(mean_2, arg156_1, arg157_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_2 = arg156_1 = arg157_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    relu_23: "f32[8, 120, 1, 1]" = torch.ops.aten.relu.default(convolution_52);  convolution_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_53: "f32[8, 480, 1, 1]" = torch.ops.aten.convolution.default(relu_23, arg158_1, arg159_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_23 = arg158_1 = arg159_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    add_107: "f32[8, 480, 1, 1]" = torch.ops.aten.add.Tensor(convolution_53, 3);  convolution_53 = None
    clamp_min_2: "f32[8, 480, 1, 1]" = torch.ops.aten.clamp_min.default(add_107, 0);  add_107 = None
    clamp_max_2: "f32[8, 480, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_2, 6);  clamp_min_2 = None
    div_2: "f32[8, 480, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_2, 6);  clamp_max_2 = None
    mul_146: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(cat_18, div_2);  cat_18 = div_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_54: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(mul_146, arg160_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_146 = arg160_1 = None
    unsqueeze_384: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg418_1, -1);  arg418_1 = None
    unsqueeze_385: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_384, -1);  unsqueeze_384 = None
    sub_48: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_385);  convolution_54 = unsqueeze_385 = None
    add_108: "f32[56]" = torch.ops.aten.add.Tensor(arg419_1, 1e-05);  arg419_1 = None
    sqrt_48: "f32[56]" = torch.ops.aten.sqrt.default(add_108);  add_108 = None
    reciprocal_48: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_48);  sqrt_48 = None
    mul_147: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_48, 1);  reciprocal_48 = None
    unsqueeze_386: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_147, -1);  mul_147 = None
    unsqueeze_387: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, -1);  unsqueeze_386 = None
    mul_148: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_48, unsqueeze_387);  sub_48 = unsqueeze_387 = None
    unsqueeze_388: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg161_1, -1);  arg161_1 = None
    unsqueeze_389: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, -1);  unsqueeze_388 = None
    mul_149: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_148, unsqueeze_389);  mul_148 = unsqueeze_389 = None
    unsqueeze_390: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg162_1, -1);  arg162_1 = None
    unsqueeze_391: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_390, -1);  unsqueeze_390 = None
    add_109: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_149, unsqueeze_391);  mul_149 = unsqueeze_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_55: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_109, arg163_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 56);  arg163_1 = None
    unsqueeze_392: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg421_1, -1);  arg421_1 = None
    unsqueeze_393: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, -1);  unsqueeze_392 = None
    sub_49: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_393);  convolution_55 = unsqueeze_393 = None
    add_110: "f32[56]" = torch.ops.aten.add.Tensor(arg422_1, 1e-05);  arg422_1 = None
    sqrt_49: "f32[56]" = torch.ops.aten.sqrt.default(add_110);  add_110 = None
    reciprocal_49: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_49);  sqrt_49 = None
    mul_150: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_49, 1);  reciprocal_49 = None
    unsqueeze_394: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_150, -1);  mul_150 = None
    unsqueeze_395: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, -1);  unsqueeze_394 = None
    mul_151: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_49, unsqueeze_395);  sub_49 = unsqueeze_395 = None
    unsqueeze_396: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg164_1, -1);  arg164_1 = None
    unsqueeze_397: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_396, -1);  unsqueeze_396 = None
    mul_152: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_151, unsqueeze_397);  mul_151 = unsqueeze_397 = None
    unsqueeze_398: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg165_1, -1);  arg165_1 = None
    unsqueeze_399: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, -1);  unsqueeze_398 = None
    add_111: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_152, unsqueeze_399);  mul_152 = unsqueeze_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat_19: "f32[8, 112, 14, 14]" = torch.ops.aten.cat.default([add_109, add_111], 1);  add_109 = add_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    convolution_56: "f32[8, 80, 14, 14]" = torch.ops.aten.convolution.default(add_102, arg166_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 80);  add_102 = arg166_1 = None
    unsqueeze_400: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg424_1, -1);  arg424_1 = None
    unsqueeze_401: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, -1);  unsqueeze_400 = None
    sub_50: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_401);  convolution_56 = unsqueeze_401 = None
    add_112: "f32[80]" = torch.ops.aten.add.Tensor(arg425_1, 1e-05);  arg425_1 = None
    sqrt_50: "f32[80]" = torch.ops.aten.sqrt.default(add_112);  add_112 = None
    reciprocal_50: "f32[80]" = torch.ops.aten.reciprocal.default(sqrt_50);  sqrt_50 = None
    mul_153: "f32[80]" = torch.ops.aten.mul.Tensor(reciprocal_50, 1);  reciprocal_50 = None
    unsqueeze_402: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(mul_153, -1);  mul_153 = None
    unsqueeze_403: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, -1);  unsqueeze_402 = None
    mul_154: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_50, unsqueeze_403);  sub_50 = unsqueeze_403 = None
    unsqueeze_404: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg167_1, -1);  arg167_1 = None
    unsqueeze_405: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, -1);  unsqueeze_404 = None
    mul_155: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(mul_154, unsqueeze_405);  mul_154 = unsqueeze_405 = None
    unsqueeze_406: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg168_1, -1);  arg168_1 = None
    unsqueeze_407: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, -1);  unsqueeze_406 = None
    add_113: "f32[8, 80, 14, 14]" = torch.ops.aten.add.Tensor(mul_155, unsqueeze_407);  mul_155 = unsqueeze_407 = None
    convolution_57: "f32[8, 112, 14, 14]" = torch.ops.aten.convolution.default(add_113, arg169_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_113 = arg169_1 = None
    unsqueeze_408: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg427_1, -1);  arg427_1 = None
    unsqueeze_409: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_408, -1);  unsqueeze_408 = None
    sub_51: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_57, unsqueeze_409);  convolution_57 = unsqueeze_409 = None
    add_114: "f32[112]" = torch.ops.aten.add.Tensor(arg428_1, 1e-05);  arg428_1 = None
    sqrt_51: "f32[112]" = torch.ops.aten.sqrt.default(add_114);  add_114 = None
    reciprocal_51: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_51);  sqrt_51 = None
    mul_156: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_51, 1);  reciprocal_51 = None
    unsqueeze_410: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_156, -1);  mul_156 = None
    unsqueeze_411: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, -1);  unsqueeze_410 = None
    mul_157: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_51, unsqueeze_411);  sub_51 = unsqueeze_411 = None
    unsqueeze_412: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg170_1, -1);  arg170_1 = None
    unsqueeze_413: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, -1);  unsqueeze_412 = None
    mul_158: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(mul_157, unsqueeze_413);  mul_157 = unsqueeze_413 = None
    unsqueeze_414: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg171_1, -1);  arg171_1 = None
    unsqueeze_415: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, -1);  unsqueeze_414 = None
    add_115: "f32[8, 112, 14, 14]" = torch.ops.aten.add.Tensor(mul_158, unsqueeze_415);  mul_158 = unsqueeze_415 = None
    add_116: "f32[8, 112, 14, 14]" = torch.ops.aten.add.Tensor(cat_19, add_115);  cat_19 = add_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_58: "f32[8, 336, 14, 14]" = torch.ops.aten.convolution.default(add_116, arg172_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg172_1 = None
    unsqueeze_416: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg430_1, -1);  arg430_1 = None
    unsqueeze_417: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, -1);  unsqueeze_416 = None
    sub_52: "f32[8, 336, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_58, unsqueeze_417);  convolution_58 = unsqueeze_417 = None
    add_117: "f32[336]" = torch.ops.aten.add.Tensor(arg431_1, 1e-05);  arg431_1 = None
    sqrt_52: "f32[336]" = torch.ops.aten.sqrt.default(add_117);  add_117 = None
    reciprocal_52: "f32[336]" = torch.ops.aten.reciprocal.default(sqrt_52);  sqrt_52 = None
    mul_159: "f32[336]" = torch.ops.aten.mul.Tensor(reciprocal_52, 1);  reciprocal_52 = None
    unsqueeze_418: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(mul_159, -1);  mul_159 = None
    unsqueeze_419: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, -1);  unsqueeze_418 = None
    mul_160: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(sub_52, unsqueeze_419);  sub_52 = unsqueeze_419 = None
    unsqueeze_420: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg173_1, -1);  arg173_1 = None
    unsqueeze_421: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_420, -1);  unsqueeze_420 = None
    mul_161: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(mul_160, unsqueeze_421);  mul_160 = unsqueeze_421 = None
    unsqueeze_422: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg174_1, -1);  arg174_1 = None
    unsqueeze_423: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, -1);  unsqueeze_422 = None
    add_118: "f32[8, 336, 14, 14]" = torch.ops.aten.add.Tensor(mul_161, unsqueeze_423);  mul_161 = unsqueeze_423 = None
    relu_24: "f32[8, 336, 14, 14]" = torch.ops.aten.relu.default(add_118);  add_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_59: "f32[8, 336, 14, 14]" = torch.ops.aten.convolution.default(relu_24, arg175_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 336);  arg175_1 = None
    unsqueeze_424: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg433_1, -1);  arg433_1 = None
    unsqueeze_425: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_424, -1);  unsqueeze_424 = None
    sub_53: "f32[8, 336, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_425);  convolution_59 = unsqueeze_425 = None
    add_119: "f32[336]" = torch.ops.aten.add.Tensor(arg434_1, 1e-05);  arg434_1 = None
    sqrt_53: "f32[336]" = torch.ops.aten.sqrt.default(add_119);  add_119 = None
    reciprocal_53: "f32[336]" = torch.ops.aten.reciprocal.default(sqrt_53);  sqrt_53 = None
    mul_162: "f32[336]" = torch.ops.aten.mul.Tensor(reciprocal_53, 1);  reciprocal_53 = None
    unsqueeze_426: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(mul_162, -1);  mul_162 = None
    unsqueeze_427: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, -1);  unsqueeze_426 = None
    mul_163: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(sub_53, unsqueeze_427);  sub_53 = unsqueeze_427 = None
    unsqueeze_428: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg176_1, -1);  arg176_1 = None
    unsqueeze_429: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, -1);  unsqueeze_428 = None
    mul_164: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(mul_163, unsqueeze_429);  mul_163 = unsqueeze_429 = None
    unsqueeze_430: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg177_1, -1);  arg177_1 = None
    unsqueeze_431: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, -1);  unsqueeze_430 = None
    add_120: "f32[8, 336, 14, 14]" = torch.ops.aten.add.Tensor(mul_164, unsqueeze_431);  mul_164 = unsqueeze_431 = None
    relu_25: "f32[8, 336, 14, 14]" = torch.ops.aten.relu.default(add_120);  add_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat_20: "f32[8, 672, 14, 14]" = torch.ops.aten.cat.default([relu_24, relu_25], 1);  relu_24 = relu_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_3: "f32[8, 672, 1, 1]" = torch.ops.aten.mean.dim(cat_20, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_60: "f32[8, 168, 1, 1]" = torch.ops.aten.convolution.default(mean_3, arg178_1, arg179_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_3 = arg178_1 = arg179_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    relu_26: "f32[8, 168, 1, 1]" = torch.ops.aten.relu.default(convolution_60);  convolution_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_61: "f32[8, 672, 1, 1]" = torch.ops.aten.convolution.default(relu_26, arg180_1, arg181_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_26 = arg180_1 = arg181_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    add_121: "f32[8, 672, 1, 1]" = torch.ops.aten.add.Tensor(convolution_61, 3);  convolution_61 = None
    clamp_min_3: "f32[8, 672, 1, 1]" = torch.ops.aten.clamp_min.default(add_121, 0);  add_121 = None
    clamp_max_3: "f32[8, 672, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_3, 6);  clamp_min_3 = None
    div_3: "f32[8, 672, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_3, 6);  clamp_max_3 = None
    mul_165: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(cat_20, div_3);  cat_20 = div_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_62: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(mul_165, arg182_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_165 = arg182_1 = None
    unsqueeze_432: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg436_1, -1);  arg436_1 = None
    unsqueeze_433: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_432, -1);  unsqueeze_432 = None
    sub_54: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_62, unsqueeze_433);  convolution_62 = unsqueeze_433 = None
    add_122: "f32[56]" = torch.ops.aten.add.Tensor(arg437_1, 1e-05);  arg437_1 = None
    sqrt_54: "f32[56]" = torch.ops.aten.sqrt.default(add_122);  add_122 = None
    reciprocal_54: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_54);  sqrt_54 = None
    mul_166: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_54, 1);  reciprocal_54 = None
    unsqueeze_434: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_166, -1);  mul_166 = None
    unsqueeze_435: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, -1);  unsqueeze_434 = None
    mul_167: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_54, unsqueeze_435);  sub_54 = unsqueeze_435 = None
    unsqueeze_436: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg183_1, -1);  arg183_1 = None
    unsqueeze_437: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_436, -1);  unsqueeze_436 = None
    mul_168: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_167, unsqueeze_437);  mul_167 = unsqueeze_437 = None
    unsqueeze_438: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg184_1, -1);  arg184_1 = None
    unsqueeze_439: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_438, -1);  unsqueeze_438 = None
    add_123: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_168, unsqueeze_439);  mul_168 = unsqueeze_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_63: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_123, arg185_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 56);  arg185_1 = None
    unsqueeze_440: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg439_1, -1);  arg439_1 = None
    unsqueeze_441: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, -1);  unsqueeze_440 = None
    sub_55: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_63, unsqueeze_441);  convolution_63 = unsqueeze_441 = None
    add_124: "f32[56]" = torch.ops.aten.add.Tensor(arg440_1, 1e-05);  arg440_1 = None
    sqrt_55: "f32[56]" = torch.ops.aten.sqrt.default(add_124);  add_124 = None
    reciprocal_55: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_55);  sqrt_55 = None
    mul_169: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_55, 1);  reciprocal_55 = None
    unsqueeze_442: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_169, -1);  mul_169 = None
    unsqueeze_443: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_442, -1);  unsqueeze_442 = None
    mul_170: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_55, unsqueeze_443);  sub_55 = unsqueeze_443 = None
    unsqueeze_444: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg186_1, -1);  arg186_1 = None
    unsqueeze_445: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_444, -1);  unsqueeze_444 = None
    mul_171: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_170, unsqueeze_445);  mul_170 = unsqueeze_445 = None
    unsqueeze_446: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg187_1, -1);  arg187_1 = None
    unsqueeze_447: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, -1);  unsqueeze_446 = None
    add_125: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_171, unsqueeze_447);  mul_171 = unsqueeze_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat_21: "f32[8, 112, 14, 14]" = torch.ops.aten.cat.default([add_123, add_125], 1);  add_123 = add_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    add_126: "f32[8, 112, 14, 14]" = torch.ops.aten.add.Tensor(cat_21, add_116);  cat_21 = add_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_64: "f32[8, 336, 14, 14]" = torch.ops.aten.convolution.default(add_126, arg188_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg188_1 = None
    unsqueeze_448: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg442_1, -1);  arg442_1 = None
    unsqueeze_449: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_448, -1);  unsqueeze_448 = None
    sub_56: "f32[8, 336, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_64, unsqueeze_449);  convolution_64 = unsqueeze_449 = None
    add_127: "f32[336]" = torch.ops.aten.add.Tensor(arg443_1, 1e-05);  arg443_1 = None
    sqrt_56: "f32[336]" = torch.ops.aten.sqrt.default(add_127);  add_127 = None
    reciprocal_56: "f32[336]" = torch.ops.aten.reciprocal.default(sqrt_56);  sqrt_56 = None
    mul_172: "f32[336]" = torch.ops.aten.mul.Tensor(reciprocal_56, 1);  reciprocal_56 = None
    unsqueeze_450: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(mul_172, -1);  mul_172 = None
    unsqueeze_451: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_450, -1);  unsqueeze_450 = None
    mul_173: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(sub_56, unsqueeze_451);  sub_56 = unsqueeze_451 = None
    unsqueeze_452: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg189_1, -1);  arg189_1 = None
    unsqueeze_453: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, -1);  unsqueeze_452 = None
    mul_174: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(mul_173, unsqueeze_453);  mul_173 = unsqueeze_453 = None
    unsqueeze_454: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg190_1, -1);  arg190_1 = None
    unsqueeze_455: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_454, -1);  unsqueeze_454 = None
    add_128: "f32[8, 336, 14, 14]" = torch.ops.aten.add.Tensor(mul_174, unsqueeze_455);  mul_174 = unsqueeze_455 = None
    relu_27: "f32[8, 336, 14, 14]" = torch.ops.aten.relu.default(add_128);  add_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_65: "f32[8, 336, 14, 14]" = torch.ops.aten.convolution.default(relu_27, arg191_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 336);  arg191_1 = None
    unsqueeze_456: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg445_1, -1);  arg445_1 = None
    unsqueeze_457: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_456, -1);  unsqueeze_456 = None
    sub_57: "f32[8, 336, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_65, unsqueeze_457);  convolution_65 = unsqueeze_457 = None
    add_129: "f32[336]" = torch.ops.aten.add.Tensor(arg446_1, 1e-05);  arg446_1 = None
    sqrt_57: "f32[336]" = torch.ops.aten.sqrt.default(add_129);  add_129 = None
    reciprocal_57: "f32[336]" = torch.ops.aten.reciprocal.default(sqrt_57);  sqrt_57 = None
    mul_175: "f32[336]" = torch.ops.aten.mul.Tensor(reciprocal_57, 1);  reciprocal_57 = None
    unsqueeze_458: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(mul_175, -1);  mul_175 = None
    unsqueeze_459: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, -1);  unsqueeze_458 = None
    mul_176: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(sub_57, unsqueeze_459);  sub_57 = unsqueeze_459 = None
    unsqueeze_460: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg192_1, -1);  arg192_1 = None
    unsqueeze_461: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_460, -1);  unsqueeze_460 = None
    mul_177: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(mul_176, unsqueeze_461);  mul_176 = unsqueeze_461 = None
    unsqueeze_462: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg193_1, -1);  arg193_1 = None
    unsqueeze_463: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_462, -1);  unsqueeze_462 = None
    add_130: "f32[8, 336, 14, 14]" = torch.ops.aten.add.Tensor(mul_177, unsqueeze_463);  mul_177 = unsqueeze_463 = None
    relu_28: "f32[8, 336, 14, 14]" = torch.ops.aten.relu.default(add_130);  add_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat_22: "f32[8, 672, 14, 14]" = torch.ops.aten.cat.default([relu_27, relu_28], 1);  relu_27 = relu_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:172, code: x = self.conv_dw(x)
    convolution_66: "f32[8, 672, 7, 7]" = torch.ops.aten.convolution.default(cat_22, arg194_1, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 672);  cat_22 = arg194_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:173, code: x = self.bn_dw(x)
    unsqueeze_464: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(arg448_1, -1);  arg448_1 = None
    unsqueeze_465: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, -1);  unsqueeze_464 = None
    sub_58: "f32[8, 672, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_66, unsqueeze_465);  convolution_66 = unsqueeze_465 = None
    add_131: "f32[672]" = torch.ops.aten.add.Tensor(arg449_1, 1e-05);  arg449_1 = None
    sqrt_58: "f32[672]" = torch.ops.aten.sqrt.default(add_131);  add_131 = None
    reciprocal_58: "f32[672]" = torch.ops.aten.reciprocal.default(sqrt_58);  sqrt_58 = None
    mul_178: "f32[672]" = torch.ops.aten.mul.Tensor(reciprocal_58, 1);  reciprocal_58 = None
    unsqueeze_466: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(mul_178, -1);  mul_178 = None
    unsqueeze_467: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_466, -1);  unsqueeze_466 = None
    mul_179: "f32[8, 672, 7, 7]" = torch.ops.aten.mul.Tensor(sub_58, unsqueeze_467);  sub_58 = unsqueeze_467 = None
    unsqueeze_468: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(arg195_1, -1);  arg195_1 = None
    unsqueeze_469: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_468, -1);  unsqueeze_468 = None
    mul_180: "f32[8, 672, 7, 7]" = torch.ops.aten.mul.Tensor(mul_179, unsqueeze_469);  mul_179 = unsqueeze_469 = None
    unsqueeze_470: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(arg196_1, -1);  arg196_1 = None
    unsqueeze_471: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, -1);  unsqueeze_470 = None
    add_132: "f32[8, 672, 7, 7]" = torch.ops.aten.add.Tensor(mul_180, unsqueeze_471);  mul_180 = unsqueeze_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_4: "f32[8, 672, 1, 1]" = torch.ops.aten.mean.dim(add_132, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_67: "f32[8, 168, 1, 1]" = torch.ops.aten.convolution.default(mean_4, arg197_1, arg198_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_4 = arg197_1 = arg198_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    relu_29: "f32[8, 168, 1, 1]" = torch.ops.aten.relu.default(convolution_67);  convolution_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_68: "f32[8, 672, 1, 1]" = torch.ops.aten.convolution.default(relu_29, arg199_1, arg200_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_29 = arg199_1 = arg200_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    add_133: "f32[8, 672, 1, 1]" = torch.ops.aten.add.Tensor(convolution_68, 3);  convolution_68 = None
    clamp_min_4: "f32[8, 672, 1, 1]" = torch.ops.aten.clamp_min.default(add_133, 0);  add_133 = None
    clamp_max_4: "f32[8, 672, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_4, 6);  clamp_min_4 = None
    div_4: "f32[8, 672, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_4, 6);  clamp_max_4 = None
    mul_181: "f32[8, 672, 7, 7]" = torch.ops.aten.mul.Tensor(add_132, div_4);  add_132 = div_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_69: "f32[8, 80, 7, 7]" = torch.ops.aten.convolution.default(mul_181, arg201_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_181 = arg201_1 = None
    unsqueeze_472: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg451_1, -1);  arg451_1 = None
    unsqueeze_473: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_472, -1);  unsqueeze_472 = None
    sub_59: "f32[8, 80, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_69, unsqueeze_473);  convolution_69 = unsqueeze_473 = None
    add_134: "f32[80]" = torch.ops.aten.add.Tensor(arg452_1, 1e-05);  arg452_1 = None
    sqrt_59: "f32[80]" = torch.ops.aten.sqrt.default(add_134);  add_134 = None
    reciprocal_59: "f32[80]" = torch.ops.aten.reciprocal.default(sqrt_59);  sqrt_59 = None
    mul_182: "f32[80]" = torch.ops.aten.mul.Tensor(reciprocal_59, 1);  reciprocal_59 = None
    unsqueeze_474: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(mul_182, -1);  mul_182 = None
    unsqueeze_475: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_474, -1);  unsqueeze_474 = None
    mul_183: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(sub_59, unsqueeze_475);  sub_59 = unsqueeze_475 = None
    unsqueeze_476: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg202_1, -1);  arg202_1 = None
    unsqueeze_477: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, -1);  unsqueeze_476 = None
    mul_184: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(mul_183, unsqueeze_477);  mul_183 = unsqueeze_477 = None
    unsqueeze_478: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg203_1, -1);  arg203_1 = None
    unsqueeze_479: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_478, -1);  unsqueeze_478 = None
    add_135: "f32[8, 80, 7, 7]" = torch.ops.aten.add.Tensor(mul_184, unsqueeze_479);  mul_184 = unsqueeze_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_70: "f32[8, 80, 7, 7]" = torch.ops.aten.convolution.default(add_135, arg204_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 80);  arg204_1 = None
    unsqueeze_480: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg454_1, -1);  arg454_1 = None
    unsqueeze_481: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_480, -1);  unsqueeze_480 = None
    sub_60: "f32[8, 80, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_70, unsqueeze_481);  convolution_70 = unsqueeze_481 = None
    add_136: "f32[80]" = torch.ops.aten.add.Tensor(arg455_1, 1e-05);  arg455_1 = None
    sqrt_60: "f32[80]" = torch.ops.aten.sqrt.default(add_136);  add_136 = None
    reciprocal_60: "f32[80]" = torch.ops.aten.reciprocal.default(sqrt_60);  sqrt_60 = None
    mul_185: "f32[80]" = torch.ops.aten.mul.Tensor(reciprocal_60, 1);  reciprocal_60 = None
    unsqueeze_482: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(mul_185, -1);  mul_185 = None
    unsqueeze_483: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, -1);  unsqueeze_482 = None
    mul_186: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(sub_60, unsqueeze_483);  sub_60 = unsqueeze_483 = None
    unsqueeze_484: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg205_1, -1);  arg205_1 = None
    unsqueeze_485: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_484, -1);  unsqueeze_484 = None
    mul_187: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(mul_186, unsqueeze_485);  mul_186 = unsqueeze_485 = None
    unsqueeze_486: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg206_1, -1);  arg206_1 = None
    unsqueeze_487: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_486, -1);  unsqueeze_486 = None
    add_137: "f32[8, 80, 7, 7]" = torch.ops.aten.add.Tensor(mul_187, unsqueeze_487);  mul_187 = unsqueeze_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat_23: "f32[8, 160, 7, 7]" = torch.ops.aten.cat.default([add_135, add_137], 1);  add_135 = add_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    convolution_71: "f32[8, 112, 7, 7]" = torch.ops.aten.convolution.default(add_126, arg207_1, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 112);  add_126 = arg207_1 = None
    unsqueeze_488: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg457_1, -1);  arg457_1 = None
    unsqueeze_489: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, -1);  unsqueeze_488 = None
    sub_61: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_71, unsqueeze_489);  convolution_71 = unsqueeze_489 = None
    add_138: "f32[112]" = torch.ops.aten.add.Tensor(arg458_1, 1e-05);  arg458_1 = None
    sqrt_61: "f32[112]" = torch.ops.aten.sqrt.default(add_138);  add_138 = None
    reciprocal_61: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_61);  sqrt_61 = None
    mul_188: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_61, 1);  reciprocal_61 = None
    unsqueeze_490: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_188, -1);  mul_188 = None
    unsqueeze_491: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_490, -1);  unsqueeze_490 = None
    mul_189: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_61, unsqueeze_491);  sub_61 = unsqueeze_491 = None
    unsqueeze_492: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg208_1, -1);  arg208_1 = None
    unsqueeze_493: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_492, -1);  unsqueeze_492 = None
    mul_190: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(mul_189, unsqueeze_493);  mul_189 = unsqueeze_493 = None
    unsqueeze_494: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg209_1, -1);  arg209_1 = None
    unsqueeze_495: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, -1);  unsqueeze_494 = None
    add_139: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(mul_190, unsqueeze_495);  mul_190 = unsqueeze_495 = None
    convolution_72: "f32[8, 160, 7, 7]" = torch.ops.aten.convolution.default(add_139, arg210_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_139 = arg210_1 = None
    unsqueeze_496: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg460_1, -1);  arg460_1 = None
    unsqueeze_497: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_496, -1);  unsqueeze_496 = None
    sub_62: "f32[8, 160, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_72, unsqueeze_497);  convolution_72 = unsqueeze_497 = None
    add_140: "f32[160]" = torch.ops.aten.add.Tensor(arg461_1, 1e-05);  arg461_1 = None
    sqrt_62: "f32[160]" = torch.ops.aten.sqrt.default(add_140);  add_140 = None
    reciprocal_62: "f32[160]" = torch.ops.aten.reciprocal.default(sqrt_62);  sqrt_62 = None
    mul_191: "f32[160]" = torch.ops.aten.mul.Tensor(reciprocal_62, 1);  reciprocal_62 = None
    unsqueeze_498: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(mul_191, -1);  mul_191 = None
    unsqueeze_499: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_498, -1);  unsqueeze_498 = None
    mul_192: "f32[8, 160, 7, 7]" = torch.ops.aten.mul.Tensor(sub_62, unsqueeze_499);  sub_62 = unsqueeze_499 = None
    unsqueeze_500: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg211_1, -1);  arg211_1 = None
    unsqueeze_501: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_500, -1);  unsqueeze_500 = None
    mul_193: "f32[8, 160, 7, 7]" = torch.ops.aten.mul.Tensor(mul_192, unsqueeze_501);  mul_192 = unsqueeze_501 = None
    unsqueeze_502: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg212_1, -1);  arg212_1 = None
    unsqueeze_503: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_502, -1);  unsqueeze_502 = None
    add_141: "f32[8, 160, 7, 7]" = torch.ops.aten.add.Tensor(mul_193, unsqueeze_503);  mul_193 = unsqueeze_503 = None
    add_142: "f32[8, 160, 7, 7]" = torch.ops.aten.add.Tensor(cat_23, add_141);  cat_23 = add_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_73: "f32[8, 480, 7, 7]" = torch.ops.aten.convolution.default(add_142, arg213_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg213_1 = None
    unsqueeze_504: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg463_1, -1);  arg463_1 = None
    unsqueeze_505: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_504, -1);  unsqueeze_504 = None
    sub_63: "f32[8, 480, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_73, unsqueeze_505);  convolution_73 = unsqueeze_505 = None
    add_143: "f32[480]" = torch.ops.aten.add.Tensor(arg464_1, 1e-05);  arg464_1 = None
    sqrt_63: "f32[480]" = torch.ops.aten.sqrt.default(add_143);  add_143 = None
    reciprocal_63: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_63);  sqrt_63 = None
    mul_194: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_63, 1);  reciprocal_63 = None
    unsqueeze_506: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_194, -1);  mul_194 = None
    unsqueeze_507: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_506, -1);  unsqueeze_506 = None
    mul_195: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(sub_63, unsqueeze_507);  sub_63 = unsqueeze_507 = None
    unsqueeze_508: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg214_1, -1);  arg214_1 = None
    unsqueeze_509: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_508, -1);  unsqueeze_508 = None
    mul_196: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(mul_195, unsqueeze_509);  mul_195 = unsqueeze_509 = None
    unsqueeze_510: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg215_1, -1);  arg215_1 = None
    unsqueeze_511: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_510, -1);  unsqueeze_510 = None
    add_144: "f32[8, 480, 7, 7]" = torch.ops.aten.add.Tensor(mul_196, unsqueeze_511);  mul_196 = unsqueeze_511 = None
    relu_30: "f32[8, 480, 7, 7]" = torch.ops.aten.relu.default(add_144);  add_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_74: "f32[8, 480, 7, 7]" = torch.ops.aten.convolution.default(relu_30, arg216_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 480);  arg216_1 = None
    unsqueeze_512: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg466_1, -1);  arg466_1 = None
    unsqueeze_513: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_512, -1);  unsqueeze_512 = None
    sub_64: "f32[8, 480, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_74, unsqueeze_513);  convolution_74 = unsqueeze_513 = None
    add_145: "f32[480]" = torch.ops.aten.add.Tensor(arg467_1, 1e-05);  arg467_1 = None
    sqrt_64: "f32[480]" = torch.ops.aten.sqrt.default(add_145);  add_145 = None
    reciprocal_64: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_64);  sqrt_64 = None
    mul_197: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_64, 1);  reciprocal_64 = None
    unsqueeze_514: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_197, -1);  mul_197 = None
    unsqueeze_515: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_514, -1);  unsqueeze_514 = None
    mul_198: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_515);  sub_64 = unsqueeze_515 = None
    unsqueeze_516: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg217_1, -1);  arg217_1 = None
    unsqueeze_517: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_516, -1);  unsqueeze_516 = None
    mul_199: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(mul_198, unsqueeze_517);  mul_198 = unsqueeze_517 = None
    unsqueeze_518: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg218_1, -1);  arg218_1 = None
    unsqueeze_519: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_518, -1);  unsqueeze_518 = None
    add_146: "f32[8, 480, 7, 7]" = torch.ops.aten.add.Tensor(mul_199, unsqueeze_519);  mul_199 = unsqueeze_519 = None
    relu_31: "f32[8, 480, 7, 7]" = torch.ops.aten.relu.default(add_146);  add_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat_24: "f32[8, 960, 7, 7]" = torch.ops.aten.cat.default([relu_30, relu_31], 1);  relu_30 = relu_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_75: "f32[8, 80, 7, 7]" = torch.ops.aten.convolution.default(cat_24, arg219_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_24 = arg219_1 = None
    unsqueeze_520: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg469_1, -1);  arg469_1 = None
    unsqueeze_521: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_520, -1);  unsqueeze_520 = None
    sub_65: "f32[8, 80, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_75, unsqueeze_521);  convolution_75 = unsqueeze_521 = None
    add_147: "f32[80]" = torch.ops.aten.add.Tensor(arg470_1, 1e-05);  arg470_1 = None
    sqrt_65: "f32[80]" = torch.ops.aten.sqrt.default(add_147);  add_147 = None
    reciprocal_65: "f32[80]" = torch.ops.aten.reciprocal.default(sqrt_65);  sqrt_65 = None
    mul_200: "f32[80]" = torch.ops.aten.mul.Tensor(reciprocal_65, 1);  reciprocal_65 = None
    unsqueeze_522: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(mul_200, -1);  mul_200 = None
    unsqueeze_523: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_522, -1);  unsqueeze_522 = None
    mul_201: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(sub_65, unsqueeze_523);  sub_65 = unsqueeze_523 = None
    unsqueeze_524: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg220_1, -1);  arg220_1 = None
    unsqueeze_525: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_524, -1);  unsqueeze_524 = None
    mul_202: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(mul_201, unsqueeze_525);  mul_201 = unsqueeze_525 = None
    unsqueeze_526: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg221_1, -1);  arg221_1 = None
    unsqueeze_527: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_526, -1);  unsqueeze_526 = None
    add_148: "f32[8, 80, 7, 7]" = torch.ops.aten.add.Tensor(mul_202, unsqueeze_527);  mul_202 = unsqueeze_527 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_76: "f32[8, 80, 7, 7]" = torch.ops.aten.convolution.default(add_148, arg222_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 80);  arg222_1 = None
    unsqueeze_528: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg472_1, -1);  arg472_1 = None
    unsqueeze_529: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_528, -1);  unsqueeze_528 = None
    sub_66: "f32[8, 80, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_76, unsqueeze_529);  convolution_76 = unsqueeze_529 = None
    add_149: "f32[80]" = torch.ops.aten.add.Tensor(arg473_1, 1e-05);  arg473_1 = None
    sqrt_66: "f32[80]" = torch.ops.aten.sqrt.default(add_149);  add_149 = None
    reciprocal_66: "f32[80]" = torch.ops.aten.reciprocal.default(sqrt_66);  sqrt_66 = None
    mul_203: "f32[80]" = torch.ops.aten.mul.Tensor(reciprocal_66, 1);  reciprocal_66 = None
    unsqueeze_530: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(mul_203, -1);  mul_203 = None
    unsqueeze_531: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_530, -1);  unsqueeze_530 = None
    mul_204: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(sub_66, unsqueeze_531);  sub_66 = unsqueeze_531 = None
    unsqueeze_532: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg223_1, -1);  arg223_1 = None
    unsqueeze_533: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_532, -1);  unsqueeze_532 = None
    mul_205: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(mul_204, unsqueeze_533);  mul_204 = unsqueeze_533 = None
    unsqueeze_534: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg224_1, -1);  arg224_1 = None
    unsqueeze_535: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_534, -1);  unsqueeze_534 = None
    add_150: "f32[8, 80, 7, 7]" = torch.ops.aten.add.Tensor(mul_205, unsqueeze_535);  mul_205 = unsqueeze_535 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat_25: "f32[8, 160, 7, 7]" = torch.ops.aten.cat.default([add_148, add_150], 1);  add_148 = add_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    add_151: "f32[8, 160, 7, 7]" = torch.ops.aten.add.Tensor(cat_25, add_142);  cat_25 = add_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_77: "f32[8, 480, 7, 7]" = torch.ops.aten.convolution.default(add_151, arg225_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg225_1 = None
    unsqueeze_536: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg475_1, -1);  arg475_1 = None
    unsqueeze_537: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_536, -1);  unsqueeze_536 = None
    sub_67: "f32[8, 480, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_77, unsqueeze_537);  convolution_77 = unsqueeze_537 = None
    add_152: "f32[480]" = torch.ops.aten.add.Tensor(arg476_1, 1e-05);  arg476_1 = None
    sqrt_67: "f32[480]" = torch.ops.aten.sqrt.default(add_152);  add_152 = None
    reciprocal_67: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_67);  sqrt_67 = None
    mul_206: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_67, 1);  reciprocal_67 = None
    unsqueeze_538: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_206, -1);  mul_206 = None
    unsqueeze_539: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_538, -1);  unsqueeze_538 = None
    mul_207: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_539);  sub_67 = unsqueeze_539 = None
    unsqueeze_540: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg226_1, -1);  arg226_1 = None
    unsqueeze_541: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_540, -1);  unsqueeze_540 = None
    mul_208: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(mul_207, unsqueeze_541);  mul_207 = unsqueeze_541 = None
    unsqueeze_542: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg227_1, -1);  arg227_1 = None
    unsqueeze_543: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_542, -1);  unsqueeze_542 = None
    add_153: "f32[8, 480, 7, 7]" = torch.ops.aten.add.Tensor(mul_208, unsqueeze_543);  mul_208 = unsqueeze_543 = None
    relu_32: "f32[8, 480, 7, 7]" = torch.ops.aten.relu.default(add_153);  add_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_78: "f32[8, 480, 7, 7]" = torch.ops.aten.convolution.default(relu_32, arg228_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 480);  arg228_1 = None
    unsqueeze_544: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg478_1, -1);  arg478_1 = None
    unsqueeze_545: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_544, -1);  unsqueeze_544 = None
    sub_68: "f32[8, 480, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_78, unsqueeze_545);  convolution_78 = unsqueeze_545 = None
    add_154: "f32[480]" = torch.ops.aten.add.Tensor(arg479_1, 1e-05);  arg479_1 = None
    sqrt_68: "f32[480]" = torch.ops.aten.sqrt.default(add_154);  add_154 = None
    reciprocal_68: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_68);  sqrt_68 = None
    mul_209: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_68, 1);  reciprocal_68 = None
    unsqueeze_546: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_209, -1);  mul_209 = None
    unsqueeze_547: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_546, -1);  unsqueeze_546 = None
    mul_210: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(sub_68, unsqueeze_547);  sub_68 = unsqueeze_547 = None
    unsqueeze_548: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg229_1, -1);  arg229_1 = None
    unsqueeze_549: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_548, -1);  unsqueeze_548 = None
    mul_211: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(mul_210, unsqueeze_549);  mul_210 = unsqueeze_549 = None
    unsqueeze_550: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg230_1, -1);  arg230_1 = None
    unsqueeze_551: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_550, -1);  unsqueeze_550 = None
    add_155: "f32[8, 480, 7, 7]" = torch.ops.aten.add.Tensor(mul_211, unsqueeze_551);  mul_211 = unsqueeze_551 = None
    relu_33: "f32[8, 480, 7, 7]" = torch.ops.aten.relu.default(add_155);  add_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat_26: "f32[8, 960, 7, 7]" = torch.ops.aten.cat.default([relu_32, relu_33], 1);  relu_32 = relu_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_5: "f32[8, 960, 1, 1]" = torch.ops.aten.mean.dim(cat_26, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_79: "f32[8, 240, 1, 1]" = torch.ops.aten.convolution.default(mean_5, arg231_1, arg232_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_5 = arg231_1 = arg232_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    relu_34: "f32[8, 240, 1, 1]" = torch.ops.aten.relu.default(convolution_79);  convolution_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_80: "f32[8, 960, 1, 1]" = torch.ops.aten.convolution.default(relu_34, arg233_1, arg234_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_34 = arg233_1 = arg234_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    add_156: "f32[8, 960, 1, 1]" = torch.ops.aten.add.Tensor(convolution_80, 3);  convolution_80 = None
    clamp_min_5: "f32[8, 960, 1, 1]" = torch.ops.aten.clamp_min.default(add_156, 0);  add_156 = None
    clamp_max_5: "f32[8, 960, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_5, 6);  clamp_min_5 = None
    div_5: "f32[8, 960, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_5, 6);  clamp_max_5 = None
    mul_212: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(cat_26, div_5);  cat_26 = div_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_81: "f32[8, 80, 7, 7]" = torch.ops.aten.convolution.default(mul_212, arg235_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_212 = arg235_1 = None
    unsqueeze_552: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg481_1, -1);  arg481_1 = None
    unsqueeze_553: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_552, -1);  unsqueeze_552 = None
    sub_69: "f32[8, 80, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_81, unsqueeze_553);  convolution_81 = unsqueeze_553 = None
    add_157: "f32[80]" = torch.ops.aten.add.Tensor(arg482_1, 1e-05);  arg482_1 = None
    sqrt_69: "f32[80]" = torch.ops.aten.sqrt.default(add_157);  add_157 = None
    reciprocal_69: "f32[80]" = torch.ops.aten.reciprocal.default(sqrt_69);  sqrt_69 = None
    mul_213: "f32[80]" = torch.ops.aten.mul.Tensor(reciprocal_69, 1);  reciprocal_69 = None
    unsqueeze_554: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(mul_213, -1);  mul_213 = None
    unsqueeze_555: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_554, -1);  unsqueeze_554 = None
    mul_214: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(sub_69, unsqueeze_555);  sub_69 = unsqueeze_555 = None
    unsqueeze_556: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg236_1, -1);  arg236_1 = None
    unsqueeze_557: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_556, -1);  unsqueeze_556 = None
    mul_215: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(mul_214, unsqueeze_557);  mul_214 = unsqueeze_557 = None
    unsqueeze_558: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg237_1, -1);  arg237_1 = None
    unsqueeze_559: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_558, -1);  unsqueeze_558 = None
    add_158: "f32[8, 80, 7, 7]" = torch.ops.aten.add.Tensor(mul_215, unsqueeze_559);  mul_215 = unsqueeze_559 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_82: "f32[8, 80, 7, 7]" = torch.ops.aten.convolution.default(add_158, arg238_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 80);  arg238_1 = None
    unsqueeze_560: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg484_1, -1);  arg484_1 = None
    unsqueeze_561: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_560, -1);  unsqueeze_560 = None
    sub_70: "f32[8, 80, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_82, unsqueeze_561);  convolution_82 = unsqueeze_561 = None
    add_159: "f32[80]" = torch.ops.aten.add.Tensor(arg485_1, 1e-05);  arg485_1 = None
    sqrt_70: "f32[80]" = torch.ops.aten.sqrt.default(add_159);  add_159 = None
    reciprocal_70: "f32[80]" = torch.ops.aten.reciprocal.default(sqrt_70);  sqrt_70 = None
    mul_216: "f32[80]" = torch.ops.aten.mul.Tensor(reciprocal_70, 1);  reciprocal_70 = None
    unsqueeze_562: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(mul_216, -1);  mul_216 = None
    unsqueeze_563: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_562, -1);  unsqueeze_562 = None
    mul_217: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(sub_70, unsqueeze_563);  sub_70 = unsqueeze_563 = None
    unsqueeze_564: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg239_1, -1);  arg239_1 = None
    unsqueeze_565: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_564, -1);  unsqueeze_564 = None
    mul_218: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(mul_217, unsqueeze_565);  mul_217 = unsqueeze_565 = None
    unsqueeze_566: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg240_1, -1);  arg240_1 = None
    unsqueeze_567: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_566, -1);  unsqueeze_566 = None
    add_160: "f32[8, 80, 7, 7]" = torch.ops.aten.add.Tensor(mul_218, unsqueeze_567);  mul_218 = unsqueeze_567 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat_27: "f32[8, 160, 7, 7]" = torch.ops.aten.cat.default([add_158, add_160], 1);  add_158 = add_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    add_161: "f32[8, 160, 7, 7]" = torch.ops.aten.add.Tensor(cat_27, add_151);  cat_27 = add_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_83: "f32[8, 480, 7, 7]" = torch.ops.aten.convolution.default(add_161, arg241_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg241_1 = None
    unsqueeze_568: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg487_1, -1);  arg487_1 = None
    unsqueeze_569: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_568, -1);  unsqueeze_568 = None
    sub_71: "f32[8, 480, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_83, unsqueeze_569);  convolution_83 = unsqueeze_569 = None
    add_162: "f32[480]" = torch.ops.aten.add.Tensor(arg488_1, 1e-05);  arg488_1 = None
    sqrt_71: "f32[480]" = torch.ops.aten.sqrt.default(add_162);  add_162 = None
    reciprocal_71: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_71);  sqrt_71 = None
    mul_219: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_71, 1);  reciprocal_71 = None
    unsqueeze_570: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_219, -1);  mul_219 = None
    unsqueeze_571: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_570, -1);  unsqueeze_570 = None
    mul_220: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(sub_71, unsqueeze_571);  sub_71 = unsqueeze_571 = None
    unsqueeze_572: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg242_1, -1);  arg242_1 = None
    unsqueeze_573: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_572, -1);  unsqueeze_572 = None
    mul_221: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(mul_220, unsqueeze_573);  mul_220 = unsqueeze_573 = None
    unsqueeze_574: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg243_1, -1);  arg243_1 = None
    unsqueeze_575: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_574, -1);  unsqueeze_574 = None
    add_163: "f32[8, 480, 7, 7]" = torch.ops.aten.add.Tensor(mul_221, unsqueeze_575);  mul_221 = unsqueeze_575 = None
    relu_35: "f32[8, 480, 7, 7]" = torch.ops.aten.relu.default(add_163);  add_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_84: "f32[8, 480, 7, 7]" = torch.ops.aten.convolution.default(relu_35, arg244_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 480);  arg244_1 = None
    unsqueeze_576: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg490_1, -1);  arg490_1 = None
    unsqueeze_577: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_576, -1);  unsqueeze_576 = None
    sub_72: "f32[8, 480, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_84, unsqueeze_577);  convolution_84 = unsqueeze_577 = None
    add_164: "f32[480]" = torch.ops.aten.add.Tensor(arg491_1, 1e-05);  arg491_1 = None
    sqrt_72: "f32[480]" = torch.ops.aten.sqrt.default(add_164);  add_164 = None
    reciprocal_72: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_72);  sqrt_72 = None
    mul_222: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_72, 1);  reciprocal_72 = None
    unsqueeze_578: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_222, -1);  mul_222 = None
    unsqueeze_579: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_578, -1);  unsqueeze_578 = None
    mul_223: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(sub_72, unsqueeze_579);  sub_72 = unsqueeze_579 = None
    unsqueeze_580: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg245_1, -1);  arg245_1 = None
    unsqueeze_581: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_580, -1);  unsqueeze_580 = None
    mul_224: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(mul_223, unsqueeze_581);  mul_223 = unsqueeze_581 = None
    unsqueeze_582: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg246_1, -1);  arg246_1 = None
    unsqueeze_583: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_582, -1);  unsqueeze_582 = None
    add_165: "f32[8, 480, 7, 7]" = torch.ops.aten.add.Tensor(mul_224, unsqueeze_583);  mul_224 = unsqueeze_583 = None
    relu_36: "f32[8, 480, 7, 7]" = torch.ops.aten.relu.default(add_165);  add_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat_28: "f32[8, 960, 7, 7]" = torch.ops.aten.cat.default([relu_35, relu_36], 1);  relu_35 = relu_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_85: "f32[8, 80, 7, 7]" = torch.ops.aten.convolution.default(cat_28, arg247_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_28 = arg247_1 = None
    unsqueeze_584: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg493_1, -1);  arg493_1 = None
    unsqueeze_585: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_584, -1);  unsqueeze_584 = None
    sub_73: "f32[8, 80, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_85, unsqueeze_585);  convolution_85 = unsqueeze_585 = None
    add_166: "f32[80]" = torch.ops.aten.add.Tensor(arg494_1, 1e-05);  arg494_1 = None
    sqrt_73: "f32[80]" = torch.ops.aten.sqrt.default(add_166);  add_166 = None
    reciprocal_73: "f32[80]" = torch.ops.aten.reciprocal.default(sqrt_73);  sqrt_73 = None
    mul_225: "f32[80]" = torch.ops.aten.mul.Tensor(reciprocal_73, 1);  reciprocal_73 = None
    unsqueeze_586: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(mul_225, -1);  mul_225 = None
    unsqueeze_587: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_586, -1);  unsqueeze_586 = None
    mul_226: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(sub_73, unsqueeze_587);  sub_73 = unsqueeze_587 = None
    unsqueeze_588: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg248_1, -1);  arg248_1 = None
    unsqueeze_589: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_588, -1);  unsqueeze_588 = None
    mul_227: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(mul_226, unsqueeze_589);  mul_226 = unsqueeze_589 = None
    unsqueeze_590: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg249_1, -1);  arg249_1 = None
    unsqueeze_591: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_590, -1);  unsqueeze_590 = None
    add_167: "f32[8, 80, 7, 7]" = torch.ops.aten.add.Tensor(mul_227, unsqueeze_591);  mul_227 = unsqueeze_591 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_86: "f32[8, 80, 7, 7]" = torch.ops.aten.convolution.default(add_167, arg250_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 80);  arg250_1 = None
    unsqueeze_592: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg496_1, -1);  arg496_1 = None
    unsqueeze_593: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_592, -1);  unsqueeze_592 = None
    sub_74: "f32[8, 80, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_86, unsqueeze_593);  convolution_86 = unsqueeze_593 = None
    add_168: "f32[80]" = torch.ops.aten.add.Tensor(arg497_1, 1e-05);  arg497_1 = None
    sqrt_74: "f32[80]" = torch.ops.aten.sqrt.default(add_168);  add_168 = None
    reciprocal_74: "f32[80]" = torch.ops.aten.reciprocal.default(sqrt_74);  sqrt_74 = None
    mul_228: "f32[80]" = torch.ops.aten.mul.Tensor(reciprocal_74, 1);  reciprocal_74 = None
    unsqueeze_594: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(mul_228, -1);  mul_228 = None
    unsqueeze_595: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_594, -1);  unsqueeze_594 = None
    mul_229: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(sub_74, unsqueeze_595);  sub_74 = unsqueeze_595 = None
    unsqueeze_596: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg251_1, -1);  arg251_1 = None
    unsqueeze_597: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_596, -1);  unsqueeze_596 = None
    mul_230: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(mul_229, unsqueeze_597);  mul_229 = unsqueeze_597 = None
    unsqueeze_598: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg252_1, -1);  arg252_1 = None
    unsqueeze_599: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_598, -1);  unsqueeze_598 = None
    add_169: "f32[8, 80, 7, 7]" = torch.ops.aten.add.Tensor(mul_230, unsqueeze_599);  mul_230 = unsqueeze_599 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat_29: "f32[8, 160, 7, 7]" = torch.ops.aten.cat.default([add_167, add_169], 1);  add_167 = add_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    add_170: "f32[8, 160, 7, 7]" = torch.ops.aten.add.Tensor(cat_29, add_161);  cat_29 = add_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_87: "f32[8, 480, 7, 7]" = torch.ops.aten.convolution.default(add_170, arg253_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg253_1 = None
    unsqueeze_600: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg499_1, -1);  arg499_1 = None
    unsqueeze_601: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_600, -1);  unsqueeze_600 = None
    sub_75: "f32[8, 480, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_87, unsqueeze_601);  convolution_87 = unsqueeze_601 = None
    add_171: "f32[480]" = torch.ops.aten.add.Tensor(arg500_1, 1e-05);  arg500_1 = None
    sqrt_75: "f32[480]" = torch.ops.aten.sqrt.default(add_171);  add_171 = None
    reciprocal_75: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_75);  sqrt_75 = None
    mul_231: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_75, 1);  reciprocal_75 = None
    unsqueeze_602: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_231, -1);  mul_231 = None
    unsqueeze_603: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_602, -1);  unsqueeze_602 = None
    mul_232: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(sub_75, unsqueeze_603);  sub_75 = unsqueeze_603 = None
    unsqueeze_604: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg254_1, -1);  arg254_1 = None
    unsqueeze_605: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_604, -1);  unsqueeze_604 = None
    mul_233: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(mul_232, unsqueeze_605);  mul_232 = unsqueeze_605 = None
    unsqueeze_606: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg255_1, -1);  arg255_1 = None
    unsqueeze_607: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_606, -1);  unsqueeze_606 = None
    add_172: "f32[8, 480, 7, 7]" = torch.ops.aten.add.Tensor(mul_233, unsqueeze_607);  mul_233 = unsqueeze_607 = None
    relu_37: "f32[8, 480, 7, 7]" = torch.ops.aten.relu.default(add_172);  add_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_88: "f32[8, 480, 7, 7]" = torch.ops.aten.convolution.default(relu_37, arg256_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 480);  arg256_1 = None
    unsqueeze_608: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg502_1, -1);  arg502_1 = None
    unsqueeze_609: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_608, -1);  unsqueeze_608 = None
    sub_76: "f32[8, 480, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_88, unsqueeze_609);  convolution_88 = unsqueeze_609 = None
    add_173: "f32[480]" = torch.ops.aten.add.Tensor(arg503_1, 1e-05);  arg503_1 = None
    sqrt_76: "f32[480]" = torch.ops.aten.sqrt.default(add_173);  add_173 = None
    reciprocal_76: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_76);  sqrt_76 = None
    mul_234: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_76, 1);  reciprocal_76 = None
    unsqueeze_610: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_234, -1);  mul_234 = None
    unsqueeze_611: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_610, -1);  unsqueeze_610 = None
    mul_235: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_611);  sub_76 = unsqueeze_611 = None
    unsqueeze_612: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg257_1, -1);  arg257_1 = None
    unsqueeze_613: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_612, -1);  unsqueeze_612 = None
    mul_236: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(mul_235, unsqueeze_613);  mul_235 = unsqueeze_613 = None
    unsqueeze_614: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg258_1, -1);  arg258_1 = None
    unsqueeze_615: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_614, -1);  unsqueeze_614 = None
    add_174: "f32[8, 480, 7, 7]" = torch.ops.aten.add.Tensor(mul_236, unsqueeze_615);  mul_236 = unsqueeze_615 = None
    relu_38: "f32[8, 480, 7, 7]" = torch.ops.aten.relu.default(add_174);  add_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat_30: "f32[8, 960, 7, 7]" = torch.ops.aten.cat.default([relu_37, relu_38], 1);  relu_37 = relu_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_6: "f32[8, 960, 1, 1]" = torch.ops.aten.mean.dim(cat_30, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_89: "f32[8, 240, 1, 1]" = torch.ops.aten.convolution.default(mean_6, arg259_1, arg260_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_6 = arg259_1 = arg260_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    relu_39: "f32[8, 240, 1, 1]" = torch.ops.aten.relu.default(convolution_89);  convolution_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_90: "f32[8, 960, 1, 1]" = torch.ops.aten.convolution.default(relu_39, arg261_1, arg262_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_39 = arg261_1 = arg262_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    add_175: "f32[8, 960, 1, 1]" = torch.ops.aten.add.Tensor(convolution_90, 3);  convolution_90 = None
    clamp_min_6: "f32[8, 960, 1, 1]" = torch.ops.aten.clamp_min.default(add_175, 0);  add_175 = None
    clamp_max_6: "f32[8, 960, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_6, 6);  clamp_min_6 = None
    div_6: "f32[8, 960, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_6, 6);  clamp_max_6 = None
    mul_237: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(cat_30, div_6);  cat_30 = div_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_91: "f32[8, 80, 7, 7]" = torch.ops.aten.convolution.default(mul_237, arg263_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_237 = arg263_1 = None
    unsqueeze_616: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg505_1, -1);  arg505_1 = None
    unsqueeze_617: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_616, -1);  unsqueeze_616 = None
    sub_77: "f32[8, 80, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_91, unsqueeze_617);  convolution_91 = unsqueeze_617 = None
    add_176: "f32[80]" = torch.ops.aten.add.Tensor(arg506_1, 1e-05);  arg506_1 = None
    sqrt_77: "f32[80]" = torch.ops.aten.sqrt.default(add_176);  add_176 = None
    reciprocal_77: "f32[80]" = torch.ops.aten.reciprocal.default(sqrt_77);  sqrt_77 = None
    mul_238: "f32[80]" = torch.ops.aten.mul.Tensor(reciprocal_77, 1);  reciprocal_77 = None
    unsqueeze_618: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(mul_238, -1);  mul_238 = None
    unsqueeze_619: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_618, -1);  unsqueeze_618 = None
    mul_239: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(sub_77, unsqueeze_619);  sub_77 = unsqueeze_619 = None
    unsqueeze_620: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg264_1, -1);  arg264_1 = None
    unsqueeze_621: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_620, -1);  unsqueeze_620 = None
    mul_240: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(mul_239, unsqueeze_621);  mul_239 = unsqueeze_621 = None
    unsqueeze_622: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg265_1, -1);  arg265_1 = None
    unsqueeze_623: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_622, -1);  unsqueeze_622 = None
    add_177: "f32[8, 80, 7, 7]" = torch.ops.aten.add.Tensor(mul_240, unsqueeze_623);  mul_240 = unsqueeze_623 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_92: "f32[8, 80, 7, 7]" = torch.ops.aten.convolution.default(add_177, arg266_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 80);  arg266_1 = None
    unsqueeze_624: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg508_1, -1);  arg508_1 = None
    unsqueeze_625: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_624, -1);  unsqueeze_624 = None
    sub_78: "f32[8, 80, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_92, unsqueeze_625);  convolution_92 = unsqueeze_625 = None
    add_178: "f32[80]" = torch.ops.aten.add.Tensor(arg509_1, 1e-05);  arg509_1 = None
    sqrt_78: "f32[80]" = torch.ops.aten.sqrt.default(add_178);  add_178 = None
    reciprocal_78: "f32[80]" = torch.ops.aten.reciprocal.default(sqrt_78);  sqrt_78 = None
    mul_241: "f32[80]" = torch.ops.aten.mul.Tensor(reciprocal_78, 1);  reciprocal_78 = None
    unsqueeze_626: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(mul_241, -1);  mul_241 = None
    unsqueeze_627: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_626, -1);  unsqueeze_626 = None
    mul_242: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(sub_78, unsqueeze_627);  sub_78 = unsqueeze_627 = None
    unsqueeze_628: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg267_1, -1);  arg267_1 = None
    unsqueeze_629: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_628, -1);  unsqueeze_628 = None
    mul_243: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(mul_242, unsqueeze_629);  mul_242 = unsqueeze_629 = None
    unsqueeze_630: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg268_1, -1);  arg268_1 = None
    unsqueeze_631: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_630, -1);  unsqueeze_630 = None
    add_179: "f32[8, 80, 7, 7]" = torch.ops.aten.add.Tensor(mul_243, unsqueeze_631);  mul_243 = unsqueeze_631 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat_31: "f32[8, 160, 7, 7]" = torch.ops.aten.cat.default([add_177, add_179], 1);  add_177 = add_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    add_180: "f32[8, 160, 7, 7]" = torch.ops.aten.add.Tensor(cat_31, add_170);  cat_31 = add_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:82, code: x = self.conv(x)
    convolution_93: "f32[8, 960, 7, 7]" = torch.ops.aten.convolution.default(add_180, arg269_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_180 = arg269_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_632: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(arg272_1, -1);  arg272_1 = None
    unsqueeze_633: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_632, -1);  unsqueeze_632 = None
    sub_79: "f32[8, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_93, unsqueeze_633);  convolution_93 = unsqueeze_633 = None
    add_181: "f32[960]" = torch.ops.aten.add.Tensor(arg273_1, 1e-05);  arg273_1 = None
    sqrt_79: "f32[960]" = torch.ops.aten.sqrt.default(add_181);  add_181 = None
    reciprocal_79: "f32[960]" = torch.ops.aten.reciprocal.default(sqrt_79);  sqrt_79 = None
    mul_244: "f32[960]" = torch.ops.aten.mul.Tensor(reciprocal_79, 1);  reciprocal_79 = None
    unsqueeze_634: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(mul_244, -1);  mul_244 = None
    unsqueeze_635: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_634, -1);  unsqueeze_634 = None
    mul_245: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_635);  sub_79 = unsqueeze_635 = None
    unsqueeze_636: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(arg0_1, -1);  arg0_1 = None
    unsqueeze_637: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_636, -1);  unsqueeze_636 = None
    mul_246: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(mul_245, unsqueeze_637);  mul_245 = unsqueeze_637 = None
    unsqueeze_638: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(arg1_1, -1);  arg1_1 = None
    unsqueeze_639: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_638, -1);  unsqueeze_638 = None
    add_182: "f32[8, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_246, unsqueeze_639);  mul_246 = unsqueeze_639 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_40: "f32[8, 960, 7, 7]" = torch.ops.aten.relu.default(add_182);  add_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean_7: "f32[8, 960, 1, 1]" = torch.ops.aten.mean.dim(relu_40, [-1, -2], True);  relu_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:293, code: x = self.conv_head(x)
    convolution_94: "f32[8, 1280, 1, 1]" = torch.ops.aten.convolution.default(mean_7, arg270_1, arg271_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_7 = arg270_1 = arg271_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:294, code: x = self.act2(x)
    relu_41: "f32[8, 1280, 1, 1]" = torch.ops.aten.relu.default(convolution_94);  convolution_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/linear.py:19, code: return F.linear(input, self.weight, self.bias)
    view_1: "f32[8, 1280]" = torch.ops.aten.reshape.default(relu_41, [8, 1280]);  relu_41 = None
    permute: "f32[1280, 1000]" = torch.ops.aten.permute.default(arg2_1, [1, 0]);  arg2_1 = None
    addmm: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg3_1, view_1, permute);  arg3_1 = view_1 = permute = None
    return (addmm,)
    