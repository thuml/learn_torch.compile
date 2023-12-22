from __future__ import annotations



def forward(self, arg0_1: "f32[16]", arg1_1: "f32[16]", arg2_1: "f32[16]", arg3_1: "f32[16]", arg4_1: "f32[16]", arg5_1: "f32[16]", arg6_1: "f32[16]", arg7_1: "f32[16]", arg8_1: "f32[16]", arg9_1: "f32[16]", arg10_1: "f32[64]", arg11_1: "f32[64]", arg12_1: "f32[64]", arg13_1: "f32[64]", arg14_1: "f32[24]", arg15_1: "f32[24]", arg16_1: "f32[48]", arg17_1: "f32[48]", arg18_1: "f32[48]", arg19_1: "f32[48]", arg20_1: "f32[24]", arg21_1: "f32[24]", arg22_1: "f32[48]", arg23_1: "f32[48]", arg24_1: "f32[48]", arg25_1: "f32[48]", arg26_1: "f32[24]", arg27_1: "f32[24]", arg28_1: "f32[48]", arg29_1: "f32[48]", arg30_1: "f32[48]", arg31_1: "f32[48]", arg32_1: "f32[24]", arg33_1: "f32[24]", arg34_1: "f32[120]", arg35_1: "f32[120]", arg36_1: "f32[120]", arg37_1: "f32[120]", arg38_1: "f32[40]", arg39_1: "f32[40]", arg40_1: "f32[120]", arg41_1: "f32[120]", arg42_1: "f32[120]", arg43_1: "f32[120]", arg44_1: "f32[40]", arg45_1: "f32[40]", arg46_1: "f32[120]", arg47_1: "f32[120]", arg48_1: "f32[120]", arg49_1: "f32[120]", arg50_1: "f32[40]", arg51_1: "f32[40]", arg52_1: "f32[120]", arg53_1: "f32[120]", arg54_1: "f32[120]", arg55_1: "f32[120]", arg56_1: "f32[40]", arg57_1: "f32[40]", arg58_1: "f32[120]", arg59_1: "f32[120]", arg60_1: "f32[120]", arg61_1: "f32[120]", arg62_1: "f32[40]", arg63_1: "f32[40]", arg64_1: "f32[200]", arg65_1: "f32[200]", arg66_1: "f32[200]", arg67_1: "f32[200]", arg68_1: "f32[72]", arg69_1: "f32[72]", arg70_1: "f32[216]", arg71_1: "f32[216]", arg72_1: "f32[216]", arg73_1: "f32[216]", arg74_1: "f32[72]", arg75_1: "f32[72]", arg76_1: "f32[216]", arg77_1: "f32[216]", arg78_1: "f32[216]", arg79_1: "f32[216]", arg80_1: "f32[72]", arg81_1: "f32[72]", arg82_1: "f32[216]", arg83_1: "f32[216]", arg84_1: "f32[216]", arg85_1: "f32[216]", arg86_1: "f32[72]", arg87_1: "f32[72]", arg88_1: "f32[216]", arg89_1: "f32[216]", arg90_1: "f32[216]", arg91_1: "f32[216]", arg92_1: "f32[72]", arg93_1: "f32[72]", arg94_1: "f32[360]", arg95_1: "f32[360]", arg96_1: "f32[360]", arg97_1: "f32[360]", arg98_1: "f32[120]", arg99_1: "f32[120]", arg100_1: "f32[360]", arg101_1: "f32[360]", arg102_1: "f32[360]", arg103_1: "f32[360]", arg104_1: "f32[120]", arg105_1: "f32[120]", arg106_1: "f32[360]", arg107_1: "f32[360]", arg108_1: "f32[360]", arg109_1: "f32[360]", arg110_1: "f32[120]", arg111_1: "f32[120]", arg112_1: "f32[360]", arg113_1: "f32[360]", arg114_1: "f32[360]", arg115_1: "f32[360]", arg116_1: "f32[120]", arg117_1: "f32[120]", arg118_1: "f32[360]", arg119_1: "f32[360]", arg120_1: "f32[360]", arg121_1: "f32[360]", arg122_1: "f32[120]", arg123_1: "f32[120]", arg124_1: "f32[360]", arg125_1: "f32[360]", arg126_1: "f32[360]", arg127_1: "f32[360]", arg128_1: "f32[120]", arg129_1: "f32[120]", arg130_1: "f32[720]", arg131_1: "f32[720]", arg132_1: "f32[720]", arg133_1: "f32[720]", arg134_1: "f32[184]", arg135_1: "f32[184]", arg136_1: "f32[736]", arg137_1: "f32[736]", arg138_1: "f32[736]", arg139_1: "f32[736]", arg140_1: "f32[184]", arg141_1: "f32[184]", arg142_1: "f32[736]", arg143_1: "f32[736]", arg144_1: "f32[736]", arg145_1: "f32[736]", arg146_1: "f32[184]", arg147_1: "f32[184]", arg148_1: "f32[736]", arg149_1: "f32[736]", arg150_1: "f32[736]", arg151_1: "f32[736]", arg152_1: "f32[184]", arg153_1: "f32[184]", arg154_1: "f32[736]", arg155_1: "f32[736]", arg156_1: "f32[736]", arg157_1: "f32[736]", arg158_1: "f32[184]", arg159_1: "f32[184]", arg160_1: "f32[736]", arg161_1: "f32[736]", arg162_1: "f32[736]", arg163_1: "f32[736]", arg164_1: "f32[184]", arg165_1: "f32[184]", arg166_1: "f32[1104]", arg167_1: "f32[1104]", arg168_1: "f32[1104]", arg169_1: "f32[1104]", arg170_1: "f32[224]", arg171_1: "f32[224]", arg172_1: "f32[1344]", arg173_1: "f32[1344]", arg174_1: "f32[1000, 1984]", arg175_1: "f32[1000]", arg176_1: "f32[16, 3, 3, 3]", arg177_1: "f32[16, 1, 3, 3]", arg178_1: "f32[16, 16, 1, 1]", arg179_1: "f32[16, 1, 3, 3]", arg180_1: "f32[16, 16, 1, 1]", arg181_1: "f32[64, 16, 1, 1]", arg182_1: "f32[64, 1, 5, 5]", arg183_1: "f32[24, 64, 1, 1]", arg184_1: "f32[48, 24, 1, 1]", arg185_1: "f32[48, 1, 5, 5]", arg186_1: "f32[24, 48, 1, 1]", arg187_1: "f32[48, 24, 1, 1]", arg188_1: "f32[48, 1, 5, 5]", arg189_1: "f32[24, 48, 1, 1]", arg190_1: "f32[48, 24, 1, 1]", arg191_1: "f32[48, 1, 5, 5]", arg192_1: "f32[24, 48, 1, 1]", arg193_1: "f32[120, 24, 1, 1]", arg194_1: "f32[120, 1, 5, 5]", arg195_1: "f32[8, 120, 1, 1]", arg196_1: "f32[8]", arg197_1: "f32[120, 8, 1, 1]", arg198_1: "f32[120]", arg199_1: "f32[40, 120, 1, 1]", arg200_1: "f32[120, 40, 1, 1]", arg201_1: "f32[120, 1, 5, 5]", arg202_1: "f32[16, 120, 1, 1]", arg203_1: "f32[16]", arg204_1: "f32[120, 16, 1, 1]", arg205_1: "f32[120]", arg206_1: "f32[40, 120, 1, 1]", arg207_1: "f32[120, 40, 1, 1]", arg208_1: "f32[120, 1, 5, 5]", arg209_1: "f32[16, 120, 1, 1]", arg210_1: "f32[16]", arg211_1: "f32[120, 16, 1, 1]", arg212_1: "f32[120]", arg213_1: "f32[40, 120, 1, 1]", arg214_1: "f32[120, 40, 1, 1]", arg215_1: "f32[120, 1, 5, 5]", arg216_1: "f32[16, 120, 1, 1]", arg217_1: "f32[16]", arg218_1: "f32[120, 16, 1, 1]", arg219_1: "f32[120]", arg220_1: "f32[40, 120, 1, 1]", arg221_1: "f32[120, 40, 1, 1]", arg222_1: "f32[120, 1, 5, 5]", arg223_1: "f32[16, 120, 1, 1]", arg224_1: "f32[16]", arg225_1: "f32[120, 16, 1, 1]", arg226_1: "f32[120]", arg227_1: "f32[40, 120, 1, 1]", arg228_1: "f32[200, 40, 1, 1]", arg229_1: "f32[200, 1, 5, 5]", arg230_1: "f32[72, 200, 1, 1]", arg231_1: "f32[216, 72, 1, 1]", arg232_1: "f32[216, 1, 3, 3]", arg233_1: "f32[72, 216, 1, 1]", arg234_1: "f32[216, 72, 1, 1]", arg235_1: "f32[216, 1, 3, 3]", arg236_1: "f32[72, 216, 1, 1]", arg237_1: "f32[216, 72, 1, 1]", arg238_1: "f32[216, 1, 3, 3]", arg239_1: "f32[72, 216, 1, 1]", arg240_1: "f32[216, 72, 1, 1]", arg241_1: "f32[216, 1, 3, 3]", arg242_1: "f32[72, 216, 1, 1]", arg243_1: "f32[360, 72, 1, 1]", arg244_1: "f32[360, 1, 3, 3]", arg245_1: "f32[24, 360, 1, 1]", arg246_1: "f32[24]", arg247_1: "f32[360, 24, 1, 1]", arg248_1: "f32[360]", arg249_1: "f32[120, 360, 1, 1]", arg250_1: "f32[360, 120, 1, 1]", arg251_1: "f32[360, 1, 5, 5]", arg252_1: "f32[32, 360, 1, 1]", arg253_1: "f32[32]", arg254_1: "f32[360, 32, 1, 1]", arg255_1: "f32[360]", arg256_1: "f32[120, 360, 1, 1]", arg257_1: "f32[360, 120, 1, 1]", arg258_1: "f32[360, 1, 5, 5]", arg259_1: "f32[32, 360, 1, 1]", arg260_1: "f32[32]", arg261_1: "f32[360, 32, 1, 1]", arg262_1: "f32[360]", arg263_1: "f32[120, 360, 1, 1]", arg264_1: "f32[360, 120, 1, 1]", arg265_1: "f32[360, 1, 5, 5]", arg266_1: "f32[32, 360, 1, 1]", arg267_1: "f32[32]", arg268_1: "f32[360, 32, 1, 1]", arg269_1: "f32[360]", arg270_1: "f32[120, 360, 1, 1]", arg271_1: "f32[360, 120, 1, 1]", arg272_1: "f32[360, 1, 5, 5]", arg273_1: "f32[32, 360, 1, 1]", arg274_1: "f32[32]", arg275_1: "f32[360, 32, 1, 1]", arg276_1: "f32[360]", arg277_1: "f32[120, 360, 1, 1]", arg278_1: "f32[360, 120, 1, 1]", arg279_1: "f32[360, 1, 5, 5]", arg280_1: "f32[32, 360, 1, 1]", arg281_1: "f32[32]", arg282_1: "f32[360, 32, 1, 1]", arg283_1: "f32[360]", arg284_1: "f32[120, 360, 1, 1]", arg285_1: "f32[720, 120, 1, 1]", arg286_1: "f32[720, 1, 3, 3]", arg287_1: "f32[32, 720, 1, 1]", arg288_1: "f32[32]", arg289_1: "f32[720, 32, 1, 1]", arg290_1: "f32[720]", arg291_1: "f32[184, 720, 1, 1]", arg292_1: "f32[736, 184, 1, 1]", arg293_1: "f32[736, 1, 5, 5]", arg294_1: "f32[48, 736, 1, 1]", arg295_1: "f32[48]", arg296_1: "f32[736, 48, 1, 1]", arg297_1: "f32[736]", arg298_1: "f32[184, 736, 1, 1]", arg299_1: "f32[736, 184, 1, 1]", arg300_1: "f32[736, 1, 5, 5]", arg301_1: "f32[48, 736, 1, 1]", arg302_1: "f32[48]", arg303_1: "f32[736, 48, 1, 1]", arg304_1: "f32[736]", arg305_1: "f32[184, 736, 1, 1]", arg306_1: "f32[736, 184, 1, 1]", arg307_1: "f32[736, 1, 5, 5]", arg308_1: "f32[48, 736, 1, 1]", arg309_1: "f32[48]", arg310_1: "f32[736, 48, 1, 1]", arg311_1: "f32[736]", arg312_1: "f32[184, 736, 1, 1]", arg313_1: "f32[736, 184, 1, 1]", arg314_1: "f32[736, 1, 5, 5]", arg315_1: "f32[48, 736, 1, 1]", arg316_1: "f32[48]", arg317_1: "f32[736, 48, 1, 1]", arg318_1: "f32[736]", arg319_1: "f32[184, 736, 1, 1]", arg320_1: "f32[736, 184, 1, 1]", arg321_1: "f32[736, 1, 5, 5]", arg322_1: "f32[48, 736, 1, 1]", arg323_1: "f32[48]", arg324_1: "f32[736, 48, 1, 1]", arg325_1: "f32[736]", arg326_1: "f32[184, 736, 1, 1]", arg327_1: "f32[1104, 184, 1, 1]", arg328_1: "f32[1104, 1, 5, 5]", arg329_1: "f32[48, 1104, 1, 1]", arg330_1: "f32[48]", arg331_1: "f32[1104, 48, 1, 1]", arg332_1: "f32[1104]", arg333_1: "f32[224, 1104, 1, 1]", arg334_1: "f32[1344, 224, 1, 1]", arg335_1: "f32[1984, 1344, 1, 1]", arg336_1: "f32[16]", arg337_1: "f32[16]", arg338_1: "f32[16]", arg339_1: "f32[16]", arg340_1: "f32[16]", arg341_1: "f32[16]", arg342_1: "f32[16]", arg343_1: "f32[16]", arg344_1: "f32[16]", arg345_1: "f32[16]", arg346_1: "f32[64]", arg347_1: "f32[64]", arg348_1: "f32[64]", arg349_1: "f32[64]", arg350_1: "f32[24]", arg351_1: "f32[24]", arg352_1: "f32[48]", arg353_1: "f32[48]", arg354_1: "f32[48]", arg355_1: "f32[48]", arg356_1: "f32[24]", arg357_1: "f32[24]", arg358_1: "f32[48]", arg359_1: "f32[48]", arg360_1: "f32[48]", arg361_1: "f32[48]", arg362_1: "f32[24]", arg363_1: "f32[24]", arg364_1: "f32[48]", arg365_1: "f32[48]", arg366_1: "f32[48]", arg367_1: "f32[48]", arg368_1: "f32[24]", arg369_1: "f32[24]", arg370_1: "f32[120]", arg371_1: "f32[120]", arg372_1: "f32[120]", arg373_1: "f32[120]", arg374_1: "f32[40]", arg375_1: "f32[40]", arg376_1: "f32[120]", arg377_1: "f32[120]", arg378_1: "f32[120]", arg379_1: "f32[120]", arg380_1: "f32[40]", arg381_1: "f32[40]", arg382_1: "f32[120]", arg383_1: "f32[120]", arg384_1: "f32[120]", arg385_1: "f32[120]", arg386_1: "f32[40]", arg387_1: "f32[40]", arg388_1: "f32[120]", arg389_1: "f32[120]", arg390_1: "f32[120]", arg391_1: "f32[120]", arg392_1: "f32[40]", arg393_1: "f32[40]", arg394_1: "f32[120]", arg395_1: "f32[120]", arg396_1: "f32[120]", arg397_1: "f32[120]", arg398_1: "f32[40]", arg399_1: "f32[40]", arg400_1: "f32[200]", arg401_1: "f32[200]", arg402_1: "f32[200]", arg403_1: "f32[200]", arg404_1: "f32[72]", arg405_1: "f32[72]", arg406_1: "f32[216]", arg407_1: "f32[216]", arg408_1: "f32[216]", arg409_1: "f32[216]", arg410_1: "f32[72]", arg411_1: "f32[72]", arg412_1: "f32[216]", arg413_1: "f32[216]", arg414_1: "f32[216]", arg415_1: "f32[216]", arg416_1: "f32[72]", arg417_1: "f32[72]", arg418_1: "f32[216]", arg419_1: "f32[216]", arg420_1: "f32[216]", arg421_1: "f32[216]", arg422_1: "f32[72]", arg423_1: "f32[72]", arg424_1: "f32[216]", arg425_1: "f32[216]", arg426_1: "f32[216]", arg427_1: "f32[216]", arg428_1: "f32[72]", arg429_1: "f32[72]", arg430_1: "f32[360]", arg431_1: "f32[360]", arg432_1: "f32[360]", arg433_1: "f32[360]", arg434_1: "f32[120]", arg435_1: "f32[120]", arg436_1: "f32[360]", arg437_1: "f32[360]", arg438_1: "f32[360]", arg439_1: "f32[360]", arg440_1: "f32[120]", arg441_1: "f32[120]", arg442_1: "f32[360]", arg443_1: "f32[360]", arg444_1: "f32[360]", arg445_1: "f32[360]", arg446_1: "f32[120]", arg447_1: "f32[120]", arg448_1: "f32[360]", arg449_1: "f32[360]", arg450_1: "f32[360]", arg451_1: "f32[360]", arg452_1: "f32[120]", arg453_1: "f32[120]", arg454_1: "f32[360]", arg455_1: "f32[360]", arg456_1: "f32[360]", arg457_1: "f32[360]", arg458_1: "f32[120]", arg459_1: "f32[120]", arg460_1: "f32[360]", arg461_1: "f32[360]", arg462_1: "f32[360]", arg463_1: "f32[360]", arg464_1: "f32[120]", arg465_1: "f32[120]", arg466_1: "f32[720]", arg467_1: "f32[720]", arg468_1: "f32[720]", arg469_1: "f32[720]", arg470_1: "f32[184]", arg471_1: "f32[184]", arg472_1: "f32[736]", arg473_1: "f32[736]", arg474_1: "f32[736]", arg475_1: "f32[736]", arg476_1: "f32[184]", arg477_1: "f32[184]", arg478_1: "f32[736]", arg479_1: "f32[736]", arg480_1: "f32[736]", arg481_1: "f32[736]", arg482_1: "f32[184]", arg483_1: "f32[184]", arg484_1: "f32[736]", arg485_1: "f32[736]", arg486_1: "f32[736]", arg487_1: "f32[736]", arg488_1: "f32[184]", arg489_1: "f32[184]", arg490_1: "f32[736]", arg491_1: "f32[736]", arg492_1: "f32[736]", arg493_1: "f32[736]", arg494_1: "f32[184]", arg495_1: "f32[184]", arg496_1: "f32[736]", arg497_1: "f32[736]", arg498_1: "f32[736]", arg499_1: "f32[736]", arg500_1: "f32[184]", arg501_1: "f32[184]", arg502_1: "f32[1104]", arg503_1: "f32[1104]", arg504_1: "f32[1104]", arg505_1: "f32[1104]", arg506_1: "f32[224]", arg507_1: "f32[224]", arg508_1: "f32[1344]", arg509_1: "f32[1344]", arg510_1: "f32[8, 3, 256, 256]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilenetv3.py:135, code: x = self.conv_stem(x)
    convolution: "f32[8, 16, 128, 128]" = torch.ops.aten.convolution.default(arg510_1, arg176_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg510_1 = arg176_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type: "f32[16]" = torch.ops.prims.convert_element_type.default(arg336_1, torch.float32);  arg336_1 = None
    convert_element_type_1: "f32[16]" = torch.ops.prims.convert_element_type.default(arg337_1, torch.float32);  arg337_1 = None
    add: "f32[16]" = torch.ops.aten.add.Tensor(convert_element_type_1, 1e-05);  convert_element_type_1 = None
    sqrt: "f32[16]" = torch.ops.aten.sqrt.default(add);  add = None
    reciprocal: "f32[16]" = torch.ops.aten.reciprocal.default(sqrt);  sqrt = None
    mul: "f32[16]" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
    unsqueeze: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type, -1);  convert_element_type = None
    unsqueeze_1: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    unsqueeze_2: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(mul, -1);  mul = None
    unsqueeze_3: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    sub: "f32[8, 16, 128, 128]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1);  convolution = unsqueeze_1 = None
    mul_1: "f32[8, 16, 128, 128]" = torch.ops.aten.mul.Tensor(sub, unsqueeze_3);  sub = unsqueeze_3 = None
    unsqueeze_4: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg0_1, -1);  arg0_1 = None
    unsqueeze_5: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_2: "f32[8, 16, 128, 128]" = torch.ops.aten.mul.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None
    unsqueeze_6: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg1_1, -1);  arg1_1 = None
    unsqueeze_7: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_1: "f32[8, 16, 128, 128]" = torch.ops.aten.add.Tensor(mul_2, unsqueeze_7);  mul_2 = unsqueeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_2: "f32[8, 16, 128, 128]" = torch.ops.aten.add.Tensor(add_1, 3)
    clamp_min: "f32[8, 16, 128, 128]" = torch.ops.aten.clamp_min.default(add_2, 0);  add_2 = None
    clamp_max: "f32[8, 16, 128, 128]" = torch.ops.aten.clamp_max.default(clamp_min, 6);  clamp_min = None
    mul_3: "f32[8, 16, 128, 128]" = torch.ops.aten.mul.Tensor(add_1, clamp_max);  add_1 = clamp_max = None
    div: "f32[8, 16, 128, 128]" = torch.ops.aten.div.Tensor(mul_3, 6);  mul_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_1: "f32[8, 16, 128, 128]" = torch.ops.aten.convolution.default(div, arg177_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 16);  arg177_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_2: "f32[16]" = torch.ops.prims.convert_element_type.default(arg338_1, torch.float32);  arg338_1 = None
    convert_element_type_3: "f32[16]" = torch.ops.prims.convert_element_type.default(arg339_1, torch.float32);  arg339_1 = None
    add_3: "f32[16]" = torch.ops.aten.add.Tensor(convert_element_type_3, 1e-05);  convert_element_type_3 = None
    sqrt_1: "f32[16]" = torch.ops.aten.sqrt.default(add_3);  add_3 = None
    reciprocal_1: "f32[16]" = torch.ops.aten.reciprocal.default(sqrt_1);  sqrt_1 = None
    mul_4: "f32[16]" = torch.ops.aten.mul.Tensor(reciprocal_1, 1);  reciprocal_1 = None
    unsqueeze_8: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, -1);  convert_element_type_2 = None
    unsqueeze_9: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    unsqueeze_10: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(mul_4, -1);  mul_4 = None
    unsqueeze_11: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    sub_1: "f32[8, 16, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_9);  convolution_1 = unsqueeze_9 = None
    mul_5: "f32[8, 16, 128, 128]" = torch.ops.aten.mul.Tensor(sub_1, unsqueeze_11);  sub_1 = unsqueeze_11 = None
    unsqueeze_12: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
    unsqueeze_13: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_6: "f32[8, 16, 128, 128]" = torch.ops.aten.mul.Tensor(mul_5, unsqueeze_13);  mul_5 = unsqueeze_13 = None
    unsqueeze_14: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, -1);  arg3_1 = None
    unsqueeze_15: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_4: "f32[8, 16, 128, 128]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_15);  mul_6 = unsqueeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_5: "f32[8, 16, 128, 128]" = torch.ops.aten.add.Tensor(add_4, 3)
    clamp_min_1: "f32[8, 16, 128, 128]" = torch.ops.aten.clamp_min.default(add_5, 0);  add_5 = None
    clamp_max_1: "f32[8, 16, 128, 128]" = torch.ops.aten.clamp_max.default(clamp_min_1, 6);  clamp_min_1 = None
    mul_7: "f32[8, 16, 128, 128]" = torch.ops.aten.mul.Tensor(add_4, clamp_max_1);  add_4 = clamp_max_1 = None
    div_1: "f32[8, 16, 128, 128]" = torch.ops.aten.div.Tensor(mul_7, 6);  mul_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_2: "f32[8, 16, 128, 128]" = torch.ops.aten.convolution.default(div_1, arg178_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_1 = arg178_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_4: "f32[16]" = torch.ops.prims.convert_element_type.default(arg340_1, torch.float32);  arg340_1 = None
    convert_element_type_5: "f32[16]" = torch.ops.prims.convert_element_type.default(arg341_1, torch.float32);  arg341_1 = None
    add_6: "f32[16]" = torch.ops.aten.add.Tensor(convert_element_type_5, 1e-05);  convert_element_type_5 = None
    sqrt_2: "f32[16]" = torch.ops.aten.sqrt.default(add_6);  add_6 = None
    reciprocal_2: "f32[16]" = torch.ops.aten.reciprocal.default(sqrt_2);  sqrt_2 = None
    mul_8: "f32[16]" = torch.ops.aten.mul.Tensor(reciprocal_2, 1);  reciprocal_2 = None
    unsqueeze_16: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_4, -1);  convert_element_type_4 = None
    unsqueeze_17: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    unsqueeze_18: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(mul_8, -1);  mul_8 = None
    unsqueeze_19: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    sub_2: "f32[8, 16, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_17);  convolution_2 = unsqueeze_17 = None
    mul_9: "f32[8, 16, 128, 128]" = torch.ops.aten.mul.Tensor(sub_2, unsqueeze_19);  sub_2 = unsqueeze_19 = None
    unsqueeze_20: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
    unsqueeze_21: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_10: "f32[8, 16, 128, 128]" = torch.ops.aten.mul.Tensor(mul_9, unsqueeze_21);  mul_9 = unsqueeze_21 = None
    unsqueeze_22: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
    unsqueeze_23: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_7: "f32[8, 16, 128, 128]" = torch.ops.aten.add.Tensor(mul_10, unsqueeze_23);  mul_10 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:129, code: x = self.drop_path(x) + shortcut
    add_8: "f32[8, 16, 128, 128]" = torch.ops.aten.add.Tensor(add_7, div);  add_7 = div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_3: "f32[8, 16, 128, 128]" = torch.ops.aten.convolution.default(add_8, arg179_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 16);  arg179_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_6: "f32[16]" = torch.ops.prims.convert_element_type.default(arg342_1, torch.float32);  arg342_1 = None
    convert_element_type_7: "f32[16]" = torch.ops.prims.convert_element_type.default(arg343_1, torch.float32);  arg343_1 = None
    add_9: "f32[16]" = torch.ops.aten.add.Tensor(convert_element_type_7, 1e-05);  convert_element_type_7 = None
    sqrt_3: "f32[16]" = torch.ops.aten.sqrt.default(add_9);  add_9 = None
    reciprocal_3: "f32[16]" = torch.ops.aten.reciprocal.default(sqrt_3);  sqrt_3 = None
    mul_11: "f32[16]" = torch.ops.aten.mul.Tensor(reciprocal_3, 1);  reciprocal_3 = None
    unsqueeze_24: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_6, -1);  convert_element_type_6 = None
    unsqueeze_25: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
    unsqueeze_26: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(mul_11, -1);  mul_11 = None
    unsqueeze_27: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
    sub_3: "f32[8, 16, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_25);  convolution_3 = unsqueeze_25 = None
    mul_12: "f32[8, 16, 128, 128]" = torch.ops.aten.mul.Tensor(sub_3, unsqueeze_27);  sub_3 = unsqueeze_27 = None
    unsqueeze_28: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg6_1, -1);  arg6_1 = None
    unsqueeze_29: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    mul_13: "f32[8, 16, 128, 128]" = torch.ops.aten.mul.Tensor(mul_12, unsqueeze_29);  mul_12 = unsqueeze_29 = None
    unsqueeze_30: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
    unsqueeze_31: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
    add_10: "f32[8, 16, 128, 128]" = torch.ops.aten.add.Tensor(mul_13, unsqueeze_31);  mul_13 = unsqueeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_11: "f32[8, 16, 128, 128]" = torch.ops.aten.add.Tensor(add_10, 3)
    clamp_min_2: "f32[8, 16, 128, 128]" = torch.ops.aten.clamp_min.default(add_11, 0);  add_11 = None
    clamp_max_2: "f32[8, 16, 128, 128]" = torch.ops.aten.clamp_max.default(clamp_min_2, 6);  clamp_min_2 = None
    mul_14: "f32[8, 16, 128, 128]" = torch.ops.aten.mul.Tensor(add_10, clamp_max_2);  add_10 = clamp_max_2 = None
    div_2: "f32[8, 16, 128, 128]" = torch.ops.aten.div.Tensor(mul_14, 6);  mul_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_4: "f32[8, 16, 128, 128]" = torch.ops.aten.convolution.default(div_2, arg180_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_2 = arg180_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_8: "f32[16]" = torch.ops.prims.convert_element_type.default(arg344_1, torch.float32);  arg344_1 = None
    convert_element_type_9: "f32[16]" = torch.ops.prims.convert_element_type.default(arg345_1, torch.float32);  arg345_1 = None
    add_12: "f32[16]" = torch.ops.aten.add.Tensor(convert_element_type_9, 1e-05);  convert_element_type_9 = None
    sqrt_4: "f32[16]" = torch.ops.aten.sqrt.default(add_12);  add_12 = None
    reciprocal_4: "f32[16]" = torch.ops.aten.reciprocal.default(sqrt_4);  sqrt_4 = None
    mul_15: "f32[16]" = torch.ops.aten.mul.Tensor(reciprocal_4, 1);  reciprocal_4 = None
    unsqueeze_32: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_8, -1);  convert_element_type_8 = None
    unsqueeze_33: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    unsqueeze_34: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(mul_15, -1);  mul_15 = None
    unsqueeze_35: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
    sub_4: "f32[8, 16, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_33);  convolution_4 = unsqueeze_33 = None
    mul_16: "f32[8, 16, 128, 128]" = torch.ops.aten.mul.Tensor(sub_4, unsqueeze_35);  sub_4 = unsqueeze_35 = None
    unsqueeze_36: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg8_1, -1);  arg8_1 = None
    unsqueeze_37: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
    mul_17: "f32[8, 16, 128, 128]" = torch.ops.aten.mul.Tensor(mul_16, unsqueeze_37);  mul_16 = unsqueeze_37 = None
    unsqueeze_38: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
    unsqueeze_39: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
    add_13: "f32[8, 16, 128, 128]" = torch.ops.aten.add.Tensor(mul_17, unsqueeze_39);  mul_17 = unsqueeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:129, code: x = self.drop_path(x) + shortcut
    add_14: "f32[8, 16, 128, 128]" = torch.ops.aten.add.Tensor(add_13, add_8);  add_13 = add_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_5: "f32[8, 64, 128, 128]" = torch.ops.aten.convolution.default(add_14, arg181_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_14 = arg181_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_10: "f32[64]" = torch.ops.prims.convert_element_type.default(arg346_1, torch.float32);  arg346_1 = None
    convert_element_type_11: "f32[64]" = torch.ops.prims.convert_element_type.default(arg347_1, torch.float32);  arg347_1 = None
    add_15: "f32[64]" = torch.ops.aten.add.Tensor(convert_element_type_11, 1e-05);  convert_element_type_11 = None
    sqrt_5: "f32[64]" = torch.ops.aten.sqrt.default(add_15);  add_15 = None
    reciprocal_5: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_5);  sqrt_5 = None
    mul_18: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_5, 1);  reciprocal_5 = None
    unsqueeze_40: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_10, -1);  convert_element_type_10 = None
    unsqueeze_41: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    unsqueeze_42: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_18, -1);  mul_18 = None
    unsqueeze_43: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    sub_5: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_41);  convolution_5 = unsqueeze_41 = None
    mul_19: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_5, unsqueeze_43);  sub_5 = unsqueeze_43 = None
    unsqueeze_44: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
    unsqueeze_45: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    mul_20: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(mul_19, unsqueeze_45);  mul_19 = unsqueeze_45 = None
    unsqueeze_46: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg11_1, -1);  arg11_1 = None
    unsqueeze_47: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
    add_16: "f32[8, 64, 128, 128]" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_47);  mul_20 = unsqueeze_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_17: "f32[8, 64, 128, 128]" = torch.ops.aten.add.Tensor(add_16, 3)
    clamp_min_3: "f32[8, 64, 128, 128]" = torch.ops.aten.clamp_min.default(add_17, 0);  add_17 = None
    clamp_max_3: "f32[8, 64, 128, 128]" = torch.ops.aten.clamp_max.default(clamp_min_3, 6);  clamp_min_3 = None
    mul_21: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(add_16, clamp_max_3);  add_16 = clamp_max_3 = None
    div_3: "f32[8, 64, 128, 128]" = torch.ops.aten.div.Tensor(mul_21, 6);  mul_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_6: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(div_3, arg182_1, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 64);  div_3 = arg182_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_12: "f32[64]" = torch.ops.prims.convert_element_type.default(arg348_1, torch.float32);  arg348_1 = None
    convert_element_type_13: "f32[64]" = torch.ops.prims.convert_element_type.default(arg349_1, torch.float32);  arg349_1 = None
    add_18: "f32[64]" = torch.ops.aten.add.Tensor(convert_element_type_13, 1e-05);  convert_element_type_13 = None
    sqrt_6: "f32[64]" = torch.ops.aten.sqrt.default(add_18);  add_18 = None
    reciprocal_6: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_6);  sqrt_6 = None
    mul_22: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_6, 1);  reciprocal_6 = None
    unsqueeze_48: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_12, -1);  convert_element_type_12 = None
    unsqueeze_49: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
    unsqueeze_50: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_22, -1);  mul_22 = None
    unsqueeze_51: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
    sub_6: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_49);  convolution_6 = unsqueeze_49 = None
    mul_23: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_6, unsqueeze_51);  sub_6 = unsqueeze_51 = None
    unsqueeze_52: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg12_1, -1);  arg12_1 = None
    unsqueeze_53: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    mul_24: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_23, unsqueeze_53);  mul_23 = unsqueeze_53 = None
    unsqueeze_54: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg13_1, -1);  arg13_1 = None
    unsqueeze_55: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
    add_19: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_24, unsqueeze_55);  mul_24 = unsqueeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_20: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(add_19, 3)
    clamp_min_4: "f32[8, 64, 64, 64]" = torch.ops.aten.clamp_min.default(add_20, 0);  add_20 = None
    clamp_max_4: "f32[8, 64, 64, 64]" = torch.ops.aten.clamp_max.default(clamp_min_4, 6);  clamp_min_4 = None
    mul_25: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_19, clamp_max_4);  add_19 = clamp_max_4 = None
    div_4: "f32[8, 64, 64, 64]" = torch.ops.aten.div.Tensor(mul_25, 6);  mul_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_7: "f32[8, 24, 64, 64]" = torch.ops.aten.convolution.default(div_4, arg183_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_4 = arg183_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_14: "f32[24]" = torch.ops.prims.convert_element_type.default(arg350_1, torch.float32);  arg350_1 = None
    convert_element_type_15: "f32[24]" = torch.ops.prims.convert_element_type.default(arg351_1, torch.float32);  arg351_1 = None
    add_21: "f32[24]" = torch.ops.aten.add.Tensor(convert_element_type_15, 1e-05);  convert_element_type_15 = None
    sqrt_7: "f32[24]" = torch.ops.aten.sqrt.default(add_21);  add_21 = None
    reciprocal_7: "f32[24]" = torch.ops.aten.reciprocal.default(sqrt_7);  sqrt_7 = None
    mul_26: "f32[24]" = torch.ops.aten.mul.Tensor(reciprocal_7, 1);  reciprocal_7 = None
    unsqueeze_56: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_14, -1);  convert_element_type_14 = None
    unsqueeze_57: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    unsqueeze_58: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(mul_26, -1);  mul_26 = None
    unsqueeze_59: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
    sub_7: "f32[8, 24, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_57);  convolution_7 = unsqueeze_57 = None
    mul_27: "f32[8, 24, 64, 64]" = torch.ops.aten.mul.Tensor(sub_7, unsqueeze_59);  sub_7 = unsqueeze_59 = None
    unsqueeze_60: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
    unsqueeze_61: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
    mul_28: "f32[8, 24, 64, 64]" = torch.ops.aten.mul.Tensor(mul_27, unsqueeze_61);  mul_27 = unsqueeze_61 = None
    unsqueeze_62: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
    unsqueeze_63: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
    add_22: "f32[8, 24, 64, 64]" = torch.ops.aten.add.Tensor(mul_28, unsqueeze_63);  mul_28 = unsqueeze_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_8: "f32[8, 48, 64, 64]" = torch.ops.aten.convolution.default(add_22, arg184_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg184_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_16: "f32[48]" = torch.ops.prims.convert_element_type.default(arg352_1, torch.float32);  arg352_1 = None
    convert_element_type_17: "f32[48]" = torch.ops.prims.convert_element_type.default(arg353_1, torch.float32);  arg353_1 = None
    add_23: "f32[48]" = torch.ops.aten.add.Tensor(convert_element_type_17, 1e-05);  convert_element_type_17 = None
    sqrt_8: "f32[48]" = torch.ops.aten.sqrt.default(add_23);  add_23 = None
    reciprocal_8: "f32[48]" = torch.ops.aten.reciprocal.default(sqrt_8);  sqrt_8 = None
    mul_29: "f32[48]" = torch.ops.aten.mul.Tensor(reciprocal_8, 1);  reciprocal_8 = None
    unsqueeze_64: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_16, -1);  convert_element_type_16 = None
    unsqueeze_65: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
    unsqueeze_66: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(mul_29, -1);  mul_29 = None
    unsqueeze_67: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
    sub_8: "f32[8, 48, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_65);  convolution_8 = unsqueeze_65 = None
    mul_30: "f32[8, 48, 64, 64]" = torch.ops.aten.mul.Tensor(sub_8, unsqueeze_67);  sub_8 = unsqueeze_67 = None
    unsqueeze_68: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(arg16_1, -1);  arg16_1 = None
    unsqueeze_69: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    mul_31: "f32[8, 48, 64, 64]" = torch.ops.aten.mul.Tensor(mul_30, unsqueeze_69);  mul_30 = unsqueeze_69 = None
    unsqueeze_70: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(arg17_1, -1);  arg17_1 = None
    unsqueeze_71: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    add_24: "f32[8, 48, 64, 64]" = torch.ops.aten.add.Tensor(mul_31, unsqueeze_71);  mul_31 = unsqueeze_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_25: "f32[8, 48, 64, 64]" = torch.ops.aten.add.Tensor(add_24, 3)
    clamp_min_5: "f32[8, 48, 64, 64]" = torch.ops.aten.clamp_min.default(add_25, 0);  add_25 = None
    clamp_max_5: "f32[8, 48, 64, 64]" = torch.ops.aten.clamp_max.default(clamp_min_5, 6);  clamp_min_5 = None
    mul_32: "f32[8, 48, 64, 64]" = torch.ops.aten.mul.Tensor(add_24, clamp_max_5);  add_24 = clamp_max_5 = None
    div_5: "f32[8, 48, 64, 64]" = torch.ops.aten.div.Tensor(mul_32, 6);  mul_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_9: "f32[8, 48, 64, 64]" = torch.ops.aten.convolution.default(div_5, arg185_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 48);  div_5 = arg185_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_18: "f32[48]" = torch.ops.prims.convert_element_type.default(arg354_1, torch.float32);  arg354_1 = None
    convert_element_type_19: "f32[48]" = torch.ops.prims.convert_element_type.default(arg355_1, torch.float32);  arg355_1 = None
    add_26: "f32[48]" = torch.ops.aten.add.Tensor(convert_element_type_19, 1e-05);  convert_element_type_19 = None
    sqrt_9: "f32[48]" = torch.ops.aten.sqrt.default(add_26);  add_26 = None
    reciprocal_9: "f32[48]" = torch.ops.aten.reciprocal.default(sqrt_9);  sqrt_9 = None
    mul_33: "f32[48]" = torch.ops.aten.mul.Tensor(reciprocal_9, 1);  reciprocal_9 = None
    unsqueeze_72: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_18, -1);  convert_element_type_18 = None
    unsqueeze_73: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
    unsqueeze_74: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(mul_33, -1);  mul_33 = None
    unsqueeze_75: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
    sub_9: "f32[8, 48, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_73);  convolution_9 = unsqueeze_73 = None
    mul_34: "f32[8, 48, 64, 64]" = torch.ops.aten.mul.Tensor(sub_9, unsqueeze_75);  sub_9 = unsqueeze_75 = None
    unsqueeze_76: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(arg18_1, -1);  arg18_1 = None
    unsqueeze_77: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
    mul_35: "f32[8, 48, 64, 64]" = torch.ops.aten.mul.Tensor(mul_34, unsqueeze_77);  mul_34 = unsqueeze_77 = None
    unsqueeze_78: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(arg19_1, -1);  arg19_1 = None
    unsqueeze_79: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    add_27: "f32[8, 48, 64, 64]" = torch.ops.aten.add.Tensor(mul_35, unsqueeze_79);  mul_35 = unsqueeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_28: "f32[8, 48, 64, 64]" = torch.ops.aten.add.Tensor(add_27, 3)
    clamp_min_6: "f32[8, 48, 64, 64]" = torch.ops.aten.clamp_min.default(add_28, 0);  add_28 = None
    clamp_max_6: "f32[8, 48, 64, 64]" = torch.ops.aten.clamp_max.default(clamp_min_6, 6);  clamp_min_6 = None
    mul_36: "f32[8, 48, 64, 64]" = torch.ops.aten.mul.Tensor(add_27, clamp_max_6);  add_27 = clamp_max_6 = None
    div_6: "f32[8, 48, 64, 64]" = torch.ops.aten.div.Tensor(mul_36, 6);  mul_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_10: "f32[8, 24, 64, 64]" = torch.ops.aten.convolution.default(div_6, arg186_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_6 = arg186_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_20: "f32[24]" = torch.ops.prims.convert_element_type.default(arg356_1, torch.float32);  arg356_1 = None
    convert_element_type_21: "f32[24]" = torch.ops.prims.convert_element_type.default(arg357_1, torch.float32);  arg357_1 = None
    add_29: "f32[24]" = torch.ops.aten.add.Tensor(convert_element_type_21, 1e-05);  convert_element_type_21 = None
    sqrt_10: "f32[24]" = torch.ops.aten.sqrt.default(add_29);  add_29 = None
    reciprocal_10: "f32[24]" = torch.ops.aten.reciprocal.default(sqrt_10);  sqrt_10 = None
    mul_37: "f32[24]" = torch.ops.aten.mul.Tensor(reciprocal_10, 1);  reciprocal_10 = None
    unsqueeze_80: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_20, -1);  convert_element_type_20 = None
    unsqueeze_81: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
    unsqueeze_82: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(mul_37, -1);  mul_37 = None
    unsqueeze_83: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    sub_10: "f32[8, 24, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_81);  convolution_10 = unsqueeze_81 = None
    mul_38: "f32[8, 24, 64, 64]" = torch.ops.aten.mul.Tensor(sub_10, unsqueeze_83);  sub_10 = unsqueeze_83 = None
    unsqueeze_84: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg20_1, -1);  arg20_1 = None
    unsqueeze_85: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    mul_39: "f32[8, 24, 64, 64]" = torch.ops.aten.mul.Tensor(mul_38, unsqueeze_85);  mul_38 = unsqueeze_85 = None
    unsqueeze_86: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg21_1, -1);  arg21_1 = None
    unsqueeze_87: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
    add_30: "f32[8, 24, 64, 64]" = torch.ops.aten.add.Tensor(mul_39, unsqueeze_87);  mul_39 = unsqueeze_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_31: "f32[8, 24, 64, 64]" = torch.ops.aten.add.Tensor(add_30, add_22);  add_30 = add_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_11: "f32[8, 48, 64, 64]" = torch.ops.aten.convolution.default(add_31, arg187_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg187_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_22: "f32[48]" = torch.ops.prims.convert_element_type.default(arg358_1, torch.float32);  arg358_1 = None
    convert_element_type_23: "f32[48]" = torch.ops.prims.convert_element_type.default(arg359_1, torch.float32);  arg359_1 = None
    add_32: "f32[48]" = torch.ops.aten.add.Tensor(convert_element_type_23, 1e-05);  convert_element_type_23 = None
    sqrt_11: "f32[48]" = torch.ops.aten.sqrt.default(add_32);  add_32 = None
    reciprocal_11: "f32[48]" = torch.ops.aten.reciprocal.default(sqrt_11);  sqrt_11 = None
    mul_40: "f32[48]" = torch.ops.aten.mul.Tensor(reciprocal_11, 1);  reciprocal_11 = None
    unsqueeze_88: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_22, -1);  convert_element_type_22 = None
    unsqueeze_89: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
    unsqueeze_90: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(mul_40, -1);  mul_40 = None
    unsqueeze_91: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    sub_11: "f32[8, 48, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_89);  convolution_11 = unsqueeze_89 = None
    mul_41: "f32[8, 48, 64, 64]" = torch.ops.aten.mul.Tensor(sub_11, unsqueeze_91);  sub_11 = unsqueeze_91 = None
    unsqueeze_92: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(arg22_1, -1);  arg22_1 = None
    unsqueeze_93: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
    mul_42: "f32[8, 48, 64, 64]" = torch.ops.aten.mul.Tensor(mul_41, unsqueeze_93);  mul_41 = unsqueeze_93 = None
    unsqueeze_94: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(arg23_1, -1);  arg23_1 = None
    unsqueeze_95: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
    add_33: "f32[8, 48, 64, 64]" = torch.ops.aten.add.Tensor(mul_42, unsqueeze_95);  mul_42 = unsqueeze_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_34: "f32[8, 48, 64, 64]" = torch.ops.aten.add.Tensor(add_33, 3)
    clamp_min_7: "f32[8, 48, 64, 64]" = torch.ops.aten.clamp_min.default(add_34, 0);  add_34 = None
    clamp_max_7: "f32[8, 48, 64, 64]" = torch.ops.aten.clamp_max.default(clamp_min_7, 6);  clamp_min_7 = None
    mul_43: "f32[8, 48, 64, 64]" = torch.ops.aten.mul.Tensor(add_33, clamp_max_7);  add_33 = clamp_max_7 = None
    div_7: "f32[8, 48, 64, 64]" = torch.ops.aten.div.Tensor(mul_43, 6);  mul_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_12: "f32[8, 48, 64, 64]" = torch.ops.aten.convolution.default(div_7, arg188_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 48);  div_7 = arg188_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_24: "f32[48]" = torch.ops.prims.convert_element_type.default(arg360_1, torch.float32);  arg360_1 = None
    convert_element_type_25: "f32[48]" = torch.ops.prims.convert_element_type.default(arg361_1, torch.float32);  arg361_1 = None
    add_35: "f32[48]" = torch.ops.aten.add.Tensor(convert_element_type_25, 1e-05);  convert_element_type_25 = None
    sqrt_12: "f32[48]" = torch.ops.aten.sqrt.default(add_35);  add_35 = None
    reciprocal_12: "f32[48]" = torch.ops.aten.reciprocal.default(sqrt_12);  sqrt_12 = None
    mul_44: "f32[48]" = torch.ops.aten.mul.Tensor(reciprocal_12, 1);  reciprocal_12 = None
    unsqueeze_96: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_24, -1);  convert_element_type_24 = None
    unsqueeze_97: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
    unsqueeze_98: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(mul_44, -1);  mul_44 = None
    unsqueeze_99: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
    sub_12: "f32[8, 48, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_97);  convolution_12 = unsqueeze_97 = None
    mul_45: "f32[8, 48, 64, 64]" = torch.ops.aten.mul.Tensor(sub_12, unsqueeze_99);  sub_12 = unsqueeze_99 = None
    unsqueeze_100: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(arg24_1, -1);  arg24_1 = None
    unsqueeze_101: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
    mul_46: "f32[8, 48, 64, 64]" = torch.ops.aten.mul.Tensor(mul_45, unsqueeze_101);  mul_45 = unsqueeze_101 = None
    unsqueeze_102: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(arg25_1, -1);  arg25_1 = None
    unsqueeze_103: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    add_36: "f32[8, 48, 64, 64]" = torch.ops.aten.add.Tensor(mul_46, unsqueeze_103);  mul_46 = unsqueeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_37: "f32[8, 48, 64, 64]" = torch.ops.aten.add.Tensor(add_36, 3)
    clamp_min_8: "f32[8, 48, 64, 64]" = torch.ops.aten.clamp_min.default(add_37, 0);  add_37 = None
    clamp_max_8: "f32[8, 48, 64, 64]" = torch.ops.aten.clamp_max.default(clamp_min_8, 6);  clamp_min_8 = None
    mul_47: "f32[8, 48, 64, 64]" = torch.ops.aten.mul.Tensor(add_36, clamp_max_8);  add_36 = clamp_max_8 = None
    div_8: "f32[8, 48, 64, 64]" = torch.ops.aten.div.Tensor(mul_47, 6);  mul_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_13: "f32[8, 24, 64, 64]" = torch.ops.aten.convolution.default(div_8, arg189_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_8 = arg189_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_26: "f32[24]" = torch.ops.prims.convert_element_type.default(arg362_1, torch.float32);  arg362_1 = None
    convert_element_type_27: "f32[24]" = torch.ops.prims.convert_element_type.default(arg363_1, torch.float32);  arg363_1 = None
    add_38: "f32[24]" = torch.ops.aten.add.Tensor(convert_element_type_27, 1e-05);  convert_element_type_27 = None
    sqrt_13: "f32[24]" = torch.ops.aten.sqrt.default(add_38);  add_38 = None
    reciprocal_13: "f32[24]" = torch.ops.aten.reciprocal.default(sqrt_13);  sqrt_13 = None
    mul_48: "f32[24]" = torch.ops.aten.mul.Tensor(reciprocal_13, 1);  reciprocal_13 = None
    unsqueeze_104: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_26, -1);  convert_element_type_26 = None
    unsqueeze_105: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
    unsqueeze_106: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(mul_48, -1);  mul_48 = None
    unsqueeze_107: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
    sub_13: "f32[8, 24, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_105);  convolution_13 = unsqueeze_105 = None
    mul_49: "f32[8, 24, 64, 64]" = torch.ops.aten.mul.Tensor(sub_13, unsqueeze_107);  sub_13 = unsqueeze_107 = None
    unsqueeze_108: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg26_1, -1);  arg26_1 = None
    unsqueeze_109: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
    mul_50: "f32[8, 24, 64, 64]" = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_109);  mul_49 = unsqueeze_109 = None
    unsqueeze_110: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg27_1, -1);  arg27_1 = None
    unsqueeze_111: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
    add_39: "f32[8, 24, 64, 64]" = torch.ops.aten.add.Tensor(mul_50, unsqueeze_111);  mul_50 = unsqueeze_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_40: "f32[8, 24, 64, 64]" = torch.ops.aten.add.Tensor(add_39, add_31);  add_39 = add_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_14: "f32[8, 48, 64, 64]" = torch.ops.aten.convolution.default(add_40, arg190_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg190_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_28: "f32[48]" = torch.ops.prims.convert_element_type.default(arg364_1, torch.float32);  arg364_1 = None
    convert_element_type_29: "f32[48]" = torch.ops.prims.convert_element_type.default(arg365_1, torch.float32);  arg365_1 = None
    add_41: "f32[48]" = torch.ops.aten.add.Tensor(convert_element_type_29, 1e-05);  convert_element_type_29 = None
    sqrt_14: "f32[48]" = torch.ops.aten.sqrt.default(add_41);  add_41 = None
    reciprocal_14: "f32[48]" = torch.ops.aten.reciprocal.default(sqrt_14);  sqrt_14 = None
    mul_51: "f32[48]" = torch.ops.aten.mul.Tensor(reciprocal_14, 1);  reciprocal_14 = None
    unsqueeze_112: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_28, -1);  convert_element_type_28 = None
    unsqueeze_113: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
    unsqueeze_114: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(mul_51, -1);  mul_51 = None
    unsqueeze_115: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
    sub_14: "f32[8, 48, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_113);  convolution_14 = unsqueeze_113 = None
    mul_52: "f32[8, 48, 64, 64]" = torch.ops.aten.mul.Tensor(sub_14, unsqueeze_115);  sub_14 = unsqueeze_115 = None
    unsqueeze_116: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(arg28_1, -1);  arg28_1 = None
    unsqueeze_117: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
    mul_53: "f32[8, 48, 64, 64]" = torch.ops.aten.mul.Tensor(mul_52, unsqueeze_117);  mul_52 = unsqueeze_117 = None
    unsqueeze_118: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(arg29_1, -1);  arg29_1 = None
    unsqueeze_119: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
    add_42: "f32[8, 48, 64, 64]" = torch.ops.aten.add.Tensor(mul_53, unsqueeze_119);  mul_53 = unsqueeze_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_43: "f32[8, 48, 64, 64]" = torch.ops.aten.add.Tensor(add_42, 3)
    clamp_min_9: "f32[8, 48, 64, 64]" = torch.ops.aten.clamp_min.default(add_43, 0);  add_43 = None
    clamp_max_9: "f32[8, 48, 64, 64]" = torch.ops.aten.clamp_max.default(clamp_min_9, 6);  clamp_min_9 = None
    mul_54: "f32[8, 48, 64, 64]" = torch.ops.aten.mul.Tensor(add_42, clamp_max_9);  add_42 = clamp_max_9 = None
    div_9: "f32[8, 48, 64, 64]" = torch.ops.aten.div.Tensor(mul_54, 6);  mul_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_15: "f32[8, 48, 64, 64]" = torch.ops.aten.convolution.default(div_9, arg191_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 48);  div_9 = arg191_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_30: "f32[48]" = torch.ops.prims.convert_element_type.default(arg366_1, torch.float32);  arg366_1 = None
    convert_element_type_31: "f32[48]" = torch.ops.prims.convert_element_type.default(arg367_1, torch.float32);  arg367_1 = None
    add_44: "f32[48]" = torch.ops.aten.add.Tensor(convert_element_type_31, 1e-05);  convert_element_type_31 = None
    sqrt_15: "f32[48]" = torch.ops.aten.sqrt.default(add_44);  add_44 = None
    reciprocal_15: "f32[48]" = torch.ops.aten.reciprocal.default(sqrt_15);  sqrt_15 = None
    mul_55: "f32[48]" = torch.ops.aten.mul.Tensor(reciprocal_15, 1);  reciprocal_15 = None
    unsqueeze_120: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_30, -1);  convert_element_type_30 = None
    unsqueeze_121: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
    unsqueeze_122: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(mul_55, -1);  mul_55 = None
    unsqueeze_123: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
    sub_15: "f32[8, 48, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_121);  convolution_15 = unsqueeze_121 = None
    mul_56: "f32[8, 48, 64, 64]" = torch.ops.aten.mul.Tensor(sub_15, unsqueeze_123);  sub_15 = unsqueeze_123 = None
    unsqueeze_124: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(arg30_1, -1);  arg30_1 = None
    unsqueeze_125: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
    mul_57: "f32[8, 48, 64, 64]" = torch.ops.aten.mul.Tensor(mul_56, unsqueeze_125);  mul_56 = unsqueeze_125 = None
    unsqueeze_126: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(arg31_1, -1);  arg31_1 = None
    unsqueeze_127: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
    add_45: "f32[8, 48, 64, 64]" = torch.ops.aten.add.Tensor(mul_57, unsqueeze_127);  mul_57 = unsqueeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_46: "f32[8, 48, 64, 64]" = torch.ops.aten.add.Tensor(add_45, 3)
    clamp_min_10: "f32[8, 48, 64, 64]" = torch.ops.aten.clamp_min.default(add_46, 0);  add_46 = None
    clamp_max_10: "f32[8, 48, 64, 64]" = torch.ops.aten.clamp_max.default(clamp_min_10, 6);  clamp_min_10 = None
    mul_58: "f32[8, 48, 64, 64]" = torch.ops.aten.mul.Tensor(add_45, clamp_max_10);  add_45 = clamp_max_10 = None
    div_10: "f32[8, 48, 64, 64]" = torch.ops.aten.div.Tensor(mul_58, 6);  mul_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_16: "f32[8, 24, 64, 64]" = torch.ops.aten.convolution.default(div_10, arg192_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_10 = arg192_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_32: "f32[24]" = torch.ops.prims.convert_element_type.default(arg368_1, torch.float32);  arg368_1 = None
    convert_element_type_33: "f32[24]" = torch.ops.prims.convert_element_type.default(arg369_1, torch.float32);  arg369_1 = None
    add_47: "f32[24]" = torch.ops.aten.add.Tensor(convert_element_type_33, 1e-05);  convert_element_type_33 = None
    sqrt_16: "f32[24]" = torch.ops.aten.sqrt.default(add_47);  add_47 = None
    reciprocal_16: "f32[24]" = torch.ops.aten.reciprocal.default(sqrt_16);  sqrt_16 = None
    mul_59: "f32[24]" = torch.ops.aten.mul.Tensor(reciprocal_16, 1);  reciprocal_16 = None
    unsqueeze_128: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_32, -1);  convert_element_type_32 = None
    unsqueeze_129: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
    unsqueeze_130: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(mul_59, -1);  mul_59 = None
    unsqueeze_131: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
    sub_16: "f32[8, 24, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_129);  convolution_16 = unsqueeze_129 = None
    mul_60: "f32[8, 24, 64, 64]" = torch.ops.aten.mul.Tensor(sub_16, unsqueeze_131);  sub_16 = unsqueeze_131 = None
    unsqueeze_132: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg32_1, -1);  arg32_1 = None
    unsqueeze_133: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
    mul_61: "f32[8, 24, 64, 64]" = torch.ops.aten.mul.Tensor(mul_60, unsqueeze_133);  mul_60 = unsqueeze_133 = None
    unsqueeze_134: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg33_1, -1);  arg33_1 = None
    unsqueeze_135: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
    add_48: "f32[8, 24, 64, 64]" = torch.ops.aten.add.Tensor(mul_61, unsqueeze_135);  mul_61 = unsqueeze_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_49: "f32[8, 24, 64, 64]" = torch.ops.aten.add.Tensor(add_48, add_40);  add_48 = add_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_17: "f32[8, 120, 64, 64]" = torch.ops.aten.convolution.default(add_49, arg193_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_49 = arg193_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_34: "f32[120]" = torch.ops.prims.convert_element_type.default(arg370_1, torch.float32);  arg370_1 = None
    convert_element_type_35: "f32[120]" = torch.ops.prims.convert_element_type.default(arg371_1, torch.float32);  arg371_1 = None
    add_50: "f32[120]" = torch.ops.aten.add.Tensor(convert_element_type_35, 1e-05);  convert_element_type_35 = None
    sqrt_17: "f32[120]" = torch.ops.aten.sqrt.default(add_50);  add_50 = None
    reciprocal_17: "f32[120]" = torch.ops.aten.reciprocal.default(sqrt_17);  sqrt_17 = None
    mul_62: "f32[120]" = torch.ops.aten.mul.Tensor(reciprocal_17, 1);  reciprocal_17 = None
    unsqueeze_136: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_34, -1);  convert_element_type_34 = None
    unsqueeze_137: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
    unsqueeze_138: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(mul_62, -1);  mul_62 = None
    unsqueeze_139: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
    sub_17: "f32[8, 120, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_137);  convolution_17 = unsqueeze_137 = None
    mul_63: "f32[8, 120, 64, 64]" = torch.ops.aten.mul.Tensor(sub_17, unsqueeze_139);  sub_17 = unsqueeze_139 = None
    unsqueeze_140: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg34_1, -1);  arg34_1 = None
    unsqueeze_141: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
    mul_64: "f32[8, 120, 64, 64]" = torch.ops.aten.mul.Tensor(mul_63, unsqueeze_141);  mul_63 = unsqueeze_141 = None
    unsqueeze_142: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg35_1, -1);  arg35_1 = None
    unsqueeze_143: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
    add_51: "f32[8, 120, 64, 64]" = torch.ops.aten.add.Tensor(mul_64, unsqueeze_143);  mul_64 = unsqueeze_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_52: "f32[8, 120, 64, 64]" = torch.ops.aten.add.Tensor(add_51, 3)
    clamp_min_11: "f32[8, 120, 64, 64]" = torch.ops.aten.clamp_min.default(add_52, 0);  add_52 = None
    clamp_max_11: "f32[8, 120, 64, 64]" = torch.ops.aten.clamp_max.default(clamp_min_11, 6);  clamp_min_11 = None
    mul_65: "f32[8, 120, 64, 64]" = torch.ops.aten.mul.Tensor(add_51, clamp_max_11);  add_51 = clamp_max_11 = None
    div_11: "f32[8, 120, 64, 64]" = torch.ops.aten.div.Tensor(mul_65, 6);  mul_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_18: "f32[8, 120, 32, 32]" = torch.ops.aten.convolution.default(div_11, arg194_1, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 120);  div_11 = arg194_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_36: "f32[120]" = torch.ops.prims.convert_element_type.default(arg372_1, torch.float32);  arg372_1 = None
    convert_element_type_37: "f32[120]" = torch.ops.prims.convert_element_type.default(arg373_1, torch.float32);  arg373_1 = None
    add_53: "f32[120]" = torch.ops.aten.add.Tensor(convert_element_type_37, 1e-05);  convert_element_type_37 = None
    sqrt_18: "f32[120]" = torch.ops.aten.sqrt.default(add_53);  add_53 = None
    reciprocal_18: "f32[120]" = torch.ops.aten.reciprocal.default(sqrt_18);  sqrt_18 = None
    mul_66: "f32[120]" = torch.ops.aten.mul.Tensor(reciprocal_18, 1);  reciprocal_18 = None
    unsqueeze_144: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_36, -1);  convert_element_type_36 = None
    unsqueeze_145: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
    unsqueeze_146: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(mul_66, -1);  mul_66 = None
    unsqueeze_147: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
    sub_18: "f32[8, 120, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_145);  convolution_18 = unsqueeze_145 = None
    mul_67: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(sub_18, unsqueeze_147);  sub_18 = unsqueeze_147 = None
    unsqueeze_148: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg36_1, -1);  arg36_1 = None
    unsqueeze_149: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
    mul_68: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(mul_67, unsqueeze_149);  mul_67 = unsqueeze_149 = None
    unsqueeze_150: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg37_1, -1);  arg37_1 = None
    unsqueeze_151: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
    add_54: "f32[8, 120, 32, 32]" = torch.ops.aten.add.Tensor(mul_68, unsqueeze_151);  mul_68 = unsqueeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_55: "f32[8, 120, 32, 32]" = torch.ops.aten.add.Tensor(add_54, 3)
    clamp_min_12: "f32[8, 120, 32, 32]" = torch.ops.aten.clamp_min.default(add_55, 0);  add_55 = None
    clamp_max_12: "f32[8, 120, 32, 32]" = torch.ops.aten.clamp_max.default(clamp_min_12, 6);  clamp_min_12 = None
    mul_69: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(add_54, clamp_max_12);  add_54 = clamp_max_12 = None
    div_12: "f32[8, 120, 32, 32]" = torch.ops.aten.div.Tensor(mul_69, 6);  mul_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean: "f32[8, 120, 1, 1]" = torch.ops.aten.mean.dim(div_12, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_19: "f32[8, 8, 1, 1]" = torch.ops.aten.convolution.default(mean, arg195_1, arg196_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean = arg195_1 = arg196_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    add_56: "f32[8, 8, 1, 1]" = torch.ops.aten.add.Tensor(convolution_19, 3)
    clamp_min_13: "f32[8, 8, 1, 1]" = torch.ops.aten.clamp_min.default(add_56, 0);  add_56 = None
    clamp_max_13: "f32[8, 8, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_13, 6);  clamp_min_13 = None
    mul_70: "f32[8, 8, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_19, clamp_max_13);  convolution_19 = clamp_max_13 = None
    div_13: "f32[8, 8, 1, 1]" = torch.ops.aten.div.Tensor(mul_70, 6);  mul_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_20: "f32[8, 120, 1, 1]" = torch.ops.aten.convolution.default(div_13, arg197_1, arg198_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_13 = arg197_1 = arg198_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    add_57: "f32[8, 120, 1, 1]" = torch.ops.aten.add.Tensor(convolution_20, 3);  convolution_20 = None
    clamp_min_14: "f32[8, 120, 1, 1]" = torch.ops.aten.clamp_min.default(add_57, 0);  add_57 = None
    clamp_max_14: "f32[8, 120, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_14, 6);  clamp_min_14 = None
    div_14: "f32[8, 120, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_14, 6);  clamp_max_14 = None
    mul_71: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(div_12, div_14);  div_12 = div_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_21: "f32[8, 40, 32, 32]" = torch.ops.aten.convolution.default(mul_71, arg199_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_71 = arg199_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_38: "f32[40]" = torch.ops.prims.convert_element_type.default(arg374_1, torch.float32);  arg374_1 = None
    convert_element_type_39: "f32[40]" = torch.ops.prims.convert_element_type.default(arg375_1, torch.float32);  arg375_1 = None
    add_58: "f32[40]" = torch.ops.aten.add.Tensor(convert_element_type_39, 1e-05);  convert_element_type_39 = None
    sqrt_19: "f32[40]" = torch.ops.aten.sqrt.default(add_58);  add_58 = None
    reciprocal_19: "f32[40]" = torch.ops.aten.reciprocal.default(sqrt_19);  sqrt_19 = None
    mul_72: "f32[40]" = torch.ops.aten.mul.Tensor(reciprocal_19, 1);  reciprocal_19 = None
    unsqueeze_152: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_38, -1);  convert_element_type_38 = None
    unsqueeze_153: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
    unsqueeze_154: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(mul_72, -1);  mul_72 = None
    unsqueeze_155: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
    sub_19: "f32[8, 40, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_153);  convolution_21 = unsqueeze_153 = None
    mul_73: "f32[8, 40, 32, 32]" = torch.ops.aten.mul.Tensor(sub_19, unsqueeze_155);  sub_19 = unsqueeze_155 = None
    unsqueeze_156: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg38_1, -1);  arg38_1 = None
    unsqueeze_157: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
    mul_74: "f32[8, 40, 32, 32]" = torch.ops.aten.mul.Tensor(mul_73, unsqueeze_157);  mul_73 = unsqueeze_157 = None
    unsqueeze_158: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg39_1, -1);  arg39_1 = None
    unsqueeze_159: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
    add_59: "f32[8, 40, 32, 32]" = torch.ops.aten.add.Tensor(mul_74, unsqueeze_159);  mul_74 = unsqueeze_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_22: "f32[8, 120, 32, 32]" = torch.ops.aten.convolution.default(add_59, arg200_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg200_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_40: "f32[120]" = torch.ops.prims.convert_element_type.default(arg376_1, torch.float32);  arg376_1 = None
    convert_element_type_41: "f32[120]" = torch.ops.prims.convert_element_type.default(arg377_1, torch.float32);  arg377_1 = None
    add_60: "f32[120]" = torch.ops.aten.add.Tensor(convert_element_type_41, 1e-05);  convert_element_type_41 = None
    sqrt_20: "f32[120]" = torch.ops.aten.sqrt.default(add_60);  add_60 = None
    reciprocal_20: "f32[120]" = torch.ops.aten.reciprocal.default(sqrt_20);  sqrt_20 = None
    mul_75: "f32[120]" = torch.ops.aten.mul.Tensor(reciprocal_20, 1);  reciprocal_20 = None
    unsqueeze_160: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_40, -1);  convert_element_type_40 = None
    unsqueeze_161: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
    unsqueeze_162: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(mul_75, -1);  mul_75 = None
    unsqueeze_163: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
    sub_20: "f32[8, 120, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_161);  convolution_22 = unsqueeze_161 = None
    mul_76: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(sub_20, unsqueeze_163);  sub_20 = unsqueeze_163 = None
    unsqueeze_164: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg40_1, -1);  arg40_1 = None
    unsqueeze_165: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
    mul_77: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(mul_76, unsqueeze_165);  mul_76 = unsqueeze_165 = None
    unsqueeze_166: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg41_1, -1);  arg41_1 = None
    unsqueeze_167: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
    add_61: "f32[8, 120, 32, 32]" = torch.ops.aten.add.Tensor(mul_77, unsqueeze_167);  mul_77 = unsqueeze_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_62: "f32[8, 120, 32, 32]" = torch.ops.aten.add.Tensor(add_61, 3)
    clamp_min_15: "f32[8, 120, 32, 32]" = torch.ops.aten.clamp_min.default(add_62, 0);  add_62 = None
    clamp_max_15: "f32[8, 120, 32, 32]" = torch.ops.aten.clamp_max.default(clamp_min_15, 6);  clamp_min_15 = None
    mul_78: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(add_61, clamp_max_15);  add_61 = clamp_max_15 = None
    div_15: "f32[8, 120, 32, 32]" = torch.ops.aten.div.Tensor(mul_78, 6);  mul_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_23: "f32[8, 120, 32, 32]" = torch.ops.aten.convolution.default(div_15, arg201_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 120);  div_15 = arg201_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_42: "f32[120]" = torch.ops.prims.convert_element_type.default(arg378_1, torch.float32);  arg378_1 = None
    convert_element_type_43: "f32[120]" = torch.ops.prims.convert_element_type.default(arg379_1, torch.float32);  arg379_1 = None
    add_63: "f32[120]" = torch.ops.aten.add.Tensor(convert_element_type_43, 1e-05);  convert_element_type_43 = None
    sqrt_21: "f32[120]" = torch.ops.aten.sqrt.default(add_63);  add_63 = None
    reciprocal_21: "f32[120]" = torch.ops.aten.reciprocal.default(sqrt_21);  sqrt_21 = None
    mul_79: "f32[120]" = torch.ops.aten.mul.Tensor(reciprocal_21, 1);  reciprocal_21 = None
    unsqueeze_168: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_42, -1);  convert_element_type_42 = None
    unsqueeze_169: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
    unsqueeze_170: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(mul_79, -1);  mul_79 = None
    unsqueeze_171: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
    sub_21: "f32[8, 120, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_169);  convolution_23 = unsqueeze_169 = None
    mul_80: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(sub_21, unsqueeze_171);  sub_21 = unsqueeze_171 = None
    unsqueeze_172: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg42_1, -1);  arg42_1 = None
    unsqueeze_173: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
    mul_81: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(mul_80, unsqueeze_173);  mul_80 = unsqueeze_173 = None
    unsqueeze_174: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg43_1, -1);  arg43_1 = None
    unsqueeze_175: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
    add_64: "f32[8, 120, 32, 32]" = torch.ops.aten.add.Tensor(mul_81, unsqueeze_175);  mul_81 = unsqueeze_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_65: "f32[8, 120, 32, 32]" = torch.ops.aten.add.Tensor(add_64, 3)
    clamp_min_16: "f32[8, 120, 32, 32]" = torch.ops.aten.clamp_min.default(add_65, 0);  add_65 = None
    clamp_max_16: "f32[8, 120, 32, 32]" = torch.ops.aten.clamp_max.default(clamp_min_16, 6);  clamp_min_16 = None
    mul_82: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(add_64, clamp_max_16);  add_64 = clamp_max_16 = None
    div_16: "f32[8, 120, 32, 32]" = torch.ops.aten.div.Tensor(mul_82, 6);  mul_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_1: "f32[8, 120, 1, 1]" = torch.ops.aten.mean.dim(div_16, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_24: "f32[8, 16, 1, 1]" = torch.ops.aten.convolution.default(mean_1, arg202_1, arg203_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_1 = arg202_1 = arg203_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    add_66: "f32[8, 16, 1, 1]" = torch.ops.aten.add.Tensor(convolution_24, 3)
    clamp_min_17: "f32[8, 16, 1, 1]" = torch.ops.aten.clamp_min.default(add_66, 0);  add_66 = None
    clamp_max_17: "f32[8, 16, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_17, 6);  clamp_min_17 = None
    mul_83: "f32[8, 16, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_24, clamp_max_17);  convolution_24 = clamp_max_17 = None
    div_17: "f32[8, 16, 1, 1]" = torch.ops.aten.div.Tensor(mul_83, 6);  mul_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_25: "f32[8, 120, 1, 1]" = torch.ops.aten.convolution.default(div_17, arg204_1, arg205_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_17 = arg204_1 = arg205_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    add_67: "f32[8, 120, 1, 1]" = torch.ops.aten.add.Tensor(convolution_25, 3);  convolution_25 = None
    clamp_min_18: "f32[8, 120, 1, 1]" = torch.ops.aten.clamp_min.default(add_67, 0);  add_67 = None
    clamp_max_18: "f32[8, 120, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_18, 6);  clamp_min_18 = None
    div_18: "f32[8, 120, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_18, 6);  clamp_max_18 = None
    mul_84: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(div_16, div_18);  div_16 = div_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_26: "f32[8, 40, 32, 32]" = torch.ops.aten.convolution.default(mul_84, arg206_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_84 = arg206_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_44: "f32[40]" = torch.ops.prims.convert_element_type.default(arg380_1, torch.float32);  arg380_1 = None
    convert_element_type_45: "f32[40]" = torch.ops.prims.convert_element_type.default(arg381_1, torch.float32);  arg381_1 = None
    add_68: "f32[40]" = torch.ops.aten.add.Tensor(convert_element_type_45, 1e-05);  convert_element_type_45 = None
    sqrt_22: "f32[40]" = torch.ops.aten.sqrt.default(add_68);  add_68 = None
    reciprocal_22: "f32[40]" = torch.ops.aten.reciprocal.default(sqrt_22);  sqrt_22 = None
    mul_85: "f32[40]" = torch.ops.aten.mul.Tensor(reciprocal_22, 1);  reciprocal_22 = None
    unsqueeze_176: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_44, -1);  convert_element_type_44 = None
    unsqueeze_177: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
    unsqueeze_178: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(mul_85, -1);  mul_85 = None
    unsqueeze_179: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
    sub_22: "f32[8, 40, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_177);  convolution_26 = unsqueeze_177 = None
    mul_86: "f32[8, 40, 32, 32]" = torch.ops.aten.mul.Tensor(sub_22, unsqueeze_179);  sub_22 = unsqueeze_179 = None
    unsqueeze_180: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg44_1, -1);  arg44_1 = None
    unsqueeze_181: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
    mul_87: "f32[8, 40, 32, 32]" = torch.ops.aten.mul.Tensor(mul_86, unsqueeze_181);  mul_86 = unsqueeze_181 = None
    unsqueeze_182: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg45_1, -1);  arg45_1 = None
    unsqueeze_183: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
    add_69: "f32[8, 40, 32, 32]" = torch.ops.aten.add.Tensor(mul_87, unsqueeze_183);  mul_87 = unsqueeze_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_70: "f32[8, 40, 32, 32]" = torch.ops.aten.add.Tensor(add_69, add_59);  add_69 = add_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_27: "f32[8, 120, 32, 32]" = torch.ops.aten.convolution.default(add_70, arg207_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg207_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_46: "f32[120]" = torch.ops.prims.convert_element_type.default(arg382_1, torch.float32);  arg382_1 = None
    convert_element_type_47: "f32[120]" = torch.ops.prims.convert_element_type.default(arg383_1, torch.float32);  arg383_1 = None
    add_71: "f32[120]" = torch.ops.aten.add.Tensor(convert_element_type_47, 1e-05);  convert_element_type_47 = None
    sqrt_23: "f32[120]" = torch.ops.aten.sqrt.default(add_71);  add_71 = None
    reciprocal_23: "f32[120]" = torch.ops.aten.reciprocal.default(sqrt_23);  sqrt_23 = None
    mul_88: "f32[120]" = torch.ops.aten.mul.Tensor(reciprocal_23, 1);  reciprocal_23 = None
    unsqueeze_184: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_46, -1);  convert_element_type_46 = None
    unsqueeze_185: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
    unsqueeze_186: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(mul_88, -1);  mul_88 = None
    unsqueeze_187: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
    sub_23: "f32[8, 120, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_185);  convolution_27 = unsqueeze_185 = None
    mul_89: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(sub_23, unsqueeze_187);  sub_23 = unsqueeze_187 = None
    unsqueeze_188: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg46_1, -1);  arg46_1 = None
    unsqueeze_189: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
    mul_90: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(mul_89, unsqueeze_189);  mul_89 = unsqueeze_189 = None
    unsqueeze_190: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg47_1, -1);  arg47_1 = None
    unsqueeze_191: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
    add_72: "f32[8, 120, 32, 32]" = torch.ops.aten.add.Tensor(mul_90, unsqueeze_191);  mul_90 = unsqueeze_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_73: "f32[8, 120, 32, 32]" = torch.ops.aten.add.Tensor(add_72, 3)
    clamp_min_19: "f32[8, 120, 32, 32]" = torch.ops.aten.clamp_min.default(add_73, 0);  add_73 = None
    clamp_max_19: "f32[8, 120, 32, 32]" = torch.ops.aten.clamp_max.default(clamp_min_19, 6);  clamp_min_19 = None
    mul_91: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(add_72, clamp_max_19);  add_72 = clamp_max_19 = None
    div_19: "f32[8, 120, 32, 32]" = torch.ops.aten.div.Tensor(mul_91, 6);  mul_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_28: "f32[8, 120, 32, 32]" = torch.ops.aten.convolution.default(div_19, arg208_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 120);  div_19 = arg208_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_48: "f32[120]" = torch.ops.prims.convert_element_type.default(arg384_1, torch.float32);  arg384_1 = None
    convert_element_type_49: "f32[120]" = torch.ops.prims.convert_element_type.default(arg385_1, torch.float32);  arg385_1 = None
    add_74: "f32[120]" = torch.ops.aten.add.Tensor(convert_element_type_49, 1e-05);  convert_element_type_49 = None
    sqrt_24: "f32[120]" = torch.ops.aten.sqrt.default(add_74);  add_74 = None
    reciprocal_24: "f32[120]" = torch.ops.aten.reciprocal.default(sqrt_24);  sqrt_24 = None
    mul_92: "f32[120]" = torch.ops.aten.mul.Tensor(reciprocal_24, 1);  reciprocal_24 = None
    unsqueeze_192: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_48, -1);  convert_element_type_48 = None
    unsqueeze_193: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
    unsqueeze_194: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(mul_92, -1);  mul_92 = None
    unsqueeze_195: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
    sub_24: "f32[8, 120, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_193);  convolution_28 = unsqueeze_193 = None
    mul_93: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(sub_24, unsqueeze_195);  sub_24 = unsqueeze_195 = None
    unsqueeze_196: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg48_1, -1);  arg48_1 = None
    unsqueeze_197: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
    mul_94: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(mul_93, unsqueeze_197);  mul_93 = unsqueeze_197 = None
    unsqueeze_198: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg49_1, -1);  arg49_1 = None
    unsqueeze_199: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
    add_75: "f32[8, 120, 32, 32]" = torch.ops.aten.add.Tensor(mul_94, unsqueeze_199);  mul_94 = unsqueeze_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_76: "f32[8, 120, 32, 32]" = torch.ops.aten.add.Tensor(add_75, 3)
    clamp_min_20: "f32[8, 120, 32, 32]" = torch.ops.aten.clamp_min.default(add_76, 0);  add_76 = None
    clamp_max_20: "f32[8, 120, 32, 32]" = torch.ops.aten.clamp_max.default(clamp_min_20, 6);  clamp_min_20 = None
    mul_95: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(add_75, clamp_max_20);  add_75 = clamp_max_20 = None
    div_20: "f32[8, 120, 32, 32]" = torch.ops.aten.div.Tensor(mul_95, 6);  mul_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_2: "f32[8, 120, 1, 1]" = torch.ops.aten.mean.dim(div_20, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_29: "f32[8, 16, 1, 1]" = torch.ops.aten.convolution.default(mean_2, arg209_1, arg210_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_2 = arg209_1 = arg210_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    add_77: "f32[8, 16, 1, 1]" = torch.ops.aten.add.Tensor(convolution_29, 3)
    clamp_min_21: "f32[8, 16, 1, 1]" = torch.ops.aten.clamp_min.default(add_77, 0);  add_77 = None
    clamp_max_21: "f32[8, 16, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_21, 6);  clamp_min_21 = None
    mul_96: "f32[8, 16, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_29, clamp_max_21);  convolution_29 = clamp_max_21 = None
    div_21: "f32[8, 16, 1, 1]" = torch.ops.aten.div.Tensor(mul_96, 6);  mul_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_30: "f32[8, 120, 1, 1]" = torch.ops.aten.convolution.default(div_21, arg211_1, arg212_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_21 = arg211_1 = arg212_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    add_78: "f32[8, 120, 1, 1]" = torch.ops.aten.add.Tensor(convolution_30, 3);  convolution_30 = None
    clamp_min_22: "f32[8, 120, 1, 1]" = torch.ops.aten.clamp_min.default(add_78, 0);  add_78 = None
    clamp_max_22: "f32[8, 120, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_22, 6);  clamp_min_22 = None
    div_22: "f32[8, 120, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_22, 6);  clamp_max_22 = None
    mul_97: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(div_20, div_22);  div_20 = div_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_31: "f32[8, 40, 32, 32]" = torch.ops.aten.convolution.default(mul_97, arg213_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_97 = arg213_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_50: "f32[40]" = torch.ops.prims.convert_element_type.default(arg386_1, torch.float32);  arg386_1 = None
    convert_element_type_51: "f32[40]" = torch.ops.prims.convert_element_type.default(arg387_1, torch.float32);  arg387_1 = None
    add_79: "f32[40]" = torch.ops.aten.add.Tensor(convert_element_type_51, 1e-05);  convert_element_type_51 = None
    sqrt_25: "f32[40]" = torch.ops.aten.sqrt.default(add_79);  add_79 = None
    reciprocal_25: "f32[40]" = torch.ops.aten.reciprocal.default(sqrt_25);  sqrt_25 = None
    mul_98: "f32[40]" = torch.ops.aten.mul.Tensor(reciprocal_25, 1);  reciprocal_25 = None
    unsqueeze_200: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_50, -1);  convert_element_type_50 = None
    unsqueeze_201: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
    unsqueeze_202: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(mul_98, -1);  mul_98 = None
    unsqueeze_203: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
    sub_25: "f32[8, 40, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_201);  convolution_31 = unsqueeze_201 = None
    mul_99: "f32[8, 40, 32, 32]" = torch.ops.aten.mul.Tensor(sub_25, unsqueeze_203);  sub_25 = unsqueeze_203 = None
    unsqueeze_204: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg50_1, -1);  arg50_1 = None
    unsqueeze_205: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
    mul_100: "f32[8, 40, 32, 32]" = torch.ops.aten.mul.Tensor(mul_99, unsqueeze_205);  mul_99 = unsqueeze_205 = None
    unsqueeze_206: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg51_1, -1);  arg51_1 = None
    unsqueeze_207: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
    add_80: "f32[8, 40, 32, 32]" = torch.ops.aten.add.Tensor(mul_100, unsqueeze_207);  mul_100 = unsqueeze_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_81: "f32[8, 40, 32, 32]" = torch.ops.aten.add.Tensor(add_80, add_70);  add_80 = add_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_32: "f32[8, 120, 32, 32]" = torch.ops.aten.convolution.default(add_81, arg214_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg214_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_52: "f32[120]" = torch.ops.prims.convert_element_type.default(arg388_1, torch.float32);  arg388_1 = None
    convert_element_type_53: "f32[120]" = torch.ops.prims.convert_element_type.default(arg389_1, torch.float32);  arg389_1 = None
    add_82: "f32[120]" = torch.ops.aten.add.Tensor(convert_element_type_53, 1e-05);  convert_element_type_53 = None
    sqrt_26: "f32[120]" = torch.ops.aten.sqrt.default(add_82);  add_82 = None
    reciprocal_26: "f32[120]" = torch.ops.aten.reciprocal.default(sqrt_26);  sqrt_26 = None
    mul_101: "f32[120]" = torch.ops.aten.mul.Tensor(reciprocal_26, 1);  reciprocal_26 = None
    unsqueeze_208: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_52, -1);  convert_element_type_52 = None
    unsqueeze_209: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
    unsqueeze_210: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(mul_101, -1);  mul_101 = None
    unsqueeze_211: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
    sub_26: "f32[8, 120, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_209);  convolution_32 = unsqueeze_209 = None
    mul_102: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(sub_26, unsqueeze_211);  sub_26 = unsqueeze_211 = None
    unsqueeze_212: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg52_1, -1);  arg52_1 = None
    unsqueeze_213: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
    mul_103: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(mul_102, unsqueeze_213);  mul_102 = unsqueeze_213 = None
    unsqueeze_214: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg53_1, -1);  arg53_1 = None
    unsqueeze_215: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
    add_83: "f32[8, 120, 32, 32]" = torch.ops.aten.add.Tensor(mul_103, unsqueeze_215);  mul_103 = unsqueeze_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_84: "f32[8, 120, 32, 32]" = torch.ops.aten.add.Tensor(add_83, 3)
    clamp_min_23: "f32[8, 120, 32, 32]" = torch.ops.aten.clamp_min.default(add_84, 0);  add_84 = None
    clamp_max_23: "f32[8, 120, 32, 32]" = torch.ops.aten.clamp_max.default(clamp_min_23, 6);  clamp_min_23 = None
    mul_104: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(add_83, clamp_max_23);  add_83 = clamp_max_23 = None
    div_23: "f32[8, 120, 32, 32]" = torch.ops.aten.div.Tensor(mul_104, 6);  mul_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_33: "f32[8, 120, 32, 32]" = torch.ops.aten.convolution.default(div_23, arg215_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 120);  div_23 = arg215_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_54: "f32[120]" = torch.ops.prims.convert_element_type.default(arg390_1, torch.float32);  arg390_1 = None
    convert_element_type_55: "f32[120]" = torch.ops.prims.convert_element_type.default(arg391_1, torch.float32);  arg391_1 = None
    add_85: "f32[120]" = torch.ops.aten.add.Tensor(convert_element_type_55, 1e-05);  convert_element_type_55 = None
    sqrt_27: "f32[120]" = torch.ops.aten.sqrt.default(add_85);  add_85 = None
    reciprocal_27: "f32[120]" = torch.ops.aten.reciprocal.default(sqrt_27);  sqrt_27 = None
    mul_105: "f32[120]" = torch.ops.aten.mul.Tensor(reciprocal_27, 1);  reciprocal_27 = None
    unsqueeze_216: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_54, -1);  convert_element_type_54 = None
    unsqueeze_217: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
    unsqueeze_218: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(mul_105, -1);  mul_105 = None
    unsqueeze_219: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, -1);  unsqueeze_218 = None
    sub_27: "f32[8, 120, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_217);  convolution_33 = unsqueeze_217 = None
    mul_106: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(sub_27, unsqueeze_219);  sub_27 = unsqueeze_219 = None
    unsqueeze_220: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg54_1, -1);  arg54_1 = None
    unsqueeze_221: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, -1);  unsqueeze_220 = None
    mul_107: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(mul_106, unsqueeze_221);  mul_106 = unsqueeze_221 = None
    unsqueeze_222: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg55_1, -1);  arg55_1 = None
    unsqueeze_223: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, -1);  unsqueeze_222 = None
    add_86: "f32[8, 120, 32, 32]" = torch.ops.aten.add.Tensor(mul_107, unsqueeze_223);  mul_107 = unsqueeze_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_87: "f32[8, 120, 32, 32]" = torch.ops.aten.add.Tensor(add_86, 3)
    clamp_min_24: "f32[8, 120, 32, 32]" = torch.ops.aten.clamp_min.default(add_87, 0);  add_87 = None
    clamp_max_24: "f32[8, 120, 32, 32]" = torch.ops.aten.clamp_max.default(clamp_min_24, 6);  clamp_min_24 = None
    mul_108: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(add_86, clamp_max_24);  add_86 = clamp_max_24 = None
    div_24: "f32[8, 120, 32, 32]" = torch.ops.aten.div.Tensor(mul_108, 6);  mul_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_3: "f32[8, 120, 1, 1]" = torch.ops.aten.mean.dim(div_24, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_34: "f32[8, 16, 1, 1]" = torch.ops.aten.convolution.default(mean_3, arg216_1, arg217_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_3 = arg216_1 = arg217_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    add_88: "f32[8, 16, 1, 1]" = torch.ops.aten.add.Tensor(convolution_34, 3)
    clamp_min_25: "f32[8, 16, 1, 1]" = torch.ops.aten.clamp_min.default(add_88, 0);  add_88 = None
    clamp_max_25: "f32[8, 16, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_25, 6);  clamp_min_25 = None
    mul_109: "f32[8, 16, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_34, clamp_max_25);  convolution_34 = clamp_max_25 = None
    div_25: "f32[8, 16, 1, 1]" = torch.ops.aten.div.Tensor(mul_109, 6);  mul_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_35: "f32[8, 120, 1, 1]" = torch.ops.aten.convolution.default(div_25, arg218_1, arg219_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_25 = arg218_1 = arg219_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    add_89: "f32[8, 120, 1, 1]" = torch.ops.aten.add.Tensor(convolution_35, 3);  convolution_35 = None
    clamp_min_26: "f32[8, 120, 1, 1]" = torch.ops.aten.clamp_min.default(add_89, 0);  add_89 = None
    clamp_max_26: "f32[8, 120, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_26, 6);  clamp_min_26 = None
    div_26: "f32[8, 120, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_26, 6);  clamp_max_26 = None
    mul_110: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(div_24, div_26);  div_24 = div_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_36: "f32[8, 40, 32, 32]" = torch.ops.aten.convolution.default(mul_110, arg220_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_110 = arg220_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_56: "f32[40]" = torch.ops.prims.convert_element_type.default(arg392_1, torch.float32);  arg392_1 = None
    convert_element_type_57: "f32[40]" = torch.ops.prims.convert_element_type.default(arg393_1, torch.float32);  arg393_1 = None
    add_90: "f32[40]" = torch.ops.aten.add.Tensor(convert_element_type_57, 1e-05);  convert_element_type_57 = None
    sqrt_28: "f32[40]" = torch.ops.aten.sqrt.default(add_90);  add_90 = None
    reciprocal_28: "f32[40]" = torch.ops.aten.reciprocal.default(sqrt_28);  sqrt_28 = None
    mul_111: "f32[40]" = torch.ops.aten.mul.Tensor(reciprocal_28, 1);  reciprocal_28 = None
    unsqueeze_224: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_56, -1);  convert_element_type_56 = None
    unsqueeze_225: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, -1);  unsqueeze_224 = None
    unsqueeze_226: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(mul_111, -1);  mul_111 = None
    unsqueeze_227: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, -1);  unsqueeze_226 = None
    sub_28: "f32[8, 40, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_225);  convolution_36 = unsqueeze_225 = None
    mul_112: "f32[8, 40, 32, 32]" = torch.ops.aten.mul.Tensor(sub_28, unsqueeze_227);  sub_28 = unsqueeze_227 = None
    unsqueeze_228: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg56_1, -1);  arg56_1 = None
    unsqueeze_229: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, -1);  unsqueeze_228 = None
    mul_113: "f32[8, 40, 32, 32]" = torch.ops.aten.mul.Tensor(mul_112, unsqueeze_229);  mul_112 = unsqueeze_229 = None
    unsqueeze_230: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg57_1, -1);  arg57_1 = None
    unsqueeze_231: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, -1);  unsqueeze_230 = None
    add_91: "f32[8, 40, 32, 32]" = torch.ops.aten.add.Tensor(mul_113, unsqueeze_231);  mul_113 = unsqueeze_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_92: "f32[8, 40, 32, 32]" = torch.ops.aten.add.Tensor(add_91, add_81);  add_91 = add_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_37: "f32[8, 120, 32, 32]" = torch.ops.aten.convolution.default(add_92, arg221_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg221_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_58: "f32[120]" = torch.ops.prims.convert_element_type.default(arg394_1, torch.float32);  arg394_1 = None
    convert_element_type_59: "f32[120]" = torch.ops.prims.convert_element_type.default(arg395_1, torch.float32);  arg395_1 = None
    add_93: "f32[120]" = torch.ops.aten.add.Tensor(convert_element_type_59, 1e-05);  convert_element_type_59 = None
    sqrt_29: "f32[120]" = torch.ops.aten.sqrt.default(add_93);  add_93 = None
    reciprocal_29: "f32[120]" = torch.ops.aten.reciprocal.default(sqrt_29);  sqrt_29 = None
    mul_114: "f32[120]" = torch.ops.aten.mul.Tensor(reciprocal_29, 1);  reciprocal_29 = None
    unsqueeze_232: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_58, -1);  convert_element_type_58 = None
    unsqueeze_233: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, -1);  unsqueeze_232 = None
    unsqueeze_234: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(mul_114, -1);  mul_114 = None
    unsqueeze_235: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, -1);  unsqueeze_234 = None
    sub_29: "f32[8, 120, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_233);  convolution_37 = unsqueeze_233 = None
    mul_115: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(sub_29, unsqueeze_235);  sub_29 = unsqueeze_235 = None
    unsqueeze_236: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg58_1, -1);  arg58_1 = None
    unsqueeze_237: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, -1);  unsqueeze_236 = None
    mul_116: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(mul_115, unsqueeze_237);  mul_115 = unsqueeze_237 = None
    unsqueeze_238: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg59_1, -1);  arg59_1 = None
    unsqueeze_239: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, -1);  unsqueeze_238 = None
    add_94: "f32[8, 120, 32, 32]" = torch.ops.aten.add.Tensor(mul_116, unsqueeze_239);  mul_116 = unsqueeze_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_95: "f32[8, 120, 32, 32]" = torch.ops.aten.add.Tensor(add_94, 3)
    clamp_min_27: "f32[8, 120, 32, 32]" = torch.ops.aten.clamp_min.default(add_95, 0);  add_95 = None
    clamp_max_27: "f32[8, 120, 32, 32]" = torch.ops.aten.clamp_max.default(clamp_min_27, 6);  clamp_min_27 = None
    mul_117: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(add_94, clamp_max_27);  add_94 = clamp_max_27 = None
    div_27: "f32[8, 120, 32, 32]" = torch.ops.aten.div.Tensor(mul_117, 6);  mul_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_38: "f32[8, 120, 32, 32]" = torch.ops.aten.convolution.default(div_27, arg222_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 120);  div_27 = arg222_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_60: "f32[120]" = torch.ops.prims.convert_element_type.default(arg396_1, torch.float32);  arg396_1 = None
    convert_element_type_61: "f32[120]" = torch.ops.prims.convert_element_type.default(arg397_1, torch.float32);  arg397_1 = None
    add_96: "f32[120]" = torch.ops.aten.add.Tensor(convert_element_type_61, 1e-05);  convert_element_type_61 = None
    sqrt_30: "f32[120]" = torch.ops.aten.sqrt.default(add_96);  add_96 = None
    reciprocal_30: "f32[120]" = torch.ops.aten.reciprocal.default(sqrt_30);  sqrt_30 = None
    mul_118: "f32[120]" = torch.ops.aten.mul.Tensor(reciprocal_30, 1);  reciprocal_30 = None
    unsqueeze_240: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_60, -1);  convert_element_type_60 = None
    unsqueeze_241: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, -1);  unsqueeze_240 = None
    unsqueeze_242: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(mul_118, -1);  mul_118 = None
    unsqueeze_243: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, -1);  unsqueeze_242 = None
    sub_30: "f32[8, 120, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_241);  convolution_38 = unsqueeze_241 = None
    mul_119: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(sub_30, unsqueeze_243);  sub_30 = unsqueeze_243 = None
    unsqueeze_244: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg60_1, -1);  arg60_1 = None
    unsqueeze_245: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, -1);  unsqueeze_244 = None
    mul_120: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(mul_119, unsqueeze_245);  mul_119 = unsqueeze_245 = None
    unsqueeze_246: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg61_1, -1);  arg61_1 = None
    unsqueeze_247: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, -1);  unsqueeze_246 = None
    add_97: "f32[8, 120, 32, 32]" = torch.ops.aten.add.Tensor(mul_120, unsqueeze_247);  mul_120 = unsqueeze_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_98: "f32[8, 120, 32, 32]" = torch.ops.aten.add.Tensor(add_97, 3)
    clamp_min_28: "f32[8, 120, 32, 32]" = torch.ops.aten.clamp_min.default(add_98, 0);  add_98 = None
    clamp_max_28: "f32[8, 120, 32, 32]" = torch.ops.aten.clamp_max.default(clamp_min_28, 6);  clamp_min_28 = None
    mul_121: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(add_97, clamp_max_28);  add_97 = clamp_max_28 = None
    div_28: "f32[8, 120, 32, 32]" = torch.ops.aten.div.Tensor(mul_121, 6);  mul_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_4: "f32[8, 120, 1, 1]" = torch.ops.aten.mean.dim(div_28, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_39: "f32[8, 16, 1, 1]" = torch.ops.aten.convolution.default(mean_4, arg223_1, arg224_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_4 = arg223_1 = arg224_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    add_99: "f32[8, 16, 1, 1]" = torch.ops.aten.add.Tensor(convolution_39, 3)
    clamp_min_29: "f32[8, 16, 1, 1]" = torch.ops.aten.clamp_min.default(add_99, 0);  add_99 = None
    clamp_max_29: "f32[8, 16, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_29, 6);  clamp_min_29 = None
    mul_122: "f32[8, 16, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_39, clamp_max_29);  convolution_39 = clamp_max_29 = None
    div_29: "f32[8, 16, 1, 1]" = torch.ops.aten.div.Tensor(mul_122, 6);  mul_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_40: "f32[8, 120, 1, 1]" = torch.ops.aten.convolution.default(div_29, arg225_1, arg226_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_29 = arg225_1 = arg226_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    add_100: "f32[8, 120, 1, 1]" = torch.ops.aten.add.Tensor(convolution_40, 3);  convolution_40 = None
    clamp_min_30: "f32[8, 120, 1, 1]" = torch.ops.aten.clamp_min.default(add_100, 0);  add_100 = None
    clamp_max_30: "f32[8, 120, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_30, 6);  clamp_min_30 = None
    div_30: "f32[8, 120, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_30, 6);  clamp_max_30 = None
    mul_123: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(div_28, div_30);  div_28 = div_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_41: "f32[8, 40, 32, 32]" = torch.ops.aten.convolution.default(mul_123, arg227_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_123 = arg227_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_62: "f32[40]" = torch.ops.prims.convert_element_type.default(arg398_1, torch.float32);  arg398_1 = None
    convert_element_type_63: "f32[40]" = torch.ops.prims.convert_element_type.default(arg399_1, torch.float32);  arg399_1 = None
    add_101: "f32[40]" = torch.ops.aten.add.Tensor(convert_element_type_63, 1e-05);  convert_element_type_63 = None
    sqrt_31: "f32[40]" = torch.ops.aten.sqrt.default(add_101);  add_101 = None
    reciprocal_31: "f32[40]" = torch.ops.aten.reciprocal.default(sqrt_31);  sqrt_31 = None
    mul_124: "f32[40]" = torch.ops.aten.mul.Tensor(reciprocal_31, 1);  reciprocal_31 = None
    unsqueeze_248: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_62, -1);  convert_element_type_62 = None
    unsqueeze_249: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, -1);  unsqueeze_248 = None
    unsqueeze_250: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(mul_124, -1);  mul_124 = None
    unsqueeze_251: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, -1);  unsqueeze_250 = None
    sub_31: "f32[8, 40, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_249);  convolution_41 = unsqueeze_249 = None
    mul_125: "f32[8, 40, 32, 32]" = torch.ops.aten.mul.Tensor(sub_31, unsqueeze_251);  sub_31 = unsqueeze_251 = None
    unsqueeze_252: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg62_1, -1);  arg62_1 = None
    unsqueeze_253: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, -1);  unsqueeze_252 = None
    mul_126: "f32[8, 40, 32, 32]" = torch.ops.aten.mul.Tensor(mul_125, unsqueeze_253);  mul_125 = unsqueeze_253 = None
    unsqueeze_254: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg63_1, -1);  arg63_1 = None
    unsqueeze_255: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, -1);  unsqueeze_254 = None
    add_102: "f32[8, 40, 32, 32]" = torch.ops.aten.add.Tensor(mul_126, unsqueeze_255);  mul_126 = unsqueeze_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_103: "f32[8, 40, 32, 32]" = torch.ops.aten.add.Tensor(add_102, add_92);  add_102 = add_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_42: "f32[8, 200, 32, 32]" = torch.ops.aten.convolution.default(add_103, arg228_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_103 = arg228_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_64: "f32[200]" = torch.ops.prims.convert_element_type.default(arg400_1, torch.float32);  arg400_1 = None
    convert_element_type_65: "f32[200]" = torch.ops.prims.convert_element_type.default(arg401_1, torch.float32);  arg401_1 = None
    add_104: "f32[200]" = torch.ops.aten.add.Tensor(convert_element_type_65, 1e-05);  convert_element_type_65 = None
    sqrt_32: "f32[200]" = torch.ops.aten.sqrt.default(add_104);  add_104 = None
    reciprocal_32: "f32[200]" = torch.ops.aten.reciprocal.default(sqrt_32);  sqrt_32 = None
    mul_127: "f32[200]" = torch.ops.aten.mul.Tensor(reciprocal_32, 1);  reciprocal_32 = None
    unsqueeze_256: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_64, -1);  convert_element_type_64 = None
    unsqueeze_257: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, -1);  unsqueeze_256 = None
    unsqueeze_258: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(mul_127, -1);  mul_127 = None
    unsqueeze_259: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, -1);  unsqueeze_258 = None
    sub_32: "f32[8, 200, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_257);  convolution_42 = unsqueeze_257 = None
    mul_128: "f32[8, 200, 32, 32]" = torch.ops.aten.mul.Tensor(sub_32, unsqueeze_259);  sub_32 = unsqueeze_259 = None
    unsqueeze_260: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg64_1, -1);  arg64_1 = None
    unsqueeze_261: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, -1);  unsqueeze_260 = None
    mul_129: "f32[8, 200, 32, 32]" = torch.ops.aten.mul.Tensor(mul_128, unsqueeze_261);  mul_128 = unsqueeze_261 = None
    unsqueeze_262: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg65_1, -1);  arg65_1 = None
    unsqueeze_263: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, -1);  unsqueeze_262 = None
    add_105: "f32[8, 200, 32, 32]" = torch.ops.aten.add.Tensor(mul_129, unsqueeze_263);  mul_129 = unsqueeze_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_106: "f32[8, 200, 32, 32]" = torch.ops.aten.add.Tensor(add_105, 3)
    clamp_min_31: "f32[8, 200, 32, 32]" = torch.ops.aten.clamp_min.default(add_106, 0);  add_106 = None
    clamp_max_31: "f32[8, 200, 32, 32]" = torch.ops.aten.clamp_max.default(clamp_min_31, 6);  clamp_min_31 = None
    mul_130: "f32[8, 200, 32, 32]" = torch.ops.aten.mul.Tensor(add_105, clamp_max_31);  add_105 = clamp_max_31 = None
    div_31: "f32[8, 200, 32, 32]" = torch.ops.aten.div.Tensor(mul_130, 6);  mul_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_43: "f32[8, 200, 16, 16]" = torch.ops.aten.convolution.default(div_31, arg229_1, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 200);  div_31 = arg229_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_66: "f32[200]" = torch.ops.prims.convert_element_type.default(arg402_1, torch.float32);  arg402_1 = None
    convert_element_type_67: "f32[200]" = torch.ops.prims.convert_element_type.default(arg403_1, torch.float32);  arg403_1 = None
    add_107: "f32[200]" = torch.ops.aten.add.Tensor(convert_element_type_67, 1e-05);  convert_element_type_67 = None
    sqrt_33: "f32[200]" = torch.ops.aten.sqrt.default(add_107);  add_107 = None
    reciprocal_33: "f32[200]" = torch.ops.aten.reciprocal.default(sqrt_33);  sqrt_33 = None
    mul_131: "f32[200]" = torch.ops.aten.mul.Tensor(reciprocal_33, 1);  reciprocal_33 = None
    unsqueeze_264: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_66, -1);  convert_element_type_66 = None
    unsqueeze_265: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, -1);  unsqueeze_264 = None
    unsqueeze_266: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(mul_131, -1);  mul_131 = None
    unsqueeze_267: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, -1);  unsqueeze_266 = None
    sub_33: "f32[8, 200, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_265);  convolution_43 = unsqueeze_265 = None
    mul_132: "f32[8, 200, 16, 16]" = torch.ops.aten.mul.Tensor(sub_33, unsqueeze_267);  sub_33 = unsqueeze_267 = None
    unsqueeze_268: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg66_1, -1);  arg66_1 = None
    unsqueeze_269: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, -1);  unsqueeze_268 = None
    mul_133: "f32[8, 200, 16, 16]" = torch.ops.aten.mul.Tensor(mul_132, unsqueeze_269);  mul_132 = unsqueeze_269 = None
    unsqueeze_270: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg67_1, -1);  arg67_1 = None
    unsqueeze_271: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, -1);  unsqueeze_270 = None
    add_108: "f32[8, 200, 16, 16]" = torch.ops.aten.add.Tensor(mul_133, unsqueeze_271);  mul_133 = unsqueeze_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_109: "f32[8, 200, 16, 16]" = torch.ops.aten.add.Tensor(add_108, 3)
    clamp_min_32: "f32[8, 200, 16, 16]" = torch.ops.aten.clamp_min.default(add_109, 0);  add_109 = None
    clamp_max_32: "f32[8, 200, 16, 16]" = torch.ops.aten.clamp_max.default(clamp_min_32, 6);  clamp_min_32 = None
    mul_134: "f32[8, 200, 16, 16]" = torch.ops.aten.mul.Tensor(add_108, clamp_max_32);  add_108 = clamp_max_32 = None
    div_32: "f32[8, 200, 16, 16]" = torch.ops.aten.div.Tensor(mul_134, 6);  mul_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_44: "f32[8, 72, 16, 16]" = torch.ops.aten.convolution.default(div_32, arg230_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_32 = arg230_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_68: "f32[72]" = torch.ops.prims.convert_element_type.default(arg404_1, torch.float32);  arg404_1 = None
    convert_element_type_69: "f32[72]" = torch.ops.prims.convert_element_type.default(arg405_1, torch.float32);  arg405_1 = None
    add_110: "f32[72]" = torch.ops.aten.add.Tensor(convert_element_type_69, 1e-05);  convert_element_type_69 = None
    sqrt_34: "f32[72]" = torch.ops.aten.sqrt.default(add_110);  add_110 = None
    reciprocal_34: "f32[72]" = torch.ops.aten.reciprocal.default(sqrt_34);  sqrt_34 = None
    mul_135: "f32[72]" = torch.ops.aten.mul.Tensor(reciprocal_34, 1);  reciprocal_34 = None
    unsqueeze_272: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_68, -1);  convert_element_type_68 = None
    unsqueeze_273: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, -1);  unsqueeze_272 = None
    unsqueeze_274: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(mul_135, -1);  mul_135 = None
    unsqueeze_275: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, -1);  unsqueeze_274 = None
    sub_34: "f32[8, 72, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_273);  convolution_44 = unsqueeze_273 = None
    mul_136: "f32[8, 72, 16, 16]" = torch.ops.aten.mul.Tensor(sub_34, unsqueeze_275);  sub_34 = unsqueeze_275 = None
    unsqueeze_276: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(arg68_1, -1);  arg68_1 = None
    unsqueeze_277: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, -1);  unsqueeze_276 = None
    mul_137: "f32[8, 72, 16, 16]" = torch.ops.aten.mul.Tensor(mul_136, unsqueeze_277);  mul_136 = unsqueeze_277 = None
    unsqueeze_278: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(arg69_1, -1);  arg69_1 = None
    unsqueeze_279: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, -1);  unsqueeze_278 = None
    add_111: "f32[8, 72, 16, 16]" = torch.ops.aten.add.Tensor(mul_137, unsqueeze_279);  mul_137 = unsqueeze_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_45: "f32[8, 216, 16, 16]" = torch.ops.aten.convolution.default(add_111, arg231_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg231_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_70: "f32[216]" = torch.ops.prims.convert_element_type.default(arg406_1, torch.float32);  arg406_1 = None
    convert_element_type_71: "f32[216]" = torch.ops.prims.convert_element_type.default(arg407_1, torch.float32);  arg407_1 = None
    add_112: "f32[216]" = torch.ops.aten.add.Tensor(convert_element_type_71, 1e-05);  convert_element_type_71 = None
    sqrt_35: "f32[216]" = torch.ops.aten.sqrt.default(add_112);  add_112 = None
    reciprocal_35: "f32[216]" = torch.ops.aten.reciprocal.default(sqrt_35);  sqrt_35 = None
    mul_138: "f32[216]" = torch.ops.aten.mul.Tensor(reciprocal_35, 1);  reciprocal_35 = None
    unsqueeze_280: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_70, -1);  convert_element_type_70 = None
    unsqueeze_281: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, -1);  unsqueeze_280 = None
    unsqueeze_282: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(mul_138, -1);  mul_138 = None
    unsqueeze_283: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, -1);  unsqueeze_282 = None
    sub_35: "f32[8, 216, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_281);  convolution_45 = unsqueeze_281 = None
    mul_139: "f32[8, 216, 16, 16]" = torch.ops.aten.mul.Tensor(sub_35, unsqueeze_283);  sub_35 = unsqueeze_283 = None
    unsqueeze_284: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(arg70_1, -1);  arg70_1 = None
    unsqueeze_285: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, -1);  unsqueeze_284 = None
    mul_140: "f32[8, 216, 16, 16]" = torch.ops.aten.mul.Tensor(mul_139, unsqueeze_285);  mul_139 = unsqueeze_285 = None
    unsqueeze_286: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(arg71_1, -1);  arg71_1 = None
    unsqueeze_287: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, -1);  unsqueeze_286 = None
    add_113: "f32[8, 216, 16, 16]" = torch.ops.aten.add.Tensor(mul_140, unsqueeze_287);  mul_140 = unsqueeze_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_114: "f32[8, 216, 16, 16]" = torch.ops.aten.add.Tensor(add_113, 3)
    clamp_min_33: "f32[8, 216, 16, 16]" = torch.ops.aten.clamp_min.default(add_114, 0);  add_114 = None
    clamp_max_33: "f32[8, 216, 16, 16]" = torch.ops.aten.clamp_max.default(clamp_min_33, 6);  clamp_min_33 = None
    mul_141: "f32[8, 216, 16, 16]" = torch.ops.aten.mul.Tensor(add_113, clamp_max_33);  add_113 = clamp_max_33 = None
    div_33: "f32[8, 216, 16, 16]" = torch.ops.aten.div.Tensor(mul_141, 6);  mul_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_46: "f32[8, 216, 16, 16]" = torch.ops.aten.convolution.default(div_33, arg232_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216);  div_33 = arg232_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_72: "f32[216]" = torch.ops.prims.convert_element_type.default(arg408_1, torch.float32);  arg408_1 = None
    convert_element_type_73: "f32[216]" = torch.ops.prims.convert_element_type.default(arg409_1, torch.float32);  arg409_1 = None
    add_115: "f32[216]" = torch.ops.aten.add.Tensor(convert_element_type_73, 1e-05);  convert_element_type_73 = None
    sqrt_36: "f32[216]" = torch.ops.aten.sqrt.default(add_115);  add_115 = None
    reciprocal_36: "f32[216]" = torch.ops.aten.reciprocal.default(sqrt_36);  sqrt_36 = None
    mul_142: "f32[216]" = torch.ops.aten.mul.Tensor(reciprocal_36, 1);  reciprocal_36 = None
    unsqueeze_288: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_72, -1);  convert_element_type_72 = None
    unsqueeze_289: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, -1);  unsqueeze_288 = None
    unsqueeze_290: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(mul_142, -1);  mul_142 = None
    unsqueeze_291: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, -1);  unsqueeze_290 = None
    sub_36: "f32[8, 216, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_289);  convolution_46 = unsqueeze_289 = None
    mul_143: "f32[8, 216, 16, 16]" = torch.ops.aten.mul.Tensor(sub_36, unsqueeze_291);  sub_36 = unsqueeze_291 = None
    unsqueeze_292: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(arg72_1, -1);  arg72_1 = None
    unsqueeze_293: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, -1);  unsqueeze_292 = None
    mul_144: "f32[8, 216, 16, 16]" = torch.ops.aten.mul.Tensor(mul_143, unsqueeze_293);  mul_143 = unsqueeze_293 = None
    unsqueeze_294: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(arg73_1, -1);  arg73_1 = None
    unsqueeze_295: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, -1);  unsqueeze_294 = None
    add_116: "f32[8, 216, 16, 16]" = torch.ops.aten.add.Tensor(mul_144, unsqueeze_295);  mul_144 = unsqueeze_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_117: "f32[8, 216, 16, 16]" = torch.ops.aten.add.Tensor(add_116, 3)
    clamp_min_34: "f32[8, 216, 16, 16]" = torch.ops.aten.clamp_min.default(add_117, 0);  add_117 = None
    clamp_max_34: "f32[8, 216, 16, 16]" = torch.ops.aten.clamp_max.default(clamp_min_34, 6);  clamp_min_34 = None
    mul_145: "f32[8, 216, 16, 16]" = torch.ops.aten.mul.Tensor(add_116, clamp_max_34);  add_116 = clamp_max_34 = None
    div_34: "f32[8, 216, 16, 16]" = torch.ops.aten.div.Tensor(mul_145, 6);  mul_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_47: "f32[8, 72, 16, 16]" = torch.ops.aten.convolution.default(div_34, arg233_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_34 = arg233_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_74: "f32[72]" = torch.ops.prims.convert_element_type.default(arg410_1, torch.float32);  arg410_1 = None
    convert_element_type_75: "f32[72]" = torch.ops.prims.convert_element_type.default(arg411_1, torch.float32);  arg411_1 = None
    add_118: "f32[72]" = torch.ops.aten.add.Tensor(convert_element_type_75, 1e-05);  convert_element_type_75 = None
    sqrt_37: "f32[72]" = torch.ops.aten.sqrt.default(add_118);  add_118 = None
    reciprocal_37: "f32[72]" = torch.ops.aten.reciprocal.default(sqrt_37);  sqrt_37 = None
    mul_146: "f32[72]" = torch.ops.aten.mul.Tensor(reciprocal_37, 1);  reciprocal_37 = None
    unsqueeze_296: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_74, -1);  convert_element_type_74 = None
    unsqueeze_297: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, -1);  unsqueeze_296 = None
    unsqueeze_298: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(mul_146, -1);  mul_146 = None
    unsqueeze_299: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, -1);  unsqueeze_298 = None
    sub_37: "f32[8, 72, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_297);  convolution_47 = unsqueeze_297 = None
    mul_147: "f32[8, 72, 16, 16]" = torch.ops.aten.mul.Tensor(sub_37, unsqueeze_299);  sub_37 = unsqueeze_299 = None
    unsqueeze_300: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(arg74_1, -1);  arg74_1 = None
    unsqueeze_301: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, -1);  unsqueeze_300 = None
    mul_148: "f32[8, 72, 16, 16]" = torch.ops.aten.mul.Tensor(mul_147, unsqueeze_301);  mul_147 = unsqueeze_301 = None
    unsqueeze_302: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(arg75_1, -1);  arg75_1 = None
    unsqueeze_303: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, -1);  unsqueeze_302 = None
    add_119: "f32[8, 72, 16, 16]" = torch.ops.aten.add.Tensor(mul_148, unsqueeze_303);  mul_148 = unsqueeze_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_120: "f32[8, 72, 16, 16]" = torch.ops.aten.add.Tensor(add_119, add_111);  add_119 = add_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_48: "f32[8, 216, 16, 16]" = torch.ops.aten.convolution.default(add_120, arg234_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg234_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_76: "f32[216]" = torch.ops.prims.convert_element_type.default(arg412_1, torch.float32);  arg412_1 = None
    convert_element_type_77: "f32[216]" = torch.ops.prims.convert_element_type.default(arg413_1, torch.float32);  arg413_1 = None
    add_121: "f32[216]" = torch.ops.aten.add.Tensor(convert_element_type_77, 1e-05);  convert_element_type_77 = None
    sqrt_38: "f32[216]" = torch.ops.aten.sqrt.default(add_121);  add_121 = None
    reciprocal_38: "f32[216]" = torch.ops.aten.reciprocal.default(sqrt_38);  sqrt_38 = None
    mul_149: "f32[216]" = torch.ops.aten.mul.Tensor(reciprocal_38, 1);  reciprocal_38 = None
    unsqueeze_304: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_76, -1);  convert_element_type_76 = None
    unsqueeze_305: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, -1);  unsqueeze_304 = None
    unsqueeze_306: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(mul_149, -1);  mul_149 = None
    unsqueeze_307: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, -1);  unsqueeze_306 = None
    sub_38: "f32[8, 216, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_305);  convolution_48 = unsqueeze_305 = None
    mul_150: "f32[8, 216, 16, 16]" = torch.ops.aten.mul.Tensor(sub_38, unsqueeze_307);  sub_38 = unsqueeze_307 = None
    unsqueeze_308: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(arg76_1, -1);  arg76_1 = None
    unsqueeze_309: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, -1);  unsqueeze_308 = None
    mul_151: "f32[8, 216, 16, 16]" = torch.ops.aten.mul.Tensor(mul_150, unsqueeze_309);  mul_150 = unsqueeze_309 = None
    unsqueeze_310: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(arg77_1, -1);  arg77_1 = None
    unsqueeze_311: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, -1);  unsqueeze_310 = None
    add_122: "f32[8, 216, 16, 16]" = torch.ops.aten.add.Tensor(mul_151, unsqueeze_311);  mul_151 = unsqueeze_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_123: "f32[8, 216, 16, 16]" = torch.ops.aten.add.Tensor(add_122, 3)
    clamp_min_35: "f32[8, 216, 16, 16]" = torch.ops.aten.clamp_min.default(add_123, 0);  add_123 = None
    clamp_max_35: "f32[8, 216, 16, 16]" = torch.ops.aten.clamp_max.default(clamp_min_35, 6);  clamp_min_35 = None
    mul_152: "f32[8, 216, 16, 16]" = torch.ops.aten.mul.Tensor(add_122, clamp_max_35);  add_122 = clamp_max_35 = None
    div_35: "f32[8, 216, 16, 16]" = torch.ops.aten.div.Tensor(mul_152, 6);  mul_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_49: "f32[8, 216, 16, 16]" = torch.ops.aten.convolution.default(div_35, arg235_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216);  div_35 = arg235_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_78: "f32[216]" = torch.ops.prims.convert_element_type.default(arg414_1, torch.float32);  arg414_1 = None
    convert_element_type_79: "f32[216]" = torch.ops.prims.convert_element_type.default(arg415_1, torch.float32);  arg415_1 = None
    add_124: "f32[216]" = torch.ops.aten.add.Tensor(convert_element_type_79, 1e-05);  convert_element_type_79 = None
    sqrt_39: "f32[216]" = torch.ops.aten.sqrt.default(add_124);  add_124 = None
    reciprocal_39: "f32[216]" = torch.ops.aten.reciprocal.default(sqrt_39);  sqrt_39 = None
    mul_153: "f32[216]" = torch.ops.aten.mul.Tensor(reciprocal_39, 1);  reciprocal_39 = None
    unsqueeze_312: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_78, -1);  convert_element_type_78 = None
    unsqueeze_313: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, -1);  unsqueeze_312 = None
    unsqueeze_314: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(mul_153, -1);  mul_153 = None
    unsqueeze_315: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, -1);  unsqueeze_314 = None
    sub_39: "f32[8, 216, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_313);  convolution_49 = unsqueeze_313 = None
    mul_154: "f32[8, 216, 16, 16]" = torch.ops.aten.mul.Tensor(sub_39, unsqueeze_315);  sub_39 = unsqueeze_315 = None
    unsqueeze_316: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(arg78_1, -1);  arg78_1 = None
    unsqueeze_317: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, -1);  unsqueeze_316 = None
    mul_155: "f32[8, 216, 16, 16]" = torch.ops.aten.mul.Tensor(mul_154, unsqueeze_317);  mul_154 = unsqueeze_317 = None
    unsqueeze_318: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(arg79_1, -1);  arg79_1 = None
    unsqueeze_319: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, -1);  unsqueeze_318 = None
    add_125: "f32[8, 216, 16, 16]" = torch.ops.aten.add.Tensor(mul_155, unsqueeze_319);  mul_155 = unsqueeze_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_126: "f32[8, 216, 16, 16]" = torch.ops.aten.add.Tensor(add_125, 3)
    clamp_min_36: "f32[8, 216, 16, 16]" = torch.ops.aten.clamp_min.default(add_126, 0);  add_126 = None
    clamp_max_36: "f32[8, 216, 16, 16]" = torch.ops.aten.clamp_max.default(clamp_min_36, 6);  clamp_min_36 = None
    mul_156: "f32[8, 216, 16, 16]" = torch.ops.aten.mul.Tensor(add_125, clamp_max_36);  add_125 = clamp_max_36 = None
    div_36: "f32[8, 216, 16, 16]" = torch.ops.aten.div.Tensor(mul_156, 6);  mul_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_50: "f32[8, 72, 16, 16]" = torch.ops.aten.convolution.default(div_36, arg236_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_36 = arg236_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_80: "f32[72]" = torch.ops.prims.convert_element_type.default(arg416_1, torch.float32);  arg416_1 = None
    convert_element_type_81: "f32[72]" = torch.ops.prims.convert_element_type.default(arg417_1, torch.float32);  arg417_1 = None
    add_127: "f32[72]" = torch.ops.aten.add.Tensor(convert_element_type_81, 1e-05);  convert_element_type_81 = None
    sqrt_40: "f32[72]" = torch.ops.aten.sqrt.default(add_127);  add_127 = None
    reciprocal_40: "f32[72]" = torch.ops.aten.reciprocal.default(sqrt_40);  sqrt_40 = None
    mul_157: "f32[72]" = torch.ops.aten.mul.Tensor(reciprocal_40, 1);  reciprocal_40 = None
    unsqueeze_320: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_80, -1);  convert_element_type_80 = None
    unsqueeze_321: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, -1);  unsqueeze_320 = None
    unsqueeze_322: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(mul_157, -1);  mul_157 = None
    unsqueeze_323: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, -1);  unsqueeze_322 = None
    sub_40: "f32[8, 72, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_321);  convolution_50 = unsqueeze_321 = None
    mul_158: "f32[8, 72, 16, 16]" = torch.ops.aten.mul.Tensor(sub_40, unsqueeze_323);  sub_40 = unsqueeze_323 = None
    unsqueeze_324: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(arg80_1, -1);  arg80_1 = None
    unsqueeze_325: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, -1);  unsqueeze_324 = None
    mul_159: "f32[8, 72, 16, 16]" = torch.ops.aten.mul.Tensor(mul_158, unsqueeze_325);  mul_158 = unsqueeze_325 = None
    unsqueeze_326: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(arg81_1, -1);  arg81_1 = None
    unsqueeze_327: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, -1);  unsqueeze_326 = None
    add_128: "f32[8, 72, 16, 16]" = torch.ops.aten.add.Tensor(mul_159, unsqueeze_327);  mul_159 = unsqueeze_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_129: "f32[8, 72, 16, 16]" = torch.ops.aten.add.Tensor(add_128, add_120);  add_128 = add_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_51: "f32[8, 216, 16, 16]" = torch.ops.aten.convolution.default(add_129, arg237_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg237_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_82: "f32[216]" = torch.ops.prims.convert_element_type.default(arg418_1, torch.float32);  arg418_1 = None
    convert_element_type_83: "f32[216]" = torch.ops.prims.convert_element_type.default(arg419_1, torch.float32);  arg419_1 = None
    add_130: "f32[216]" = torch.ops.aten.add.Tensor(convert_element_type_83, 1e-05);  convert_element_type_83 = None
    sqrt_41: "f32[216]" = torch.ops.aten.sqrt.default(add_130);  add_130 = None
    reciprocal_41: "f32[216]" = torch.ops.aten.reciprocal.default(sqrt_41);  sqrt_41 = None
    mul_160: "f32[216]" = torch.ops.aten.mul.Tensor(reciprocal_41, 1);  reciprocal_41 = None
    unsqueeze_328: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_82, -1);  convert_element_type_82 = None
    unsqueeze_329: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, -1);  unsqueeze_328 = None
    unsqueeze_330: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(mul_160, -1);  mul_160 = None
    unsqueeze_331: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, -1);  unsqueeze_330 = None
    sub_41: "f32[8, 216, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_329);  convolution_51 = unsqueeze_329 = None
    mul_161: "f32[8, 216, 16, 16]" = torch.ops.aten.mul.Tensor(sub_41, unsqueeze_331);  sub_41 = unsqueeze_331 = None
    unsqueeze_332: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(arg82_1, -1);  arg82_1 = None
    unsqueeze_333: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, -1);  unsqueeze_332 = None
    mul_162: "f32[8, 216, 16, 16]" = torch.ops.aten.mul.Tensor(mul_161, unsqueeze_333);  mul_161 = unsqueeze_333 = None
    unsqueeze_334: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(arg83_1, -1);  arg83_1 = None
    unsqueeze_335: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, -1);  unsqueeze_334 = None
    add_131: "f32[8, 216, 16, 16]" = torch.ops.aten.add.Tensor(mul_162, unsqueeze_335);  mul_162 = unsqueeze_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_132: "f32[8, 216, 16, 16]" = torch.ops.aten.add.Tensor(add_131, 3)
    clamp_min_37: "f32[8, 216, 16, 16]" = torch.ops.aten.clamp_min.default(add_132, 0);  add_132 = None
    clamp_max_37: "f32[8, 216, 16, 16]" = torch.ops.aten.clamp_max.default(clamp_min_37, 6);  clamp_min_37 = None
    mul_163: "f32[8, 216, 16, 16]" = torch.ops.aten.mul.Tensor(add_131, clamp_max_37);  add_131 = clamp_max_37 = None
    div_37: "f32[8, 216, 16, 16]" = torch.ops.aten.div.Tensor(mul_163, 6);  mul_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_52: "f32[8, 216, 16, 16]" = torch.ops.aten.convolution.default(div_37, arg238_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216);  div_37 = arg238_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_84: "f32[216]" = torch.ops.prims.convert_element_type.default(arg420_1, torch.float32);  arg420_1 = None
    convert_element_type_85: "f32[216]" = torch.ops.prims.convert_element_type.default(arg421_1, torch.float32);  arg421_1 = None
    add_133: "f32[216]" = torch.ops.aten.add.Tensor(convert_element_type_85, 1e-05);  convert_element_type_85 = None
    sqrt_42: "f32[216]" = torch.ops.aten.sqrt.default(add_133);  add_133 = None
    reciprocal_42: "f32[216]" = torch.ops.aten.reciprocal.default(sqrt_42);  sqrt_42 = None
    mul_164: "f32[216]" = torch.ops.aten.mul.Tensor(reciprocal_42, 1);  reciprocal_42 = None
    unsqueeze_336: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_84, -1);  convert_element_type_84 = None
    unsqueeze_337: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, -1);  unsqueeze_336 = None
    unsqueeze_338: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(mul_164, -1);  mul_164 = None
    unsqueeze_339: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, -1);  unsqueeze_338 = None
    sub_42: "f32[8, 216, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_337);  convolution_52 = unsqueeze_337 = None
    mul_165: "f32[8, 216, 16, 16]" = torch.ops.aten.mul.Tensor(sub_42, unsqueeze_339);  sub_42 = unsqueeze_339 = None
    unsqueeze_340: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(arg84_1, -1);  arg84_1 = None
    unsqueeze_341: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, -1);  unsqueeze_340 = None
    mul_166: "f32[8, 216, 16, 16]" = torch.ops.aten.mul.Tensor(mul_165, unsqueeze_341);  mul_165 = unsqueeze_341 = None
    unsqueeze_342: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(arg85_1, -1);  arg85_1 = None
    unsqueeze_343: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, -1);  unsqueeze_342 = None
    add_134: "f32[8, 216, 16, 16]" = torch.ops.aten.add.Tensor(mul_166, unsqueeze_343);  mul_166 = unsqueeze_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_135: "f32[8, 216, 16, 16]" = torch.ops.aten.add.Tensor(add_134, 3)
    clamp_min_38: "f32[8, 216, 16, 16]" = torch.ops.aten.clamp_min.default(add_135, 0);  add_135 = None
    clamp_max_38: "f32[8, 216, 16, 16]" = torch.ops.aten.clamp_max.default(clamp_min_38, 6);  clamp_min_38 = None
    mul_167: "f32[8, 216, 16, 16]" = torch.ops.aten.mul.Tensor(add_134, clamp_max_38);  add_134 = clamp_max_38 = None
    div_38: "f32[8, 216, 16, 16]" = torch.ops.aten.div.Tensor(mul_167, 6);  mul_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_53: "f32[8, 72, 16, 16]" = torch.ops.aten.convolution.default(div_38, arg239_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_38 = arg239_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_86: "f32[72]" = torch.ops.prims.convert_element_type.default(arg422_1, torch.float32);  arg422_1 = None
    convert_element_type_87: "f32[72]" = torch.ops.prims.convert_element_type.default(arg423_1, torch.float32);  arg423_1 = None
    add_136: "f32[72]" = torch.ops.aten.add.Tensor(convert_element_type_87, 1e-05);  convert_element_type_87 = None
    sqrt_43: "f32[72]" = torch.ops.aten.sqrt.default(add_136);  add_136 = None
    reciprocal_43: "f32[72]" = torch.ops.aten.reciprocal.default(sqrt_43);  sqrt_43 = None
    mul_168: "f32[72]" = torch.ops.aten.mul.Tensor(reciprocal_43, 1);  reciprocal_43 = None
    unsqueeze_344: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_86, -1);  convert_element_type_86 = None
    unsqueeze_345: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, -1);  unsqueeze_344 = None
    unsqueeze_346: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(mul_168, -1);  mul_168 = None
    unsqueeze_347: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, -1);  unsqueeze_346 = None
    sub_43: "f32[8, 72, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_345);  convolution_53 = unsqueeze_345 = None
    mul_169: "f32[8, 72, 16, 16]" = torch.ops.aten.mul.Tensor(sub_43, unsqueeze_347);  sub_43 = unsqueeze_347 = None
    unsqueeze_348: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(arg86_1, -1);  arg86_1 = None
    unsqueeze_349: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, -1);  unsqueeze_348 = None
    mul_170: "f32[8, 72, 16, 16]" = torch.ops.aten.mul.Tensor(mul_169, unsqueeze_349);  mul_169 = unsqueeze_349 = None
    unsqueeze_350: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(arg87_1, -1);  arg87_1 = None
    unsqueeze_351: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, -1);  unsqueeze_350 = None
    add_137: "f32[8, 72, 16, 16]" = torch.ops.aten.add.Tensor(mul_170, unsqueeze_351);  mul_170 = unsqueeze_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_138: "f32[8, 72, 16, 16]" = torch.ops.aten.add.Tensor(add_137, add_129);  add_137 = add_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_54: "f32[8, 216, 16, 16]" = torch.ops.aten.convolution.default(add_138, arg240_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg240_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_88: "f32[216]" = torch.ops.prims.convert_element_type.default(arg424_1, torch.float32);  arg424_1 = None
    convert_element_type_89: "f32[216]" = torch.ops.prims.convert_element_type.default(arg425_1, torch.float32);  arg425_1 = None
    add_139: "f32[216]" = torch.ops.aten.add.Tensor(convert_element_type_89, 1e-05);  convert_element_type_89 = None
    sqrt_44: "f32[216]" = torch.ops.aten.sqrt.default(add_139);  add_139 = None
    reciprocal_44: "f32[216]" = torch.ops.aten.reciprocal.default(sqrt_44);  sqrt_44 = None
    mul_171: "f32[216]" = torch.ops.aten.mul.Tensor(reciprocal_44, 1);  reciprocal_44 = None
    unsqueeze_352: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_88, -1);  convert_element_type_88 = None
    unsqueeze_353: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, -1);  unsqueeze_352 = None
    unsqueeze_354: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(mul_171, -1);  mul_171 = None
    unsqueeze_355: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, -1);  unsqueeze_354 = None
    sub_44: "f32[8, 216, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_353);  convolution_54 = unsqueeze_353 = None
    mul_172: "f32[8, 216, 16, 16]" = torch.ops.aten.mul.Tensor(sub_44, unsqueeze_355);  sub_44 = unsqueeze_355 = None
    unsqueeze_356: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(arg88_1, -1);  arg88_1 = None
    unsqueeze_357: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, -1);  unsqueeze_356 = None
    mul_173: "f32[8, 216, 16, 16]" = torch.ops.aten.mul.Tensor(mul_172, unsqueeze_357);  mul_172 = unsqueeze_357 = None
    unsqueeze_358: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(arg89_1, -1);  arg89_1 = None
    unsqueeze_359: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, -1);  unsqueeze_358 = None
    add_140: "f32[8, 216, 16, 16]" = torch.ops.aten.add.Tensor(mul_173, unsqueeze_359);  mul_173 = unsqueeze_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_141: "f32[8, 216, 16, 16]" = torch.ops.aten.add.Tensor(add_140, 3)
    clamp_min_39: "f32[8, 216, 16, 16]" = torch.ops.aten.clamp_min.default(add_141, 0);  add_141 = None
    clamp_max_39: "f32[8, 216, 16, 16]" = torch.ops.aten.clamp_max.default(clamp_min_39, 6);  clamp_min_39 = None
    mul_174: "f32[8, 216, 16, 16]" = torch.ops.aten.mul.Tensor(add_140, clamp_max_39);  add_140 = clamp_max_39 = None
    div_39: "f32[8, 216, 16, 16]" = torch.ops.aten.div.Tensor(mul_174, 6);  mul_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_55: "f32[8, 216, 16, 16]" = torch.ops.aten.convolution.default(div_39, arg241_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216);  div_39 = arg241_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_90: "f32[216]" = torch.ops.prims.convert_element_type.default(arg426_1, torch.float32);  arg426_1 = None
    convert_element_type_91: "f32[216]" = torch.ops.prims.convert_element_type.default(arg427_1, torch.float32);  arg427_1 = None
    add_142: "f32[216]" = torch.ops.aten.add.Tensor(convert_element_type_91, 1e-05);  convert_element_type_91 = None
    sqrt_45: "f32[216]" = torch.ops.aten.sqrt.default(add_142);  add_142 = None
    reciprocal_45: "f32[216]" = torch.ops.aten.reciprocal.default(sqrt_45);  sqrt_45 = None
    mul_175: "f32[216]" = torch.ops.aten.mul.Tensor(reciprocal_45, 1);  reciprocal_45 = None
    unsqueeze_360: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_90, -1);  convert_element_type_90 = None
    unsqueeze_361: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, -1);  unsqueeze_360 = None
    unsqueeze_362: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(mul_175, -1);  mul_175 = None
    unsqueeze_363: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, -1);  unsqueeze_362 = None
    sub_45: "f32[8, 216, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_361);  convolution_55 = unsqueeze_361 = None
    mul_176: "f32[8, 216, 16, 16]" = torch.ops.aten.mul.Tensor(sub_45, unsqueeze_363);  sub_45 = unsqueeze_363 = None
    unsqueeze_364: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(arg90_1, -1);  arg90_1 = None
    unsqueeze_365: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, -1);  unsqueeze_364 = None
    mul_177: "f32[8, 216, 16, 16]" = torch.ops.aten.mul.Tensor(mul_176, unsqueeze_365);  mul_176 = unsqueeze_365 = None
    unsqueeze_366: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(arg91_1, -1);  arg91_1 = None
    unsqueeze_367: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, -1);  unsqueeze_366 = None
    add_143: "f32[8, 216, 16, 16]" = torch.ops.aten.add.Tensor(mul_177, unsqueeze_367);  mul_177 = unsqueeze_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_144: "f32[8, 216, 16, 16]" = torch.ops.aten.add.Tensor(add_143, 3)
    clamp_min_40: "f32[8, 216, 16, 16]" = torch.ops.aten.clamp_min.default(add_144, 0);  add_144 = None
    clamp_max_40: "f32[8, 216, 16, 16]" = torch.ops.aten.clamp_max.default(clamp_min_40, 6);  clamp_min_40 = None
    mul_178: "f32[8, 216, 16, 16]" = torch.ops.aten.mul.Tensor(add_143, clamp_max_40);  add_143 = clamp_max_40 = None
    div_40: "f32[8, 216, 16, 16]" = torch.ops.aten.div.Tensor(mul_178, 6);  mul_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_56: "f32[8, 72, 16, 16]" = torch.ops.aten.convolution.default(div_40, arg242_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_40 = arg242_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_92: "f32[72]" = torch.ops.prims.convert_element_type.default(arg428_1, torch.float32);  arg428_1 = None
    convert_element_type_93: "f32[72]" = torch.ops.prims.convert_element_type.default(arg429_1, torch.float32);  arg429_1 = None
    add_145: "f32[72]" = torch.ops.aten.add.Tensor(convert_element_type_93, 1e-05);  convert_element_type_93 = None
    sqrt_46: "f32[72]" = torch.ops.aten.sqrt.default(add_145);  add_145 = None
    reciprocal_46: "f32[72]" = torch.ops.aten.reciprocal.default(sqrt_46);  sqrt_46 = None
    mul_179: "f32[72]" = torch.ops.aten.mul.Tensor(reciprocal_46, 1);  reciprocal_46 = None
    unsqueeze_368: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_92, -1);  convert_element_type_92 = None
    unsqueeze_369: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, -1);  unsqueeze_368 = None
    unsqueeze_370: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(mul_179, -1);  mul_179 = None
    unsqueeze_371: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, -1);  unsqueeze_370 = None
    sub_46: "f32[8, 72, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_369);  convolution_56 = unsqueeze_369 = None
    mul_180: "f32[8, 72, 16, 16]" = torch.ops.aten.mul.Tensor(sub_46, unsqueeze_371);  sub_46 = unsqueeze_371 = None
    unsqueeze_372: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(arg92_1, -1);  arg92_1 = None
    unsqueeze_373: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_372, -1);  unsqueeze_372 = None
    mul_181: "f32[8, 72, 16, 16]" = torch.ops.aten.mul.Tensor(mul_180, unsqueeze_373);  mul_180 = unsqueeze_373 = None
    unsqueeze_374: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(arg93_1, -1);  arg93_1 = None
    unsqueeze_375: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, -1);  unsqueeze_374 = None
    add_146: "f32[8, 72, 16, 16]" = torch.ops.aten.add.Tensor(mul_181, unsqueeze_375);  mul_181 = unsqueeze_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_147: "f32[8, 72, 16, 16]" = torch.ops.aten.add.Tensor(add_146, add_138);  add_146 = add_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_57: "f32[8, 360, 16, 16]" = torch.ops.aten.convolution.default(add_147, arg243_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_147 = arg243_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_94: "f32[360]" = torch.ops.prims.convert_element_type.default(arg430_1, torch.float32);  arg430_1 = None
    convert_element_type_95: "f32[360]" = torch.ops.prims.convert_element_type.default(arg431_1, torch.float32);  arg431_1 = None
    add_148: "f32[360]" = torch.ops.aten.add.Tensor(convert_element_type_95, 1e-05);  convert_element_type_95 = None
    sqrt_47: "f32[360]" = torch.ops.aten.sqrt.default(add_148);  add_148 = None
    reciprocal_47: "f32[360]" = torch.ops.aten.reciprocal.default(sqrt_47);  sqrt_47 = None
    mul_182: "f32[360]" = torch.ops.aten.mul.Tensor(reciprocal_47, 1);  reciprocal_47 = None
    unsqueeze_376: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_94, -1);  convert_element_type_94 = None
    unsqueeze_377: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, -1);  unsqueeze_376 = None
    unsqueeze_378: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(mul_182, -1);  mul_182 = None
    unsqueeze_379: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, -1);  unsqueeze_378 = None
    sub_47: "f32[8, 360, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_57, unsqueeze_377);  convolution_57 = unsqueeze_377 = None
    mul_183: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(sub_47, unsqueeze_379);  sub_47 = unsqueeze_379 = None
    unsqueeze_380: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(arg94_1, -1);  arg94_1 = None
    unsqueeze_381: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, -1);  unsqueeze_380 = None
    mul_184: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(mul_183, unsqueeze_381);  mul_183 = unsqueeze_381 = None
    unsqueeze_382: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(arg95_1, -1);  arg95_1 = None
    unsqueeze_383: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, -1);  unsqueeze_382 = None
    add_149: "f32[8, 360, 16, 16]" = torch.ops.aten.add.Tensor(mul_184, unsqueeze_383);  mul_184 = unsqueeze_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_150: "f32[8, 360, 16, 16]" = torch.ops.aten.add.Tensor(add_149, 3)
    clamp_min_41: "f32[8, 360, 16, 16]" = torch.ops.aten.clamp_min.default(add_150, 0);  add_150 = None
    clamp_max_41: "f32[8, 360, 16, 16]" = torch.ops.aten.clamp_max.default(clamp_min_41, 6);  clamp_min_41 = None
    mul_185: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(add_149, clamp_max_41);  add_149 = clamp_max_41 = None
    div_41: "f32[8, 360, 16, 16]" = torch.ops.aten.div.Tensor(mul_185, 6);  mul_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_58: "f32[8, 360, 16, 16]" = torch.ops.aten.convolution.default(div_41, arg244_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 360);  div_41 = arg244_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_96: "f32[360]" = torch.ops.prims.convert_element_type.default(arg432_1, torch.float32);  arg432_1 = None
    convert_element_type_97: "f32[360]" = torch.ops.prims.convert_element_type.default(arg433_1, torch.float32);  arg433_1 = None
    add_151: "f32[360]" = torch.ops.aten.add.Tensor(convert_element_type_97, 1e-05);  convert_element_type_97 = None
    sqrt_48: "f32[360]" = torch.ops.aten.sqrt.default(add_151);  add_151 = None
    reciprocal_48: "f32[360]" = torch.ops.aten.reciprocal.default(sqrt_48);  sqrt_48 = None
    mul_186: "f32[360]" = torch.ops.aten.mul.Tensor(reciprocal_48, 1);  reciprocal_48 = None
    unsqueeze_384: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_96, -1);  convert_element_type_96 = None
    unsqueeze_385: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_384, -1);  unsqueeze_384 = None
    unsqueeze_386: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(mul_186, -1);  mul_186 = None
    unsqueeze_387: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, -1);  unsqueeze_386 = None
    sub_48: "f32[8, 360, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_58, unsqueeze_385);  convolution_58 = unsqueeze_385 = None
    mul_187: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(sub_48, unsqueeze_387);  sub_48 = unsqueeze_387 = None
    unsqueeze_388: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(arg96_1, -1);  arg96_1 = None
    unsqueeze_389: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, -1);  unsqueeze_388 = None
    mul_188: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(mul_187, unsqueeze_389);  mul_187 = unsqueeze_389 = None
    unsqueeze_390: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(arg97_1, -1);  arg97_1 = None
    unsqueeze_391: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_390, -1);  unsqueeze_390 = None
    add_152: "f32[8, 360, 16, 16]" = torch.ops.aten.add.Tensor(mul_188, unsqueeze_391);  mul_188 = unsqueeze_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_153: "f32[8, 360, 16, 16]" = torch.ops.aten.add.Tensor(add_152, 3)
    clamp_min_42: "f32[8, 360, 16, 16]" = torch.ops.aten.clamp_min.default(add_153, 0);  add_153 = None
    clamp_max_42: "f32[8, 360, 16, 16]" = torch.ops.aten.clamp_max.default(clamp_min_42, 6);  clamp_min_42 = None
    mul_189: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(add_152, clamp_max_42);  add_152 = clamp_max_42 = None
    div_42: "f32[8, 360, 16, 16]" = torch.ops.aten.div.Tensor(mul_189, 6);  mul_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_5: "f32[8, 360, 1, 1]" = torch.ops.aten.mean.dim(div_42, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_59: "f32[8, 24, 1, 1]" = torch.ops.aten.convolution.default(mean_5, arg245_1, arg246_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_5 = arg245_1 = arg246_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    add_154: "f32[8, 24, 1, 1]" = torch.ops.aten.add.Tensor(convolution_59, 3)
    clamp_min_43: "f32[8, 24, 1, 1]" = torch.ops.aten.clamp_min.default(add_154, 0);  add_154 = None
    clamp_max_43: "f32[8, 24, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_43, 6);  clamp_min_43 = None
    mul_190: "f32[8, 24, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_59, clamp_max_43);  convolution_59 = clamp_max_43 = None
    div_43: "f32[8, 24, 1, 1]" = torch.ops.aten.div.Tensor(mul_190, 6);  mul_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_60: "f32[8, 360, 1, 1]" = torch.ops.aten.convolution.default(div_43, arg247_1, arg248_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_43 = arg247_1 = arg248_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    add_155: "f32[8, 360, 1, 1]" = torch.ops.aten.add.Tensor(convolution_60, 3);  convolution_60 = None
    clamp_min_44: "f32[8, 360, 1, 1]" = torch.ops.aten.clamp_min.default(add_155, 0);  add_155 = None
    clamp_max_44: "f32[8, 360, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_44, 6);  clamp_min_44 = None
    div_44: "f32[8, 360, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_44, 6);  clamp_max_44 = None
    mul_191: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(div_42, div_44);  div_42 = div_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_61: "f32[8, 120, 16, 16]" = torch.ops.aten.convolution.default(mul_191, arg249_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_191 = arg249_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_98: "f32[120]" = torch.ops.prims.convert_element_type.default(arg434_1, torch.float32);  arg434_1 = None
    convert_element_type_99: "f32[120]" = torch.ops.prims.convert_element_type.default(arg435_1, torch.float32);  arg435_1 = None
    add_156: "f32[120]" = torch.ops.aten.add.Tensor(convert_element_type_99, 1e-05);  convert_element_type_99 = None
    sqrt_49: "f32[120]" = torch.ops.aten.sqrt.default(add_156);  add_156 = None
    reciprocal_49: "f32[120]" = torch.ops.aten.reciprocal.default(sqrt_49);  sqrt_49 = None
    mul_192: "f32[120]" = torch.ops.aten.mul.Tensor(reciprocal_49, 1);  reciprocal_49 = None
    unsqueeze_392: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_98, -1);  convert_element_type_98 = None
    unsqueeze_393: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, -1);  unsqueeze_392 = None
    unsqueeze_394: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(mul_192, -1);  mul_192 = None
    unsqueeze_395: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, -1);  unsqueeze_394 = None
    sub_49: "f32[8, 120, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_393);  convolution_61 = unsqueeze_393 = None
    mul_193: "f32[8, 120, 16, 16]" = torch.ops.aten.mul.Tensor(sub_49, unsqueeze_395);  sub_49 = unsqueeze_395 = None
    unsqueeze_396: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg98_1, -1);  arg98_1 = None
    unsqueeze_397: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_396, -1);  unsqueeze_396 = None
    mul_194: "f32[8, 120, 16, 16]" = torch.ops.aten.mul.Tensor(mul_193, unsqueeze_397);  mul_193 = unsqueeze_397 = None
    unsqueeze_398: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg99_1, -1);  arg99_1 = None
    unsqueeze_399: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, -1);  unsqueeze_398 = None
    add_157: "f32[8, 120, 16, 16]" = torch.ops.aten.add.Tensor(mul_194, unsqueeze_399);  mul_194 = unsqueeze_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_62: "f32[8, 360, 16, 16]" = torch.ops.aten.convolution.default(add_157, arg250_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg250_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_100: "f32[360]" = torch.ops.prims.convert_element_type.default(arg436_1, torch.float32);  arg436_1 = None
    convert_element_type_101: "f32[360]" = torch.ops.prims.convert_element_type.default(arg437_1, torch.float32);  arg437_1 = None
    add_158: "f32[360]" = torch.ops.aten.add.Tensor(convert_element_type_101, 1e-05);  convert_element_type_101 = None
    sqrt_50: "f32[360]" = torch.ops.aten.sqrt.default(add_158);  add_158 = None
    reciprocal_50: "f32[360]" = torch.ops.aten.reciprocal.default(sqrt_50);  sqrt_50 = None
    mul_195: "f32[360]" = torch.ops.aten.mul.Tensor(reciprocal_50, 1);  reciprocal_50 = None
    unsqueeze_400: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_100, -1);  convert_element_type_100 = None
    unsqueeze_401: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, -1);  unsqueeze_400 = None
    unsqueeze_402: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(mul_195, -1);  mul_195 = None
    unsqueeze_403: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, -1);  unsqueeze_402 = None
    sub_50: "f32[8, 360, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_62, unsqueeze_401);  convolution_62 = unsqueeze_401 = None
    mul_196: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(sub_50, unsqueeze_403);  sub_50 = unsqueeze_403 = None
    unsqueeze_404: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(arg100_1, -1);  arg100_1 = None
    unsqueeze_405: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, -1);  unsqueeze_404 = None
    mul_197: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(mul_196, unsqueeze_405);  mul_196 = unsqueeze_405 = None
    unsqueeze_406: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(arg101_1, -1);  arg101_1 = None
    unsqueeze_407: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, -1);  unsqueeze_406 = None
    add_159: "f32[8, 360, 16, 16]" = torch.ops.aten.add.Tensor(mul_197, unsqueeze_407);  mul_197 = unsqueeze_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_160: "f32[8, 360, 16, 16]" = torch.ops.aten.add.Tensor(add_159, 3)
    clamp_min_45: "f32[8, 360, 16, 16]" = torch.ops.aten.clamp_min.default(add_160, 0);  add_160 = None
    clamp_max_45: "f32[8, 360, 16, 16]" = torch.ops.aten.clamp_max.default(clamp_min_45, 6);  clamp_min_45 = None
    mul_198: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(add_159, clamp_max_45);  add_159 = clamp_max_45 = None
    div_45: "f32[8, 360, 16, 16]" = torch.ops.aten.div.Tensor(mul_198, 6);  mul_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_63: "f32[8, 360, 16, 16]" = torch.ops.aten.convolution.default(div_45, arg251_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 360);  div_45 = arg251_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_102: "f32[360]" = torch.ops.prims.convert_element_type.default(arg438_1, torch.float32);  arg438_1 = None
    convert_element_type_103: "f32[360]" = torch.ops.prims.convert_element_type.default(arg439_1, torch.float32);  arg439_1 = None
    add_161: "f32[360]" = torch.ops.aten.add.Tensor(convert_element_type_103, 1e-05);  convert_element_type_103 = None
    sqrt_51: "f32[360]" = torch.ops.aten.sqrt.default(add_161);  add_161 = None
    reciprocal_51: "f32[360]" = torch.ops.aten.reciprocal.default(sqrt_51);  sqrt_51 = None
    mul_199: "f32[360]" = torch.ops.aten.mul.Tensor(reciprocal_51, 1);  reciprocal_51 = None
    unsqueeze_408: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_102, -1);  convert_element_type_102 = None
    unsqueeze_409: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_408, -1);  unsqueeze_408 = None
    unsqueeze_410: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(mul_199, -1);  mul_199 = None
    unsqueeze_411: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, -1);  unsqueeze_410 = None
    sub_51: "f32[8, 360, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_63, unsqueeze_409);  convolution_63 = unsqueeze_409 = None
    mul_200: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(sub_51, unsqueeze_411);  sub_51 = unsqueeze_411 = None
    unsqueeze_412: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(arg102_1, -1);  arg102_1 = None
    unsqueeze_413: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, -1);  unsqueeze_412 = None
    mul_201: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(mul_200, unsqueeze_413);  mul_200 = unsqueeze_413 = None
    unsqueeze_414: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(arg103_1, -1);  arg103_1 = None
    unsqueeze_415: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, -1);  unsqueeze_414 = None
    add_162: "f32[8, 360, 16, 16]" = torch.ops.aten.add.Tensor(mul_201, unsqueeze_415);  mul_201 = unsqueeze_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_163: "f32[8, 360, 16, 16]" = torch.ops.aten.add.Tensor(add_162, 3)
    clamp_min_46: "f32[8, 360, 16, 16]" = torch.ops.aten.clamp_min.default(add_163, 0);  add_163 = None
    clamp_max_46: "f32[8, 360, 16, 16]" = torch.ops.aten.clamp_max.default(clamp_min_46, 6);  clamp_min_46 = None
    mul_202: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(add_162, clamp_max_46);  add_162 = clamp_max_46 = None
    div_46: "f32[8, 360, 16, 16]" = torch.ops.aten.div.Tensor(mul_202, 6);  mul_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_6: "f32[8, 360, 1, 1]" = torch.ops.aten.mean.dim(div_46, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_64: "f32[8, 32, 1, 1]" = torch.ops.aten.convolution.default(mean_6, arg252_1, arg253_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_6 = arg252_1 = arg253_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    add_164: "f32[8, 32, 1, 1]" = torch.ops.aten.add.Tensor(convolution_64, 3)
    clamp_min_47: "f32[8, 32, 1, 1]" = torch.ops.aten.clamp_min.default(add_164, 0);  add_164 = None
    clamp_max_47: "f32[8, 32, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_47, 6);  clamp_min_47 = None
    mul_203: "f32[8, 32, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_64, clamp_max_47);  convolution_64 = clamp_max_47 = None
    div_47: "f32[8, 32, 1, 1]" = torch.ops.aten.div.Tensor(mul_203, 6);  mul_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_65: "f32[8, 360, 1, 1]" = torch.ops.aten.convolution.default(div_47, arg254_1, arg255_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_47 = arg254_1 = arg255_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    add_165: "f32[8, 360, 1, 1]" = torch.ops.aten.add.Tensor(convolution_65, 3);  convolution_65 = None
    clamp_min_48: "f32[8, 360, 1, 1]" = torch.ops.aten.clamp_min.default(add_165, 0);  add_165 = None
    clamp_max_48: "f32[8, 360, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_48, 6);  clamp_min_48 = None
    div_48: "f32[8, 360, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_48, 6);  clamp_max_48 = None
    mul_204: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(div_46, div_48);  div_46 = div_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_66: "f32[8, 120, 16, 16]" = torch.ops.aten.convolution.default(mul_204, arg256_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_204 = arg256_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_104: "f32[120]" = torch.ops.prims.convert_element_type.default(arg440_1, torch.float32);  arg440_1 = None
    convert_element_type_105: "f32[120]" = torch.ops.prims.convert_element_type.default(arg441_1, torch.float32);  arg441_1 = None
    add_166: "f32[120]" = torch.ops.aten.add.Tensor(convert_element_type_105, 1e-05);  convert_element_type_105 = None
    sqrt_52: "f32[120]" = torch.ops.aten.sqrt.default(add_166);  add_166 = None
    reciprocal_52: "f32[120]" = torch.ops.aten.reciprocal.default(sqrt_52);  sqrt_52 = None
    mul_205: "f32[120]" = torch.ops.aten.mul.Tensor(reciprocal_52, 1);  reciprocal_52 = None
    unsqueeze_416: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_104, -1);  convert_element_type_104 = None
    unsqueeze_417: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, -1);  unsqueeze_416 = None
    unsqueeze_418: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(mul_205, -1);  mul_205 = None
    unsqueeze_419: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, -1);  unsqueeze_418 = None
    sub_52: "f32[8, 120, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_66, unsqueeze_417);  convolution_66 = unsqueeze_417 = None
    mul_206: "f32[8, 120, 16, 16]" = torch.ops.aten.mul.Tensor(sub_52, unsqueeze_419);  sub_52 = unsqueeze_419 = None
    unsqueeze_420: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg104_1, -1);  arg104_1 = None
    unsqueeze_421: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_420, -1);  unsqueeze_420 = None
    mul_207: "f32[8, 120, 16, 16]" = torch.ops.aten.mul.Tensor(mul_206, unsqueeze_421);  mul_206 = unsqueeze_421 = None
    unsqueeze_422: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg105_1, -1);  arg105_1 = None
    unsqueeze_423: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, -1);  unsqueeze_422 = None
    add_167: "f32[8, 120, 16, 16]" = torch.ops.aten.add.Tensor(mul_207, unsqueeze_423);  mul_207 = unsqueeze_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_168: "f32[8, 120, 16, 16]" = torch.ops.aten.add.Tensor(add_167, add_157);  add_167 = add_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_67: "f32[8, 360, 16, 16]" = torch.ops.aten.convolution.default(add_168, arg257_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg257_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_106: "f32[360]" = torch.ops.prims.convert_element_type.default(arg442_1, torch.float32);  arg442_1 = None
    convert_element_type_107: "f32[360]" = torch.ops.prims.convert_element_type.default(arg443_1, torch.float32);  arg443_1 = None
    add_169: "f32[360]" = torch.ops.aten.add.Tensor(convert_element_type_107, 1e-05);  convert_element_type_107 = None
    sqrt_53: "f32[360]" = torch.ops.aten.sqrt.default(add_169);  add_169 = None
    reciprocal_53: "f32[360]" = torch.ops.aten.reciprocal.default(sqrt_53);  sqrt_53 = None
    mul_208: "f32[360]" = torch.ops.aten.mul.Tensor(reciprocal_53, 1);  reciprocal_53 = None
    unsqueeze_424: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_106, -1);  convert_element_type_106 = None
    unsqueeze_425: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_424, -1);  unsqueeze_424 = None
    unsqueeze_426: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(mul_208, -1);  mul_208 = None
    unsqueeze_427: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, -1);  unsqueeze_426 = None
    sub_53: "f32[8, 360, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_67, unsqueeze_425);  convolution_67 = unsqueeze_425 = None
    mul_209: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(sub_53, unsqueeze_427);  sub_53 = unsqueeze_427 = None
    unsqueeze_428: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(arg106_1, -1);  arg106_1 = None
    unsqueeze_429: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, -1);  unsqueeze_428 = None
    mul_210: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(mul_209, unsqueeze_429);  mul_209 = unsqueeze_429 = None
    unsqueeze_430: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(arg107_1, -1);  arg107_1 = None
    unsqueeze_431: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, -1);  unsqueeze_430 = None
    add_170: "f32[8, 360, 16, 16]" = torch.ops.aten.add.Tensor(mul_210, unsqueeze_431);  mul_210 = unsqueeze_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_171: "f32[8, 360, 16, 16]" = torch.ops.aten.add.Tensor(add_170, 3)
    clamp_min_49: "f32[8, 360, 16, 16]" = torch.ops.aten.clamp_min.default(add_171, 0);  add_171 = None
    clamp_max_49: "f32[8, 360, 16, 16]" = torch.ops.aten.clamp_max.default(clamp_min_49, 6);  clamp_min_49 = None
    mul_211: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(add_170, clamp_max_49);  add_170 = clamp_max_49 = None
    div_49: "f32[8, 360, 16, 16]" = torch.ops.aten.div.Tensor(mul_211, 6);  mul_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_68: "f32[8, 360, 16, 16]" = torch.ops.aten.convolution.default(div_49, arg258_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 360);  div_49 = arg258_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_108: "f32[360]" = torch.ops.prims.convert_element_type.default(arg444_1, torch.float32);  arg444_1 = None
    convert_element_type_109: "f32[360]" = torch.ops.prims.convert_element_type.default(arg445_1, torch.float32);  arg445_1 = None
    add_172: "f32[360]" = torch.ops.aten.add.Tensor(convert_element_type_109, 1e-05);  convert_element_type_109 = None
    sqrt_54: "f32[360]" = torch.ops.aten.sqrt.default(add_172);  add_172 = None
    reciprocal_54: "f32[360]" = torch.ops.aten.reciprocal.default(sqrt_54);  sqrt_54 = None
    mul_212: "f32[360]" = torch.ops.aten.mul.Tensor(reciprocal_54, 1);  reciprocal_54 = None
    unsqueeze_432: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_108, -1);  convert_element_type_108 = None
    unsqueeze_433: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_432, -1);  unsqueeze_432 = None
    unsqueeze_434: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(mul_212, -1);  mul_212 = None
    unsqueeze_435: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, -1);  unsqueeze_434 = None
    sub_54: "f32[8, 360, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_68, unsqueeze_433);  convolution_68 = unsqueeze_433 = None
    mul_213: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(sub_54, unsqueeze_435);  sub_54 = unsqueeze_435 = None
    unsqueeze_436: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(arg108_1, -1);  arg108_1 = None
    unsqueeze_437: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_436, -1);  unsqueeze_436 = None
    mul_214: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(mul_213, unsqueeze_437);  mul_213 = unsqueeze_437 = None
    unsqueeze_438: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(arg109_1, -1);  arg109_1 = None
    unsqueeze_439: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_438, -1);  unsqueeze_438 = None
    add_173: "f32[8, 360, 16, 16]" = torch.ops.aten.add.Tensor(mul_214, unsqueeze_439);  mul_214 = unsqueeze_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_174: "f32[8, 360, 16, 16]" = torch.ops.aten.add.Tensor(add_173, 3)
    clamp_min_50: "f32[8, 360, 16, 16]" = torch.ops.aten.clamp_min.default(add_174, 0);  add_174 = None
    clamp_max_50: "f32[8, 360, 16, 16]" = torch.ops.aten.clamp_max.default(clamp_min_50, 6);  clamp_min_50 = None
    mul_215: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(add_173, clamp_max_50);  add_173 = clamp_max_50 = None
    div_50: "f32[8, 360, 16, 16]" = torch.ops.aten.div.Tensor(mul_215, 6);  mul_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_7: "f32[8, 360, 1, 1]" = torch.ops.aten.mean.dim(div_50, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_69: "f32[8, 32, 1, 1]" = torch.ops.aten.convolution.default(mean_7, arg259_1, arg260_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_7 = arg259_1 = arg260_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    add_175: "f32[8, 32, 1, 1]" = torch.ops.aten.add.Tensor(convolution_69, 3)
    clamp_min_51: "f32[8, 32, 1, 1]" = torch.ops.aten.clamp_min.default(add_175, 0);  add_175 = None
    clamp_max_51: "f32[8, 32, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_51, 6);  clamp_min_51 = None
    mul_216: "f32[8, 32, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_69, clamp_max_51);  convolution_69 = clamp_max_51 = None
    div_51: "f32[8, 32, 1, 1]" = torch.ops.aten.div.Tensor(mul_216, 6);  mul_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_70: "f32[8, 360, 1, 1]" = torch.ops.aten.convolution.default(div_51, arg261_1, arg262_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_51 = arg261_1 = arg262_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    add_176: "f32[8, 360, 1, 1]" = torch.ops.aten.add.Tensor(convolution_70, 3);  convolution_70 = None
    clamp_min_52: "f32[8, 360, 1, 1]" = torch.ops.aten.clamp_min.default(add_176, 0);  add_176 = None
    clamp_max_52: "f32[8, 360, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_52, 6);  clamp_min_52 = None
    div_52: "f32[8, 360, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_52, 6);  clamp_max_52 = None
    mul_217: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(div_50, div_52);  div_50 = div_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_71: "f32[8, 120, 16, 16]" = torch.ops.aten.convolution.default(mul_217, arg263_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_217 = arg263_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_110: "f32[120]" = torch.ops.prims.convert_element_type.default(arg446_1, torch.float32);  arg446_1 = None
    convert_element_type_111: "f32[120]" = torch.ops.prims.convert_element_type.default(arg447_1, torch.float32);  arg447_1 = None
    add_177: "f32[120]" = torch.ops.aten.add.Tensor(convert_element_type_111, 1e-05);  convert_element_type_111 = None
    sqrt_55: "f32[120]" = torch.ops.aten.sqrt.default(add_177);  add_177 = None
    reciprocal_55: "f32[120]" = torch.ops.aten.reciprocal.default(sqrt_55);  sqrt_55 = None
    mul_218: "f32[120]" = torch.ops.aten.mul.Tensor(reciprocal_55, 1);  reciprocal_55 = None
    unsqueeze_440: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_110, -1);  convert_element_type_110 = None
    unsqueeze_441: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, -1);  unsqueeze_440 = None
    unsqueeze_442: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(mul_218, -1);  mul_218 = None
    unsqueeze_443: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_442, -1);  unsqueeze_442 = None
    sub_55: "f32[8, 120, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_71, unsqueeze_441);  convolution_71 = unsqueeze_441 = None
    mul_219: "f32[8, 120, 16, 16]" = torch.ops.aten.mul.Tensor(sub_55, unsqueeze_443);  sub_55 = unsqueeze_443 = None
    unsqueeze_444: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg110_1, -1);  arg110_1 = None
    unsqueeze_445: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_444, -1);  unsqueeze_444 = None
    mul_220: "f32[8, 120, 16, 16]" = torch.ops.aten.mul.Tensor(mul_219, unsqueeze_445);  mul_219 = unsqueeze_445 = None
    unsqueeze_446: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg111_1, -1);  arg111_1 = None
    unsqueeze_447: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, -1);  unsqueeze_446 = None
    add_178: "f32[8, 120, 16, 16]" = torch.ops.aten.add.Tensor(mul_220, unsqueeze_447);  mul_220 = unsqueeze_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_179: "f32[8, 120, 16, 16]" = torch.ops.aten.add.Tensor(add_178, add_168);  add_178 = add_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_72: "f32[8, 360, 16, 16]" = torch.ops.aten.convolution.default(add_179, arg264_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg264_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_112: "f32[360]" = torch.ops.prims.convert_element_type.default(arg448_1, torch.float32);  arg448_1 = None
    convert_element_type_113: "f32[360]" = torch.ops.prims.convert_element_type.default(arg449_1, torch.float32);  arg449_1 = None
    add_180: "f32[360]" = torch.ops.aten.add.Tensor(convert_element_type_113, 1e-05);  convert_element_type_113 = None
    sqrt_56: "f32[360]" = torch.ops.aten.sqrt.default(add_180);  add_180 = None
    reciprocal_56: "f32[360]" = torch.ops.aten.reciprocal.default(sqrt_56);  sqrt_56 = None
    mul_221: "f32[360]" = torch.ops.aten.mul.Tensor(reciprocal_56, 1);  reciprocal_56 = None
    unsqueeze_448: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_112, -1);  convert_element_type_112 = None
    unsqueeze_449: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_448, -1);  unsqueeze_448 = None
    unsqueeze_450: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(mul_221, -1);  mul_221 = None
    unsqueeze_451: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_450, -1);  unsqueeze_450 = None
    sub_56: "f32[8, 360, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_72, unsqueeze_449);  convolution_72 = unsqueeze_449 = None
    mul_222: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(sub_56, unsqueeze_451);  sub_56 = unsqueeze_451 = None
    unsqueeze_452: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(arg112_1, -1);  arg112_1 = None
    unsqueeze_453: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, -1);  unsqueeze_452 = None
    mul_223: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(mul_222, unsqueeze_453);  mul_222 = unsqueeze_453 = None
    unsqueeze_454: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(arg113_1, -1);  arg113_1 = None
    unsqueeze_455: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_454, -1);  unsqueeze_454 = None
    add_181: "f32[8, 360, 16, 16]" = torch.ops.aten.add.Tensor(mul_223, unsqueeze_455);  mul_223 = unsqueeze_455 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_182: "f32[8, 360, 16, 16]" = torch.ops.aten.add.Tensor(add_181, 3)
    clamp_min_53: "f32[8, 360, 16, 16]" = torch.ops.aten.clamp_min.default(add_182, 0);  add_182 = None
    clamp_max_53: "f32[8, 360, 16, 16]" = torch.ops.aten.clamp_max.default(clamp_min_53, 6);  clamp_min_53 = None
    mul_224: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(add_181, clamp_max_53);  add_181 = clamp_max_53 = None
    div_53: "f32[8, 360, 16, 16]" = torch.ops.aten.div.Tensor(mul_224, 6);  mul_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_73: "f32[8, 360, 16, 16]" = torch.ops.aten.convolution.default(div_53, arg265_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 360);  div_53 = arg265_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_114: "f32[360]" = torch.ops.prims.convert_element_type.default(arg450_1, torch.float32);  arg450_1 = None
    convert_element_type_115: "f32[360]" = torch.ops.prims.convert_element_type.default(arg451_1, torch.float32);  arg451_1 = None
    add_183: "f32[360]" = torch.ops.aten.add.Tensor(convert_element_type_115, 1e-05);  convert_element_type_115 = None
    sqrt_57: "f32[360]" = torch.ops.aten.sqrt.default(add_183);  add_183 = None
    reciprocal_57: "f32[360]" = torch.ops.aten.reciprocal.default(sqrt_57);  sqrt_57 = None
    mul_225: "f32[360]" = torch.ops.aten.mul.Tensor(reciprocal_57, 1);  reciprocal_57 = None
    unsqueeze_456: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_114, -1);  convert_element_type_114 = None
    unsqueeze_457: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_456, -1);  unsqueeze_456 = None
    unsqueeze_458: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(mul_225, -1);  mul_225 = None
    unsqueeze_459: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, -1);  unsqueeze_458 = None
    sub_57: "f32[8, 360, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_73, unsqueeze_457);  convolution_73 = unsqueeze_457 = None
    mul_226: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(sub_57, unsqueeze_459);  sub_57 = unsqueeze_459 = None
    unsqueeze_460: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(arg114_1, -1);  arg114_1 = None
    unsqueeze_461: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_460, -1);  unsqueeze_460 = None
    mul_227: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(mul_226, unsqueeze_461);  mul_226 = unsqueeze_461 = None
    unsqueeze_462: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(arg115_1, -1);  arg115_1 = None
    unsqueeze_463: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_462, -1);  unsqueeze_462 = None
    add_184: "f32[8, 360, 16, 16]" = torch.ops.aten.add.Tensor(mul_227, unsqueeze_463);  mul_227 = unsqueeze_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_185: "f32[8, 360, 16, 16]" = torch.ops.aten.add.Tensor(add_184, 3)
    clamp_min_54: "f32[8, 360, 16, 16]" = torch.ops.aten.clamp_min.default(add_185, 0);  add_185 = None
    clamp_max_54: "f32[8, 360, 16, 16]" = torch.ops.aten.clamp_max.default(clamp_min_54, 6);  clamp_min_54 = None
    mul_228: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(add_184, clamp_max_54);  add_184 = clamp_max_54 = None
    div_54: "f32[8, 360, 16, 16]" = torch.ops.aten.div.Tensor(mul_228, 6);  mul_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_8: "f32[8, 360, 1, 1]" = torch.ops.aten.mean.dim(div_54, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_74: "f32[8, 32, 1, 1]" = torch.ops.aten.convolution.default(mean_8, arg266_1, arg267_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_8 = arg266_1 = arg267_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    add_186: "f32[8, 32, 1, 1]" = torch.ops.aten.add.Tensor(convolution_74, 3)
    clamp_min_55: "f32[8, 32, 1, 1]" = torch.ops.aten.clamp_min.default(add_186, 0);  add_186 = None
    clamp_max_55: "f32[8, 32, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_55, 6);  clamp_min_55 = None
    mul_229: "f32[8, 32, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_74, clamp_max_55);  convolution_74 = clamp_max_55 = None
    div_55: "f32[8, 32, 1, 1]" = torch.ops.aten.div.Tensor(mul_229, 6);  mul_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_75: "f32[8, 360, 1, 1]" = torch.ops.aten.convolution.default(div_55, arg268_1, arg269_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_55 = arg268_1 = arg269_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    add_187: "f32[8, 360, 1, 1]" = torch.ops.aten.add.Tensor(convolution_75, 3);  convolution_75 = None
    clamp_min_56: "f32[8, 360, 1, 1]" = torch.ops.aten.clamp_min.default(add_187, 0);  add_187 = None
    clamp_max_56: "f32[8, 360, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_56, 6);  clamp_min_56 = None
    div_56: "f32[8, 360, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_56, 6);  clamp_max_56 = None
    mul_230: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(div_54, div_56);  div_54 = div_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_76: "f32[8, 120, 16, 16]" = torch.ops.aten.convolution.default(mul_230, arg270_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_230 = arg270_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_116: "f32[120]" = torch.ops.prims.convert_element_type.default(arg452_1, torch.float32);  arg452_1 = None
    convert_element_type_117: "f32[120]" = torch.ops.prims.convert_element_type.default(arg453_1, torch.float32);  arg453_1 = None
    add_188: "f32[120]" = torch.ops.aten.add.Tensor(convert_element_type_117, 1e-05);  convert_element_type_117 = None
    sqrt_58: "f32[120]" = torch.ops.aten.sqrt.default(add_188);  add_188 = None
    reciprocal_58: "f32[120]" = torch.ops.aten.reciprocal.default(sqrt_58);  sqrt_58 = None
    mul_231: "f32[120]" = torch.ops.aten.mul.Tensor(reciprocal_58, 1);  reciprocal_58 = None
    unsqueeze_464: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_116, -1);  convert_element_type_116 = None
    unsqueeze_465: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, -1);  unsqueeze_464 = None
    unsqueeze_466: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(mul_231, -1);  mul_231 = None
    unsqueeze_467: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_466, -1);  unsqueeze_466 = None
    sub_58: "f32[8, 120, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_76, unsqueeze_465);  convolution_76 = unsqueeze_465 = None
    mul_232: "f32[8, 120, 16, 16]" = torch.ops.aten.mul.Tensor(sub_58, unsqueeze_467);  sub_58 = unsqueeze_467 = None
    unsqueeze_468: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg116_1, -1);  arg116_1 = None
    unsqueeze_469: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_468, -1);  unsqueeze_468 = None
    mul_233: "f32[8, 120, 16, 16]" = torch.ops.aten.mul.Tensor(mul_232, unsqueeze_469);  mul_232 = unsqueeze_469 = None
    unsqueeze_470: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg117_1, -1);  arg117_1 = None
    unsqueeze_471: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, -1);  unsqueeze_470 = None
    add_189: "f32[8, 120, 16, 16]" = torch.ops.aten.add.Tensor(mul_233, unsqueeze_471);  mul_233 = unsqueeze_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_190: "f32[8, 120, 16, 16]" = torch.ops.aten.add.Tensor(add_189, add_179);  add_189 = add_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_77: "f32[8, 360, 16, 16]" = torch.ops.aten.convolution.default(add_190, arg271_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg271_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_118: "f32[360]" = torch.ops.prims.convert_element_type.default(arg454_1, torch.float32);  arg454_1 = None
    convert_element_type_119: "f32[360]" = torch.ops.prims.convert_element_type.default(arg455_1, torch.float32);  arg455_1 = None
    add_191: "f32[360]" = torch.ops.aten.add.Tensor(convert_element_type_119, 1e-05);  convert_element_type_119 = None
    sqrt_59: "f32[360]" = torch.ops.aten.sqrt.default(add_191);  add_191 = None
    reciprocal_59: "f32[360]" = torch.ops.aten.reciprocal.default(sqrt_59);  sqrt_59 = None
    mul_234: "f32[360]" = torch.ops.aten.mul.Tensor(reciprocal_59, 1);  reciprocal_59 = None
    unsqueeze_472: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_118, -1);  convert_element_type_118 = None
    unsqueeze_473: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_472, -1);  unsqueeze_472 = None
    unsqueeze_474: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(mul_234, -1);  mul_234 = None
    unsqueeze_475: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_474, -1);  unsqueeze_474 = None
    sub_59: "f32[8, 360, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_77, unsqueeze_473);  convolution_77 = unsqueeze_473 = None
    mul_235: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(sub_59, unsqueeze_475);  sub_59 = unsqueeze_475 = None
    unsqueeze_476: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(arg118_1, -1);  arg118_1 = None
    unsqueeze_477: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, -1);  unsqueeze_476 = None
    mul_236: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(mul_235, unsqueeze_477);  mul_235 = unsqueeze_477 = None
    unsqueeze_478: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(arg119_1, -1);  arg119_1 = None
    unsqueeze_479: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_478, -1);  unsqueeze_478 = None
    add_192: "f32[8, 360, 16, 16]" = torch.ops.aten.add.Tensor(mul_236, unsqueeze_479);  mul_236 = unsqueeze_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_193: "f32[8, 360, 16, 16]" = torch.ops.aten.add.Tensor(add_192, 3)
    clamp_min_57: "f32[8, 360, 16, 16]" = torch.ops.aten.clamp_min.default(add_193, 0);  add_193 = None
    clamp_max_57: "f32[8, 360, 16, 16]" = torch.ops.aten.clamp_max.default(clamp_min_57, 6);  clamp_min_57 = None
    mul_237: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(add_192, clamp_max_57);  add_192 = clamp_max_57 = None
    div_57: "f32[8, 360, 16, 16]" = torch.ops.aten.div.Tensor(mul_237, 6);  mul_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_78: "f32[8, 360, 16, 16]" = torch.ops.aten.convolution.default(div_57, arg272_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 360);  div_57 = arg272_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_120: "f32[360]" = torch.ops.prims.convert_element_type.default(arg456_1, torch.float32);  arg456_1 = None
    convert_element_type_121: "f32[360]" = torch.ops.prims.convert_element_type.default(arg457_1, torch.float32);  arg457_1 = None
    add_194: "f32[360]" = torch.ops.aten.add.Tensor(convert_element_type_121, 1e-05);  convert_element_type_121 = None
    sqrt_60: "f32[360]" = torch.ops.aten.sqrt.default(add_194);  add_194 = None
    reciprocal_60: "f32[360]" = torch.ops.aten.reciprocal.default(sqrt_60);  sqrt_60 = None
    mul_238: "f32[360]" = torch.ops.aten.mul.Tensor(reciprocal_60, 1);  reciprocal_60 = None
    unsqueeze_480: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_120, -1);  convert_element_type_120 = None
    unsqueeze_481: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_480, -1);  unsqueeze_480 = None
    unsqueeze_482: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(mul_238, -1);  mul_238 = None
    unsqueeze_483: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, -1);  unsqueeze_482 = None
    sub_60: "f32[8, 360, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_78, unsqueeze_481);  convolution_78 = unsqueeze_481 = None
    mul_239: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(sub_60, unsqueeze_483);  sub_60 = unsqueeze_483 = None
    unsqueeze_484: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(arg120_1, -1);  arg120_1 = None
    unsqueeze_485: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_484, -1);  unsqueeze_484 = None
    mul_240: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(mul_239, unsqueeze_485);  mul_239 = unsqueeze_485 = None
    unsqueeze_486: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(arg121_1, -1);  arg121_1 = None
    unsqueeze_487: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_486, -1);  unsqueeze_486 = None
    add_195: "f32[8, 360, 16, 16]" = torch.ops.aten.add.Tensor(mul_240, unsqueeze_487);  mul_240 = unsqueeze_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_196: "f32[8, 360, 16, 16]" = torch.ops.aten.add.Tensor(add_195, 3)
    clamp_min_58: "f32[8, 360, 16, 16]" = torch.ops.aten.clamp_min.default(add_196, 0);  add_196 = None
    clamp_max_58: "f32[8, 360, 16, 16]" = torch.ops.aten.clamp_max.default(clamp_min_58, 6);  clamp_min_58 = None
    mul_241: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(add_195, clamp_max_58);  add_195 = clamp_max_58 = None
    div_58: "f32[8, 360, 16, 16]" = torch.ops.aten.div.Tensor(mul_241, 6);  mul_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_9: "f32[8, 360, 1, 1]" = torch.ops.aten.mean.dim(div_58, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_79: "f32[8, 32, 1, 1]" = torch.ops.aten.convolution.default(mean_9, arg273_1, arg274_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_9 = arg273_1 = arg274_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    add_197: "f32[8, 32, 1, 1]" = torch.ops.aten.add.Tensor(convolution_79, 3)
    clamp_min_59: "f32[8, 32, 1, 1]" = torch.ops.aten.clamp_min.default(add_197, 0);  add_197 = None
    clamp_max_59: "f32[8, 32, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_59, 6);  clamp_min_59 = None
    mul_242: "f32[8, 32, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_79, clamp_max_59);  convolution_79 = clamp_max_59 = None
    div_59: "f32[8, 32, 1, 1]" = torch.ops.aten.div.Tensor(mul_242, 6);  mul_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_80: "f32[8, 360, 1, 1]" = torch.ops.aten.convolution.default(div_59, arg275_1, arg276_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_59 = arg275_1 = arg276_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    add_198: "f32[8, 360, 1, 1]" = torch.ops.aten.add.Tensor(convolution_80, 3);  convolution_80 = None
    clamp_min_60: "f32[8, 360, 1, 1]" = torch.ops.aten.clamp_min.default(add_198, 0);  add_198 = None
    clamp_max_60: "f32[8, 360, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_60, 6);  clamp_min_60 = None
    div_60: "f32[8, 360, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_60, 6);  clamp_max_60 = None
    mul_243: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(div_58, div_60);  div_58 = div_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_81: "f32[8, 120, 16, 16]" = torch.ops.aten.convolution.default(mul_243, arg277_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_243 = arg277_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_122: "f32[120]" = torch.ops.prims.convert_element_type.default(arg458_1, torch.float32);  arg458_1 = None
    convert_element_type_123: "f32[120]" = torch.ops.prims.convert_element_type.default(arg459_1, torch.float32);  arg459_1 = None
    add_199: "f32[120]" = torch.ops.aten.add.Tensor(convert_element_type_123, 1e-05);  convert_element_type_123 = None
    sqrt_61: "f32[120]" = torch.ops.aten.sqrt.default(add_199);  add_199 = None
    reciprocal_61: "f32[120]" = torch.ops.aten.reciprocal.default(sqrt_61);  sqrt_61 = None
    mul_244: "f32[120]" = torch.ops.aten.mul.Tensor(reciprocal_61, 1);  reciprocal_61 = None
    unsqueeze_488: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_122, -1);  convert_element_type_122 = None
    unsqueeze_489: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, -1);  unsqueeze_488 = None
    unsqueeze_490: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(mul_244, -1);  mul_244 = None
    unsqueeze_491: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_490, -1);  unsqueeze_490 = None
    sub_61: "f32[8, 120, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_81, unsqueeze_489);  convolution_81 = unsqueeze_489 = None
    mul_245: "f32[8, 120, 16, 16]" = torch.ops.aten.mul.Tensor(sub_61, unsqueeze_491);  sub_61 = unsqueeze_491 = None
    unsqueeze_492: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg122_1, -1);  arg122_1 = None
    unsqueeze_493: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_492, -1);  unsqueeze_492 = None
    mul_246: "f32[8, 120, 16, 16]" = torch.ops.aten.mul.Tensor(mul_245, unsqueeze_493);  mul_245 = unsqueeze_493 = None
    unsqueeze_494: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg123_1, -1);  arg123_1 = None
    unsqueeze_495: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, -1);  unsqueeze_494 = None
    add_200: "f32[8, 120, 16, 16]" = torch.ops.aten.add.Tensor(mul_246, unsqueeze_495);  mul_246 = unsqueeze_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_201: "f32[8, 120, 16, 16]" = torch.ops.aten.add.Tensor(add_200, add_190);  add_200 = add_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_82: "f32[8, 360, 16, 16]" = torch.ops.aten.convolution.default(add_201, arg278_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg278_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_124: "f32[360]" = torch.ops.prims.convert_element_type.default(arg460_1, torch.float32);  arg460_1 = None
    convert_element_type_125: "f32[360]" = torch.ops.prims.convert_element_type.default(arg461_1, torch.float32);  arg461_1 = None
    add_202: "f32[360]" = torch.ops.aten.add.Tensor(convert_element_type_125, 1e-05);  convert_element_type_125 = None
    sqrt_62: "f32[360]" = torch.ops.aten.sqrt.default(add_202);  add_202 = None
    reciprocal_62: "f32[360]" = torch.ops.aten.reciprocal.default(sqrt_62);  sqrt_62 = None
    mul_247: "f32[360]" = torch.ops.aten.mul.Tensor(reciprocal_62, 1);  reciprocal_62 = None
    unsqueeze_496: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_124, -1);  convert_element_type_124 = None
    unsqueeze_497: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_496, -1);  unsqueeze_496 = None
    unsqueeze_498: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(mul_247, -1);  mul_247 = None
    unsqueeze_499: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_498, -1);  unsqueeze_498 = None
    sub_62: "f32[8, 360, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_82, unsqueeze_497);  convolution_82 = unsqueeze_497 = None
    mul_248: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(sub_62, unsqueeze_499);  sub_62 = unsqueeze_499 = None
    unsqueeze_500: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(arg124_1, -1);  arg124_1 = None
    unsqueeze_501: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_500, -1);  unsqueeze_500 = None
    mul_249: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(mul_248, unsqueeze_501);  mul_248 = unsqueeze_501 = None
    unsqueeze_502: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(arg125_1, -1);  arg125_1 = None
    unsqueeze_503: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_502, -1);  unsqueeze_502 = None
    add_203: "f32[8, 360, 16, 16]" = torch.ops.aten.add.Tensor(mul_249, unsqueeze_503);  mul_249 = unsqueeze_503 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_204: "f32[8, 360, 16, 16]" = torch.ops.aten.add.Tensor(add_203, 3)
    clamp_min_61: "f32[8, 360, 16, 16]" = torch.ops.aten.clamp_min.default(add_204, 0);  add_204 = None
    clamp_max_61: "f32[8, 360, 16, 16]" = torch.ops.aten.clamp_max.default(clamp_min_61, 6);  clamp_min_61 = None
    mul_250: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(add_203, clamp_max_61);  add_203 = clamp_max_61 = None
    div_61: "f32[8, 360, 16, 16]" = torch.ops.aten.div.Tensor(mul_250, 6);  mul_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_83: "f32[8, 360, 16, 16]" = torch.ops.aten.convolution.default(div_61, arg279_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 360);  div_61 = arg279_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_126: "f32[360]" = torch.ops.prims.convert_element_type.default(arg462_1, torch.float32);  arg462_1 = None
    convert_element_type_127: "f32[360]" = torch.ops.prims.convert_element_type.default(arg463_1, torch.float32);  arg463_1 = None
    add_205: "f32[360]" = torch.ops.aten.add.Tensor(convert_element_type_127, 1e-05);  convert_element_type_127 = None
    sqrt_63: "f32[360]" = torch.ops.aten.sqrt.default(add_205);  add_205 = None
    reciprocal_63: "f32[360]" = torch.ops.aten.reciprocal.default(sqrt_63);  sqrt_63 = None
    mul_251: "f32[360]" = torch.ops.aten.mul.Tensor(reciprocal_63, 1);  reciprocal_63 = None
    unsqueeze_504: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_126, -1);  convert_element_type_126 = None
    unsqueeze_505: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_504, -1);  unsqueeze_504 = None
    unsqueeze_506: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(mul_251, -1);  mul_251 = None
    unsqueeze_507: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_506, -1);  unsqueeze_506 = None
    sub_63: "f32[8, 360, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_83, unsqueeze_505);  convolution_83 = unsqueeze_505 = None
    mul_252: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(sub_63, unsqueeze_507);  sub_63 = unsqueeze_507 = None
    unsqueeze_508: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(arg126_1, -1);  arg126_1 = None
    unsqueeze_509: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_508, -1);  unsqueeze_508 = None
    mul_253: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(mul_252, unsqueeze_509);  mul_252 = unsqueeze_509 = None
    unsqueeze_510: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(arg127_1, -1);  arg127_1 = None
    unsqueeze_511: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_510, -1);  unsqueeze_510 = None
    add_206: "f32[8, 360, 16, 16]" = torch.ops.aten.add.Tensor(mul_253, unsqueeze_511);  mul_253 = unsqueeze_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_207: "f32[8, 360, 16, 16]" = torch.ops.aten.add.Tensor(add_206, 3)
    clamp_min_62: "f32[8, 360, 16, 16]" = torch.ops.aten.clamp_min.default(add_207, 0);  add_207 = None
    clamp_max_62: "f32[8, 360, 16, 16]" = torch.ops.aten.clamp_max.default(clamp_min_62, 6);  clamp_min_62 = None
    mul_254: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(add_206, clamp_max_62);  add_206 = clamp_max_62 = None
    div_62: "f32[8, 360, 16, 16]" = torch.ops.aten.div.Tensor(mul_254, 6);  mul_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_10: "f32[8, 360, 1, 1]" = torch.ops.aten.mean.dim(div_62, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_84: "f32[8, 32, 1, 1]" = torch.ops.aten.convolution.default(mean_10, arg280_1, arg281_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_10 = arg280_1 = arg281_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    add_208: "f32[8, 32, 1, 1]" = torch.ops.aten.add.Tensor(convolution_84, 3)
    clamp_min_63: "f32[8, 32, 1, 1]" = torch.ops.aten.clamp_min.default(add_208, 0);  add_208 = None
    clamp_max_63: "f32[8, 32, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_63, 6);  clamp_min_63 = None
    mul_255: "f32[8, 32, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_84, clamp_max_63);  convolution_84 = clamp_max_63 = None
    div_63: "f32[8, 32, 1, 1]" = torch.ops.aten.div.Tensor(mul_255, 6);  mul_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_85: "f32[8, 360, 1, 1]" = torch.ops.aten.convolution.default(div_63, arg282_1, arg283_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_63 = arg282_1 = arg283_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    add_209: "f32[8, 360, 1, 1]" = torch.ops.aten.add.Tensor(convolution_85, 3);  convolution_85 = None
    clamp_min_64: "f32[8, 360, 1, 1]" = torch.ops.aten.clamp_min.default(add_209, 0);  add_209 = None
    clamp_max_64: "f32[8, 360, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_64, 6);  clamp_min_64 = None
    div_64: "f32[8, 360, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_64, 6);  clamp_max_64 = None
    mul_256: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(div_62, div_64);  div_62 = div_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_86: "f32[8, 120, 16, 16]" = torch.ops.aten.convolution.default(mul_256, arg284_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_256 = arg284_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_128: "f32[120]" = torch.ops.prims.convert_element_type.default(arg464_1, torch.float32);  arg464_1 = None
    convert_element_type_129: "f32[120]" = torch.ops.prims.convert_element_type.default(arg465_1, torch.float32);  arg465_1 = None
    add_210: "f32[120]" = torch.ops.aten.add.Tensor(convert_element_type_129, 1e-05);  convert_element_type_129 = None
    sqrt_64: "f32[120]" = torch.ops.aten.sqrt.default(add_210);  add_210 = None
    reciprocal_64: "f32[120]" = torch.ops.aten.reciprocal.default(sqrt_64);  sqrt_64 = None
    mul_257: "f32[120]" = torch.ops.aten.mul.Tensor(reciprocal_64, 1);  reciprocal_64 = None
    unsqueeze_512: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_128, -1);  convert_element_type_128 = None
    unsqueeze_513: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_512, -1);  unsqueeze_512 = None
    unsqueeze_514: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(mul_257, -1);  mul_257 = None
    unsqueeze_515: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_514, -1);  unsqueeze_514 = None
    sub_64: "f32[8, 120, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_86, unsqueeze_513);  convolution_86 = unsqueeze_513 = None
    mul_258: "f32[8, 120, 16, 16]" = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_515);  sub_64 = unsqueeze_515 = None
    unsqueeze_516: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg128_1, -1);  arg128_1 = None
    unsqueeze_517: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_516, -1);  unsqueeze_516 = None
    mul_259: "f32[8, 120, 16, 16]" = torch.ops.aten.mul.Tensor(mul_258, unsqueeze_517);  mul_258 = unsqueeze_517 = None
    unsqueeze_518: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg129_1, -1);  arg129_1 = None
    unsqueeze_519: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_518, -1);  unsqueeze_518 = None
    add_211: "f32[8, 120, 16, 16]" = torch.ops.aten.add.Tensor(mul_259, unsqueeze_519);  mul_259 = unsqueeze_519 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_212: "f32[8, 120, 16, 16]" = torch.ops.aten.add.Tensor(add_211, add_201);  add_211 = add_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_87: "f32[8, 720, 16, 16]" = torch.ops.aten.convolution.default(add_212, arg285_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_212 = arg285_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_130: "f32[720]" = torch.ops.prims.convert_element_type.default(arg466_1, torch.float32);  arg466_1 = None
    convert_element_type_131: "f32[720]" = torch.ops.prims.convert_element_type.default(arg467_1, torch.float32);  arg467_1 = None
    add_213: "f32[720]" = torch.ops.aten.add.Tensor(convert_element_type_131, 1e-05);  convert_element_type_131 = None
    sqrt_65: "f32[720]" = torch.ops.aten.sqrt.default(add_213);  add_213 = None
    reciprocal_65: "f32[720]" = torch.ops.aten.reciprocal.default(sqrt_65);  sqrt_65 = None
    mul_260: "f32[720]" = torch.ops.aten.mul.Tensor(reciprocal_65, 1);  reciprocal_65 = None
    unsqueeze_520: "f32[720, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_130, -1);  convert_element_type_130 = None
    unsqueeze_521: "f32[720, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_520, -1);  unsqueeze_520 = None
    unsqueeze_522: "f32[720, 1]" = torch.ops.aten.unsqueeze.default(mul_260, -1);  mul_260 = None
    unsqueeze_523: "f32[720, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_522, -1);  unsqueeze_522 = None
    sub_65: "f32[8, 720, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_87, unsqueeze_521);  convolution_87 = unsqueeze_521 = None
    mul_261: "f32[8, 720, 16, 16]" = torch.ops.aten.mul.Tensor(sub_65, unsqueeze_523);  sub_65 = unsqueeze_523 = None
    unsqueeze_524: "f32[720, 1]" = torch.ops.aten.unsqueeze.default(arg130_1, -1);  arg130_1 = None
    unsqueeze_525: "f32[720, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_524, -1);  unsqueeze_524 = None
    mul_262: "f32[8, 720, 16, 16]" = torch.ops.aten.mul.Tensor(mul_261, unsqueeze_525);  mul_261 = unsqueeze_525 = None
    unsqueeze_526: "f32[720, 1]" = torch.ops.aten.unsqueeze.default(arg131_1, -1);  arg131_1 = None
    unsqueeze_527: "f32[720, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_526, -1);  unsqueeze_526 = None
    add_214: "f32[8, 720, 16, 16]" = torch.ops.aten.add.Tensor(mul_262, unsqueeze_527);  mul_262 = unsqueeze_527 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_215: "f32[8, 720, 16, 16]" = torch.ops.aten.add.Tensor(add_214, 3)
    clamp_min_65: "f32[8, 720, 16, 16]" = torch.ops.aten.clamp_min.default(add_215, 0);  add_215 = None
    clamp_max_65: "f32[8, 720, 16, 16]" = torch.ops.aten.clamp_max.default(clamp_min_65, 6);  clamp_min_65 = None
    mul_263: "f32[8, 720, 16, 16]" = torch.ops.aten.mul.Tensor(add_214, clamp_max_65);  add_214 = clamp_max_65 = None
    div_65: "f32[8, 720, 16, 16]" = torch.ops.aten.div.Tensor(mul_263, 6);  mul_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_88: "f32[8, 720, 8, 8]" = torch.ops.aten.convolution.default(div_65, arg286_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 720);  div_65 = arg286_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_132: "f32[720]" = torch.ops.prims.convert_element_type.default(arg468_1, torch.float32);  arg468_1 = None
    convert_element_type_133: "f32[720]" = torch.ops.prims.convert_element_type.default(arg469_1, torch.float32);  arg469_1 = None
    add_216: "f32[720]" = torch.ops.aten.add.Tensor(convert_element_type_133, 1e-05);  convert_element_type_133 = None
    sqrt_66: "f32[720]" = torch.ops.aten.sqrt.default(add_216);  add_216 = None
    reciprocal_66: "f32[720]" = torch.ops.aten.reciprocal.default(sqrt_66);  sqrt_66 = None
    mul_264: "f32[720]" = torch.ops.aten.mul.Tensor(reciprocal_66, 1);  reciprocal_66 = None
    unsqueeze_528: "f32[720, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_132, -1);  convert_element_type_132 = None
    unsqueeze_529: "f32[720, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_528, -1);  unsqueeze_528 = None
    unsqueeze_530: "f32[720, 1]" = torch.ops.aten.unsqueeze.default(mul_264, -1);  mul_264 = None
    unsqueeze_531: "f32[720, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_530, -1);  unsqueeze_530 = None
    sub_66: "f32[8, 720, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_88, unsqueeze_529);  convolution_88 = unsqueeze_529 = None
    mul_265: "f32[8, 720, 8, 8]" = torch.ops.aten.mul.Tensor(sub_66, unsqueeze_531);  sub_66 = unsqueeze_531 = None
    unsqueeze_532: "f32[720, 1]" = torch.ops.aten.unsqueeze.default(arg132_1, -1);  arg132_1 = None
    unsqueeze_533: "f32[720, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_532, -1);  unsqueeze_532 = None
    mul_266: "f32[8, 720, 8, 8]" = torch.ops.aten.mul.Tensor(mul_265, unsqueeze_533);  mul_265 = unsqueeze_533 = None
    unsqueeze_534: "f32[720, 1]" = torch.ops.aten.unsqueeze.default(arg133_1, -1);  arg133_1 = None
    unsqueeze_535: "f32[720, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_534, -1);  unsqueeze_534 = None
    add_217: "f32[8, 720, 8, 8]" = torch.ops.aten.add.Tensor(mul_266, unsqueeze_535);  mul_266 = unsqueeze_535 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_218: "f32[8, 720, 8, 8]" = torch.ops.aten.add.Tensor(add_217, 3)
    clamp_min_66: "f32[8, 720, 8, 8]" = torch.ops.aten.clamp_min.default(add_218, 0);  add_218 = None
    clamp_max_66: "f32[8, 720, 8, 8]" = torch.ops.aten.clamp_max.default(clamp_min_66, 6);  clamp_min_66 = None
    mul_267: "f32[8, 720, 8, 8]" = torch.ops.aten.mul.Tensor(add_217, clamp_max_66);  add_217 = clamp_max_66 = None
    div_66: "f32[8, 720, 8, 8]" = torch.ops.aten.div.Tensor(mul_267, 6);  mul_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_11: "f32[8, 720, 1, 1]" = torch.ops.aten.mean.dim(div_66, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_89: "f32[8, 32, 1, 1]" = torch.ops.aten.convolution.default(mean_11, arg287_1, arg288_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_11 = arg287_1 = arg288_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    add_219: "f32[8, 32, 1, 1]" = torch.ops.aten.add.Tensor(convolution_89, 3)
    clamp_min_67: "f32[8, 32, 1, 1]" = torch.ops.aten.clamp_min.default(add_219, 0);  add_219 = None
    clamp_max_67: "f32[8, 32, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_67, 6);  clamp_min_67 = None
    mul_268: "f32[8, 32, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_89, clamp_max_67);  convolution_89 = clamp_max_67 = None
    div_67: "f32[8, 32, 1, 1]" = torch.ops.aten.div.Tensor(mul_268, 6);  mul_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_90: "f32[8, 720, 1, 1]" = torch.ops.aten.convolution.default(div_67, arg289_1, arg290_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_67 = arg289_1 = arg290_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    add_220: "f32[8, 720, 1, 1]" = torch.ops.aten.add.Tensor(convolution_90, 3);  convolution_90 = None
    clamp_min_68: "f32[8, 720, 1, 1]" = torch.ops.aten.clamp_min.default(add_220, 0);  add_220 = None
    clamp_max_68: "f32[8, 720, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_68, 6);  clamp_min_68 = None
    div_68: "f32[8, 720, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_68, 6);  clamp_max_68 = None
    mul_269: "f32[8, 720, 8, 8]" = torch.ops.aten.mul.Tensor(div_66, div_68);  div_66 = div_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_91: "f32[8, 184, 8, 8]" = torch.ops.aten.convolution.default(mul_269, arg291_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_269 = arg291_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_134: "f32[184]" = torch.ops.prims.convert_element_type.default(arg470_1, torch.float32);  arg470_1 = None
    convert_element_type_135: "f32[184]" = torch.ops.prims.convert_element_type.default(arg471_1, torch.float32);  arg471_1 = None
    add_221: "f32[184]" = torch.ops.aten.add.Tensor(convert_element_type_135, 1e-05);  convert_element_type_135 = None
    sqrt_67: "f32[184]" = torch.ops.aten.sqrt.default(add_221);  add_221 = None
    reciprocal_67: "f32[184]" = torch.ops.aten.reciprocal.default(sqrt_67);  sqrt_67 = None
    mul_270: "f32[184]" = torch.ops.aten.mul.Tensor(reciprocal_67, 1);  reciprocal_67 = None
    unsqueeze_536: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_134, -1);  convert_element_type_134 = None
    unsqueeze_537: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_536, -1);  unsqueeze_536 = None
    unsqueeze_538: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(mul_270, -1);  mul_270 = None
    unsqueeze_539: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_538, -1);  unsqueeze_538 = None
    sub_67: "f32[8, 184, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_91, unsqueeze_537);  convolution_91 = unsqueeze_537 = None
    mul_271: "f32[8, 184, 8, 8]" = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_539);  sub_67 = unsqueeze_539 = None
    unsqueeze_540: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(arg134_1, -1);  arg134_1 = None
    unsqueeze_541: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_540, -1);  unsqueeze_540 = None
    mul_272: "f32[8, 184, 8, 8]" = torch.ops.aten.mul.Tensor(mul_271, unsqueeze_541);  mul_271 = unsqueeze_541 = None
    unsqueeze_542: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(arg135_1, -1);  arg135_1 = None
    unsqueeze_543: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_542, -1);  unsqueeze_542 = None
    add_222: "f32[8, 184, 8, 8]" = torch.ops.aten.add.Tensor(mul_272, unsqueeze_543);  mul_272 = unsqueeze_543 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_92: "f32[8, 736, 8, 8]" = torch.ops.aten.convolution.default(add_222, arg292_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg292_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_136: "f32[736]" = torch.ops.prims.convert_element_type.default(arg472_1, torch.float32);  arg472_1 = None
    convert_element_type_137: "f32[736]" = torch.ops.prims.convert_element_type.default(arg473_1, torch.float32);  arg473_1 = None
    add_223: "f32[736]" = torch.ops.aten.add.Tensor(convert_element_type_137, 1e-05);  convert_element_type_137 = None
    sqrt_68: "f32[736]" = torch.ops.aten.sqrt.default(add_223);  add_223 = None
    reciprocal_68: "f32[736]" = torch.ops.aten.reciprocal.default(sqrt_68);  sqrt_68 = None
    mul_273: "f32[736]" = torch.ops.aten.mul.Tensor(reciprocal_68, 1);  reciprocal_68 = None
    unsqueeze_544: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_136, -1);  convert_element_type_136 = None
    unsqueeze_545: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_544, -1);  unsqueeze_544 = None
    unsqueeze_546: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(mul_273, -1);  mul_273 = None
    unsqueeze_547: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_546, -1);  unsqueeze_546 = None
    sub_68: "f32[8, 736, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_92, unsqueeze_545);  convolution_92 = unsqueeze_545 = None
    mul_274: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(sub_68, unsqueeze_547);  sub_68 = unsqueeze_547 = None
    unsqueeze_548: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(arg136_1, -1);  arg136_1 = None
    unsqueeze_549: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_548, -1);  unsqueeze_548 = None
    mul_275: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(mul_274, unsqueeze_549);  mul_274 = unsqueeze_549 = None
    unsqueeze_550: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(arg137_1, -1);  arg137_1 = None
    unsqueeze_551: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_550, -1);  unsqueeze_550 = None
    add_224: "f32[8, 736, 8, 8]" = torch.ops.aten.add.Tensor(mul_275, unsqueeze_551);  mul_275 = unsqueeze_551 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_225: "f32[8, 736, 8, 8]" = torch.ops.aten.add.Tensor(add_224, 3)
    clamp_min_69: "f32[8, 736, 8, 8]" = torch.ops.aten.clamp_min.default(add_225, 0);  add_225 = None
    clamp_max_69: "f32[8, 736, 8, 8]" = torch.ops.aten.clamp_max.default(clamp_min_69, 6);  clamp_min_69 = None
    mul_276: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(add_224, clamp_max_69);  add_224 = clamp_max_69 = None
    div_69: "f32[8, 736, 8, 8]" = torch.ops.aten.div.Tensor(mul_276, 6);  mul_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_93: "f32[8, 736, 8, 8]" = torch.ops.aten.convolution.default(div_69, arg293_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 736);  div_69 = arg293_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_138: "f32[736]" = torch.ops.prims.convert_element_type.default(arg474_1, torch.float32);  arg474_1 = None
    convert_element_type_139: "f32[736]" = torch.ops.prims.convert_element_type.default(arg475_1, torch.float32);  arg475_1 = None
    add_226: "f32[736]" = torch.ops.aten.add.Tensor(convert_element_type_139, 1e-05);  convert_element_type_139 = None
    sqrt_69: "f32[736]" = torch.ops.aten.sqrt.default(add_226);  add_226 = None
    reciprocal_69: "f32[736]" = torch.ops.aten.reciprocal.default(sqrt_69);  sqrt_69 = None
    mul_277: "f32[736]" = torch.ops.aten.mul.Tensor(reciprocal_69, 1);  reciprocal_69 = None
    unsqueeze_552: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_138, -1);  convert_element_type_138 = None
    unsqueeze_553: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_552, -1);  unsqueeze_552 = None
    unsqueeze_554: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(mul_277, -1);  mul_277 = None
    unsqueeze_555: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_554, -1);  unsqueeze_554 = None
    sub_69: "f32[8, 736, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_93, unsqueeze_553);  convolution_93 = unsqueeze_553 = None
    mul_278: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(sub_69, unsqueeze_555);  sub_69 = unsqueeze_555 = None
    unsqueeze_556: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(arg138_1, -1);  arg138_1 = None
    unsqueeze_557: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_556, -1);  unsqueeze_556 = None
    mul_279: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(mul_278, unsqueeze_557);  mul_278 = unsqueeze_557 = None
    unsqueeze_558: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(arg139_1, -1);  arg139_1 = None
    unsqueeze_559: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_558, -1);  unsqueeze_558 = None
    add_227: "f32[8, 736, 8, 8]" = torch.ops.aten.add.Tensor(mul_279, unsqueeze_559);  mul_279 = unsqueeze_559 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_228: "f32[8, 736, 8, 8]" = torch.ops.aten.add.Tensor(add_227, 3)
    clamp_min_70: "f32[8, 736, 8, 8]" = torch.ops.aten.clamp_min.default(add_228, 0);  add_228 = None
    clamp_max_70: "f32[8, 736, 8, 8]" = torch.ops.aten.clamp_max.default(clamp_min_70, 6);  clamp_min_70 = None
    mul_280: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(add_227, clamp_max_70);  add_227 = clamp_max_70 = None
    div_70: "f32[8, 736, 8, 8]" = torch.ops.aten.div.Tensor(mul_280, 6);  mul_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_12: "f32[8, 736, 1, 1]" = torch.ops.aten.mean.dim(div_70, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_94: "f32[8, 48, 1, 1]" = torch.ops.aten.convolution.default(mean_12, arg294_1, arg295_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_12 = arg294_1 = arg295_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    add_229: "f32[8, 48, 1, 1]" = torch.ops.aten.add.Tensor(convolution_94, 3)
    clamp_min_71: "f32[8, 48, 1, 1]" = torch.ops.aten.clamp_min.default(add_229, 0);  add_229 = None
    clamp_max_71: "f32[8, 48, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_71, 6);  clamp_min_71 = None
    mul_281: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_94, clamp_max_71);  convolution_94 = clamp_max_71 = None
    div_71: "f32[8, 48, 1, 1]" = torch.ops.aten.div.Tensor(mul_281, 6);  mul_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_95: "f32[8, 736, 1, 1]" = torch.ops.aten.convolution.default(div_71, arg296_1, arg297_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_71 = arg296_1 = arg297_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    add_230: "f32[8, 736, 1, 1]" = torch.ops.aten.add.Tensor(convolution_95, 3);  convolution_95 = None
    clamp_min_72: "f32[8, 736, 1, 1]" = torch.ops.aten.clamp_min.default(add_230, 0);  add_230 = None
    clamp_max_72: "f32[8, 736, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_72, 6);  clamp_min_72 = None
    div_72: "f32[8, 736, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_72, 6);  clamp_max_72 = None
    mul_282: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(div_70, div_72);  div_70 = div_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_96: "f32[8, 184, 8, 8]" = torch.ops.aten.convolution.default(mul_282, arg298_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_282 = arg298_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_140: "f32[184]" = torch.ops.prims.convert_element_type.default(arg476_1, torch.float32);  arg476_1 = None
    convert_element_type_141: "f32[184]" = torch.ops.prims.convert_element_type.default(arg477_1, torch.float32);  arg477_1 = None
    add_231: "f32[184]" = torch.ops.aten.add.Tensor(convert_element_type_141, 1e-05);  convert_element_type_141 = None
    sqrt_70: "f32[184]" = torch.ops.aten.sqrt.default(add_231);  add_231 = None
    reciprocal_70: "f32[184]" = torch.ops.aten.reciprocal.default(sqrt_70);  sqrt_70 = None
    mul_283: "f32[184]" = torch.ops.aten.mul.Tensor(reciprocal_70, 1);  reciprocal_70 = None
    unsqueeze_560: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_140, -1);  convert_element_type_140 = None
    unsqueeze_561: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_560, -1);  unsqueeze_560 = None
    unsqueeze_562: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(mul_283, -1);  mul_283 = None
    unsqueeze_563: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_562, -1);  unsqueeze_562 = None
    sub_70: "f32[8, 184, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_96, unsqueeze_561);  convolution_96 = unsqueeze_561 = None
    mul_284: "f32[8, 184, 8, 8]" = torch.ops.aten.mul.Tensor(sub_70, unsqueeze_563);  sub_70 = unsqueeze_563 = None
    unsqueeze_564: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(arg140_1, -1);  arg140_1 = None
    unsqueeze_565: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_564, -1);  unsqueeze_564 = None
    mul_285: "f32[8, 184, 8, 8]" = torch.ops.aten.mul.Tensor(mul_284, unsqueeze_565);  mul_284 = unsqueeze_565 = None
    unsqueeze_566: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(arg141_1, -1);  arg141_1 = None
    unsqueeze_567: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_566, -1);  unsqueeze_566 = None
    add_232: "f32[8, 184, 8, 8]" = torch.ops.aten.add.Tensor(mul_285, unsqueeze_567);  mul_285 = unsqueeze_567 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_233: "f32[8, 184, 8, 8]" = torch.ops.aten.add.Tensor(add_232, add_222);  add_232 = add_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_97: "f32[8, 736, 8, 8]" = torch.ops.aten.convolution.default(add_233, arg299_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg299_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_142: "f32[736]" = torch.ops.prims.convert_element_type.default(arg478_1, torch.float32);  arg478_1 = None
    convert_element_type_143: "f32[736]" = torch.ops.prims.convert_element_type.default(arg479_1, torch.float32);  arg479_1 = None
    add_234: "f32[736]" = torch.ops.aten.add.Tensor(convert_element_type_143, 1e-05);  convert_element_type_143 = None
    sqrt_71: "f32[736]" = torch.ops.aten.sqrt.default(add_234);  add_234 = None
    reciprocal_71: "f32[736]" = torch.ops.aten.reciprocal.default(sqrt_71);  sqrt_71 = None
    mul_286: "f32[736]" = torch.ops.aten.mul.Tensor(reciprocal_71, 1);  reciprocal_71 = None
    unsqueeze_568: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_142, -1);  convert_element_type_142 = None
    unsqueeze_569: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_568, -1);  unsqueeze_568 = None
    unsqueeze_570: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(mul_286, -1);  mul_286 = None
    unsqueeze_571: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_570, -1);  unsqueeze_570 = None
    sub_71: "f32[8, 736, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_97, unsqueeze_569);  convolution_97 = unsqueeze_569 = None
    mul_287: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(sub_71, unsqueeze_571);  sub_71 = unsqueeze_571 = None
    unsqueeze_572: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(arg142_1, -1);  arg142_1 = None
    unsqueeze_573: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_572, -1);  unsqueeze_572 = None
    mul_288: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(mul_287, unsqueeze_573);  mul_287 = unsqueeze_573 = None
    unsqueeze_574: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(arg143_1, -1);  arg143_1 = None
    unsqueeze_575: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_574, -1);  unsqueeze_574 = None
    add_235: "f32[8, 736, 8, 8]" = torch.ops.aten.add.Tensor(mul_288, unsqueeze_575);  mul_288 = unsqueeze_575 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_236: "f32[8, 736, 8, 8]" = torch.ops.aten.add.Tensor(add_235, 3)
    clamp_min_73: "f32[8, 736, 8, 8]" = torch.ops.aten.clamp_min.default(add_236, 0);  add_236 = None
    clamp_max_73: "f32[8, 736, 8, 8]" = torch.ops.aten.clamp_max.default(clamp_min_73, 6);  clamp_min_73 = None
    mul_289: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(add_235, clamp_max_73);  add_235 = clamp_max_73 = None
    div_73: "f32[8, 736, 8, 8]" = torch.ops.aten.div.Tensor(mul_289, 6);  mul_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_98: "f32[8, 736, 8, 8]" = torch.ops.aten.convolution.default(div_73, arg300_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 736);  div_73 = arg300_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_144: "f32[736]" = torch.ops.prims.convert_element_type.default(arg480_1, torch.float32);  arg480_1 = None
    convert_element_type_145: "f32[736]" = torch.ops.prims.convert_element_type.default(arg481_1, torch.float32);  arg481_1 = None
    add_237: "f32[736]" = torch.ops.aten.add.Tensor(convert_element_type_145, 1e-05);  convert_element_type_145 = None
    sqrt_72: "f32[736]" = torch.ops.aten.sqrt.default(add_237);  add_237 = None
    reciprocal_72: "f32[736]" = torch.ops.aten.reciprocal.default(sqrt_72);  sqrt_72 = None
    mul_290: "f32[736]" = torch.ops.aten.mul.Tensor(reciprocal_72, 1);  reciprocal_72 = None
    unsqueeze_576: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_144, -1);  convert_element_type_144 = None
    unsqueeze_577: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_576, -1);  unsqueeze_576 = None
    unsqueeze_578: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(mul_290, -1);  mul_290 = None
    unsqueeze_579: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_578, -1);  unsqueeze_578 = None
    sub_72: "f32[8, 736, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_98, unsqueeze_577);  convolution_98 = unsqueeze_577 = None
    mul_291: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(sub_72, unsqueeze_579);  sub_72 = unsqueeze_579 = None
    unsqueeze_580: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(arg144_1, -1);  arg144_1 = None
    unsqueeze_581: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_580, -1);  unsqueeze_580 = None
    mul_292: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(mul_291, unsqueeze_581);  mul_291 = unsqueeze_581 = None
    unsqueeze_582: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(arg145_1, -1);  arg145_1 = None
    unsqueeze_583: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_582, -1);  unsqueeze_582 = None
    add_238: "f32[8, 736, 8, 8]" = torch.ops.aten.add.Tensor(mul_292, unsqueeze_583);  mul_292 = unsqueeze_583 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_239: "f32[8, 736, 8, 8]" = torch.ops.aten.add.Tensor(add_238, 3)
    clamp_min_74: "f32[8, 736, 8, 8]" = torch.ops.aten.clamp_min.default(add_239, 0);  add_239 = None
    clamp_max_74: "f32[8, 736, 8, 8]" = torch.ops.aten.clamp_max.default(clamp_min_74, 6);  clamp_min_74 = None
    mul_293: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(add_238, clamp_max_74);  add_238 = clamp_max_74 = None
    div_74: "f32[8, 736, 8, 8]" = torch.ops.aten.div.Tensor(mul_293, 6);  mul_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_13: "f32[8, 736, 1, 1]" = torch.ops.aten.mean.dim(div_74, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_99: "f32[8, 48, 1, 1]" = torch.ops.aten.convolution.default(mean_13, arg301_1, arg302_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_13 = arg301_1 = arg302_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    add_240: "f32[8, 48, 1, 1]" = torch.ops.aten.add.Tensor(convolution_99, 3)
    clamp_min_75: "f32[8, 48, 1, 1]" = torch.ops.aten.clamp_min.default(add_240, 0);  add_240 = None
    clamp_max_75: "f32[8, 48, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_75, 6);  clamp_min_75 = None
    mul_294: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_99, clamp_max_75);  convolution_99 = clamp_max_75 = None
    div_75: "f32[8, 48, 1, 1]" = torch.ops.aten.div.Tensor(mul_294, 6);  mul_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_100: "f32[8, 736, 1, 1]" = torch.ops.aten.convolution.default(div_75, arg303_1, arg304_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_75 = arg303_1 = arg304_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    add_241: "f32[8, 736, 1, 1]" = torch.ops.aten.add.Tensor(convolution_100, 3);  convolution_100 = None
    clamp_min_76: "f32[8, 736, 1, 1]" = torch.ops.aten.clamp_min.default(add_241, 0);  add_241 = None
    clamp_max_76: "f32[8, 736, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_76, 6);  clamp_min_76 = None
    div_76: "f32[8, 736, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_76, 6);  clamp_max_76 = None
    mul_295: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(div_74, div_76);  div_74 = div_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_101: "f32[8, 184, 8, 8]" = torch.ops.aten.convolution.default(mul_295, arg305_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_295 = arg305_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_146: "f32[184]" = torch.ops.prims.convert_element_type.default(arg482_1, torch.float32);  arg482_1 = None
    convert_element_type_147: "f32[184]" = torch.ops.prims.convert_element_type.default(arg483_1, torch.float32);  arg483_1 = None
    add_242: "f32[184]" = torch.ops.aten.add.Tensor(convert_element_type_147, 1e-05);  convert_element_type_147 = None
    sqrt_73: "f32[184]" = torch.ops.aten.sqrt.default(add_242);  add_242 = None
    reciprocal_73: "f32[184]" = torch.ops.aten.reciprocal.default(sqrt_73);  sqrt_73 = None
    mul_296: "f32[184]" = torch.ops.aten.mul.Tensor(reciprocal_73, 1);  reciprocal_73 = None
    unsqueeze_584: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_146, -1);  convert_element_type_146 = None
    unsqueeze_585: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_584, -1);  unsqueeze_584 = None
    unsqueeze_586: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(mul_296, -1);  mul_296 = None
    unsqueeze_587: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_586, -1);  unsqueeze_586 = None
    sub_73: "f32[8, 184, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_101, unsqueeze_585);  convolution_101 = unsqueeze_585 = None
    mul_297: "f32[8, 184, 8, 8]" = torch.ops.aten.mul.Tensor(sub_73, unsqueeze_587);  sub_73 = unsqueeze_587 = None
    unsqueeze_588: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(arg146_1, -1);  arg146_1 = None
    unsqueeze_589: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_588, -1);  unsqueeze_588 = None
    mul_298: "f32[8, 184, 8, 8]" = torch.ops.aten.mul.Tensor(mul_297, unsqueeze_589);  mul_297 = unsqueeze_589 = None
    unsqueeze_590: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(arg147_1, -1);  arg147_1 = None
    unsqueeze_591: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_590, -1);  unsqueeze_590 = None
    add_243: "f32[8, 184, 8, 8]" = torch.ops.aten.add.Tensor(mul_298, unsqueeze_591);  mul_298 = unsqueeze_591 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_244: "f32[8, 184, 8, 8]" = torch.ops.aten.add.Tensor(add_243, add_233);  add_243 = add_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_102: "f32[8, 736, 8, 8]" = torch.ops.aten.convolution.default(add_244, arg306_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg306_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_148: "f32[736]" = torch.ops.prims.convert_element_type.default(arg484_1, torch.float32);  arg484_1 = None
    convert_element_type_149: "f32[736]" = torch.ops.prims.convert_element_type.default(arg485_1, torch.float32);  arg485_1 = None
    add_245: "f32[736]" = torch.ops.aten.add.Tensor(convert_element_type_149, 1e-05);  convert_element_type_149 = None
    sqrt_74: "f32[736]" = torch.ops.aten.sqrt.default(add_245);  add_245 = None
    reciprocal_74: "f32[736]" = torch.ops.aten.reciprocal.default(sqrt_74);  sqrt_74 = None
    mul_299: "f32[736]" = torch.ops.aten.mul.Tensor(reciprocal_74, 1);  reciprocal_74 = None
    unsqueeze_592: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_148, -1);  convert_element_type_148 = None
    unsqueeze_593: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_592, -1);  unsqueeze_592 = None
    unsqueeze_594: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(mul_299, -1);  mul_299 = None
    unsqueeze_595: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_594, -1);  unsqueeze_594 = None
    sub_74: "f32[8, 736, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_102, unsqueeze_593);  convolution_102 = unsqueeze_593 = None
    mul_300: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(sub_74, unsqueeze_595);  sub_74 = unsqueeze_595 = None
    unsqueeze_596: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(arg148_1, -1);  arg148_1 = None
    unsqueeze_597: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_596, -1);  unsqueeze_596 = None
    mul_301: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(mul_300, unsqueeze_597);  mul_300 = unsqueeze_597 = None
    unsqueeze_598: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(arg149_1, -1);  arg149_1 = None
    unsqueeze_599: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_598, -1);  unsqueeze_598 = None
    add_246: "f32[8, 736, 8, 8]" = torch.ops.aten.add.Tensor(mul_301, unsqueeze_599);  mul_301 = unsqueeze_599 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_247: "f32[8, 736, 8, 8]" = torch.ops.aten.add.Tensor(add_246, 3)
    clamp_min_77: "f32[8, 736, 8, 8]" = torch.ops.aten.clamp_min.default(add_247, 0);  add_247 = None
    clamp_max_77: "f32[8, 736, 8, 8]" = torch.ops.aten.clamp_max.default(clamp_min_77, 6);  clamp_min_77 = None
    mul_302: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(add_246, clamp_max_77);  add_246 = clamp_max_77 = None
    div_77: "f32[8, 736, 8, 8]" = torch.ops.aten.div.Tensor(mul_302, 6);  mul_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_103: "f32[8, 736, 8, 8]" = torch.ops.aten.convolution.default(div_77, arg307_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 736);  div_77 = arg307_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_150: "f32[736]" = torch.ops.prims.convert_element_type.default(arg486_1, torch.float32);  arg486_1 = None
    convert_element_type_151: "f32[736]" = torch.ops.prims.convert_element_type.default(arg487_1, torch.float32);  arg487_1 = None
    add_248: "f32[736]" = torch.ops.aten.add.Tensor(convert_element_type_151, 1e-05);  convert_element_type_151 = None
    sqrt_75: "f32[736]" = torch.ops.aten.sqrt.default(add_248);  add_248 = None
    reciprocal_75: "f32[736]" = torch.ops.aten.reciprocal.default(sqrt_75);  sqrt_75 = None
    mul_303: "f32[736]" = torch.ops.aten.mul.Tensor(reciprocal_75, 1);  reciprocal_75 = None
    unsqueeze_600: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_150, -1);  convert_element_type_150 = None
    unsqueeze_601: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_600, -1);  unsqueeze_600 = None
    unsqueeze_602: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(mul_303, -1);  mul_303 = None
    unsqueeze_603: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_602, -1);  unsqueeze_602 = None
    sub_75: "f32[8, 736, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_103, unsqueeze_601);  convolution_103 = unsqueeze_601 = None
    mul_304: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(sub_75, unsqueeze_603);  sub_75 = unsqueeze_603 = None
    unsqueeze_604: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(arg150_1, -1);  arg150_1 = None
    unsqueeze_605: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_604, -1);  unsqueeze_604 = None
    mul_305: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(mul_304, unsqueeze_605);  mul_304 = unsqueeze_605 = None
    unsqueeze_606: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(arg151_1, -1);  arg151_1 = None
    unsqueeze_607: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_606, -1);  unsqueeze_606 = None
    add_249: "f32[8, 736, 8, 8]" = torch.ops.aten.add.Tensor(mul_305, unsqueeze_607);  mul_305 = unsqueeze_607 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_250: "f32[8, 736, 8, 8]" = torch.ops.aten.add.Tensor(add_249, 3)
    clamp_min_78: "f32[8, 736, 8, 8]" = torch.ops.aten.clamp_min.default(add_250, 0);  add_250 = None
    clamp_max_78: "f32[8, 736, 8, 8]" = torch.ops.aten.clamp_max.default(clamp_min_78, 6);  clamp_min_78 = None
    mul_306: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(add_249, clamp_max_78);  add_249 = clamp_max_78 = None
    div_78: "f32[8, 736, 8, 8]" = torch.ops.aten.div.Tensor(mul_306, 6);  mul_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_14: "f32[8, 736, 1, 1]" = torch.ops.aten.mean.dim(div_78, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_104: "f32[8, 48, 1, 1]" = torch.ops.aten.convolution.default(mean_14, arg308_1, arg309_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_14 = arg308_1 = arg309_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    add_251: "f32[8, 48, 1, 1]" = torch.ops.aten.add.Tensor(convolution_104, 3)
    clamp_min_79: "f32[8, 48, 1, 1]" = torch.ops.aten.clamp_min.default(add_251, 0);  add_251 = None
    clamp_max_79: "f32[8, 48, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_79, 6);  clamp_min_79 = None
    mul_307: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_104, clamp_max_79);  convolution_104 = clamp_max_79 = None
    div_79: "f32[8, 48, 1, 1]" = torch.ops.aten.div.Tensor(mul_307, 6);  mul_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_105: "f32[8, 736, 1, 1]" = torch.ops.aten.convolution.default(div_79, arg310_1, arg311_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_79 = arg310_1 = arg311_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    add_252: "f32[8, 736, 1, 1]" = torch.ops.aten.add.Tensor(convolution_105, 3);  convolution_105 = None
    clamp_min_80: "f32[8, 736, 1, 1]" = torch.ops.aten.clamp_min.default(add_252, 0);  add_252 = None
    clamp_max_80: "f32[8, 736, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_80, 6);  clamp_min_80 = None
    div_80: "f32[8, 736, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_80, 6);  clamp_max_80 = None
    mul_308: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(div_78, div_80);  div_78 = div_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_106: "f32[8, 184, 8, 8]" = torch.ops.aten.convolution.default(mul_308, arg312_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_308 = arg312_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_152: "f32[184]" = torch.ops.prims.convert_element_type.default(arg488_1, torch.float32);  arg488_1 = None
    convert_element_type_153: "f32[184]" = torch.ops.prims.convert_element_type.default(arg489_1, torch.float32);  arg489_1 = None
    add_253: "f32[184]" = torch.ops.aten.add.Tensor(convert_element_type_153, 1e-05);  convert_element_type_153 = None
    sqrt_76: "f32[184]" = torch.ops.aten.sqrt.default(add_253);  add_253 = None
    reciprocal_76: "f32[184]" = torch.ops.aten.reciprocal.default(sqrt_76);  sqrt_76 = None
    mul_309: "f32[184]" = torch.ops.aten.mul.Tensor(reciprocal_76, 1);  reciprocal_76 = None
    unsqueeze_608: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_152, -1);  convert_element_type_152 = None
    unsqueeze_609: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_608, -1);  unsqueeze_608 = None
    unsqueeze_610: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(mul_309, -1);  mul_309 = None
    unsqueeze_611: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_610, -1);  unsqueeze_610 = None
    sub_76: "f32[8, 184, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_106, unsqueeze_609);  convolution_106 = unsqueeze_609 = None
    mul_310: "f32[8, 184, 8, 8]" = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_611);  sub_76 = unsqueeze_611 = None
    unsqueeze_612: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(arg152_1, -1);  arg152_1 = None
    unsqueeze_613: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_612, -1);  unsqueeze_612 = None
    mul_311: "f32[8, 184, 8, 8]" = torch.ops.aten.mul.Tensor(mul_310, unsqueeze_613);  mul_310 = unsqueeze_613 = None
    unsqueeze_614: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(arg153_1, -1);  arg153_1 = None
    unsqueeze_615: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_614, -1);  unsqueeze_614 = None
    add_254: "f32[8, 184, 8, 8]" = torch.ops.aten.add.Tensor(mul_311, unsqueeze_615);  mul_311 = unsqueeze_615 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_255: "f32[8, 184, 8, 8]" = torch.ops.aten.add.Tensor(add_254, add_244);  add_254 = add_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_107: "f32[8, 736, 8, 8]" = torch.ops.aten.convolution.default(add_255, arg313_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg313_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_154: "f32[736]" = torch.ops.prims.convert_element_type.default(arg490_1, torch.float32);  arg490_1 = None
    convert_element_type_155: "f32[736]" = torch.ops.prims.convert_element_type.default(arg491_1, torch.float32);  arg491_1 = None
    add_256: "f32[736]" = torch.ops.aten.add.Tensor(convert_element_type_155, 1e-05);  convert_element_type_155 = None
    sqrt_77: "f32[736]" = torch.ops.aten.sqrt.default(add_256);  add_256 = None
    reciprocal_77: "f32[736]" = torch.ops.aten.reciprocal.default(sqrt_77);  sqrt_77 = None
    mul_312: "f32[736]" = torch.ops.aten.mul.Tensor(reciprocal_77, 1);  reciprocal_77 = None
    unsqueeze_616: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_154, -1);  convert_element_type_154 = None
    unsqueeze_617: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_616, -1);  unsqueeze_616 = None
    unsqueeze_618: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(mul_312, -1);  mul_312 = None
    unsqueeze_619: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_618, -1);  unsqueeze_618 = None
    sub_77: "f32[8, 736, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_107, unsqueeze_617);  convolution_107 = unsqueeze_617 = None
    mul_313: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(sub_77, unsqueeze_619);  sub_77 = unsqueeze_619 = None
    unsqueeze_620: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(arg154_1, -1);  arg154_1 = None
    unsqueeze_621: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_620, -1);  unsqueeze_620 = None
    mul_314: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(mul_313, unsqueeze_621);  mul_313 = unsqueeze_621 = None
    unsqueeze_622: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(arg155_1, -1);  arg155_1 = None
    unsqueeze_623: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_622, -1);  unsqueeze_622 = None
    add_257: "f32[8, 736, 8, 8]" = torch.ops.aten.add.Tensor(mul_314, unsqueeze_623);  mul_314 = unsqueeze_623 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_258: "f32[8, 736, 8, 8]" = torch.ops.aten.add.Tensor(add_257, 3)
    clamp_min_81: "f32[8, 736, 8, 8]" = torch.ops.aten.clamp_min.default(add_258, 0);  add_258 = None
    clamp_max_81: "f32[8, 736, 8, 8]" = torch.ops.aten.clamp_max.default(clamp_min_81, 6);  clamp_min_81 = None
    mul_315: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(add_257, clamp_max_81);  add_257 = clamp_max_81 = None
    div_81: "f32[8, 736, 8, 8]" = torch.ops.aten.div.Tensor(mul_315, 6);  mul_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_108: "f32[8, 736, 8, 8]" = torch.ops.aten.convolution.default(div_81, arg314_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 736);  div_81 = arg314_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_156: "f32[736]" = torch.ops.prims.convert_element_type.default(arg492_1, torch.float32);  arg492_1 = None
    convert_element_type_157: "f32[736]" = torch.ops.prims.convert_element_type.default(arg493_1, torch.float32);  arg493_1 = None
    add_259: "f32[736]" = torch.ops.aten.add.Tensor(convert_element_type_157, 1e-05);  convert_element_type_157 = None
    sqrt_78: "f32[736]" = torch.ops.aten.sqrt.default(add_259);  add_259 = None
    reciprocal_78: "f32[736]" = torch.ops.aten.reciprocal.default(sqrt_78);  sqrt_78 = None
    mul_316: "f32[736]" = torch.ops.aten.mul.Tensor(reciprocal_78, 1);  reciprocal_78 = None
    unsqueeze_624: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_156, -1);  convert_element_type_156 = None
    unsqueeze_625: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_624, -1);  unsqueeze_624 = None
    unsqueeze_626: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(mul_316, -1);  mul_316 = None
    unsqueeze_627: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_626, -1);  unsqueeze_626 = None
    sub_78: "f32[8, 736, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_108, unsqueeze_625);  convolution_108 = unsqueeze_625 = None
    mul_317: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(sub_78, unsqueeze_627);  sub_78 = unsqueeze_627 = None
    unsqueeze_628: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(arg156_1, -1);  arg156_1 = None
    unsqueeze_629: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_628, -1);  unsqueeze_628 = None
    mul_318: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(mul_317, unsqueeze_629);  mul_317 = unsqueeze_629 = None
    unsqueeze_630: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(arg157_1, -1);  arg157_1 = None
    unsqueeze_631: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_630, -1);  unsqueeze_630 = None
    add_260: "f32[8, 736, 8, 8]" = torch.ops.aten.add.Tensor(mul_318, unsqueeze_631);  mul_318 = unsqueeze_631 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_261: "f32[8, 736, 8, 8]" = torch.ops.aten.add.Tensor(add_260, 3)
    clamp_min_82: "f32[8, 736, 8, 8]" = torch.ops.aten.clamp_min.default(add_261, 0);  add_261 = None
    clamp_max_82: "f32[8, 736, 8, 8]" = torch.ops.aten.clamp_max.default(clamp_min_82, 6);  clamp_min_82 = None
    mul_319: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(add_260, clamp_max_82);  add_260 = clamp_max_82 = None
    div_82: "f32[8, 736, 8, 8]" = torch.ops.aten.div.Tensor(mul_319, 6);  mul_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_15: "f32[8, 736, 1, 1]" = torch.ops.aten.mean.dim(div_82, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_109: "f32[8, 48, 1, 1]" = torch.ops.aten.convolution.default(mean_15, arg315_1, arg316_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_15 = arg315_1 = arg316_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    add_262: "f32[8, 48, 1, 1]" = torch.ops.aten.add.Tensor(convolution_109, 3)
    clamp_min_83: "f32[8, 48, 1, 1]" = torch.ops.aten.clamp_min.default(add_262, 0);  add_262 = None
    clamp_max_83: "f32[8, 48, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_83, 6);  clamp_min_83 = None
    mul_320: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_109, clamp_max_83);  convolution_109 = clamp_max_83 = None
    div_83: "f32[8, 48, 1, 1]" = torch.ops.aten.div.Tensor(mul_320, 6);  mul_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_110: "f32[8, 736, 1, 1]" = torch.ops.aten.convolution.default(div_83, arg317_1, arg318_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_83 = arg317_1 = arg318_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    add_263: "f32[8, 736, 1, 1]" = torch.ops.aten.add.Tensor(convolution_110, 3);  convolution_110 = None
    clamp_min_84: "f32[8, 736, 1, 1]" = torch.ops.aten.clamp_min.default(add_263, 0);  add_263 = None
    clamp_max_84: "f32[8, 736, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_84, 6);  clamp_min_84 = None
    div_84: "f32[8, 736, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_84, 6);  clamp_max_84 = None
    mul_321: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(div_82, div_84);  div_82 = div_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_111: "f32[8, 184, 8, 8]" = torch.ops.aten.convolution.default(mul_321, arg319_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_321 = arg319_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_158: "f32[184]" = torch.ops.prims.convert_element_type.default(arg494_1, torch.float32);  arg494_1 = None
    convert_element_type_159: "f32[184]" = torch.ops.prims.convert_element_type.default(arg495_1, torch.float32);  arg495_1 = None
    add_264: "f32[184]" = torch.ops.aten.add.Tensor(convert_element_type_159, 1e-05);  convert_element_type_159 = None
    sqrt_79: "f32[184]" = torch.ops.aten.sqrt.default(add_264);  add_264 = None
    reciprocal_79: "f32[184]" = torch.ops.aten.reciprocal.default(sqrt_79);  sqrt_79 = None
    mul_322: "f32[184]" = torch.ops.aten.mul.Tensor(reciprocal_79, 1);  reciprocal_79 = None
    unsqueeze_632: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_158, -1);  convert_element_type_158 = None
    unsqueeze_633: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_632, -1);  unsqueeze_632 = None
    unsqueeze_634: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(mul_322, -1);  mul_322 = None
    unsqueeze_635: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_634, -1);  unsqueeze_634 = None
    sub_79: "f32[8, 184, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_111, unsqueeze_633);  convolution_111 = unsqueeze_633 = None
    mul_323: "f32[8, 184, 8, 8]" = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_635);  sub_79 = unsqueeze_635 = None
    unsqueeze_636: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(arg158_1, -1);  arg158_1 = None
    unsqueeze_637: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_636, -1);  unsqueeze_636 = None
    mul_324: "f32[8, 184, 8, 8]" = torch.ops.aten.mul.Tensor(mul_323, unsqueeze_637);  mul_323 = unsqueeze_637 = None
    unsqueeze_638: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(arg159_1, -1);  arg159_1 = None
    unsqueeze_639: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_638, -1);  unsqueeze_638 = None
    add_265: "f32[8, 184, 8, 8]" = torch.ops.aten.add.Tensor(mul_324, unsqueeze_639);  mul_324 = unsqueeze_639 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_266: "f32[8, 184, 8, 8]" = torch.ops.aten.add.Tensor(add_265, add_255);  add_265 = add_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_112: "f32[8, 736, 8, 8]" = torch.ops.aten.convolution.default(add_266, arg320_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg320_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_160: "f32[736]" = torch.ops.prims.convert_element_type.default(arg496_1, torch.float32);  arg496_1 = None
    convert_element_type_161: "f32[736]" = torch.ops.prims.convert_element_type.default(arg497_1, torch.float32);  arg497_1 = None
    add_267: "f32[736]" = torch.ops.aten.add.Tensor(convert_element_type_161, 1e-05);  convert_element_type_161 = None
    sqrt_80: "f32[736]" = torch.ops.aten.sqrt.default(add_267);  add_267 = None
    reciprocal_80: "f32[736]" = torch.ops.aten.reciprocal.default(sqrt_80);  sqrt_80 = None
    mul_325: "f32[736]" = torch.ops.aten.mul.Tensor(reciprocal_80, 1);  reciprocal_80 = None
    unsqueeze_640: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_160, -1);  convert_element_type_160 = None
    unsqueeze_641: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_640, -1);  unsqueeze_640 = None
    unsqueeze_642: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(mul_325, -1);  mul_325 = None
    unsqueeze_643: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_642, -1);  unsqueeze_642 = None
    sub_80: "f32[8, 736, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_112, unsqueeze_641);  convolution_112 = unsqueeze_641 = None
    mul_326: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(sub_80, unsqueeze_643);  sub_80 = unsqueeze_643 = None
    unsqueeze_644: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(arg160_1, -1);  arg160_1 = None
    unsqueeze_645: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_644, -1);  unsqueeze_644 = None
    mul_327: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(mul_326, unsqueeze_645);  mul_326 = unsqueeze_645 = None
    unsqueeze_646: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(arg161_1, -1);  arg161_1 = None
    unsqueeze_647: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_646, -1);  unsqueeze_646 = None
    add_268: "f32[8, 736, 8, 8]" = torch.ops.aten.add.Tensor(mul_327, unsqueeze_647);  mul_327 = unsqueeze_647 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_269: "f32[8, 736, 8, 8]" = torch.ops.aten.add.Tensor(add_268, 3)
    clamp_min_85: "f32[8, 736, 8, 8]" = torch.ops.aten.clamp_min.default(add_269, 0);  add_269 = None
    clamp_max_85: "f32[8, 736, 8, 8]" = torch.ops.aten.clamp_max.default(clamp_min_85, 6);  clamp_min_85 = None
    mul_328: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(add_268, clamp_max_85);  add_268 = clamp_max_85 = None
    div_85: "f32[8, 736, 8, 8]" = torch.ops.aten.div.Tensor(mul_328, 6);  mul_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_113: "f32[8, 736, 8, 8]" = torch.ops.aten.convolution.default(div_85, arg321_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 736);  div_85 = arg321_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_162: "f32[736]" = torch.ops.prims.convert_element_type.default(arg498_1, torch.float32);  arg498_1 = None
    convert_element_type_163: "f32[736]" = torch.ops.prims.convert_element_type.default(arg499_1, torch.float32);  arg499_1 = None
    add_270: "f32[736]" = torch.ops.aten.add.Tensor(convert_element_type_163, 1e-05);  convert_element_type_163 = None
    sqrt_81: "f32[736]" = torch.ops.aten.sqrt.default(add_270);  add_270 = None
    reciprocal_81: "f32[736]" = torch.ops.aten.reciprocal.default(sqrt_81);  sqrt_81 = None
    mul_329: "f32[736]" = torch.ops.aten.mul.Tensor(reciprocal_81, 1);  reciprocal_81 = None
    unsqueeze_648: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_162, -1);  convert_element_type_162 = None
    unsqueeze_649: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_648, -1);  unsqueeze_648 = None
    unsqueeze_650: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(mul_329, -1);  mul_329 = None
    unsqueeze_651: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_650, -1);  unsqueeze_650 = None
    sub_81: "f32[8, 736, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_113, unsqueeze_649);  convolution_113 = unsqueeze_649 = None
    mul_330: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(sub_81, unsqueeze_651);  sub_81 = unsqueeze_651 = None
    unsqueeze_652: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(arg162_1, -1);  arg162_1 = None
    unsqueeze_653: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_652, -1);  unsqueeze_652 = None
    mul_331: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(mul_330, unsqueeze_653);  mul_330 = unsqueeze_653 = None
    unsqueeze_654: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(arg163_1, -1);  arg163_1 = None
    unsqueeze_655: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_654, -1);  unsqueeze_654 = None
    add_271: "f32[8, 736, 8, 8]" = torch.ops.aten.add.Tensor(mul_331, unsqueeze_655);  mul_331 = unsqueeze_655 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_272: "f32[8, 736, 8, 8]" = torch.ops.aten.add.Tensor(add_271, 3)
    clamp_min_86: "f32[8, 736, 8, 8]" = torch.ops.aten.clamp_min.default(add_272, 0);  add_272 = None
    clamp_max_86: "f32[8, 736, 8, 8]" = torch.ops.aten.clamp_max.default(clamp_min_86, 6);  clamp_min_86 = None
    mul_332: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(add_271, clamp_max_86);  add_271 = clamp_max_86 = None
    div_86: "f32[8, 736, 8, 8]" = torch.ops.aten.div.Tensor(mul_332, 6);  mul_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_16: "f32[8, 736, 1, 1]" = torch.ops.aten.mean.dim(div_86, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_114: "f32[8, 48, 1, 1]" = torch.ops.aten.convolution.default(mean_16, arg322_1, arg323_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_16 = arg322_1 = arg323_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    add_273: "f32[8, 48, 1, 1]" = torch.ops.aten.add.Tensor(convolution_114, 3)
    clamp_min_87: "f32[8, 48, 1, 1]" = torch.ops.aten.clamp_min.default(add_273, 0);  add_273 = None
    clamp_max_87: "f32[8, 48, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_87, 6);  clamp_min_87 = None
    mul_333: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_114, clamp_max_87);  convolution_114 = clamp_max_87 = None
    div_87: "f32[8, 48, 1, 1]" = torch.ops.aten.div.Tensor(mul_333, 6);  mul_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_115: "f32[8, 736, 1, 1]" = torch.ops.aten.convolution.default(div_87, arg324_1, arg325_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_87 = arg324_1 = arg325_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    add_274: "f32[8, 736, 1, 1]" = torch.ops.aten.add.Tensor(convolution_115, 3);  convolution_115 = None
    clamp_min_88: "f32[8, 736, 1, 1]" = torch.ops.aten.clamp_min.default(add_274, 0);  add_274 = None
    clamp_max_88: "f32[8, 736, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_88, 6);  clamp_min_88 = None
    div_88: "f32[8, 736, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_88, 6);  clamp_max_88 = None
    mul_334: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(div_86, div_88);  div_86 = div_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_116: "f32[8, 184, 8, 8]" = torch.ops.aten.convolution.default(mul_334, arg326_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_334 = arg326_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_164: "f32[184]" = torch.ops.prims.convert_element_type.default(arg500_1, torch.float32);  arg500_1 = None
    convert_element_type_165: "f32[184]" = torch.ops.prims.convert_element_type.default(arg501_1, torch.float32);  arg501_1 = None
    add_275: "f32[184]" = torch.ops.aten.add.Tensor(convert_element_type_165, 1e-05);  convert_element_type_165 = None
    sqrt_82: "f32[184]" = torch.ops.aten.sqrt.default(add_275);  add_275 = None
    reciprocal_82: "f32[184]" = torch.ops.aten.reciprocal.default(sqrt_82);  sqrt_82 = None
    mul_335: "f32[184]" = torch.ops.aten.mul.Tensor(reciprocal_82, 1);  reciprocal_82 = None
    unsqueeze_656: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_164, -1);  convert_element_type_164 = None
    unsqueeze_657: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_656, -1);  unsqueeze_656 = None
    unsqueeze_658: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(mul_335, -1);  mul_335 = None
    unsqueeze_659: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_658, -1);  unsqueeze_658 = None
    sub_82: "f32[8, 184, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_116, unsqueeze_657);  convolution_116 = unsqueeze_657 = None
    mul_336: "f32[8, 184, 8, 8]" = torch.ops.aten.mul.Tensor(sub_82, unsqueeze_659);  sub_82 = unsqueeze_659 = None
    unsqueeze_660: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(arg164_1, -1);  arg164_1 = None
    unsqueeze_661: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_660, -1);  unsqueeze_660 = None
    mul_337: "f32[8, 184, 8, 8]" = torch.ops.aten.mul.Tensor(mul_336, unsqueeze_661);  mul_336 = unsqueeze_661 = None
    unsqueeze_662: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(arg165_1, -1);  arg165_1 = None
    unsqueeze_663: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_662, -1);  unsqueeze_662 = None
    add_276: "f32[8, 184, 8, 8]" = torch.ops.aten.add.Tensor(mul_337, unsqueeze_663);  mul_337 = unsqueeze_663 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_277: "f32[8, 184, 8, 8]" = torch.ops.aten.add.Tensor(add_276, add_266);  add_276 = add_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_117: "f32[8, 1104, 8, 8]" = torch.ops.aten.convolution.default(add_277, arg327_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_277 = arg327_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_166: "f32[1104]" = torch.ops.prims.convert_element_type.default(arg502_1, torch.float32);  arg502_1 = None
    convert_element_type_167: "f32[1104]" = torch.ops.prims.convert_element_type.default(arg503_1, torch.float32);  arg503_1 = None
    add_278: "f32[1104]" = torch.ops.aten.add.Tensor(convert_element_type_167, 1e-05);  convert_element_type_167 = None
    sqrt_83: "f32[1104]" = torch.ops.aten.sqrt.default(add_278);  add_278 = None
    reciprocal_83: "f32[1104]" = torch.ops.aten.reciprocal.default(sqrt_83);  sqrt_83 = None
    mul_338: "f32[1104]" = torch.ops.aten.mul.Tensor(reciprocal_83, 1);  reciprocal_83 = None
    unsqueeze_664: "f32[1104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_166, -1);  convert_element_type_166 = None
    unsqueeze_665: "f32[1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_664, -1);  unsqueeze_664 = None
    unsqueeze_666: "f32[1104, 1]" = torch.ops.aten.unsqueeze.default(mul_338, -1);  mul_338 = None
    unsqueeze_667: "f32[1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_666, -1);  unsqueeze_666 = None
    sub_83: "f32[8, 1104, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_117, unsqueeze_665);  convolution_117 = unsqueeze_665 = None
    mul_339: "f32[8, 1104, 8, 8]" = torch.ops.aten.mul.Tensor(sub_83, unsqueeze_667);  sub_83 = unsqueeze_667 = None
    unsqueeze_668: "f32[1104, 1]" = torch.ops.aten.unsqueeze.default(arg166_1, -1);  arg166_1 = None
    unsqueeze_669: "f32[1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_668, -1);  unsqueeze_668 = None
    mul_340: "f32[8, 1104, 8, 8]" = torch.ops.aten.mul.Tensor(mul_339, unsqueeze_669);  mul_339 = unsqueeze_669 = None
    unsqueeze_670: "f32[1104, 1]" = torch.ops.aten.unsqueeze.default(arg167_1, -1);  arg167_1 = None
    unsqueeze_671: "f32[1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_670, -1);  unsqueeze_670 = None
    add_279: "f32[8, 1104, 8, 8]" = torch.ops.aten.add.Tensor(mul_340, unsqueeze_671);  mul_340 = unsqueeze_671 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_280: "f32[8, 1104, 8, 8]" = torch.ops.aten.add.Tensor(add_279, 3)
    clamp_min_89: "f32[8, 1104, 8, 8]" = torch.ops.aten.clamp_min.default(add_280, 0);  add_280 = None
    clamp_max_89: "f32[8, 1104, 8, 8]" = torch.ops.aten.clamp_max.default(clamp_min_89, 6);  clamp_min_89 = None
    mul_341: "f32[8, 1104, 8, 8]" = torch.ops.aten.mul.Tensor(add_279, clamp_max_89);  add_279 = clamp_max_89 = None
    div_89: "f32[8, 1104, 8, 8]" = torch.ops.aten.div.Tensor(mul_341, 6);  mul_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_118: "f32[8, 1104, 8, 8]" = torch.ops.aten.convolution.default(div_89, arg328_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 1104);  div_89 = arg328_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_168: "f32[1104]" = torch.ops.prims.convert_element_type.default(arg504_1, torch.float32);  arg504_1 = None
    convert_element_type_169: "f32[1104]" = torch.ops.prims.convert_element_type.default(arg505_1, torch.float32);  arg505_1 = None
    add_281: "f32[1104]" = torch.ops.aten.add.Tensor(convert_element_type_169, 1e-05);  convert_element_type_169 = None
    sqrt_84: "f32[1104]" = torch.ops.aten.sqrt.default(add_281);  add_281 = None
    reciprocal_84: "f32[1104]" = torch.ops.aten.reciprocal.default(sqrt_84);  sqrt_84 = None
    mul_342: "f32[1104]" = torch.ops.aten.mul.Tensor(reciprocal_84, 1);  reciprocal_84 = None
    unsqueeze_672: "f32[1104, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_168, -1);  convert_element_type_168 = None
    unsqueeze_673: "f32[1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_672, -1);  unsqueeze_672 = None
    unsqueeze_674: "f32[1104, 1]" = torch.ops.aten.unsqueeze.default(mul_342, -1);  mul_342 = None
    unsqueeze_675: "f32[1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_674, -1);  unsqueeze_674 = None
    sub_84: "f32[8, 1104, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_118, unsqueeze_673);  convolution_118 = unsqueeze_673 = None
    mul_343: "f32[8, 1104, 8, 8]" = torch.ops.aten.mul.Tensor(sub_84, unsqueeze_675);  sub_84 = unsqueeze_675 = None
    unsqueeze_676: "f32[1104, 1]" = torch.ops.aten.unsqueeze.default(arg168_1, -1);  arg168_1 = None
    unsqueeze_677: "f32[1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_676, -1);  unsqueeze_676 = None
    mul_344: "f32[8, 1104, 8, 8]" = torch.ops.aten.mul.Tensor(mul_343, unsqueeze_677);  mul_343 = unsqueeze_677 = None
    unsqueeze_678: "f32[1104, 1]" = torch.ops.aten.unsqueeze.default(arg169_1, -1);  arg169_1 = None
    unsqueeze_679: "f32[1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_678, -1);  unsqueeze_678 = None
    add_282: "f32[8, 1104, 8, 8]" = torch.ops.aten.add.Tensor(mul_344, unsqueeze_679);  mul_344 = unsqueeze_679 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_283: "f32[8, 1104, 8, 8]" = torch.ops.aten.add.Tensor(add_282, 3)
    clamp_min_90: "f32[8, 1104, 8, 8]" = torch.ops.aten.clamp_min.default(add_283, 0);  add_283 = None
    clamp_max_90: "f32[8, 1104, 8, 8]" = torch.ops.aten.clamp_max.default(clamp_min_90, 6);  clamp_min_90 = None
    mul_345: "f32[8, 1104, 8, 8]" = torch.ops.aten.mul.Tensor(add_282, clamp_max_90);  add_282 = clamp_max_90 = None
    div_90: "f32[8, 1104, 8, 8]" = torch.ops.aten.div.Tensor(mul_345, 6);  mul_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_17: "f32[8, 1104, 1, 1]" = torch.ops.aten.mean.dim(div_90, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_119: "f32[8, 48, 1, 1]" = torch.ops.aten.convolution.default(mean_17, arg329_1, arg330_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_17 = arg329_1 = arg330_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    add_284: "f32[8, 48, 1, 1]" = torch.ops.aten.add.Tensor(convolution_119, 3)
    clamp_min_91: "f32[8, 48, 1, 1]" = torch.ops.aten.clamp_min.default(add_284, 0);  add_284 = None
    clamp_max_91: "f32[8, 48, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_91, 6);  clamp_min_91 = None
    mul_346: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_119, clamp_max_91);  convolution_119 = clamp_max_91 = None
    div_91: "f32[8, 48, 1, 1]" = torch.ops.aten.div.Tensor(mul_346, 6);  mul_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_120: "f32[8, 1104, 1, 1]" = torch.ops.aten.convolution.default(div_91, arg331_1, arg332_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_91 = arg331_1 = arg332_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    add_285: "f32[8, 1104, 1, 1]" = torch.ops.aten.add.Tensor(convolution_120, 3);  convolution_120 = None
    clamp_min_92: "f32[8, 1104, 1, 1]" = torch.ops.aten.clamp_min.default(add_285, 0);  add_285 = None
    clamp_max_92: "f32[8, 1104, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_92, 6);  clamp_min_92 = None
    div_92: "f32[8, 1104, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_92, 6);  clamp_max_92 = None
    mul_347: "f32[8, 1104, 8, 8]" = torch.ops.aten.mul.Tensor(div_90, div_92);  div_90 = div_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_121: "f32[8, 224, 8, 8]" = torch.ops.aten.convolution.default(mul_347, arg333_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_347 = arg333_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_170: "f32[224]" = torch.ops.prims.convert_element_type.default(arg506_1, torch.float32);  arg506_1 = None
    convert_element_type_171: "f32[224]" = torch.ops.prims.convert_element_type.default(arg507_1, torch.float32);  arg507_1 = None
    add_286: "f32[224]" = torch.ops.aten.add.Tensor(convert_element_type_171, 1e-05);  convert_element_type_171 = None
    sqrt_85: "f32[224]" = torch.ops.aten.sqrt.default(add_286);  add_286 = None
    reciprocal_85: "f32[224]" = torch.ops.aten.reciprocal.default(sqrt_85);  sqrt_85 = None
    mul_348: "f32[224]" = torch.ops.aten.mul.Tensor(reciprocal_85, 1);  reciprocal_85 = None
    unsqueeze_680: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_170, -1);  convert_element_type_170 = None
    unsqueeze_681: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_680, -1);  unsqueeze_680 = None
    unsqueeze_682: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(mul_348, -1);  mul_348 = None
    unsqueeze_683: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_682, -1);  unsqueeze_682 = None
    sub_85: "f32[8, 224, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_121, unsqueeze_681);  convolution_121 = unsqueeze_681 = None
    mul_349: "f32[8, 224, 8, 8]" = torch.ops.aten.mul.Tensor(sub_85, unsqueeze_683);  sub_85 = unsqueeze_683 = None
    unsqueeze_684: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(arg170_1, -1);  arg170_1 = None
    unsqueeze_685: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_684, -1);  unsqueeze_684 = None
    mul_350: "f32[8, 224, 8, 8]" = torch.ops.aten.mul.Tensor(mul_349, unsqueeze_685);  mul_349 = unsqueeze_685 = None
    unsqueeze_686: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(arg171_1, -1);  arg171_1 = None
    unsqueeze_687: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_686, -1);  unsqueeze_686 = None
    add_287: "f32[8, 224, 8, 8]" = torch.ops.aten.add.Tensor(mul_350, unsqueeze_687);  mul_350 = unsqueeze_687 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:82, code: x = self.conv(x)
    convolution_122: "f32[8, 1344, 8, 8]" = torch.ops.aten.convolution.default(add_287, arg334_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_287 = arg334_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_172: "f32[1344]" = torch.ops.prims.convert_element_type.default(arg508_1, torch.float32);  arg508_1 = None
    convert_element_type_173: "f32[1344]" = torch.ops.prims.convert_element_type.default(arg509_1, torch.float32);  arg509_1 = None
    add_288: "f32[1344]" = torch.ops.aten.add.Tensor(convert_element_type_173, 1e-05);  convert_element_type_173 = None
    sqrt_86: "f32[1344]" = torch.ops.aten.sqrt.default(add_288);  add_288 = None
    reciprocal_86: "f32[1344]" = torch.ops.aten.reciprocal.default(sqrt_86);  sqrt_86 = None
    mul_351: "f32[1344]" = torch.ops.aten.mul.Tensor(reciprocal_86, 1);  reciprocal_86 = None
    unsqueeze_688: "f32[1344, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_172, -1);  convert_element_type_172 = None
    unsqueeze_689: "f32[1344, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_688, -1);  unsqueeze_688 = None
    unsqueeze_690: "f32[1344, 1]" = torch.ops.aten.unsqueeze.default(mul_351, -1);  mul_351 = None
    unsqueeze_691: "f32[1344, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_690, -1);  unsqueeze_690 = None
    sub_86: "f32[8, 1344, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_122, unsqueeze_689);  convolution_122 = unsqueeze_689 = None
    mul_352: "f32[8, 1344, 8, 8]" = torch.ops.aten.mul.Tensor(sub_86, unsqueeze_691);  sub_86 = unsqueeze_691 = None
    unsqueeze_692: "f32[1344, 1]" = torch.ops.aten.unsqueeze.default(arg172_1, -1);  arg172_1 = None
    unsqueeze_693: "f32[1344, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_692, -1);  unsqueeze_692 = None
    mul_353: "f32[8, 1344, 8, 8]" = torch.ops.aten.mul.Tensor(mul_352, unsqueeze_693);  mul_352 = unsqueeze_693 = None
    unsqueeze_694: "f32[1344, 1]" = torch.ops.aten.unsqueeze.default(arg173_1, -1);  arg173_1 = None
    unsqueeze_695: "f32[1344, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_694, -1);  unsqueeze_694 = None
    add_289: "f32[8, 1344, 8, 8]" = torch.ops.aten.add.Tensor(mul_353, unsqueeze_695);  mul_353 = unsqueeze_695 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    add_290: "f32[8, 1344, 8, 8]" = torch.ops.aten.add.Tensor(add_289, 3)
    clamp_min_93: "f32[8, 1344, 8, 8]" = torch.ops.aten.clamp_min.default(add_290, 0);  add_290 = None
    clamp_max_93: "f32[8, 1344, 8, 8]" = torch.ops.aten.clamp_max.default(clamp_min_93, 6);  clamp_min_93 = None
    mul_354: "f32[8, 1344, 8, 8]" = torch.ops.aten.mul.Tensor(add_289, clamp_max_93);  add_289 = clamp_max_93 = None
    div_93: "f32[8, 1344, 8, 8]" = torch.ops.aten.div.Tensor(mul_354, 6);  mul_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean_18: "f32[8, 1344, 1, 1]" = torch.ops.aten.mean.dim(div_93, [-1, -2], True);  div_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilenetv3.py:145, code: x = self.conv_head(x)
    convolution_123: "f32[8, 1984, 1, 1]" = torch.ops.aten.convolution.default(mean_18, arg335_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_18 = arg335_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilenetv3.py:146, code: x = self.act2(x)
    add_291: "f32[8, 1984, 1, 1]" = torch.ops.aten.add.Tensor(convolution_123, 3)
    clamp_min_94: "f32[8, 1984, 1, 1]" = torch.ops.aten.clamp_min.default(add_291, 0);  add_291 = None
    clamp_max_94: "f32[8, 1984, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_94, 6);  clamp_min_94 = None
    mul_355: "f32[8, 1984, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_123, clamp_max_94);  convolution_123 = clamp_max_94 = None
    div_94: "f32[8, 1984, 1, 1]" = torch.ops.aten.div.Tensor(mul_355, 6);  mul_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/linear.py:19, code: return F.linear(input, self.weight, self.bias)
    permute: "f32[1984, 1000]" = torch.ops.aten.permute.default(arg174_1, [1, 0]);  arg174_1 = None
    view_1: "f32[8, 1984]" = torch.ops.aten.view.default(div_94, [8, 1984]);  div_94 = None
    addmm: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg175_1, view_1, permute);  arg175_1 = view_1 = permute = None
    return (addmm,)
    