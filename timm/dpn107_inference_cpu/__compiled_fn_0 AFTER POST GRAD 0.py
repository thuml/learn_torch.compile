from __future__ import annotations



def forward(self, arg0_1: "f32[128]", arg1_1: "f32[128]", arg2_1: "f32[128]", arg3_1: "f32[128]", arg4_1: "f32[128]", arg5_1: "f32[128]", arg6_1: "f32[200]", arg7_1: "f32[200]", arg8_1: "f32[200]", arg9_1: "f32[200]", arg10_1: "f32[316]", arg11_1: "f32[316]", arg12_1: "f32[200]", arg13_1: "f32[200]", arg14_1: "f32[200]", arg15_1: "f32[200]", arg16_1: "f32[336]", arg17_1: "f32[336]", arg18_1: "f32[200]", arg19_1: "f32[200]", arg20_1: "f32[200]", arg21_1: "f32[200]", arg22_1: "f32[356]", arg23_1: "f32[356]", arg24_1: "f32[200]", arg25_1: "f32[200]", arg26_1: "f32[200]", arg27_1: "f32[200]", arg28_1: "f32[376]", arg29_1: "f32[376]", arg30_1: "f32[376]", arg31_1: "f32[376]", arg32_1: "f32[400]", arg33_1: "f32[400]", arg34_1: "f32[400]", arg35_1: "f32[400]", arg36_1: "f32[704]", arg37_1: "f32[704]", arg38_1: "f32[400]", arg39_1: "f32[400]", arg40_1: "f32[400]", arg41_1: "f32[400]", arg42_1: "f32[768]", arg43_1: "f32[768]", arg44_1: "f32[400]", arg45_1: "f32[400]", arg46_1: "f32[400]", arg47_1: "f32[400]", arg48_1: "f32[832]", arg49_1: "f32[832]", arg50_1: "f32[400]", arg51_1: "f32[400]", arg52_1: "f32[400]", arg53_1: "f32[400]", arg54_1: "f32[896]", arg55_1: "f32[896]", arg56_1: "f32[400]", arg57_1: "f32[400]", arg58_1: "f32[400]", arg59_1: "f32[400]", arg60_1: "f32[960]", arg61_1: "f32[960]", arg62_1: "f32[400]", arg63_1: "f32[400]", arg64_1: "f32[400]", arg65_1: "f32[400]", arg66_1: "f32[1024]", arg67_1: "f32[1024]", arg68_1: "f32[400]", arg69_1: "f32[400]", arg70_1: "f32[400]", arg71_1: "f32[400]", arg72_1: "f32[1088]", arg73_1: "f32[1088]", arg74_1: "f32[400]", arg75_1: "f32[400]", arg76_1: "f32[400]", arg77_1: "f32[400]", arg78_1: "f32[1152]", arg79_1: "f32[1152]", arg80_1: "f32[1152]", arg81_1: "f32[1152]", arg82_1: "f32[800]", arg83_1: "f32[800]", arg84_1: "f32[800]", arg85_1: "f32[800]", arg86_1: "f32[1216]", arg87_1: "f32[1216]", arg88_1: "f32[800]", arg89_1: "f32[800]", arg90_1: "f32[800]", arg91_1: "f32[800]", arg92_1: "f32[1280]", arg93_1: "f32[1280]", arg94_1: "f32[800]", arg95_1: "f32[800]", arg96_1: "f32[800]", arg97_1: "f32[800]", arg98_1: "f32[1344]", arg99_1: "f32[1344]", arg100_1: "f32[800]", arg101_1: "f32[800]", arg102_1: "f32[800]", arg103_1: "f32[800]", arg104_1: "f32[1408]", arg105_1: "f32[1408]", arg106_1: "f32[800]", arg107_1: "f32[800]", arg108_1: "f32[800]", arg109_1: "f32[800]", arg110_1: "f32[1472]", arg111_1: "f32[1472]", arg112_1: "f32[800]", arg113_1: "f32[800]", arg114_1: "f32[800]", arg115_1: "f32[800]", arg116_1: "f32[1536]", arg117_1: "f32[1536]", arg118_1: "f32[800]", arg119_1: "f32[800]", arg120_1: "f32[800]", arg121_1: "f32[800]", arg122_1: "f32[1600]", arg123_1: "f32[1600]", arg124_1: "f32[800]", arg125_1: "f32[800]", arg126_1: "f32[800]", arg127_1: "f32[800]", arg128_1: "f32[1664]", arg129_1: "f32[1664]", arg130_1: "f32[800]", arg131_1: "f32[800]", arg132_1: "f32[800]", arg133_1: "f32[800]", arg134_1: "f32[1728]", arg135_1: "f32[1728]", arg136_1: "f32[800]", arg137_1: "f32[800]", arg138_1: "f32[800]", arg139_1: "f32[800]", arg140_1: "f32[1792]", arg141_1: "f32[1792]", arg142_1: "f32[800]", arg143_1: "f32[800]", arg144_1: "f32[800]", arg145_1: "f32[800]", arg146_1: "f32[1856]", arg147_1: "f32[1856]", arg148_1: "f32[800]", arg149_1: "f32[800]", arg150_1: "f32[800]", arg151_1: "f32[800]", arg152_1: "f32[1920]", arg153_1: "f32[1920]", arg154_1: "f32[800]", arg155_1: "f32[800]", arg156_1: "f32[800]", arg157_1: "f32[800]", arg158_1: "f32[1984]", arg159_1: "f32[1984]", arg160_1: "f32[800]", arg161_1: "f32[800]", arg162_1: "f32[800]", arg163_1: "f32[800]", arg164_1: "f32[2048]", arg165_1: "f32[2048]", arg166_1: "f32[800]", arg167_1: "f32[800]", arg168_1: "f32[800]", arg169_1: "f32[800]", arg170_1: "f32[2112]", arg171_1: "f32[2112]", arg172_1: "f32[800]", arg173_1: "f32[800]", arg174_1: "f32[800]", arg175_1: "f32[800]", arg176_1: "f32[2176]", arg177_1: "f32[2176]", arg178_1: "f32[800]", arg179_1: "f32[800]", arg180_1: "f32[800]", arg181_1: "f32[800]", arg182_1: "f32[2240]", arg183_1: "f32[2240]", arg184_1: "f32[800]", arg185_1: "f32[800]", arg186_1: "f32[800]", arg187_1: "f32[800]", arg188_1: "f32[2304]", arg189_1: "f32[2304]", arg190_1: "f32[800]", arg191_1: "f32[800]", arg192_1: "f32[800]", arg193_1: "f32[800]", arg194_1: "f32[2368]", arg195_1: "f32[2368]", arg196_1: "f32[800]", arg197_1: "f32[800]", arg198_1: "f32[800]", arg199_1: "f32[800]", arg200_1: "f32[2432]", arg201_1: "f32[2432]", arg202_1: "f32[2432]", arg203_1: "f32[2432]", arg204_1: "f32[1600]", arg205_1: "f32[1600]", arg206_1: "f32[1600]", arg207_1: "f32[1600]", arg208_1: "f32[2432]", arg209_1: "f32[2432]", arg210_1: "f32[1600]", arg211_1: "f32[1600]", arg212_1: "f32[1600]", arg213_1: "f32[1600]", arg214_1: "f32[2560]", arg215_1: "f32[2560]", arg216_1: "f32[1600]", arg217_1: "f32[1600]", arg218_1: "f32[1600]", arg219_1: "f32[1600]", arg220_1: "f32[2688]", arg221_1: "f32[2688]", arg222_1: "f32[128, 3, 7, 7]", arg223_1: "f32[296, 128, 1, 1]", arg224_1: "f32[200, 128, 1, 1]", arg225_1: "f32[200, 4, 3, 3]", arg226_1: "f32[276, 200, 1, 1]", arg227_1: "f32[200, 316, 1, 1]", arg228_1: "f32[200, 4, 3, 3]", arg229_1: "f32[276, 200, 1, 1]", arg230_1: "f32[200, 336, 1, 1]", arg231_1: "f32[200, 4, 3, 3]", arg232_1: "f32[276, 200, 1, 1]", arg233_1: "f32[200, 356, 1, 1]", arg234_1: "f32[200, 4, 3, 3]", arg235_1: "f32[276, 200, 1, 1]", arg236_1: "f32[640, 376, 1, 1]", arg237_1: "f32[400, 376, 1, 1]", arg238_1: "f32[400, 8, 3, 3]", arg239_1: "f32[576, 400, 1, 1]", arg240_1: "f32[400, 704, 1, 1]", arg241_1: "f32[400, 8, 3, 3]", arg242_1: "f32[576, 400, 1, 1]", arg243_1: "f32[400, 768, 1, 1]", arg244_1: "f32[400, 8, 3, 3]", arg245_1: "f32[576, 400, 1, 1]", arg246_1: "f32[400, 832, 1, 1]", arg247_1: "f32[400, 8, 3, 3]", arg248_1: "f32[576, 400, 1, 1]", arg249_1: "f32[400, 896, 1, 1]", arg250_1: "f32[400, 8, 3, 3]", arg251_1: "f32[576, 400, 1, 1]", arg252_1: "f32[400, 960, 1, 1]", arg253_1: "f32[400, 8, 3, 3]", arg254_1: "f32[576, 400, 1, 1]", arg255_1: "f32[400, 1024, 1, 1]", arg256_1: "f32[400, 8, 3, 3]", arg257_1: "f32[576, 400, 1, 1]", arg258_1: "f32[400, 1088, 1, 1]", arg259_1: "f32[400, 8, 3, 3]", arg260_1: "f32[576, 400, 1, 1]", arg261_1: "f32[1152, 1152, 1, 1]", arg262_1: "f32[800, 1152, 1, 1]", arg263_1: "f32[800, 16, 3, 3]", arg264_1: "f32[1088, 800, 1, 1]", arg265_1: "f32[800, 1216, 1, 1]", arg266_1: "f32[800, 16, 3, 3]", arg267_1: "f32[1088, 800, 1, 1]", arg268_1: "f32[800, 1280, 1, 1]", arg269_1: "f32[800, 16, 3, 3]", arg270_1: "f32[1088, 800, 1, 1]", arg271_1: "f32[800, 1344, 1, 1]", arg272_1: "f32[800, 16, 3, 3]", arg273_1: "f32[1088, 800, 1, 1]", arg274_1: "f32[800, 1408, 1, 1]", arg275_1: "f32[800, 16, 3, 3]", arg276_1: "f32[1088, 800, 1, 1]", arg277_1: "f32[800, 1472, 1, 1]", arg278_1: "f32[800, 16, 3, 3]", arg279_1: "f32[1088, 800, 1, 1]", arg280_1: "f32[800, 1536, 1, 1]", arg281_1: "f32[800, 16, 3, 3]", arg282_1: "f32[1088, 800, 1, 1]", arg283_1: "f32[800, 1600, 1, 1]", arg284_1: "f32[800, 16, 3, 3]", arg285_1: "f32[1088, 800, 1, 1]", arg286_1: "f32[800, 1664, 1, 1]", arg287_1: "f32[800, 16, 3, 3]", arg288_1: "f32[1088, 800, 1, 1]", arg289_1: "f32[800, 1728, 1, 1]", arg290_1: "f32[800, 16, 3, 3]", arg291_1: "f32[1088, 800, 1, 1]", arg292_1: "f32[800, 1792, 1, 1]", arg293_1: "f32[800, 16, 3, 3]", arg294_1: "f32[1088, 800, 1, 1]", arg295_1: "f32[800, 1856, 1, 1]", arg296_1: "f32[800, 16, 3, 3]", arg297_1: "f32[1088, 800, 1, 1]", arg298_1: "f32[800, 1920, 1, 1]", arg299_1: "f32[800, 16, 3, 3]", arg300_1: "f32[1088, 800, 1, 1]", arg301_1: "f32[800, 1984, 1, 1]", arg302_1: "f32[800, 16, 3, 3]", arg303_1: "f32[1088, 800, 1, 1]", arg304_1: "f32[800, 2048, 1, 1]", arg305_1: "f32[800, 16, 3, 3]", arg306_1: "f32[1088, 800, 1, 1]", arg307_1: "f32[800, 2112, 1, 1]", arg308_1: "f32[800, 16, 3, 3]", arg309_1: "f32[1088, 800, 1, 1]", arg310_1: "f32[800, 2176, 1, 1]", arg311_1: "f32[800, 16, 3, 3]", arg312_1: "f32[1088, 800, 1, 1]", arg313_1: "f32[800, 2240, 1, 1]", arg314_1: "f32[800, 16, 3, 3]", arg315_1: "f32[1088, 800, 1, 1]", arg316_1: "f32[800, 2304, 1, 1]", arg317_1: "f32[800, 16, 3, 3]", arg318_1: "f32[1088, 800, 1, 1]", arg319_1: "f32[800, 2368, 1, 1]", arg320_1: "f32[800, 16, 3, 3]", arg321_1: "f32[1088, 800, 1, 1]", arg322_1: "f32[2304, 2432, 1, 1]", arg323_1: "f32[1600, 2432, 1, 1]", arg324_1: "f32[1600, 32, 3, 3]", arg325_1: "f32[2176, 1600, 1, 1]", arg326_1: "f32[1600, 2432, 1, 1]", arg327_1: "f32[1600, 32, 3, 3]", arg328_1: "f32[2176, 1600, 1, 1]", arg329_1: "f32[1600, 2560, 1, 1]", arg330_1: "f32[1600, 32, 3, 3]", arg331_1: "f32[2176, 1600, 1, 1]", arg332_1: "f32[1000, 2688, 1, 1]", arg333_1: "f32[1000]", arg334_1: "f32[128]", arg335_1: "f32[128]", arg336_1: "f32[128]", arg337_1: "f32[128]", arg338_1: "f32[128]", arg339_1: "f32[128]", arg340_1: "f32[200]", arg341_1: "f32[200]", arg342_1: "f32[200]", arg343_1: "f32[200]", arg344_1: "f32[316]", arg345_1: "f32[316]", arg346_1: "f32[200]", arg347_1: "f32[200]", arg348_1: "f32[200]", arg349_1: "f32[200]", arg350_1: "f32[336]", arg351_1: "f32[336]", arg352_1: "f32[200]", arg353_1: "f32[200]", arg354_1: "f32[200]", arg355_1: "f32[200]", arg356_1: "f32[356]", arg357_1: "f32[356]", arg358_1: "f32[200]", arg359_1: "f32[200]", arg360_1: "f32[200]", arg361_1: "f32[200]", arg362_1: "f32[376]", arg363_1: "f32[376]", arg364_1: "f32[376]", arg365_1: "f32[376]", arg366_1: "f32[400]", arg367_1: "f32[400]", arg368_1: "f32[400]", arg369_1: "f32[400]", arg370_1: "f32[704]", arg371_1: "f32[704]", arg372_1: "f32[400]", arg373_1: "f32[400]", arg374_1: "f32[400]", arg375_1: "f32[400]", arg376_1: "f32[768]", arg377_1: "f32[768]", arg378_1: "f32[400]", arg379_1: "f32[400]", arg380_1: "f32[400]", arg381_1: "f32[400]", arg382_1: "f32[832]", arg383_1: "f32[832]", arg384_1: "f32[400]", arg385_1: "f32[400]", arg386_1: "f32[400]", arg387_1: "f32[400]", arg388_1: "f32[896]", arg389_1: "f32[896]", arg390_1: "f32[400]", arg391_1: "f32[400]", arg392_1: "f32[400]", arg393_1: "f32[400]", arg394_1: "f32[960]", arg395_1: "f32[960]", arg396_1: "f32[400]", arg397_1: "f32[400]", arg398_1: "f32[400]", arg399_1: "f32[400]", arg400_1: "f32[1024]", arg401_1: "f32[1024]", arg402_1: "f32[400]", arg403_1: "f32[400]", arg404_1: "f32[400]", arg405_1: "f32[400]", arg406_1: "f32[1088]", arg407_1: "f32[1088]", arg408_1: "f32[400]", arg409_1: "f32[400]", arg410_1: "f32[400]", arg411_1: "f32[400]", arg412_1: "f32[1152]", arg413_1: "f32[1152]", arg414_1: "f32[1152]", arg415_1: "f32[1152]", arg416_1: "f32[800]", arg417_1: "f32[800]", arg418_1: "f32[800]", arg419_1: "f32[800]", arg420_1: "f32[1216]", arg421_1: "f32[1216]", arg422_1: "f32[800]", arg423_1: "f32[800]", arg424_1: "f32[800]", arg425_1: "f32[800]", arg426_1: "f32[1280]", arg427_1: "f32[1280]", arg428_1: "f32[800]", arg429_1: "f32[800]", arg430_1: "f32[800]", arg431_1: "f32[800]", arg432_1: "f32[1344]", arg433_1: "f32[1344]", arg434_1: "f32[800]", arg435_1: "f32[800]", arg436_1: "f32[800]", arg437_1: "f32[800]", arg438_1: "f32[1408]", arg439_1: "f32[1408]", arg440_1: "f32[800]", arg441_1: "f32[800]", arg442_1: "f32[800]", arg443_1: "f32[800]", arg444_1: "f32[1472]", arg445_1: "f32[1472]", arg446_1: "f32[800]", arg447_1: "f32[800]", arg448_1: "f32[800]", arg449_1: "f32[800]", arg450_1: "f32[1536]", arg451_1: "f32[1536]", arg452_1: "f32[800]", arg453_1: "f32[800]", arg454_1: "f32[800]", arg455_1: "f32[800]", arg456_1: "f32[1600]", arg457_1: "f32[1600]", arg458_1: "f32[800]", arg459_1: "f32[800]", arg460_1: "f32[800]", arg461_1: "f32[800]", arg462_1: "f32[1664]", arg463_1: "f32[1664]", arg464_1: "f32[800]", arg465_1: "f32[800]", arg466_1: "f32[800]", arg467_1: "f32[800]", arg468_1: "f32[1728]", arg469_1: "f32[1728]", arg470_1: "f32[800]", arg471_1: "f32[800]", arg472_1: "f32[800]", arg473_1: "f32[800]", arg474_1: "f32[1792]", arg475_1: "f32[1792]", arg476_1: "f32[800]", arg477_1: "f32[800]", arg478_1: "f32[800]", arg479_1: "f32[800]", arg480_1: "f32[1856]", arg481_1: "f32[1856]", arg482_1: "f32[800]", arg483_1: "f32[800]", arg484_1: "f32[800]", arg485_1: "f32[800]", arg486_1: "f32[1920]", arg487_1: "f32[1920]", arg488_1: "f32[800]", arg489_1: "f32[800]", arg490_1: "f32[800]", arg491_1: "f32[800]", arg492_1: "f32[1984]", arg493_1: "f32[1984]", arg494_1: "f32[800]", arg495_1: "f32[800]", arg496_1: "f32[800]", arg497_1: "f32[800]", arg498_1: "f32[2048]", arg499_1: "f32[2048]", arg500_1: "f32[800]", arg501_1: "f32[800]", arg502_1: "f32[800]", arg503_1: "f32[800]", arg504_1: "f32[2112]", arg505_1: "f32[2112]", arg506_1: "f32[800]", arg507_1: "f32[800]", arg508_1: "f32[800]", arg509_1: "f32[800]", arg510_1: "f32[2176]", arg511_1: "f32[2176]", arg512_1: "f32[800]", arg513_1: "f32[800]", arg514_1: "f32[800]", arg515_1: "f32[800]", arg516_1: "f32[2240]", arg517_1: "f32[2240]", arg518_1: "f32[800]", arg519_1: "f32[800]", arg520_1: "f32[800]", arg521_1: "f32[800]", arg522_1: "f32[2304]", arg523_1: "f32[2304]", arg524_1: "f32[800]", arg525_1: "f32[800]", arg526_1: "f32[800]", arg527_1: "f32[800]", arg528_1: "f32[2368]", arg529_1: "f32[2368]", arg530_1: "f32[800]", arg531_1: "f32[800]", arg532_1: "f32[800]", arg533_1: "f32[800]", arg534_1: "f32[2432]", arg535_1: "f32[2432]", arg536_1: "f32[2432]", arg537_1: "f32[2432]", arg538_1: "f32[1600]", arg539_1: "f32[1600]", arg540_1: "f32[1600]", arg541_1: "f32[1600]", arg542_1: "f32[2432]", arg543_1: "f32[2432]", arg544_1: "f32[1600]", arg545_1: "f32[1600]", arg546_1: "f32[1600]", arg547_1: "f32[1600]", arg548_1: "f32[2560]", arg549_1: "f32[2560]", arg550_1: "f32[1600]", arg551_1: "f32[1600]", arg552_1: "f32[1600]", arg553_1: "f32[1600]", arg554_1: "f32[2688]", arg555_1: "f32[2688]", arg556_1: "f32[8, 3, 224, 224]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution: "f32[8, 128, 112, 112]" = torch.ops.aten.convolution.default(arg556_1, arg222_1, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1);  arg556_1 = arg222_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg334_1, -1);  arg334_1 = None
    unsqueeze_1: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    sub: "f32[8, 128, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1);  convolution = unsqueeze_1 = None
    add: "f32[128]" = torch.ops.aten.add.Tensor(arg335_1, 0.001);  arg335_1 = None
    sqrt: "f32[128]" = torch.ops.aten.sqrt.default(add);  add = None
    reciprocal: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt);  sqrt = None
    mul: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
    unsqueeze_2: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul, -1);  mul = None
    unsqueeze_3: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    mul_1: "f32[8, 128, 112, 112]" = torch.ops.aten.mul.Tensor(sub, unsqueeze_3);  sub = unsqueeze_3 = None
    unsqueeze_4: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg0_1, -1);  arg0_1 = None
    unsqueeze_5: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_2: "f32[8, 128, 112, 112]" = torch.ops.aten.mul.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None
    unsqueeze_6: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg1_1, -1);  arg1_1 = None
    unsqueeze_7: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_1: "f32[8, 128, 112, 112]" = torch.ops.aten.add.Tensor(mul_2, unsqueeze_7);  mul_2 = unsqueeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu: "f32[8, 128, 112, 112]" = torch.ops.aten.relu.default(add_1);  add_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:266, code: return self.features(x)
    max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices.default(relu, [3, 3], [2, 2], [1, 1]);  relu = None
    getitem: "f32[8, 128, 56, 56]" = max_pool2d_with_indices[0];  max_pool2d_with_indices = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_8: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg336_1, -1);  arg336_1 = None
    unsqueeze_9: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    sub_1: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(getitem, unsqueeze_9);  unsqueeze_9 = None
    add_2: "f32[128]" = torch.ops.aten.add.Tensor(arg337_1, 0.001);  arg337_1 = None
    sqrt_1: "f32[128]" = torch.ops.aten.sqrt.default(add_2);  add_2 = None
    reciprocal_1: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_1);  sqrt_1 = None
    mul_3: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_1, 1);  reciprocal_1 = None
    unsqueeze_10: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_3, -1);  mul_3 = None
    unsqueeze_11: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    mul_4: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_1, unsqueeze_11);  sub_1 = unsqueeze_11 = None
    unsqueeze_12: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
    unsqueeze_13: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_5: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(mul_4, unsqueeze_13);  mul_4 = unsqueeze_13 = None
    unsqueeze_14: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, -1);  arg3_1 = None
    unsqueeze_15: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_3: "f32[8, 128, 56, 56]" = torch.ops.aten.add.Tensor(mul_5, unsqueeze_15);  mul_5 = unsqueeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_1: "f32[8, 128, 56, 56]" = torch.ops.aten.relu.default(add_3);  add_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_1: "f32[8, 296, 56, 56]" = torch.ops.aten.convolution.default(relu_1, arg223_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_1 = arg223_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:133, code: x_s1 = x_s[:, :self.num_1x1_c, :, :]
    slice_2: "f32[8, 256, 56, 56]" = torch.ops.aten.slice.Tensor(convolution_1, 1, 0, 256)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_16: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg338_1, -1);  arg338_1 = None
    unsqueeze_17: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    sub_2: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(getitem, unsqueeze_17);  getitem = unsqueeze_17 = None
    add_4: "f32[128]" = torch.ops.aten.add.Tensor(arg339_1, 0.001);  arg339_1 = None
    sqrt_2: "f32[128]" = torch.ops.aten.sqrt.default(add_4);  add_4 = None
    reciprocal_2: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_2);  sqrt_2 = None
    mul_6: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_2, 1);  reciprocal_2 = None
    unsqueeze_18: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_6, -1);  mul_6 = None
    unsqueeze_19: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    mul_7: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_2, unsqueeze_19);  sub_2 = unsqueeze_19 = None
    unsqueeze_20: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
    unsqueeze_21: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_8: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_21);  mul_7 = unsqueeze_21 = None
    unsqueeze_22: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
    unsqueeze_23: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_5: "f32[8, 128, 56, 56]" = torch.ops.aten.add.Tensor(mul_8, unsqueeze_23);  mul_8 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_2: "f32[8, 128, 56, 56]" = torch.ops.aten.relu.default(add_5);  add_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_2: "f32[8, 200, 56, 56]" = torch.ops.aten.convolution.default(relu_2, arg224_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_2 = arg224_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_24: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg340_1, -1);  arg340_1 = None
    unsqueeze_25: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
    sub_3: "f32[8, 200, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_25);  convolution_2 = unsqueeze_25 = None
    add_6: "f32[200]" = torch.ops.aten.add.Tensor(arg341_1, 0.001);  arg341_1 = None
    sqrt_3: "f32[200]" = torch.ops.aten.sqrt.default(add_6);  add_6 = None
    reciprocal_3: "f32[200]" = torch.ops.aten.reciprocal.default(sqrt_3);  sqrt_3 = None
    mul_9: "f32[200]" = torch.ops.aten.mul.Tensor(reciprocal_3, 1);  reciprocal_3 = None
    unsqueeze_26: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(mul_9, -1);  mul_9 = None
    unsqueeze_27: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
    mul_10: "f32[8, 200, 56, 56]" = torch.ops.aten.mul.Tensor(sub_3, unsqueeze_27);  sub_3 = unsqueeze_27 = None
    unsqueeze_28: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg6_1, -1);  arg6_1 = None
    unsqueeze_29: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    mul_11: "f32[8, 200, 56, 56]" = torch.ops.aten.mul.Tensor(mul_10, unsqueeze_29);  mul_10 = unsqueeze_29 = None
    unsqueeze_30: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
    unsqueeze_31: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
    add_7: "f32[8, 200, 56, 56]" = torch.ops.aten.add.Tensor(mul_11, unsqueeze_31);  mul_11 = unsqueeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_3: "f32[8, 200, 56, 56]" = torch.ops.aten.relu.default(add_7);  add_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_3: "f32[8, 200, 56, 56]" = torch.ops.aten.convolution.default(relu_3, arg225_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_3 = arg225_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_32: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg342_1, -1);  arg342_1 = None
    unsqueeze_33: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    sub_4: "f32[8, 200, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_33);  convolution_3 = unsqueeze_33 = None
    add_8: "f32[200]" = torch.ops.aten.add.Tensor(arg343_1, 0.001);  arg343_1 = None
    sqrt_4: "f32[200]" = torch.ops.aten.sqrt.default(add_8);  add_8 = None
    reciprocal_4: "f32[200]" = torch.ops.aten.reciprocal.default(sqrt_4);  sqrt_4 = None
    mul_12: "f32[200]" = torch.ops.aten.mul.Tensor(reciprocal_4, 1);  reciprocal_4 = None
    unsqueeze_34: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(mul_12, -1);  mul_12 = None
    unsqueeze_35: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
    mul_13: "f32[8, 200, 56, 56]" = torch.ops.aten.mul.Tensor(sub_4, unsqueeze_35);  sub_4 = unsqueeze_35 = None
    unsqueeze_36: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg8_1, -1);  arg8_1 = None
    unsqueeze_37: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
    mul_14: "f32[8, 200, 56, 56]" = torch.ops.aten.mul.Tensor(mul_13, unsqueeze_37);  mul_13 = unsqueeze_37 = None
    unsqueeze_38: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
    unsqueeze_39: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
    add_9: "f32[8, 200, 56, 56]" = torch.ops.aten.add.Tensor(mul_14, unsqueeze_39);  mul_14 = unsqueeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_4: "f32[8, 200, 56, 56]" = torch.ops.aten.relu.default(add_9);  add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_4: "f32[8, 276, 56, 56]" = torch.ops.aten.convolution.default(relu_4, arg226_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_4 = arg226_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_10: "f32[8, 256, 56, 56]" = torch.ops.aten.slice.Tensor(convolution_4, 1, 0, 256)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    add_10: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(slice_2, slice_10);  slice_2 = slice_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:134, code: x_s2 = x_s[:, self.num_1x1_c:, :, :]
    slice_6: "f32[8, 40, 56, 56]" = torch.ops.aten.slice.Tensor(convolution_1, 1, 256, 9223372036854775807);  convolution_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_14: "f32[8, 20, 56, 56]" = torch.ops.aten.slice.Tensor(convolution_4, 1, 256, 9223372036854775807);  convolution_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    cat: "f32[8, 60, 56, 56]" = torch.ops.aten.cat.default([slice_6, slice_14], 1);  slice_6 = slice_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    cat_1: "f32[8, 316, 56, 56]" = torch.ops.aten.cat.default([add_10, cat], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_40: "f32[316, 1]" = torch.ops.aten.unsqueeze.default(arg344_1, -1);  arg344_1 = None
    unsqueeze_41: "f32[316, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    sub_5: "f32[8, 316, 56, 56]" = torch.ops.aten.sub.Tensor(cat_1, unsqueeze_41);  cat_1 = unsqueeze_41 = None
    add_11: "f32[316]" = torch.ops.aten.add.Tensor(arg345_1, 0.001);  arg345_1 = None
    sqrt_5: "f32[316]" = torch.ops.aten.sqrt.default(add_11);  add_11 = None
    reciprocal_5: "f32[316]" = torch.ops.aten.reciprocal.default(sqrt_5);  sqrt_5 = None
    mul_15: "f32[316]" = torch.ops.aten.mul.Tensor(reciprocal_5, 1);  reciprocal_5 = None
    unsqueeze_42: "f32[316, 1]" = torch.ops.aten.unsqueeze.default(mul_15, -1);  mul_15 = None
    unsqueeze_43: "f32[316, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    mul_16: "f32[8, 316, 56, 56]" = torch.ops.aten.mul.Tensor(sub_5, unsqueeze_43);  sub_5 = unsqueeze_43 = None
    unsqueeze_44: "f32[316, 1]" = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
    unsqueeze_45: "f32[316, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    mul_17: "f32[8, 316, 56, 56]" = torch.ops.aten.mul.Tensor(mul_16, unsqueeze_45);  mul_16 = unsqueeze_45 = None
    unsqueeze_46: "f32[316, 1]" = torch.ops.aten.unsqueeze.default(arg11_1, -1);  arg11_1 = None
    unsqueeze_47: "f32[316, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
    add_12: "f32[8, 316, 56, 56]" = torch.ops.aten.add.Tensor(mul_17, unsqueeze_47);  mul_17 = unsqueeze_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_5: "f32[8, 316, 56, 56]" = torch.ops.aten.relu.default(add_12);  add_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_5: "f32[8, 200, 56, 56]" = torch.ops.aten.convolution.default(relu_5, arg227_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_5 = arg227_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_48: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg346_1, -1);  arg346_1 = None
    unsqueeze_49: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
    sub_6: "f32[8, 200, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_49);  convolution_5 = unsqueeze_49 = None
    add_13: "f32[200]" = torch.ops.aten.add.Tensor(arg347_1, 0.001);  arg347_1 = None
    sqrt_6: "f32[200]" = torch.ops.aten.sqrt.default(add_13);  add_13 = None
    reciprocal_6: "f32[200]" = torch.ops.aten.reciprocal.default(sqrt_6);  sqrt_6 = None
    mul_18: "f32[200]" = torch.ops.aten.mul.Tensor(reciprocal_6, 1);  reciprocal_6 = None
    unsqueeze_50: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(mul_18, -1);  mul_18 = None
    unsqueeze_51: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
    mul_19: "f32[8, 200, 56, 56]" = torch.ops.aten.mul.Tensor(sub_6, unsqueeze_51);  sub_6 = unsqueeze_51 = None
    unsqueeze_52: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg12_1, -1);  arg12_1 = None
    unsqueeze_53: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    mul_20: "f32[8, 200, 56, 56]" = torch.ops.aten.mul.Tensor(mul_19, unsqueeze_53);  mul_19 = unsqueeze_53 = None
    unsqueeze_54: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg13_1, -1);  arg13_1 = None
    unsqueeze_55: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
    add_14: "f32[8, 200, 56, 56]" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_55);  mul_20 = unsqueeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_6: "f32[8, 200, 56, 56]" = torch.ops.aten.relu.default(add_14);  add_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_6: "f32[8, 200, 56, 56]" = torch.ops.aten.convolution.default(relu_6, arg228_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_6 = arg228_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_56: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg348_1, -1);  arg348_1 = None
    unsqueeze_57: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    sub_7: "f32[8, 200, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_57);  convolution_6 = unsqueeze_57 = None
    add_15: "f32[200]" = torch.ops.aten.add.Tensor(arg349_1, 0.001);  arg349_1 = None
    sqrt_7: "f32[200]" = torch.ops.aten.sqrt.default(add_15);  add_15 = None
    reciprocal_7: "f32[200]" = torch.ops.aten.reciprocal.default(sqrt_7);  sqrt_7 = None
    mul_21: "f32[200]" = torch.ops.aten.mul.Tensor(reciprocal_7, 1);  reciprocal_7 = None
    unsqueeze_58: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(mul_21, -1);  mul_21 = None
    unsqueeze_59: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
    mul_22: "f32[8, 200, 56, 56]" = torch.ops.aten.mul.Tensor(sub_7, unsqueeze_59);  sub_7 = unsqueeze_59 = None
    unsqueeze_60: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
    unsqueeze_61: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
    mul_23: "f32[8, 200, 56, 56]" = torch.ops.aten.mul.Tensor(mul_22, unsqueeze_61);  mul_22 = unsqueeze_61 = None
    unsqueeze_62: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
    unsqueeze_63: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
    add_16: "f32[8, 200, 56, 56]" = torch.ops.aten.add.Tensor(mul_23, unsqueeze_63);  mul_23 = unsqueeze_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_7: "f32[8, 200, 56, 56]" = torch.ops.aten.relu.default(add_16);  add_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_7: "f32[8, 276, 56, 56]" = torch.ops.aten.convolution.default(relu_7, arg229_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_7 = arg229_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_18: "f32[8, 256, 56, 56]" = torch.ops.aten.slice.Tensor(convolution_7, 1, 0, 256)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    add_17: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(add_10, slice_18);  add_10 = slice_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_22: "f32[8, 20, 56, 56]" = torch.ops.aten.slice.Tensor(convolution_7, 1, 256, 9223372036854775807);  convolution_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    cat_2: "f32[8, 80, 56, 56]" = torch.ops.aten.cat.default([cat, slice_22], 1);  cat = slice_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    cat_3: "f32[8, 336, 56, 56]" = torch.ops.aten.cat.default([add_17, cat_2], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_64: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg350_1, -1);  arg350_1 = None
    unsqueeze_65: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
    sub_8: "f32[8, 336, 56, 56]" = torch.ops.aten.sub.Tensor(cat_3, unsqueeze_65);  cat_3 = unsqueeze_65 = None
    add_18: "f32[336]" = torch.ops.aten.add.Tensor(arg351_1, 0.001);  arg351_1 = None
    sqrt_8: "f32[336]" = torch.ops.aten.sqrt.default(add_18);  add_18 = None
    reciprocal_8: "f32[336]" = torch.ops.aten.reciprocal.default(sqrt_8);  sqrt_8 = None
    mul_24: "f32[336]" = torch.ops.aten.mul.Tensor(reciprocal_8, 1);  reciprocal_8 = None
    unsqueeze_66: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(mul_24, -1);  mul_24 = None
    unsqueeze_67: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
    mul_25: "f32[8, 336, 56, 56]" = torch.ops.aten.mul.Tensor(sub_8, unsqueeze_67);  sub_8 = unsqueeze_67 = None
    unsqueeze_68: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg16_1, -1);  arg16_1 = None
    unsqueeze_69: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    mul_26: "f32[8, 336, 56, 56]" = torch.ops.aten.mul.Tensor(mul_25, unsqueeze_69);  mul_25 = unsqueeze_69 = None
    unsqueeze_70: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg17_1, -1);  arg17_1 = None
    unsqueeze_71: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    add_19: "f32[8, 336, 56, 56]" = torch.ops.aten.add.Tensor(mul_26, unsqueeze_71);  mul_26 = unsqueeze_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_8: "f32[8, 336, 56, 56]" = torch.ops.aten.relu.default(add_19);  add_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_8: "f32[8, 200, 56, 56]" = torch.ops.aten.convolution.default(relu_8, arg230_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_8 = arg230_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_72: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg352_1, -1);  arg352_1 = None
    unsqueeze_73: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
    sub_9: "f32[8, 200, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_73);  convolution_8 = unsqueeze_73 = None
    add_20: "f32[200]" = torch.ops.aten.add.Tensor(arg353_1, 0.001);  arg353_1 = None
    sqrt_9: "f32[200]" = torch.ops.aten.sqrt.default(add_20);  add_20 = None
    reciprocal_9: "f32[200]" = torch.ops.aten.reciprocal.default(sqrt_9);  sqrt_9 = None
    mul_27: "f32[200]" = torch.ops.aten.mul.Tensor(reciprocal_9, 1);  reciprocal_9 = None
    unsqueeze_74: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(mul_27, -1);  mul_27 = None
    unsqueeze_75: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
    mul_28: "f32[8, 200, 56, 56]" = torch.ops.aten.mul.Tensor(sub_9, unsqueeze_75);  sub_9 = unsqueeze_75 = None
    unsqueeze_76: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg18_1, -1);  arg18_1 = None
    unsqueeze_77: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
    mul_29: "f32[8, 200, 56, 56]" = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_77);  mul_28 = unsqueeze_77 = None
    unsqueeze_78: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg19_1, -1);  arg19_1 = None
    unsqueeze_79: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    add_21: "f32[8, 200, 56, 56]" = torch.ops.aten.add.Tensor(mul_29, unsqueeze_79);  mul_29 = unsqueeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_9: "f32[8, 200, 56, 56]" = torch.ops.aten.relu.default(add_21);  add_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_9: "f32[8, 200, 56, 56]" = torch.ops.aten.convolution.default(relu_9, arg231_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_9 = arg231_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_80: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg354_1, -1);  arg354_1 = None
    unsqueeze_81: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
    sub_10: "f32[8, 200, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_81);  convolution_9 = unsqueeze_81 = None
    add_22: "f32[200]" = torch.ops.aten.add.Tensor(arg355_1, 0.001);  arg355_1 = None
    sqrt_10: "f32[200]" = torch.ops.aten.sqrt.default(add_22);  add_22 = None
    reciprocal_10: "f32[200]" = torch.ops.aten.reciprocal.default(sqrt_10);  sqrt_10 = None
    mul_30: "f32[200]" = torch.ops.aten.mul.Tensor(reciprocal_10, 1);  reciprocal_10 = None
    unsqueeze_82: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(mul_30, -1);  mul_30 = None
    unsqueeze_83: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    mul_31: "f32[8, 200, 56, 56]" = torch.ops.aten.mul.Tensor(sub_10, unsqueeze_83);  sub_10 = unsqueeze_83 = None
    unsqueeze_84: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg20_1, -1);  arg20_1 = None
    unsqueeze_85: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    mul_32: "f32[8, 200, 56, 56]" = torch.ops.aten.mul.Tensor(mul_31, unsqueeze_85);  mul_31 = unsqueeze_85 = None
    unsqueeze_86: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg21_1, -1);  arg21_1 = None
    unsqueeze_87: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
    add_23: "f32[8, 200, 56, 56]" = torch.ops.aten.add.Tensor(mul_32, unsqueeze_87);  mul_32 = unsqueeze_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_10: "f32[8, 200, 56, 56]" = torch.ops.aten.relu.default(add_23);  add_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_10: "f32[8, 276, 56, 56]" = torch.ops.aten.convolution.default(relu_10, arg232_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_10 = arg232_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_26: "f32[8, 256, 56, 56]" = torch.ops.aten.slice.Tensor(convolution_10, 1, 0, 256)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    add_24: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(add_17, slice_26);  add_17 = slice_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_30: "f32[8, 20, 56, 56]" = torch.ops.aten.slice.Tensor(convolution_10, 1, 256, 9223372036854775807);  convolution_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    cat_4: "f32[8, 100, 56, 56]" = torch.ops.aten.cat.default([cat_2, slice_30], 1);  cat_2 = slice_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    cat_5: "f32[8, 356, 56, 56]" = torch.ops.aten.cat.default([add_24, cat_4], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_88: "f32[356, 1]" = torch.ops.aten.unsqueeze.default(arg356_1, -1);  arg356_1 = None
    unsqueeze_89: "f32[356, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
    sub_11: "f32[8, 356, 56, 56]" = torch.ops.aten.sub.Tensor(cat_5, unsqueeze_89);  cat_5 = unsqueeze_89 = None
    add_25: "f32[356]" = torch.ops.aten.add.Tensor(arg357_1, 0.001);  arg357_1 = None
    sqrt_11: "f32[356]" = torch.ops.aten.sqrt.default(add_25);  add_25 = None
    reciprocal_11: "f32[356]" = torch.ops.aten.reciprocal.default(sqrt_11);  sqrt_11 = None
    mul_33: "f32[356]" = torch.ops.aten.mul.Tensor(reciprocal_11, 1);  reciprocal_11 = None
    unsqueeze_90: "f32[356, 1]" = torch.ops.aten.unsqueeze.default(mul_33, -1);  mul_33 = None
    unsqueeze_91: "f32[356, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    mul_34: "f32[8, 356, 56, 56]" = torch.ops.aten.mul.Tensor(sub_11, unsqueeze_91);  sub_11 = unsqueeze_91 = None
    unsqueeze_92: "f32[356, 1]" = torch.ops.aten.unsqueeze.default(arg22_1, -1);  arg22_1 = None
    unsqueeze_93: "f32[356, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
    mul_35: "f32[8, 356, 56, 56]" = torch.ops.aten.mul.Tensor(mul_34, unsqueeze_93);  mul_34 = unsqueeze_93 = None
    unsqueeze_94: "f32[356, 1]" = torch.ops.aten.unsqueeze.default(arg23_1, -1);  arg23_1 = None
    unsqueeze_95: "f32[356, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
    add_26: "f32[8, 356, 56, 56]" = torch.ops.aten.add.Tensor(mul_35, unsqueeze_95);  mul_35 = unsqueeze_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_11: "f32[8, 356, 56, 56]" = torch.ops.aten.relu.default(add_26);  add_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_11: "f32[8, 200, 56, 56]" = torch.ops.aten.convolution.default(relu_11, arg233_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_11 = arg233_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_96: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg358_1, -1);  arg358_1 = None
    unsqueeze_97: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
    sub_12: "f32[8, 200, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_97);  convolution_11 = unsqueeze_97 = None
    add_27: "f32[200]" = torch.ops.aten.add.Tensor(arg359_1, 0.001);  arg359_1 = None
    sqrt_12: "f32[200]" = torch.ops.aten.sqrt.default(add_27);  add_27 = None
    reciprocal_12: "f32[200]" = torch.ops.aten.reciprocal.default(sqrt_12);  sqrt_12 = None
    mul_36: "f32[200]" = torch.ops.aten.mul.Tensor(reciprocal_12, 1);  reciprocal_12 = None
    unsqueeze_98: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(mul_36, -1);  mul_36 = None
    unsqueeze_99: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
    mul_37: "f32[8, 200, 56, 56]" = torch.ops.aten.mul.Tensor(sub_12, unsqueeze_99);  sub_12 = unsqueeze_99 = None
    unsqueeze_100: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg24_1, -1);  arg24_1 = None
    unsqueeze_101: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
    mul_38: "f32[8, 200, 56, 56]" = torch.ops.aten.mul.Tensor(mul_37, unsqueeze_101);  mul_37 = unsqueeze_101 = None
    unsqueeze_102: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg25_1, -1);  arg25_1 = None
    unsqueeze_103: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    add_28: "f32[8, 200, 56, 56]" = torch.ops.aten.add.Tensor(mul_38, unsqueeze_103);  mul_38 = unsqueeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_12: "f32[8, 200, 56, 56]" = torch.ops.aten.relu.default(add_28);  add_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_12: "f32[8, 200, 56, 56]" = torch.ops.aten.convolution.default(relu_12, arg234_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_12 = arg234_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_104: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg360_1, -1);  arg360_1 = None
    unsqueeze_105: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
    sub_13: "f32[8, 200, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_105);  convolution_12 = unsqueeze_105 = None
    add_29: "f32[200]" = torch.ops.aten.add.Tensor(arg361_1, 0.001);  arg361_1 = None
    sqrt_13: "f32[200]" = torch.ops.aten.sqrt.default(add_29);  add_29 = None
    reciprocal_13: "f32[200]" = torch.ops.aten.reciprocal.default(sqrt_13);  sqrt_13 = None
    mul_39: "f32[200]" = torch.ops.aten.mul.Tensor(reciprocal_13, 1);  reciprocal_13 = None
    unsqueeze_106: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(mul_39, -1);  mul_39 = None
    unsqueeze_107: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
    mul_40: "f32[8, 200, 56, 56]" = torch.ops.aten.mul.Tensor(sub_13, unsqueeze_107);  sub_13 = unsqueeze_107 = None
    unsqueeze_108: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg26_1, -1);  arg26_1 = None
    unsqueeze_109: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
    mul_41: "f32[8, 200, 56, 56]" = torch.ops.aten.mul.Tensor(mul_40, unsqueeze_109);  mul_40 = unsqueeze_109 = None
    unsqueeze_110: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg27_1, -1);  arg27_1 = None
    unsqueeze_111: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
    add_30: "f32[8, 200, 56, 56]" = torch.ops.aten.add.Tensor(mul_41, unsqueeze_111);  mul_41 = unsqueeze_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_13: "f32[8, 200, 56, 56]" = torch.ops.aten.relu.default(add_30);  add_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_13: "f32[8, 276, 56, 56]" = torch.ops.aten.convolution.default(relu_13, arg235_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_13 = arg235_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_34: "f32[8, 256, 56, 56]" = torch.ops.aten.slice.Tensor(convolution_13, 1, 0, 256)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    add_31: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(add_24, slice_34);  add_24 = slice_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_38: "f32[8, 20, 56, 56]" = torch.ops.aten.slice.Tensor(convolution_13, 1, 256, 9223372036854775807);  convolution_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    cat_6: "f32[8, 120, 56, 56]" = torch.ops.aten.cat.default([cat_4, slice_38], 1);  cat_4 = slice_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    cat_7: "f32[8, 376, 56, 56]" = torch.ops.aten.cat.default([add_31, cat_6], 1);  add_31 = cat_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_112: "f32[376, 1]" = torch.ops.aten.unsqueeze.default(arg362_1, -1);  arg362_1 = None
    unsqueeze_113: "f32[376, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
    sub_14: "f32[8, 376, 56, 56]" = torch.ops.aten.sub.Tensor(cat_7, unsqueeze_113);  unsqueeze_113 = None
    add_32: "f32[376]" = torch.ops.aten.add.Tensor(arg363_1, 0.001);  arg363_1 = None
    sqrt_14: "f32[376]" = torch.ops.aten.sqrt.default(add_32);  add_32 = None
    reciprocal_14: "f32[376]" = torch.ops.aten.reciprocal.default(sqrt_14);  sqrt_14 = None
    mul_42: "f32[376]" = torch.ops.aten.mul.Tensor(reciprocal_14, 1);  reciprocal_14 = None
    unsqueeze_114: "f32[376, 1]" = torch.ops.aten.unsqueeze.default(mul_42, -1);  mul_42 = None
    unsqueeze_115: "f32[376, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
    mul_43: "f32[8, 376, 56, 56]" = torch.ops.aten.mul.Tensor(sub_14, unsqueeze_115);  sub_14 = unsqueeze_115 = None
    unsqueeze_116: "f32[376, 1]" = torch.ops.aten.unsqueeze.default(arg28_1, -1);  arg28_1 = None
    unsqueeze_117: "f32[376, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
    mul_44: "f32[8, 376, 56, 56]" = torch.ops.aten.mul.Tensor(mul_43, unsqueeze_117);  mul_43 = unsqueeze_117 = None
    unsqueeze_118: "f32[376, 1]" = torch.ops.aten.unsqueeze.default(arg29_1, -1);  arg29_1 = None
    unsqueeze_119: "f32[376, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
    add_33: "f32[8, 376, 56, 56]" = torch.ops.aten.add.Tensor(mul_44, unsqueeze_119);  mul_44 = unsqueeze_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_14: "f32[8, 376, 56, 56]" = torch.ops.aten.relu.default(add_33);  add_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_14: "f32[8, 640, 28, 28]" = torch.ops.aten.convolution.default(relu_14, arg236_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  relu_14 = arg236_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:133, code: x_s1 = x_s[:, :self.num_1x1_c, :, :]
    slice_42: "f32[8, 512, 28, 28]" = torch.ops.aten.slice.Tensor(convolution_14, 1, 0, 512)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_120: "f32[376, 1]" = torch.ops.aten.unsqueeze.default(arg364_1, -1);  arg364_1 = None
    unsqueeze_121: "f32[376, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
    sub_15: "f32[8, 376, 56, 56]" = torch.ops.aten.sub.Tensor(cat_7, unsqueeze_121);  cat_7 = unsqueeze_121 = None
    add_34: "f32[376]" = torch.ops.aten.add.Tensor(arg365_1, 0.001);  arg365_1 = None
    sqrt_15: "f32[376]" = torch.ops.aten.sqrt.default(add_34);  add_34 = None
    reciprocal_15: "f32[376]" = torch.ops.aten.reciprocal.default(sqrt_15);  sqrt_15 = None
    mul_45: "f32[376]" = torch.ops.aten.mul.Tensor(reciprocal_15, 1);  reciprocal_15 = None
    unsqueeze_122: "f32[376, 1]" = torch.ops.aten.unsqueeze.default(mul_45, -1);  mul_45 = None
    unsqueeze_123: "f32[376, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
    mul_46: "f32[8, 376, 56, 56]" = torch.ops.aten.mul.Tensor(sub_15, unsqueeze_123);  sub_15 = unsqueeze_123 = None
    unsqueeze_124: "f32[376, 1]" = torch.ops.aten.unsqueeze.default(arg30_1, -1);  arg30_1 = None
    unsqueeze_125: "f32[376, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
    mul_47: "f32[8, 376, 56, 56]" = torch.ops.aten.mul.Tensor(mul_46, unsqueeze_125);  mul_46 = unsqueeze_125 = None
    unsqueeze_126: "f32[376, 1]" = torch.ops.aten.unsqueeze.default(arg31_1, -1);  arg31_1 = None
    unsqueeze_127: "f32[376, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
    add_35: "f32[8, 376, 56, 56]" = torch.ops.aten.add.Tensor(mul_47, unsqueeze_127);  mul_47 = unsqueeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_15: "f32[8, 376, 56, 56]" = torch.ops.aten.relu.default(add_35);  add_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_15: "f32[8, 400, 56, 56]" = torch.ops.aten.convolution.default(relu_15, arg237_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_15 = arg237_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_128: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg366_1, -1);  arg366_1 = None
    unsqueeze_129: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
    sub_16: "f32[8, 400, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_129);  convolution_15 = unsqueeze_129 = None
    add_36: "f32[400]" = torch.ops.aten.add.Tensor(arg367_1, 0.001);  arg367_1 = None
    sqrt_16: "f32[400]" = torch.ops.aten.sqrt.default(add_36);  add_36 = None
    reciprocal_16: "f32[400]" = torch.ops.aten.reciprocal.default(sqrt_16);  sqrt_16 = None
    mul_48: "f32[400]" = torch.ops.aten.mul.Tensor(reciprocal_16, 1);  reciprocal_16 = None
    unsqueeze_130: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(mul_48, -1);  mul_48 = None
    unsqueeze_131: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
    mul_49: "f32[8, 400, 56, 56]" = torch.ops.aten.mul.Tensor(sub_16, unsqueeze_131);  sub_16 = unsqueeze_131 = None
    unsqueeze_132: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg32_1, -1);  arg32_1 = None
    unsqueeze_133: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
    mul_50: "f32[8, 400, 56, 56]" = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_133);  mul_49 = unsqueeze_133 = None
    unsqueeze_134: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg33_1, -1);  arg33_1 = None
    unsqueeze_135: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
    add_37: "f32[8, 400, 56, 56]" = torch.ops.aten.add.Tensor(mul_50, unsqueeze_135);  mul_50 = unsqueeze_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_16: "f32[8, 400, 56, 56]" = torch.ops.aten.relu.default(add_37);  add_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_16: "f32[8, 400, 28, 28]" = torch.ops.aten.convolution.default(relu_16, arg238_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 50);  relu_16 = arg238_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_136: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg368_1, -1);  arg368_1 = None
    unsqueeze_137: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
    sub_17: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_137);  convolution_16 = unsqueeze_137 = None
    add_38: "f32[400]" = torch.ops.aten.add.Tensor(arg369_1, 0.001);  arg369_1 = None
    sqrt_17: "f32[400]" = torch.ops.aten.sqrt.default(add_38);  add_38 = None
    reciprocal_17: "f32[400]" = torch.ops.aten.reciprocal.default(sqrt_17);  sqrt_17 = None
    mul_51: "f32[400]" = torch.ops.aten.mul.Tensor(reciprocal_17, 1);  reciprocal_17 = None
    unsqueeze_138: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(mul_51, -1);  mul_51 = None
    unsqueeze_139: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
    mul_52: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(sub_17, unsqueeze_139);  sub_17 = unsqueeze_139 = None
    unsqueeze_140: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg34_1, -1);  arg34_1 = None
    unsqueeze_141: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
    mul_53: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(mul_52, unsqueeze_141);  mul_52 = unsqueeze_141 = None
    unsqueeze_142: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg35_1, -1);  arg35_1 = None
    unsqueeze_143: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
    add_39: "f32[8, 400, 28, 28]" = torch.ops.aten.add.Tensor(mul_53, unsqueeze_143);  mul_53 = unsqueeze_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_17: "f32[8, 400, 28, 28]" = torch.ops.aten.relu.default(add_39);  add_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_17: "f32[8, 576, 28, 28]" = torch.ops.aten.convolution.default(relu_17, arg239_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_17 = arg239_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_50: "f32[8, 512, 28, 28]" = torch.ops.aten.slice.Tensor(convolution_17, 1, 0, 512)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    add_40: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(slice_42, slice_50);  slice_42 = slice_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:134, code: x_s2 = x_s[:, self.num_1x1_c:, :, :]
    slice_46: "f32[8, 128, 28, 28]" = torch.ops.aten.slice.Tensor(convolution_14, 1, 512, 9223372036854775807);  convolution_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_54: "f32[8, 64, 28, 28]" = torch.ops.aten.slice.Tensor(convolution_17, 1, 512, 9223372036854775807);  convolution_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    cat_8: "f32[8, 192, 28, 28]" = torch.ops.aten.cat.default([slice_46, slice_54], 1);  slice_46 = slice_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    cat_9: "f32[8, 704, 28, 28]" = torch.ops.aten.cat.default([add_40, cat_8], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_144: "f32[704, 1]" = torch.ops.aten.unsqueeze.default(arg370_1, -1);  arg370_1 = None
    unsqueeze_145: "f32[704, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
    sub_18: "f32[8, 704, 28, 28]" = torch.ops.aten.sub.Tensor(cat_9, unsqueeze_145);  cat_9 = unsqueeze_145 = None
    add_41: "f32[704]" = torch.ops.aten.add.Tensor(arg371_1, 0.001);  arg371_1 = None
    sqrt_18: "f32[704]" = torch.ops.aten.sqrt.default(add_41);  add_41 = None
    reciprocal_18: "f32[704]" = torch.ops.aten.reciprocal.default(sqrt_18);  sqrt_18 = None
    mul_54: "f32[704]" = torch.ops.aten.mul.Tensor(reciprocal_18, 1);  reciprocal_18 = None
    unsqueeze_146: "f32[704, 1]" = torch.ops.aten.unsqueeze.default(mul_54, -1);  mul_54 = None
    unsqueeze_147: "f32[704, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
    mul_55: "f32[8, 704, 28, 28]" = torch.ops.aten.mul.Tensor(sub_18, unsqueeze_147);  sub_18 = unsqueeze_147 = None
    unsqueeze_148: "f32[704, 1]" = torch.ops.aten.unsqueeze.default(arg36_1, -1);  arg36_1 = None
    unsqueeze_149: "f32[704, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
    mul_56: "f32[8, 704, 28, 28]" = torch.ops.aten.mul.Tensor(mul_55, unsqueeze_149);  mul_55 = unsqueeze_149 = None
    unsqueeze_150: "f32[704, 1]" = torch.ops.aten.unsqueeze.default(arg37_1, -1);  arg37_1 = None
    unsqueeze_151: "f32[704, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
    add_42: "f32[8, 704, 28, 28]" = torch.ops.aten.add.Tensor(mul_56, unsqueeze_151);  mul_56 = unsqueeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_18: "f32[8, 704, 28, 28]" = torch.ops.aten.relu.default(add_42);  add_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_18: "f32[8, 400, 28, 28]" = torch.ops.aten.convolution.default(relu_18, arg240_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_18 = arg240_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_152: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg372_1, -1);  arg372_1 = None
    unsqueeze_153: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
    sub_19: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_153);  convolution_18 = unsqueeze_153 = None
    add_43: "f32[400]" = torch.ops.aten.add.Tensor(arg373_1, 0.001);  arg373_1 = None
    sqrt_19: "f32[400]" = torch.ops.aten.sqrt.default(add_43);  add_43 = None
    reciprocal_19: "f32[400]" = torch.ops.aten.reciprocal.default(sqrt_19);  sqrt_19 = None
    mul_57: "f32[400]" = torch.ops.aten.mul.Tensor(reciprocal_19, 1);  reciprocal_19 = None
    unsqueeze_154: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(mul_57, -1);  mul_57 = None
    unsqueeze_155: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
    mul_58: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(sub_19, unsqueeze_155);  sub_19 = unsqueeze_155 = None
    unsqueeze_156: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg38_1, -1);  arg38_1 = None
    unsqueeze_157: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
    mul_59: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(mul_58, unsqueeze_157);  mul_58 = unsqueeze_157 = None
    unsqueeze_158: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg39_1, -1);  arg39_1 = None
    unsqueeze_159: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
    add_44: "f32[8, 400, 28, 28]" = torch.ops.aten.add.Tensor(mul_59, unsqueeze_159);  mul_59 = unsqueeze_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_19: "f32[8, 400, 28, 28]" = torch.ops.aten.relu.default(add_44);  add_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_19: "f32[8, 400, 28, 28]" = torch.ops.aten.convolution.default(relu_19, arg241_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_19 = arg241_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_160: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg374_1, -1);  arg374_1 = None
    unsqueeze_161: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
    sub_20: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_161);  convolution_19 = unsqueeze_161 = None
    add_45: "f32[400]" = torch.ops.aten.add.Tensor(arg375_1, 0.001);  arg375_1 = None
    sqrt_20: "f32[400]" = torch.ops.aten.sqrt.default(add_45);  add_45 = None
    reciprocal_20: "f32[400]" = torch.ops.aten.reciprocal.default(sqrt_20);  sqrt_20 = None
    mul_60: "f32[400]" = torch.ops.aten.mul.Tensor(reciprocal_20, 1);  reciprocal_20 = None
    unsqueeze_162: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(mul_60, -1);  mul_60 = None
    unsqueeze_163: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
    mul_61: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(sub_20, unsqueeze_163);  sub_20 = unsqueeze_163 = None
    unsqueeze_164: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg40_1, -1);  arg40_1 = None
    unsqueeze_165: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
    mul_62: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(mul_61, unsqueeze_165);  mul_61 = unsqueeze_165 = None
    unsqueeze_166: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg41_1, -1);  arg41_1 = None
    unsqueeze_167: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
    add_46: "f32[8, 400, 28, 28]" = torch.ops.aten.add.Tensor(mul_62, unsqueeze_167);  mul_62 = unsqueeze_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_20: "f32[8, 400, 28, 28]" = torch.ops.aten.relu.default(add_46);  add_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_20: "f32[8, 576, 28, 28]" = torch.ops.aten.convolution.default(relu_20, arg242_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_20 = arg242_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_58: "f32[8, 512, 28, 28]" = torch.ops.aten.slice.Tensor(convolution_20, 1, 0, 512)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    add_47: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_40, slice_58);  add_40 = slice_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_62: "f32[8, 64, 28, 28]" = torch.ops.aten.slice.Tensor(convolution_20, 1, 512, 9223372036854775807);  convolution_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    cat_10: "f32[8, 256, 28, 28]" = torch.ops.aten.cat.default([cat_8, slice_62], 1);  cat_8 = slice_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    cat_11: "f32[8, 768, 28, 28]" = torch.ops.aten.cat.default([add_47, cat_10], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_168: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg376_1, -1);  arg376_1 = None
    unsqueeze_169: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
    sub_21: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(cat_11, unsqueeze_169);  cat_11 = unsqueeze_169 = None
    add_48: "f32[768]" = torch.ops.aten.add.Tensor(arg377_1, 0.001);  arg377_1 = None
    sqrt_21: "f32[768]" = torch.ops.aten.sqrt.default(add_48);  add_48 = None
    reciprocal_21: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_21);  sqrt_21 = None
    mul_63: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_21, 1);  reciprocal_21 = None
    unsqueeze_170: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_63, -1);  mul_63 = None
    unsqueeze_171: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
    mul_64: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_21, unsqueeze_171);  sub_21 = unsqueeze_171 = None
    unsqueeze_172: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg42_1, -1);  arg42_1 = None
    unsqueeze_173: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
    mul_65: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_64, unsqueeze_173);  mul_64 = unsqueeze_173 = None
    unsqueeze_174: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg43_1, -1);  arg43_1 = None
    unsqueeze_175: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
    add_49: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_65, unsqueeze_175);  mul_65 = unsqueeze_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_21: "f32[8, 768, 28, 28]" = torch.ops.aten.relu.default(add_49);  add_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_21: "f32[8, 400, 28, 28]" = torch.ops.aten.convolution.default(relu_21, arg243_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_21 = arg243_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_176: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg378_1, -1);  arg378_1 = None
    unsqueeze_177: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
    sub_22: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_177);  convolution_21 = unsqueeze_177 = None
    add_50: "f32[400]" = torch.ops.aten.add.Tensor(arg379_1, 0.001);  arg379_1 = None
    sqrt_22: "f32[400]" = torch.ops.aten.sqrt.default(add_50);  add_50 = None
    reciprocal_22: "f32[400]" = torch.ops.aten.reciprocal.default(sqrt_22);  sqrt_22 = None
    mul_66: "f32[400]" = torch.ops.aten.mul.Tensor(reciprocal_22, 1);  reciprocal_22 = None
    unsqueeze_178: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(mul_66, -1);  mul_66 = None
    unsqueeze_179: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
    mul_67: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(sub_22, unsqueeze_179);  sub_22 = unsqueeze_179 = None
    unsqueeze_180: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg44_1, -1);  arg44_1 = None
    unsqueeze_181: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
    mul_68: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(mul_67, unsqueeze_181);  mul_67 = unsqueeze_181 = None
    unsqueeze_182: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg45_1, -1);  arg45_1 = None
    unsqueeze_183: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
    add_51: "f32[8, 400, 28, 28]" = torch.ops.aten.add.Tensor(mul_68, unsqueeze_183);  mul_68 = unsqueeze_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_22: "f32[8, 400, 28, 28]" = torch.ops.aten.relu.default(add_51);  add_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_22: "f32[8, 400, 28, 28]" = torch.ops.aten.convolution.default(relu_22, arg244_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_22 = arg244_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_184: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg380_1, -1);  arg380_1 = None
    unsqueeze_185: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
    sub_23: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_185);  convolution_22 = unsqueeze_185 = None
    add_52: "f32[400]" = torch.ops.aten.add.Tensor(arg381_1, 0.001);  arg381_1 = None
    sqrt_23: "f32[400]" = torch.ops.aten.sqrt.default(add_52);  add_52 = None
    reciprocal_23: "f32[400]" = torch.ops.aten.reciprocal.default(sqrt_23);  sqrt_23 = None
    mul_69: "f32[400]" = torch.ops.aten.mul.Tensor(reciprocal_23, 1);  reciprocal_23 = None
    unsqueeze_186: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(mul_69, -1);  mul_69 = None
    unsqueeze_187: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
    mul_70: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(sub_23, unsqueeze_187);  sub_23 = unsqueeze_187 = None
    unsqueeze_188: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg46_1, -1);  arg46_1 = None
    unsqueeze_189: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
    mul_71: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(mul_70, unsqueeze_189);  mul_70 = unsqueeze_189 = None
    unsqueeze_190: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg47_1, -1);  arg47_1 = None
    unsqueeze_191: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
    add_53: "f32[8, 400, 28, 28]" = torch.ops.aten.add.Tensor(mul_71, unsqueeze_191);  mul_71 = unsqueeze_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_23: "f32[8, 400, 28, 28]" = torch.ops.aten.relu.default(add_53);  add_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_23: "f32[8, 576, 28, 28]" = torch.ops.aten.convolution.default(relu_23, arg245_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_23 = arg245_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_66: "f32[8, 512, 28, 28]" = torch.ops.aten.slice.Tensor(convolution_23, 1, 0, 512)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    add_54: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_47, slice_66);  add_47 = slice_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_70: "f32[8, 64, 28, 28]" = torch.ops.aten.slice.Tensor(convolution_23, 1, 512, 9223372036854775807);  convolution_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    cat_12: "f32[8, 320, 28, 28]" = torch.ops.aten.cat.default([cat_10, slice_70], 1);  cat_10 = slice_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    cat_13: "f32[8, 832, 28, 28]" = torch.ops.aten.cat.default([add_54, cat_12], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_192: "f32[832, 1]" = torch.ops.aten.unsqueeze.default(arg382_1, -1);  arg382_1 = None
    unsqueeze_193: "f32[832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
    sub_24: "f32[8, 832, 28, 28]" = torch.ops.aten.sub.Tensor(cat_13, unsqueeze_193);  cat_13 = unsqueeze_193 = None
    add_55: "f32[832]" = torch.ops.aten.add.Tensor(arg383_1, 0.001);  arg383_1 = None
    sqrt_24: "f32[832]" = torch.ops.aten.sqrt.default(add_55);  add_55 = None
    reciprocal_24: "f32[832]" = torch.ops.aten.reciprocal.default(sqrt_24);  sqrt_24 = None
    mul_72: "f32[832]" = torch.ops.aten.mul.Tensor(reciprocal_24, 1);  reciprocal_24 = None
    unsqueeze_194: "f32[832, 1]" = torch.ops.aten.unsqueeze.default(mul_72, -1);  mul_72 = None
    unsqueeze_195: "f32[832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
    mul_73: "f32[8, 832, 28, 28]" = torch.ops.aten.mul.Tensor(sub_24, unsqueeze_195);  sub_24 = unsqueeze_195 = None
    unsqueeze_196: "f32[832, 1]" = torch.ops.aten.unsqueeze.default(arg48_1, -1);  arg48_1 = None
    unsqueeze_197: "f32[832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
    mul_74: "f32[8, 832, 28, 28]" = torch.ops.aten.mul.Tensor(mul_73, unsqueeze_197);  mul_73 = unsqueeze_197 = None
    unsqueeze_198: "f32[832, 1]" = torch.ops.aten.unsqueeze.default(arg49_1, -1);  arg49_1 = None
    unsqueeze_199: "f32[832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
    add_56: "f32[8, 832, 28, 28]" = torch.ops.aten.add.Tensor(mul_74, unsqueeze_199);  mul_74 = unsqueeze_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_24: "f32[8, 832, 28, 28]" = torch.ops.aten.relu.default(add_56);  add_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_24: "f32[8, 400, 28, 28]" = torch.ops.aten.convolution.default(relu_24, arg246_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_24 = arg246_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_200: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg384_1, -1);  arg384_1 = None
    unsqueeze_201: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
    sub_25: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_201);  convolution_24 = unsqueeze_201 = None
    add_57: "f32[400]" = torch.ops.aten.add.Tensor(arg385_1, 0.001);  arg385_1 = None
    sqrt_25: "f32[400]" = torch.ops.aten.sqrt.default(add_57);  add_57 = None
    reciprocal_25: "f32[400]" = torch.ops.aten.reciprocal.default(sqrt_25);  sqrt_25 = None
    mul_75: "f32[400]" = torch.ops.aten.mul.Tensor(reciprocal_25, 1);  reciprocal_25 = None
    unsqueeze_202: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(mul_75, -1);  mul_75 = None
    unsqueeze_203: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
    mul_76: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(sub_25, unsqueeze_203);  sub_25 = unsqueeze_203 = None
    unsqueeze_204: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg50_1, -1);  arg50_1 = None
    unsqueeze_205: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
    mul_77: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(mul_76, unsqueeze_205);  mul_76 = unsqueeze_205 = None
    unsqueeze_206: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg51_1, -1);  arg51_1 = None
    unsqueeze_207: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
    add_58: "f32[8, 400, 28, 28]" = torch.ops.aten.add.Tensor(mul_77, unsqueeze_207);  mul_77 = unsqueeze_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_25: "f32[8, 400, 28, 28]" = torch.ops.aten.relu.default(add_58);  add_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_25: "f32[8, 400, 28, 28]" = torch.ops.aten.convolution.default(relu_25, arg247_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_25 = arg247_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_208: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg386_1, -1);  arg386_1 = None
    unsqueeze_209: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
    sub_26: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_209);  convolution_25 = unsqueeze_209 = None
    add_59: "f32[400]" = torch.ops.aten.add.Tensor(arg387_1, 0.001);  arg387_1 = None
    sqrt_26: "f32[400]" = torch.ops.aten.sqrt.default(add_59);  add_59 = None
    reciprocal_26: "f32[400]" = torch.ops.aten.reciprocal.default(sqrt_26);  sqrt_26 = None
    mul_78: "f32[400]" = torch.ops.aten.mul.Tensor(reciprocal_26, 1);  reciprocal_26 = None
    unsqueeze_210: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(mul_78, -1);  mul_78 = None
    unsqueeze_211: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
    mul_79: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(sub_26, unsqueeze_211);  sub_26 = unsqueeze_211 = None
    unsqueeze_212: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg52_1, -1);  arg52_1 = None
    unsqueeze_213: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
    mul_80: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(mul_79, unsqueeze_213);  mul_79 = unsqueeze_213 = None
    unsqueeze_214: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg53_1, -1);  arg53_1 = None
    unsqueeze_215: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
    add_60: "f32[8, 400, 28, 28]" = torch.ops.aten.add.Tensor(mul_80, unsqueeze_215);  mul_80 = unsqueeze_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_26: "f32[8, 400, 28, 28]" = torch.ops.aten.relu.default(add_60);  add_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_26: "f32[8, 576, 28, 28]" = torch.ops.aten.convolution.default(relu_26, arg248_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_26 = arg248_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_74: "f32[8, 512, 28, 28]" = torch.ops.aten.slice.Tensor(convolution_26, 1, 0, 512)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    add_61: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_54, slice_74);  add_54 = slice_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_78: "f32[8, 64, 28, 28]" = torch.ops.aten.slice.Tensor(convolution_26, 1, 512, 9223372036854775807);  convolution_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    cat_14: "f32[8, 384, 28, 28]" = torch.ops.aten.cat.default([cat_12, slice_78], 1);  cat_12 = slice_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    cat_15: "f32[8, 896, 28, 28]" = torch.ops.aten.cat.default([add_61, cat_14], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_216: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(arg388_1, -1);  arg388_1 = None
    unsqueeze_217: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
    sub_27: "f32[8, 896, 28, 28]" = torch.ops.aten.sub.Tensor(cat_15, unsqueeze_217);  cat_15 = unsqueeze_217 = None
    add_62: "f32[896]" = torch.ops.aten.add.Tensor(arg389_1, 0.001);  arg389_1 = None
    sqrt_27: "f32[896]" = torch.ops.aten.sqrt.default(add_62);  add_62 = None
    reciprocal_27: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_27);  sqrt_27 = None
    mul_81: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_27, 1);  reciprocal_27 = None
    unsqueeze_218: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_81, -1);  mul_81 = None
    unsqueeze_219: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, -1);  unsqueeze_218 = None
    mul_82: "f32[8, 896, 28, 28]" = torch.ops.aten.mul.Tensor(sub_27, unsqueeze_219);  sub_27 = unsqueeze_219 = None
    unsqueeze_220: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(arg54_1, -1);  arg54_1 = None
    unsqueeze_221: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, -1);  unsqueeze_220 = None
    mul_83: "f32[8, 896, 28, 28]" = torch.ops.aten.mul.Tensor(mul_82, unsqueeze_221);  mul_82 = unsqueeze_221 = None
    unsqueeze_222: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(arg55_1, -1);  arg55_1 = None
    unsqueeze_223: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, -1);  unsqueeze_222 = None
    add_63: "f32[8, 896, 28, 28]" = torch.ops.aten.add.Tensor(mul_83, unsqueeze_223);  mul_83 = unsqueeze_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_27: "f32[8, 896, 28, 28]" = torch.ops.aten.relu.default(add_63);  add_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_27: "f32[8, 400, 28, 28]" = torch.ops.aten.convolution.default(relu_27, arg249_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_27 = arg249_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_224: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg390_1, -1);  arg390_1 = None
    unsqueeze_225: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, -1);  unsqueeze_224 = None
    sub_28: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_225);  convolution_27 = unsqueeze_225 = None
    add_64: "f32[400]" = torch.ops.aten.add.Tensor(arg391_1, 0.001);  arg391_1 = None
    sqrt_28: "f32[400]" = torch.ops.aten.sqrt.default(add_64);  add_64 = None
    reciprocal_28: "f32[400]" = torch.ops.aten.reciprocal.default(sqrt_28);  sqrt_28 = None
    mul_84: "f32[400]" = torch.ops.aten.mul.Tensor(reciprocal_28, 1);  reciprocal_28 = None
    unsqueeze_226: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(mul_84, -1);  mul_84 = None
    unsqueeze_227: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, -1);  unsqueeze_226 = None
    mul_85: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(sub_28, unsqueeze_227);  sub_28 = unsqueeze_227 = None
    unsqueeze_228: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg56_1, -1);  arg56_1 = None
    unsqueeze_229: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, -1);  unsqueeze_228 = None
    mul_86: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(mul_85, unsqueeze_229);  mul_85 = unsqueeze_229 = None
    unsqueeze_230: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg57_1, -1);  arg57_1 = None
    unsqueeze_231: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, -1);  unsqueeze_230 = None
    add_65: "f32[8, 400, 28, 28]" = torch.ops.aten.add.Tensor(mul_86, unsqueeze_231);  mul_86 = unsqueeze_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_28: "f32[8, 400, 28, 28]" = torch.ops.aten.relu.default(add_65);  add_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_28: "f32[8, 400, 28, 28]" = torch.ops.aten.convolution.default(relu_28, arg250_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_28 = arg250_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_232: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg392_1, -1);  arg392_1 = None
    unsqueeze_233: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, -1);  unsqueeze_232 = None
    sub_29: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_233);  convolution_28 = unsqueeze_233 = None
    add_66: "f32[400]" = torch.ops.aten.add.Tensor(arg393_1, 0.001);  arg393_1 = None
    sqrt_29: "f32[400]" = torch.ops.aten.sqrt.default(add_66);  add_66 = None
    reciprocal_29: "f32[400]" = torch.ops.aten.reciprocal.default(sqrt_29);  sqrt_29 = None
    mul_87: "f32[400]" = torch.ops.aten.mul.Tensor(reciprocal_29, 1);  reciprocal_29 = None
    unsqueeze_234: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(mul_87, -1);  mul_87 = None
    unsqueeze_235: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, -1);  unsqueeze_234 = None
    mul_88: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(sub_29, unsqueeze_235);  sub_29 = unsqueeze_235 = None
    unsqueeze_236: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg58_1, -1);  arg58_1 = None
    unsqueeze_237: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, -1);  unsqueeze_236 = None
    mul_89: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(mul_88, unsqueeze_237);  mul_88 = unsqueeze_237 = None
    unsqueeze_238: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg59_1, -1);  arg59_1 = None
    unsqueeze_239: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, -1);  unsqueeze_238 = None
    add_67: "f32[8, 400, 28, 28]" = torch.ops.aten.add.Tensor(mul_89, unsqueeze_239);  mul_89 = unsqueeze_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_29: "f32[8, 400, 28, 28]" = torch.ops.aten.relu.default(add_67);  add_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_29: "f32[8, 576, 28, 28]" = torch.ops.aten.convolution.default(relu_29, arg251_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_29 = arg251_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_82: "f32[8, 512, 28, 28]" = torch.ops.aten.slice.Tensor(convolution_29, 1, 0, 512)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    add_68: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_61, slice_82);  add_61 = slice_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_86: "f32[8, 64, 28, 28]" = torch.ops.aten.slice.Tensor(convolution_29, 1, 512, 9223372036854775807);  convolution_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    cat_16: "f32[8, 448, 28, 28]" = torch.ops.aten.cat.default([cat_14, slice_86], 1);  cat_14 = slice_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    cat_17: "f32[8, 960, 28, 28]" = torch.ops.aten.cat.default([add_68, cat_16], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_240: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(arg394_1, -1);  arg394_1 = None
    unsqueeze_241: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, -1);  unsqueeze_240 = None
    sub_30: "f32[8, 960, 28, 28]" = torch.ops.aten.sub.Tensor(cat_17, unsqueeze_241);  cat_17 = unsqueeze_241 = None
    add_69: "f32[960]" = torch.ops.aten.add.Tensor(arg395_1, 0.001);  arg395_1 = None
    sqrt_30: "f32[960]" = torch.ops.aten.sqrt.default(add_69);  add_69 = None
    reciprocal_30: "f32[960]" = torch.ops.aten.reciprocal.default(sqrt_30);  sqrt_30 = None
    mul_90: "f32[960]" = torch.ops.aten.mul.Tensor(reciprocal_30, 1);  reciprocal_30 = None
    unsqueeze_242: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(mul_90, -1);  mul_90 = None
    unsqueeze_243: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, -1);  unsqueeze_242 = None
    mul_91: "f32[8, 960, 28, 28]" = torch.ops.aten.mul.Tensor(sub_30, unsqueeze_243);  sub_30 = unsqueeze_243 = None
    unsqueeze_244: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(arg60_1, -1);  arg60_1 = None
    unsqueeze_245: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, -1);  unsqueeze_244 = None
    mul_92: "f32[8, 960, 28, 28]" = torch.ops.aten.mul.Tensor(mul_91, unsqueeze_245);  mul_91 = unsqueeze_245 = None
    unsqueeze_246: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(arg61_1, -1);  arg61_1 = None
    unsqueeze_247: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, -1);  unsqueeze_246 = None
    add_70: "f32[8, 960, 28, 28]" = torch.ops.aten.add.Tensor(mul_92, unsqueeze_247);  mul_92 = unsqueeze_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_30: "f32[8, 960, 28, 28]" = torch.ops.aten.relu.default(add_70);  add_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_30: "f32[8, 400, 28, 28]" = torch.ops.aten.convolution.default(relu_30, arg252_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_30 = arg252_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_248: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg396_1, -1);  arg396_1 = None
    unsqueeze_249: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, -1);  unsqueeze_248 = None
    sub_31: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_249);  convolution_30 = unsqueeze_249 = None
    add_71: "f32[400]" = torch.ops.aten.add.Tensor(arg397_1, 0.001);  arg397_1 = None
    sqrt_31: "f32[400]" = torch.ops.aten.sqrt.default(add_71);  add_71 = None
    reciprocal_31: "f32[400]" = torch.ops.aten.reciprocal.default(sqrt_31);  sqrt_31 = None
    mul_93: "f32[400]" = torch.ops.aten.mul.Tensor(reciprocal_31, 1);  reciprocal_31 = None
    unsqueeze_250: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(mul_93, -1);  mul_93 = None
    unsqueeze_251: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, -1);  unsqueeze_250 = None
    mul_94: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(sub_31, unsqueeze_251);  sub_31 = unsqueeze_251 = None
    unsqueeze_252: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg62_1, -1);  arg62_1 = None
    unsqueeze_253: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, -1);  unsqueeze_252 = None
    mul_95: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(mul_94, unsqueeze_253);  mul_94 = unsqueeze_253 = None
    unsqueeze_254: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg63_1, -1);  arg63_1 = None
    unsqueeze_255: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, -1);  unsqueeze_254 = None
    add_72: "f32[8, 400, 28, 28]" = torch.ops.aten.add.Tensor(mul_95, unsqueeze_255);  mul_95 = unsqueeze_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_31: "f32[8, 400, 28, 28]" = torch.ops.aten.relu.default(add_72);  add_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_31: "f32[8, 400, 28, 28]" = torch.ops.aten.convolution.default(relu_31, arg253_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_31 = arg253_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_256: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg398_1, -1);  arg398_1 = None
    unsqueeze_257: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, -1);  unsqueeze_256 = None
    sub_32: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_257);  convolution_31 = unsqueeze_257 = None
    add_73: "f32[400]" = torch.ops.aten.add.Tensor(arg399_1, 0.001);  arg399_1 = None
    sqrt_32: "f32[400]" = torch.ops.aten.sqrt.default(add_73);  add_73 = None
    reciprocal_32: "f32[400]" = torch.ops.aten.reciprocal.default(sqrt_32);  sqrt_32 = None
    mul_96: "f32[400]" = torch.ops.aten.mul.Tensor(reciprocal_32, 1);  reciprocal_32 = None
    unsqueeze_258: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(mul_96, -1);  mul_96 = None
    unsqueeze_259: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, -1);  unsqueeze_258 = None
    mul_97: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(sub_32, unsqueeze_259);  sub_32 = unsqueeze_259 = None
    unsqueeze_260: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg64_1, -1);  arg64_1 = None
    unsqueeze_261: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, -1);  unsqueeze_260 = None
    mul_98: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(mul_97, unsqueeze_261);  mul_97 = unsqueeze_261 = None
    unsqueeze_262: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg65_1, -1);  arg65_1 = None
    unsqueeze_263: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, -1);  unsqueeze_262 = None
    add_74: "f32[8, 400, 28, 28]" = torch.ops.aten.add.Tensor(mul_98, unsqueeze_263);  mul_98 = unsqueeze_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_32: "f32[8, 400, 28, 28]" = torch.ops.aten.relu.default(add_74);  add_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_32: "f32[8, 576, 28, 28]" = torch.ops.aten.convolution.default(relu_32, arg254_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_32 = arg254_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_90: "f32[8, 512, 28, 28]" = torch.ops.aten.slice.Tensor(convolution_32, 1, 0, 512)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    add_75: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_68, slice_90);  add_68 = slice_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_94: "f32[8, 64, 28, 28]" = torch.ops.aten.slice.Tensor(convolution_32, 1, 512, 9223372036854775807);  convolution_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    cat_18: "f32[8, 512, 28, 28]" = torch.ops.aten.cat.default([cat_16, slice_94], 1);  cat_16 = slice_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    cat_19: "f32[8, 1024, 28, 28]" = torch.ops.aten.cat.default([add_75, cat_18], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_264: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg400_1, -1);  arg400_1 = None
    unsqueeze_265: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, -1);  unsqueeze_264 = None
    sub_33: "f32[8, 1024, 28, 28]" = torch.ops.aten.sub.Tensor(cat_19, unsqueeze_265);  cat_19 = unsqueeze_265 = None
    add_76: "f32[1024]" = torch.ops.aten.add.Tensor(arg401_1, 0.001);  arg401_1 = None
    sqrt_33: "f32[1024]" = torch.ops.aten.sqrt.default(add_76);  add_76 = None
    reciprocal_33: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_33);  sqrt_33 = None
    mul_99: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_33, 1);  reciprocal_33 = None
    unsqueeze_266: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_99, -1);  mul_99 = None
    unsqueeze_267: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, -1);  unsqueeze_266 = None
    mul_100: "f32[8, 1024, 28, 28]" = torch.ops.aten.mul.Tensor(sub_33, unsqueeze_267);  sub_33 = unsqueeze_267 = None
    unsqueeze_268: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg66_1, -1);  arg66_1 = None
    unsqueeze_269: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, -1);  unsqueeze_268 = None
    mul_101: "f32[8, 1024, 28, 28]" = torch.ops.aten.mul.Tensor(mul_100, unsqueeze_269);  mul_100 = unsqueeze_269 = None
    unsqueeze_270: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg67_1, -1);  arg67_1 = None
    unsqueeze_271: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, -1);  unsqueeze_270 = None
    add_77: "f32[8, 1024, 28, 28]" = torch.ops.aten.add.Tensor(mul_101, unsqueeze_271);  mul_101 = unsqueeze_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_33: "f32[8, 1024, 28, 28]" = torch.ops.aten.relu.default(add_77);  add_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_33: "f32[8, 400, 28, 28]" = torch.ops.aten.convolution.default(relu_33, arg255_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_33 = arg255_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_272: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg402_1, -1);  arg402_1 = None
    unsqueeze_273: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, -1);  unsqueeze_272 = None
    sub_34: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_273);  convolution_33 = unsqueeze_273 = None
    add_78: "f32[400]" = torch.ops.aten.add.Tensor(arg403_1, 0.001);  arg403_1 = None
    sqrt_34: "f32[400]" = torch.ops.aten.sqrt.default(add_78);  add_78 = None
    reciprocal_34: "f32[400]" = torch.ops.aten.reciprocal.default(sqrt_34);  sqrt_34 = None
    mul_102: "f32[400]" = torch.ops.aten.mul.Tensor(reciprocal_34, 1);  reciprocal_34 = None
    unsqueeze_274: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(mul_102, -1);  mul_102 = None
    unsqueeze_275: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, -1);  unsqueeze_274 = None
    mul_103: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(sub_34, unsqueeze_275);  sub_34 = unsqueeze_275 = None
    unsqueeze_276: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg68_1, -1);  arg68_1 = None
    unsqueeze_277: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, -1);  unsqueeze_276 = None
    mul_104: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(mul_103, unsqueeze_277);  mul_103 = unsqueeze_277 = None
    unsqueeze_278: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg69_1, -1);  arg69_1 = None
    unsqueeze_279: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, -1);  unsqueeze_278 = None
    add_79: "f32[8, 400, 28, 28]" = torch.ops.aten.add.Tensor(mul_104, unsqueeze_279);  mul_104 = unsqueeze_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_34: "f32[8, 400, 28, 28]" = torch.ops.aten.relu.default(add_79);  add_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_34: "f32[8, 400, 28, 28]" = torch.ops.aten.convolution.default(relu_34, arg256_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_34 = arg256_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_280: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg404_1, -1);  arg404_1 = None
    unsqueeze_281: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, -1);  unsqueeze_280 = None
    sub_35: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_281);  convolution_34 = unsqueeze_281 = None
    add_80: "f32[400]" = torch.ops.aten.add.Tensor(arg405_1, 0.001);  arg405_1 = None
    sqrt_35: "f32[400]" = torch.ops.aten.sqrt.default(add_80);  add_80 = None
    reciprocal_35: "f32[400]" = torch.ops.aten.reciprocal.default(sqrt_35);  sqrt_35 = None
    mul_105: "f32[400]" = torch.ops.aten.mul.Tensor(reciprocal_35, 1);  reciprocal_35 = None
    unsqueeze_282: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(mul_105, -1);  mul_105 = None
    unsqueeze_283: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, -1);  unsqueeze_282 = None
    mul_106: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(sub_35, unsqueeze_283);  sub_35 = unsqueeze_283 = None
    unsqueeze_284: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg70_1, -1);  arg70_1 = None
    unsqueeze_285: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, -1);  unsqueeze_284 = None
    mul_107: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(mul_106, unsqueeze_285);  mul_106 = unsqueeze_285 = None
    unsqueeze_286: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg71_1, -1);  arg71_1 = None
    unsqueeze_287: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, -1);  unsqueeze_286 = None
    add_81: "f32[8, 400, 28, 28]" = torch.ops.aten.add.Tensor(mul_107, unsqueeze_287);  mul_107 = unsqueeze_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_35: "f32[8, 400, 28, 28]" = torch.ops.aten.relu.default(add_81);  add_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_35: "f32[8, 576, 28, 28]" = torch.ops.aten.convolution.default(relu_35, arg257_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_35 = arg257_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_98: "f32[8, 512, 28, 28]" = torch.ops.aten.slice.Tensor(convolution_35, 1, 0, 512)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    add_82: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_75, slice_98);  add_75 = slice_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_102: "f32[8, 64, 28, 28]" = torch.ops.aten.slice.Tensor(convolution_35, 1, 512, 9223372036854775807);  convolution_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    cat_20: "f32[8, 576, 28, 28]" = torch.ops.aten.cat.default([cat_18, slice_102], 1);  cat_18 = slice_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    cat_21: "f32[8, 1088, 28, 28]" = torch.ops.aten.cat.default([add_82, cat_20], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_288: "f32[1088, 1]" = torch.ops.aten.unsqueeze.default(arg406_1, -1);  arg406_1 = None
    unsqueeze_289: "f32[1088, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, -1);  unsqueeze_288 = None
    sub_36: "f32[8, 1088, 28, 28]" = torch.ops.aten.sub.Tensor(cat_21, unsqueeze_289);  cat_21 = unsqueeze_289 = None
    add_83: "f32[1088]" = torch.ops.aten.add.Tensor(arg407_1, 0.001);  arg407_1 = None
    sqrt_36: "f32[1088]" = torch.ops.aten.sqrt.default(add_83);  add_83 = None
    reciprocal_36: "f32[1088]" = torch.ops.aten.reciprocal.default(sqrt_36);  sqrt_36 = None
    mul_108: "f32[1088]" = torch.ops.aten.mul.Tensor(reciprocal_36, 1);  reciprocal_36 = None
    unsqueeze_290: "f32[1088, 1]" = torch.ops.aten.unsqueeze.default(mul_108, -1);  mul_108 = None
    unsqueeze_291: "f32[1088, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, -1);  unsqueeze_290 = None
    mul_109: "f32[8, 1088, 28, 28]" = torch.ops.aten.mul.Tensor(sub_36, unsqueeze_291);  sub_36 = unsqueeze_291 = None
    unsqueeze_292: "f32[1088, 1]" = torch.ops.aten.unsqueeze.default(arg72_1, -1);  arg72_1 = None
    unsqueeze_293: "f32[1088, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, -1);  unsqueeze_292 = None
    mul_110: "f32[8, 1088, 28, 28]" = torch.ops.aten.mul.Tensor(mul_109, unsqueeze_293);  mul_109 = unsqueeze_293 = None
    unsqueeze_294: "f32[1088, 1]" = torch.ops.aten.unsqueeze.default(arg73_1, -1);  arg73_1 = None
    unsqueeze_295: "f32[1088, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, -1);  unsqueeze_294 = None
    add_84: "f32[8, 1088, 28, 28]" = torch.ops.aten.add.Tensor(mul_110, unsqueeze_295);  mul_110 = unsqueeze_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_36: "f32[8, 1088, 28, 28]" = torch.ops.aten.relu.default(add_84);  add_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_36: "f32[8, 400, 28, 28]" = torch.ops.aten.convolution.default(relu_36, arg258_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_36 = arg258_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_296: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg408_1, -1);  arg408_1 = None
    unsqueeze_297: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, -1);  unsqueeze_296 = None
    sub_37: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_297);  convolution_36 = unsqueeze_297 = None
    add_85: "f32[400]" = torch.ops.aten.add.Tensor(arg409_1, 0.001);  arg409_1 = None
    sqrt_37: "f32[400]" = torch.ops.aten.sqrt.default(add_85);  add_85 = None
    reciprocal_37: "f32[400]" = torch.ops.aten.reciprocal.default(sqrt_37);  sqrt_37 = None
    mul_111: "f32[400]" = torch.ops.aten.mul.Tensor(reciprocal_37, 1);  reciprocal_37 = None
    unsqueeze_298: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(mul_111, -1);  mul_111 = None
    unsqueeze_299: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, -1);  unsqueeze_298 = None
    mul_112: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(sub_37, unsqueeze_299);  sub_37 = unsqueeze_299 = None
    unsqueeze_300: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg74_1, -1);  arg74_1 = None
    unsqueeze_301: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, -1);  unsqueeze_300 = None
    mul_113: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(mul_112, unsqueeze_301);  mul_112 = unsqueeze_301 = None
    unsqueeze_302: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg75_1, -1);  arg75_1 = None
    unsqueeze_303: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, -1);  unsqueeze_302 = None
    add_86: "f32[8, 400, 28, 28]" = torch.ops.aten.add.Tensor(mul_113, unsqueeze_303);  mul_113 = unsqueeze_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_37: "f32[8, 400, 28, 28]" = torch.ops.aten.relu.default(add_86);  add_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_37: "f32[8, 400, 28, 28]" = torch.ops.aten.convolution.default(relu_37, arg259_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_37 = arg259_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_304: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg410_1, -1);  arg410_1 = None
    unsqueeze_305: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, -1);  unsqueeze_304 = None
    sub_38: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_305);  convolution_37 = unsqueeze_305 = None
    add_87: "f32[400]" = torch.ops.aten.add.Tensor(arg411_1, 0.001);  arg411_1 = None
    sqrt_38: "f32[400]" = torch.ops.aten.sqrt.default(add_87);  add_87 = None
    reciprocal_38: "f32[400]" = torch.ops.aten.reciprocal.default(sqrt_38);  sqrt_38 = None
    mul_114: "f32[400]" = torch.ops.aten.mul.Tensor(reciprocal_38, 1);  reciprocal_38 = None
    unsqueeze_306: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(mul_114, -1);  mul_114 = None
    unsqueeze_307: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, -1);  unsqueeze_306 = None
    mul_115: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(sub_38, unsqueeze_307);  sub_38 = unsqueeze_307 = None
    unsqueeze_308: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg76_1, -1);  arg76_1 = None
    unsqueeze_309: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, -1);  unsqueeze_308 = None
    mul_116: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(mul_115, unsqueeze_309);  mul_115 = unsqueeze_309 = None
    unsqueeze_310: "f32[400, 1]" = torch.ops.aten.unsqueeze.default(arg77_1, -1);  arg77_1 = None
    unsqueeze_311: "f32[400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, -1);  unsqueeze_310 = None
    add_88: "f32[8, 400, 28, 28]" = torch.ops.aten.add.Tensor(mul_116, unsqueeze_311);  mul_116 = unsqueeze_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_38: "f32[8, 400, 28, 28]" = torch.ops.aten.relu.default(add_88);  add_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_38: "f32[8, 576, 28, 28]" = torch.ops.aten.convolution.default(relu_38, arg260_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_38 = arg260_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_106: "f32[8, 512, 28, 28]" = torch.ops.aten.slice.Tensor(convolution_38, 1, 0, 512)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    add_89: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_82, slice_106);  add_82 = slice_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_110: "f32[8, 64, 28, 28]" = torch.ops.aten.slice.Tensor(convolution_38, 1, 512, 9223372036854775807);  convolution_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    cat_22: "f32[8, 640, 28, 28]" = torch.ops.aten.cat.default([cat_20, slice_110], 1);  cat_20 = slice_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    cat_23: "f32[8, 1152, 28, 28]" = torch.ops.aten.cat.default([add_89, cat_22], 1);  add_89 = cat_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_312: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg412_1, -1);  arg412_1 = None
    unsqueeze_313: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, -1);  unsqueeze_312 = None
    sub_39: "f32[8, 1152, 28, 28]" = torch.ops.aten.sub.Tensor(cat_23, unsqueeze_313);  unsqueeze_313 = None
    add_90: "f32[1152]" = torch.ops.aten.add.Tensor(arg413_1, 0.001);  arg413_1 = None
    sqrt_39: "f32[1152]" = torch.ops.aten.sqrt.default(add_90);  add_90 = None
    reciprocal_39: "f32[1152]" = torch.ops.aten.reciprocal.default(sqrt_39);  sqrt_39 = None
    mul_117: "f32[1152]" = torch.ops.aten.mul.Tensor(reciprocal_39, 1);  reciprocal_39 = None
    unsqueeze_314: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(mul_117, -1);  mul_117 = None
    unsqueeze_315: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, -1);  unsqueeze_314 = None
    mul_118: "f32[8, 1152, 28, 28]" = torch.ops.aten.mul.Tensor(sub_39, unsqueeze_315);  sub_39 = unsqueeze_315 = None
    unsqueeze_316: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg78_1, -1);  arg78_1 = None
    unsqueeze_317: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, -1);  unsqueeze_316 = None
    mul_119: "f32[8, 1152, 28, 28]" = torch.ops.aten.mul.Tensor(mul_118, unsqueeze_317);  mul_118 = unsqueeze_317 = None
    unsqueeze_318: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg79_1, -1);  arg79_1 = None
    unsqueeze_319: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, -1);  unsqueeze_318 = None
    add_91: "f32[8, 1152, 28, 28]" = torch.ops.aten.add.Tensor(mul_119, unsqueeze_319);  mul_119 = unsqueeze_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_39: "f32[8, 1152, 28, 28]" = torch.ops.aten.relu.default(add_91);  add_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_39: "f32[8, 1152, 14, 14]" = torch.ops.aten.convolution.default(relu_39, arg261_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  relu_39 = arg261_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:133, code: x_s1 = x_s[:, :self.num_1x1_c, :, :]
    slice_114: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_39, 1, 0, 1024)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_320: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg414_1, -1);  arg414_1 = None
    unsqueeze_321: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, -1);  unsqueeze_320 = None
    sub_40: "f32[8, 1152, 28, 28]" = torch.ops.aten.sub.Tensor(cat_23, unsqueeze_321);  cat_23 = unsqueeze_321 = None
    add_92: "f32[1152]" = torch.ops.aten.add.Tensor(arg415_1, 0.001);  arg415_1 = None
    sqrt_40: "f32[1152]" = torch.ops.aten.sqrt.default(add_92);  add_92 = None
    reciprocal_40: "f32[1152]" = torch.ops.aten.reciprocal.default(sqrt_40);  sqrt_40 = None
    mul_120: "f32[1152]" = torch.ops.aten.mul.Tensor(reciprocal_40, 1);  reciprocal_40 = None
    unsqueeze_322: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(mul_120, -1);  mul_120 = None
    unsqueeze_323: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, -1);  unsqueeze_322 = None
    mul_121: "f32[8, 1152, 28, 28]" = torch.ops.aten.mul.Tensor(sub_40, unsqueeze_323);  sub_40 = unsqueeze_323 = None
    unsqueeze_324: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg80_1, -1);  arg80_1 = None
    unsqueeze_325: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, -1);  unsqueeze_324 = None
    mul_122: "f32[8, 1152, 28, 28]" = torch.ops.aten.mul.Tensor(mul_121, unsqueeze_325);  mul_121 = unsqueeze_325 = None
    unsqueeze_326: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg81_1, -1);  arg81_1 = None
    unsqueeze_327: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, -1);  unsqueeze_326 = None
    add_93: "f32[8, 1152, 28, 28]" = torch.ops.aten.add.Tensor(mul_122, unsqueeze_327);  mul_122 = unsqueeze_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_40: "f32[8, 1152, 28, 28]" = torch.ops.aten.relu.default(add_93);  add_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_40: "f32[8, 800, 28, 28]" = torch.ops.aten.convolution.default(relu_40, arg262_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_40 = arg262_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_328: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg416_1, -1);  arg416_1 = None
    unsqueeze_329: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, -1);  unsqueeze_328 = None
    sub_41: "f32[8, 800, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_329);  convolution_40 = unsqueeze_329 = None
    add_94: "f32[800]" = torch.ops.aten.add.Tensor(arg417_1, 0.001);  arg417_1 = None
    sqrt_41: "f32[800]" = torch.ops.aten.sqrt.default(add_94);  add_94 = None
    reciprocal_41: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_41);  sqrt_41 = None
    mul_123: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_41, 1);  reciprocal_41 = None
    unsqueeze_330: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_123, -1);  mul_123 = None
    unsqueeze_331: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, -1);  unsqueeze_330 = None
    mul_124: "f32[8, 800, 28, 28]" = torch.ops.aten.mul.Tensor(sub_41, unsqueeze_331);  sub_41 = unsqueeze_331 = None
    unsqueeze_332: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg82_1, -1);  arg82_1 = None
    unsqueeze_333: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, -1);  unsqueeze_332 = None
    mul_125: "f32[8, 800, 28, 28]" = torch.ops.aten.mul.Tensor(mul_124, unsqueeze_333);  mul_124 = unsqueeze_333 = None
    unsqueeze_334: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg83_1, -1);  arg83_1 = None
    unsqueeze_335: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, -1);  unsqueeze_334 = None
    add_95: "f32[8, 800, 28, 28]" = torch.ops.aten.add.Tensor(mul_125, unsqueeze_335);  mul_125 = unsqueeze_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_41: "f32[8, 800, 28, 28]" = torch.ops.aten.relu.default(add_95);  add_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_41: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_41, arg263_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 50);  relu_41 = arg263_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_336: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg418_1, -1);  arg418_1 = None
    unsqueeze_337: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, -1);  unsqueeze_336 = None
    sub_42: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_337);  convolution_41 = unsqueeze_337 = None
    add_96: "f32[800]" = torch.ops.aten.add.Tensor(arg419_1, 0.001);  arg419_1 = None
    sqrt_42: "f32[800]" = torch.ops.aten.sqrt.default(add_96);  add_96 = None
    reciprocal_42: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_42);  sqrt_42 = None
    mul_126: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_42, 1);  reciprocal_42 = None
    unsqueeze_338: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_126, -1);  mul_126 = None
    unsqueeze_339: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, -1);  unsqueeze_338 = None
    mul_127: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_42, unsqueeze_339);  sub_42 = unsqueeze_339 = None
    unsqueeze_340: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg84_1, -1);  arg84_1 = None
    unsqueeze_341: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, -1);  unsqueeze_340 = None
    mul_128: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_127, unsqueeze_341);  mul_127 = unsqueeze_341 = None
    unsqueeze_342: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg85_1, -1);  arg85_1 = None
    unsqueeze_343: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, -1);  unsqueeze_342 = None
    add_97: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_128, unsqueeze_343);  mul_128 = unsqueeze_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_42: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_97);  add_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_42: "f32[8, 1088, 14, 14]" = torch.ops.aten.convolution.default(relu_42, arg264_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_42 = arg264_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_122: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_42, 1, 0, 1024)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    add_98: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(slice_114, slice_122);  slice_114 = slice_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:134, code: x_s2 = x_s[:, self.num_1x1_c:, :, :]
    slice_118: "f32[8, 128, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_39, 1, 1024, 9223372036854775807);  convolution_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_126: "f32[8, 64, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_42, 1, 1024, 9223372036854775807);  convolution_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    cat_24: "f32[8, 192, 14, 14]" = torch.ops.aten.cat.default([slice_118, slice_126], 1);  slice_118 = slice_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    cat_25: "f32[8, 1216, 14, 14]" = torch.ops.aten.cat.default([add_98, cat_24], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_344: "f32[1216, 1]" = torch.ops.aten.unsqueeze.default(arg420_1, -1);  arg420_1 = None
    unsqueeze_345: "f32[1216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, -1);  unsqueeze_344 = None
    sub_43: "f32[8, 1216, 14, 14]" = torch.ops.aten.sub.Tensor(cat_25, unsqueeze_345);  cat_25 = unsqueeze_345 = None
    add_99: "f32[1216]" = torch.ops.aten.add.Tensor(arg421_1, 0.001);  arg421_1 = None
    sqrt_43: "f32[1216]" = torch.ops.aten.sqrt.default(add_99);  add_99 = None
    reciprocal_43: "f32[1216]" = torch.ops.aten.reciprocal.default(sqrt_43);  sqrt_43 = None
    mul_129: "f32[1216]" = torch.ops.aten.mul.Tensor(reciprocal_43, 1);  reciprocal_43 = None
    unsqueeze_346: "f32[1216, 1]" = torch.ops.aten.unsqueeze.default(mul_129, -1);  mul_129 = None
    unsqueeze_347: "f32[1216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, -1);  unsqueeze_346 = None
    mul_130: "f32[8, 1216, 14, 14]" = torch.ops.aten.mul.Tensor(sub_43, unsqueeze_347);  sub_43 = unsqueeze_347 = None
    unsqueeze_348: "f32[1216, 1]" = torch.ops.aten.unsqueeze.default(arg86_1, -1);  arg86_1 = None
    unsqueeze_349: "f32[1216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, -1);  unsqueeze_348 = None
    mul_131: "f32[8, 1216, 14, 14]" = torch.ops.aten.mul.Tensor(mul_130, unsqueeze_349);  mul_130 = unsqueeze_349 = None
    unsqueeze_350: "f32[1216, 1]" = torch.ops.aten.unsqueeze.default(arg87_1, -1);  arg87_1 = None
    unsqueeze_351: "f32[1216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, -1);  unsqueeze_350 = None
    add_100: "f32[8, 1216, 14, 14]" = torch.ops.aten.add.Tensor(mul_131, unsqueeze_351);  mul_131 = unsqueeze_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_43: "f32[8, 1216, 14, 14]" = torch.ops.aten.relu.default(add_100);  add_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_43: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_43, arg265_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_43 = arg265_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_352: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg422_1, -1);  arg422_1 = None
    unsqueeze_353: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, -1);  unsqueeze_352 = None
    sub_44: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_353);  convolution_43 = unsqueeze_353 = None
    add_101: "f32[800]" = torch.ops.aten.add.Tensor(arg423_1, 0.001);  arg423_1 = None
    sqrt_44: "f32[800]" = torch.ops.aten.sqrt.default(add_101);  add_101 = None
    reciprocal_44: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_44);  sqrt_44 = None
    mul_132: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_44, 1);  reciprocal_44 = None
    unsqueeze_354: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_132, -1);  mul_132 = None
    unsqueeze_355: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, -1);  unsqueeze_354 = None
    mul_133: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_44, unsqueeze_355);  sub_44 = unsqueeze_355 = None
    unsqueeze_356: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg88_1, -1);  arg88_1 = None
    unsqueeze_357: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, -1);  unsqueeze_356 = None
    mul_134: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_133, unsqueeze_357);  mul_133 = unsqueeze_357 = None
    unsqueeze_358: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg89_1, -1);  arg89_1 = None
    unsqueeze_359: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, -1);  unsqueeze_358 = None
    add_102: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_134, unsqueeze_359);  mul_134 = unsqueeze_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_44: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_102);  add_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_44: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_44, arg266_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_44 = arg266_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_360: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg424_1, -1);  arg424_1 = None
    unsqueeze_361: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, -1);  unsqueeze_360 = None
    sub_45: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_361);  convolution_44 = unsqueeze_361 = None
    add_103: "f32[800]" = torch.ops.aten.add.Tensor(arg425_1, 0.001);  arg425_1 = None
    sqrt_45: "f32[800]" = torch.ops.aten.sqrt.default(add_103);  add_103 = None
    reciprocal_45: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_45);  sqrt_45 = None
    mul_135: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_45, 1);  reciprocal_45 = None
    unsqueeze_362: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_135, -1);  mul_135 = None
    unsqueeze_363: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, -1);  unsqueeze_362 = None
    mul_136: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_45, unsqueeze_363);  sub_45 = unsqueeze_363 = None
    unsqueeze_364: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg90_1, -1);  arg90_1 = None
    unsqueeze_365: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, -1);  unsqueeze_364 = None
    mul_137: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_136, unsqueeze_365);  mul_136 = unsqueeze_365 = None
    unsqueeze_366: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg91_1, -1);  arg91_1 = None
    unsqueeze_367: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, -1);  unsqueeze_366 = None
    add_104: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_137, unsqueeze_367);  mul_137 = unsqueeze_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_45: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_104);  add_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_45: "f32[8, 1088, 14, 14]" = torch.ops.aten.convolution.default(relu_45, arg267_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_45 = arg267_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_130: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_45, 1, 0, 1024)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    add_105: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_98, slice_130);  add_98 = slice_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_134: "f32[8, 64, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_45, 1, 1024, 9223372036854775807);  convolution_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    cat_26: "f32[8, 256, 14, 14]" = torch.ops.aten.cat.default([cat_24, slice_134], 1);  cat_24 = slice_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    cat_27: "f32[8, 1280, 14, 14]" = torch.ops.aten.cat.default([add_105, cat_26], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_368: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(arg426_1, -1);  arg426_1 = None
    unsqueeze_369: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, -1);  unsqueeze_368 = None
    sub_46: "f32[8, 1280, 14, 14]" = torch.ops.aten.sub.Tensor(cat_27, unsqueeze_369);  cat_27 = unsqueeze_369 = None
    add_106: "f32[1280]" = torch.ops.aten.add.Tensor(arg427_1, 0.001);  arg427_1 = None
    sqrt_46: "f32[1280]" = torch.ops.aten.sqrt.default(add_106);  add_106 = None
    reciprocal_46: "f32[1280]" = torch.ops.aten.reciprocal.default(sqrt_46);  sqrt_46 = None
    mul_138: "f32[1280]" = torch.ops.aten.mul.Tensor(reciprocal_46, 1);  reciprocal_46 = None
    unsqueeze_370: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(mul_138, -1);  mul_138 = None
    unsqueeze_371: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, -1);  unsqueeze_370 = None
    mul_139: "f32[8, 1280, 14, 14]" = torch.ops.aten.mul.Tensor(sub_46, unsqueeze_371);  sub_46 = unsqueeze_371 = None
    unsqueeze_372: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(arg92_1, -1);  arg92_1 = None
    unsqueeze_373: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_372, -1);  unsqueeze_372 = None
    mul_140: "f32[8, 1280, 14, 14]" = torch.ops.aten.mul.Tensor(mul_139, unsqueeze_373);  mul_139 = unsqueeze_373 = None
    unsqueeze_374: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(arg93_1, -1);  arg93_1 = None
    unsqueeze_375: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, -1);  unsqueeze_374 = None
    add_107: "f32[8, 1280, 14, 14]" = torch.ops.aten.add.Tensor(mul_140, unsqueeze_375);  mul_140 = unsqueeze_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_46: "f32[8, 1280, 14, 14]" = torch.ops.aten.relu.default(add_107);  add_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_46: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_46, arg268_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_46 = arg268_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_376: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg428_1, -1);  arg428_1 = None
    unsqueeze_377: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, -1);  unsqueeze_376 = None
    sub_47: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_377);  convolution_46 = unsqueeze_377 = None
    add_108: "f32[800]" = torch.ops.aten.add.Tensor(arg429_1, 0.001);  arg429_1 = None
    sqrt_47: "f32[800]" = torch.ops.aten.sqrt.default(add_108);  add_108 = None
    reciprocal_47: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_47);  sqrt_47 = None
    mul_141: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_47, 1);  reciprocal_47 = None
    unsqueeze_378: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_141, -1);  mul_141 = None
    unsqueeze_379: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, -1);  unsqueeze_378 = None
    mul_142: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_47, unsqueeze_379);  sub_47 = unsqueeze_379 = None
    unsqueeze_380: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg94_1, -1);  arg94_1 = None
    unsqueeze_381: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, -1);  unsqueeze_380 = None
    mul_143: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_142, unsqueeze_381);  mul_142 = unsqueeze_381 = None
    unsqueeze_382: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg95_1, -1);  arg95_1 = None
    unsqueeze_383: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, -1);  unsqueeze_382 = None
    add_109: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_143, unsqueeze_383);  mul_143 = unsqueeze_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_47: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_109);  add_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_47: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_47, arg269_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_47 = arg269_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_384: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg430_1, -1);  arg430_1 = None
    unsqueeze_385: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_384, -1);  unsqueeze_384 = None
    sub_48: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_385);  convolution_47 = unsqueeze_385 = None
    add_110: "f32[800]" = torch.ops.aten.add.Tensor(arg431_1, 0.001);  arg431_1 = None
    sqrt_48: "f32[800]" = torch.ops.aten.sqrt.default(add_110);  add_110 = None
    reciprocal_48: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_48);  sqrt_48 = None
    mul_144: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_48, 1);  reciprocal_48 = None
    unsqueeze_386: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_144, -1);  mul_144 = None
    unsqueeze_387: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, -1);  unsqueeze_386 = None
    mul_145: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_48, unsqueeze_387);  sub_48 = unsqueeze_387 = None
    unsqueeze_388: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg96_1, -1);  arg96_1 = None
    unsqueeze_389: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, -1);  unsqueeze_388 = None
    mul_146: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_145, unsqueeze_389);  mul_145 = unsqueeze_389 = None
    unsqueeze_390: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg97_1, -1);  arg97_1 = None
    unsqueeze_391: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_390, -1);  unsqueeze_390 = None
    add_111: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_146, unsqueeze_391);  mul_146 = unsqueeze_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_48: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_111);  add_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_48: "f32[8, 1088, 14, 14]" = torch.ops.aten.convolution.default(relu_48, arg270_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_48 = arg270_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_138: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_48, 1, 0, 1024)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    add_112: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_105, slice_138);  add_105 = slice_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_142: "f32[8, 64, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_48, 1, 1024, 9223372036854775807);  convolution_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    cat_28: "f32[8, 320, 14, 14]" = torch.ops.aten.cat.default([cat_26, slice_142], 1);  cat_26 = slice_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    cat_29: "f32[8, 1344, 14, 14]" = torch.ops.aten.cat.default([add_112, cat_28], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_392: "f32[1344, 1]" = torch.ops.aten.unsqueeze.default(arg432_1, -1);  arg432_1 = None
    unsqueeze_393: "f32[1344, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, -1);  unsqueeze_392 = None
    sub_49: "f32[8, 1344, 14, 14]" = torch.ops.aten.sub.Tensor(cat_29, unsqueeze_393);  cat_29 = unsqueeze_393 = None
    add_113: "f32[1344]" = torch.ops.aten.add.Tensor(arg433_1, 0.001);  arg433_1 = None
    sqrt_49: "f32[1344]" = torch.ops.aten.sqrt.default(add_113);  add_113 = None
    reciprocal_49: "f32[1344]" = torch.ops.aten.reciprocal.default(sqrt_49);  sqrt_49 = None
    mul_147: "f32[1344]" = torch.ops.aten.mul.Tensor(reciprocal_49, 1);  reciprocal_49 = None
    unsqueeze_394: "f32[1344, 1]" = torch.ops.aten.unsqueeze.default(mul_147, -1);  mul_147 = None
    unsqueeze_395: "f32[1344, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, -1);  unsqueeze_394 = None
    mul_148: "f32[8, 1344, 14, 14]" = torch.ops.aten.mul.Tensor(sub_49, unsqueeze_395);  sub_49 = unsqueeze_395 = None
    unsqueeze_396: "f32[1344, 1]" = torch.ops.aten.unsqueeze.default(arg98_1, -1);  arg98_1 = None
    unsqueeze_397: "f32[1344, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_396, -1);  unsqueeze_396 = None
    mul_149: "f32[8, 1344, 14, 14]" = torch.ops.aten.mul.Tensor(mul_148, unsqueeze_397);  mul_148 = unsqueeze_397 = None
    unsqueeze_398: "f32[1344, 1]" = torch.ops.aten.unsqueeze.default(arg99_1, -1);  arg99_1 = None
    unsqueeze_399: "f32[1344, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, -1);  unsqueeze_398 = None
    add_114: "f32[8, 1344, 14, 14]" = torch.ops.aten.add.Tensor(mul_149, unsqueeze_399);  mul_149 = unsqueeze_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_49: "f32[8, 1344, 14, 14]" = torch.ops.aten.relu.default(add_114);  add_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_49: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_49, arg271_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_49 = arg271_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_400: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg434_1, -1);  arg434_1 = None
    unsqueeze_401: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, -1);  unsqueeze_400 = None
    sub_50: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_401);  convolution_49 = unsqueeze_401 = None
    add_115: "f32[800]" = torch.ops.aten.add.Tensor(arg435_1, 0.001);  arg435_1 = None
    sqrt_50: "f32[800]" = torch.ops.aten.sqrt.default(add_115);  add_115 = None
    reciprocal_50: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_50);  sqrt_50 = None
    mul_150: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_50, 1);  reciprocal_50 = None
    unsqueeze_402: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_150, -1);  mul_150 = None
    unsqueeze_403: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, -1);  unsqueeze_402 = None
    mul_151: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_50, unsqueeze_403);  sub_50 = unsqueeze_403 = None
    unsqueeze_404: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg100_1, -1);  arg100_1 = None
    unsqueeze_405: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, -1);  unsqueeze_404 = None
    mul_152: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_151, unsqueeze_405);  mul_151 = unsqueeze_405 = None
    unsqueeze_406: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg101_1, -1);  arg101_1 = None
    unsqueeze_407: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, -1);  unsqueeze_406 = None
    add_116: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_152, unsqueeze_407);  mul_152 = unsqueeze_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_50: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_116);  add_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_50: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_50, arg272_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_50 = arg272_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_408: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg436_1, -1);  arg436_1 = None
    unsqueeze_409: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_408, -1);  unsqueeze_408 = None
    sub_51: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_409);  convolution_50 = unsqueeze_409 = None
    add_117: "f32[800]" = torch.ops.aten.add.Tensor(arg437_1, 0.001);  arg437_1 = None
    sqrt_51: "f32[800]" = torch.ops.aten.sqrt.default(add_117);  add_117 = None
    reciprocal_51: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_51);  sqrt_51 = None
    mul_153: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_51, 1);  reciprocal_51 = None
    unsqueeze_410: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_153, -1);  mul_153 = None
    unsqueeze_411: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, -1);  unsqueeze_410 = None
    mul_154: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_51, unsqueeze_411);  sub_51 = unsqueeze_411 = None
    unsqueeze_412: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg102_1, -1);  arg102_1 = None
    unsqueeze_413: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, -1);  unsqueeze_412 = None
    mul_155: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_154, unsqueeze_413);  mul_154 = unsqueeze_413 = None
    unsqueeze_414: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg103_1, -1);  arg103_1 = None
    unsqueeze_415: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, -1);  unsqueeze_414 = None
    add_118: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_155, unsqueeze_415);  mul_155 = unsqueeze_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_51: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_118);  add_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_51: "f32[8, 1088, 14, 14]" = torch.ops.aten.convolution.default(relu_51, arg273_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_51 = arg273_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_146: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_51, 1, 0, 1024)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    add_119: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_112, slice_146);  add_112 = slice_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_150: "f32[8, 64, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_51, 1, 1024, 9223372036854775807);  convolution_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    cat_30: "f32[8, 384, 14, 14]" = torch.ops.aten.cat.default([cat_28, slice_150], 1);  cat_28 = slice_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    cat_31: "f32[8, 1408, 14, 14]" = torch.ops.aten.cat.default([add_119, cat_30], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_416: "f32[1408, 1]" = torch.ops.aten.unsqueeze.default(arg438_1, -1);  arg438_1 = None
    unsqueeze_417: "f32[1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, -1);  unsqueeze_416 = None
    sub_52: "f32[8, 1408, 14, 14]" = torch.ops.aten.sub.Tensor(cat_31, unsqueeze_417);  cat_31 = unsqueeze_417 = None
    add_120: "f32[1408]" = torch.ops.aten.add.Tensor(arg439_1, 0.001);  arg439_1 = None
    sqrt_52: "f32[1408]" = torch.ops.aten.sqrt.default(add_120);  add_120 = None
    reciprocal_52: "f32[1408]" = torch.ops.aten.reciprocal.default(sqrt_52);  sqrt_52 = None
    mul_156: "f32[1408]" = torch.ops.aten.mul.Tensor(reciprocal_52, 1);  reciprocal_52 = None
    unsqueeze_418: "f32[1408, 1]" = torch.ops.aten.unsqueeze.default(mul_156, -1);  mul_156 = None
    unsqueeze_419: "f32[1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, -1);  unsqueeze_418 = None
    mul_157: "f32[8, 1408, 14, 14]" = torch.ops.aten.mul.Tensor(sub_52, unsqueeze_419);  sub_52 = unsqueeze_419 = None
    unsqueeze_420: "f32[1408, 1]" = torch.ops.aten.unsqueeze.default(arg104_1, -1);  arg104_1 = None
    unsqueeze_421: "f32[1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_420, -1);  unsqueeze_420 = None
    mul_158: "f32[8, 1408, 14, 14]" = torch.ops.aten.mul.Tensor(mul_157, unsqueeze_421);  mul_157 = unsqueeze_421 = None
    unsqueeze_422: "f32[1408, 1]" = torch.ops.aten.unsqueeze.default(arg105_1, -1);  arg105_1 = None
    unsqueeze_423: "f32[1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, -1);  unsqueeze_422 = None
    add_121: "f32[8, 1408, 14, 14]" = torch.ops.aten.add.Tensor(mul_158, unsqueeze_423);  mul_158 = unsqueeze_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_52: "f32[8, 1408, 14, 14]" = torch.ops.aten.relu.default(add_121);  add_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_52: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_52, arg274_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_52 = arg274_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_424: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg440_1, -1);  arg440_1 = None
    unsqueeze_425: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_424, -1);  unsqueeze_424 = None
    sub_53: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_425);  convolution_52 = unsqueeze_425 = None
    add_122: "f32[800]" = torch.ops.aten.add.Tensor(arg441_1, 0.001);  arg441_1 = None
    sqrt_53: "f32[800]" = torch.ops.aten.sqrt.default(add_122);  add_122 = None
    reciprocal_53: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_53);  sqrt_53 = None
    mul_159: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_53, 1);  reciprocal_53 = None
    unsqueeze_426: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_159, -1);  mul_159 = None
    unsqueeze_427: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, -1);  unsqueeze_426 = None
    mul_160: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_53, unsqueeze_427);  sub_53 = unsqueeze_427 = None
    unsqueeze_428: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg106_1, -1);  arg106_1 = None
    unsqueeze_429: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, -1);  unsqueeze_428 = None
    mul_161: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_160, unsqueeze_429);  mul_160 = unsqueeze_429 = None
    unsqueeze_430: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg107_1, -1);  arg107_1 = None
    unsqueeze_431: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, -1);  unsqueeze_430 = None
    add_123: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_161, unsqueeze_431);  mul_161 = unsqueeze_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_53: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_123);  add_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_53: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_53, arg275_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_53 = arg275_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_432: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg442_1, -1);  arg442_1 = None
    unsqueeze_433: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_432, -1);  unsqueeze_432 = None
    sub_54: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_433);  convolution_53 = unsqueeze_433 = None
    add_124: "f32[800]" = torch.ops.aten.add.Tensor(arg443_1, 0.001);  arg443_1 = None
    sqrt_54: "f32[800]" = torch.ops.aten.sqrt.default(add_124);  add_124 = None
    reciprocal_54: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_54);  sqrt_54 = None
    mul_162: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_54, 1);  reciprocal_54 = None
    unsqueeze_434: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_162, -1);  mul_162 = None
    unsqueeze_435: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, -1);  unsqueeze_434 = None
    mul_163: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_54, unsqueeze_435);  sub_54 = unsqueeze_435 = None
    unsqueeze_436: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg108_1, -1);  arg108_1 = None
    unsqueeze_437: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_436, -1);  unsqueeze_436 = None
    mul_164: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_163, unsqueeze_437);  mul_163 = unsqueeze_437 = None
    unsqueeze_438: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg109_1, -1);  arg109_1 = None
    unsqueeze_439: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_438, -1);  unsqueeze_438 = None
    add_125: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_164, unsqueeze_439);  mul_164 = unsqueeze_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_54: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_125);  add_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_54: "f32[8, 1088, 14, 14]" = torch.ops.aten.convolution.default(relu_54, arg276_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_54 = arg276_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_154: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_54, 1, 0, 1024)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    add_126: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_119, slice_154);  add_119 = slice_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_158: "f32[8, 64, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_54, 1, 1024, 9223372036854775807);  convolution_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    cat_32: "f32[8, 448, 14, 14]" = torch.ops.aten.cat.default([cat_30, slice_158], 1);  cat_30 = slice_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    cat_33: "f32[8, 1472, 14, 14]" = torch.ops.aten.cat.default([add_126, cat_32], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_440: "f32[1472, 1]" = torch.ops.aten.unsqueeze.default(arg444_1, -1);  arg444_1 = None
    unsqueeze_441: "f32[1472, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, -1);  unsqueeze_440 = None
    sub_55: "f32[8, 1472, 14, 14]" = torch.ops.aten.sub.Tensor(cat_33, unsqueeze_441);  cat_33 = unsqueeze_441 = None
    add_127: "f32[1472]" = torch.ops.aten.add.Tensor(arg445_1, 0.001);  arg445_1 = None
    sqrt_55: "f32[1472]" = torch.ops.aten.sqrt.default(add_127);  add_127 = None
    reciprocal_55: "f32[1472]" = torch.ops.aten.reciprocal.default(sqrt_55);  sqrt_55 = None
    mul_165: "f32[1472]" = torch.ops.aten.mul.Tensor(reciprocal_55, 1);  reciprocal_55 = None
    unsqueeze_442: "f32[1472, 1]" = torch.ops.aten.unsqueeze.default(mul_165, -1);  mul_165 = None
    unsqueeze_443: "f32[1472, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_442, -1);  unsqueeze_442 = None
    mul_166: "f32[8, 1472, 14, 14]" = torch.ops.aten.mul.Tensor(sub_55, unsqueeze_443);  sub_55 = unsqueeze_443 = None
    unsqueeze_444: "f32[1472, 1]" = torch.ops.aten.unsqueeze.default(arg110_1, -1);  arg110_1 = None
    unsqueeze_445: "f32[1472, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_444, -1);  unsqueeze_444 = None
    mul_167: "f32[8, 1472, 14, 14]" = torch.ops.aten.mul.Tensor(mul_166, unsqueeze_445);  mul_166 = unsqueeze_445 = None
    unsqueeze_446: "f32[1472, 1]" = torch.ops.aten.unsqueeze.default(arg111_1, -1);  arg111_1 = None
    unsqueeze_447: "f32[1472, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, -1);  unsqueeze_446 = None
    add_128: "f32[8, 1472, 14, 14]" = torch.ops.aten.add.Tensor(mul_167, unsqueeze_447);  mul_167 = unsqueeze_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_55: "f32[8, 1472, 14, 14]" = torch.ops.aten.relu.default(add_128);  add_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_55: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_55, arg277_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_55 = arg277_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_448: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg446_1, -1);  arg446_1 = None
    unsqueeze_449: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_448, -1);  unsqueeze_448 = None
    sub_56: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_449);  convolution_55 = unsqueeze_449 = None
    add_129: "f32[800]" = torch.ops.aten.add.Tensor(arg447_1, 0.001);  arg447_1 = None
    sqrt_56: "f32[800]" = torch.ops.aten.sqrt.default(add_129);  add_129 = None
    reciprocal_56: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_56);  sqrt_56 = None
    mul_168: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_56, 1);  reciprocal_56 = None
    unsqueeze_450: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_168, -1);  mul_168 = None
    unsqueeze_451: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_450, -1);  unsqueeze_450 = None
    mul_169: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_56, unsqueeze_451);  sub_56 = unsqueeze_451 = None
    unsqueeze_452: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg112_1, -1);  arg112_1 = None
    unsqueeze_453: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, -1);  unsqueeze_452 = None
    mul_170: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_169, unsqueeze_453);  mul_169 = unsqueeze_453 = None
    unsqueeze_454: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg113_1, -1);  arg113_1 = None
    unsqueeze_455: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_454, -1);  unsqueeze_454 = None
    add_130: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_170, unsqueeze_455);  mul_170 = unsqueeze_455 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_56: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_130);  add_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_56: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_56, arg278_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_56 = arg278_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_456: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg448_1, -1);  arg448_1 = None
    unsqueeze_457: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_456, -1);  unsqueeze_456 = None
    sub_57: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_457);  convolution_56 = unsqueeze_457 = None
    add_131: "f32[800]" = torch.ops.aten.add.Tensor(arg449_1, 0.001);  arg449_1 = None
    sqrt_57: "f32[800]" = torch.ops.aten.sqrt.default(add_131);  add_131 = None
    reciprocal_57: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_57);  sqrt_57 = None
    mul_171: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_57, 1);  reciprocal_57 = None
    unsqueeze_458: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_171, -1);  mul_171 = None
    unsqueeze_459: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, -1);  unsqueeze_458 = None
    mul_172: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_57, unsqueeze_459);  sub_57 = unsqueeze_459 = None
    unsqueeze_460: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg114_1, -1);  arg114_1 = None
    unsqueeze_461: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_460, -1);  unsqueeze_460 = None
    mul_173: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_172, unsqueeze_461);  mul_172 = unsqueeze_461 = None
    unsqueeze_462: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg115_1, -1);  arg115_1 = None
    unsqueeze_463: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_462, -1);  unsqueeze_462 = None
    add_132: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_173, unsqueeze_463);  mul_173 = unsqueeze_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_57: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_132);  add_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_57: "f32[8, 1088, 14, 14]" = torch.ops.aten.convolution.default(relu_57, arg279_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_57 = arg279_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_162: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_57, 1, 0, 1024)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    add_133: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_126, slice_162);  add_126 = slice_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_166: "f32[8, 64, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_57, 1, 1024, 9223372036854775807);  convolution_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    cat_34: "f32[8, 512, 14, 14]" = torch.ops.aten.cat.default([cat_32, slice_166], 1);  cat_32 = slice_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    cat_35: "f32[8, 1536, 14, 14]" = torch.ops.aten.cat.default([add_133, cat_34], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_464: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(arg450_1, -1);  arg450_1 = None
    unsqueeze_465: "f32[1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, -1);  unsqueeze_464 = None
    sub_58: "f32[8, 1536, 14, 14]" = torch.ops.aten.sub.Tensor(cat_35, unsqueeze_465);  cat_35 = unsqueeze_465 = None
    add_134: "f32[1536]" = torch.ops.aten.add.Tensor(arg451_1, 0.001);  arg451_1 = None
    sqrt_58: "f32[1536]" = torch.ops.aten.sqrt.default(add_134);  add_134 = None
    reciprocal_58: "f32[1536]" = torch.ops.aten.reciprocal.default(sqrt_58);  sqrt_58 = None
    mul_174: "f32[1536]" = torch.ops.aten.mul.Tensor(reciprocal_58, 1);  reciprocal_58 = None
    unsqueeze_466: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(mul_174, -1);  mul_174 = None
    unsqueeze_467: "f32[1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_466, -1);  unsqueeze_466 = None
    mul_175: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(sub_58, unsqueeze_467);  sub_58 = unsqueeze_467 = None
    unsqueeze_468: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(arg116_1, -1);  arg116_1 = None
    unsqueeze_469: "f32[1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_468, -1);  unsqueeze_468 = None
    mul_176: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_175, unsqueeze_469);  mul_175 = unsqueeze_469 = None
    unsqueeze_470: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(arg117_1, -1);  arg117_1 = None
    unsqueeze_471: "f32[1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, -1);  unsqueeze_470 = None
    add_135: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_176, unsqueeze_471);  mul_176 = unsqueeze_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_58: "f32[8, 1536, 14, 14]" = torch.ops.aten.relu.default(add_135);  add_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_58: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_58, arg280_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_58 = arg280_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_472: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg452_1, -1);  arg452_1 = None
    unsqueeze_473: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_472, -1);  unsqueeze_472 = None
    sub_59: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_58, unsqueeze_473);  convolution_58 = unsqueeze_473 = None
    add_136: "f32[800]" = torch.ops.aten.add.Tensor(arg453_1, 0.001);  arg453_1 = None
    sqrt_59: "f32[800]" = torch.ops.aten.sqrt.default(add_136);  add_136 = None
    reciprocal_59: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_59);  sqrt_59 = None
    mul_177: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_59, 1);  reciprocal_59 = None
    unsqueeze_474: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_177, -1);  mul_177 = None
    unsqueeze_475: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_474, -1);  unsqueeze_474 = None
    mul_178: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_59, unsqueeze_475);  sub_59 = unsqueeze_475 = None
    unsqueeze_476: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg118_1, -1);  arg118_1 = None
    unsqueeze_477: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, -1);  unsqueeze_476 = None
    mul_179: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_178, unsqueeze_477);  mul_178 = unsqueeze_477 = None
    unsqueeze_478: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg119_1, -1);  arg119_1 = None
    unsqueeze_479: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_478, -1);  unsqueeze_478 = None
    add_137: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_179, unsqueeze_479);  mul_179 = unsqueeze_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_59: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_137);  add_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_59: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_59, arg281_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_59 = arg281_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_480: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg454_1, -1);  arg454_1 = None
    unsqueeze_481: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_480, -1);  unsqueeze_480 = None
    sub_60: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_481);  convolution_59 = unsqueeze_481 = None
    add_138: "f32[800]" = torch.ops.aten.add.Tensor(arg455_1, 0.001);  arg455_1 = None
    sqrt_60: "f32[800]" = torch.ops.aten.sqrt.default(add_138);  add_138 = None
    reciprocal_60: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_60);  sqrt_60 = None
    mul_180: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_60, 1);  reciprocal_60 = None
    unsqueeze_482: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_180, -1);  mul_180 = None
    unsqueeze_483: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, -1);  unsqueeze_482 = None
    mul_181: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_60, unsqueeze_483);  sub_60 = unsqueeze_483 = None
    unsqueeze_484: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg120_1, -1);  arg120_1 = None
    unsqueeze_485: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_484, -1);  unsqueeze_484 = None
    mul_182: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_181, unsqueeze_485);  mul_181 = unsqueeze_485 = None
    unsqueeze_486: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg121_1, -1);  arg121_1 = None
    unsqueeze_487: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_486, -1);  unsqueeze_486 = None
    add_139: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_182, unsqueeze_487);  mul_182 = unsqueeze_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_60: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_139);  add_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_60: "f32[8, 1088, 14, 14]" = torch.ops.aten.convolution.default(relu_60, arg282_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_60 = arg282_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_170: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_60, 1, 0, 1024)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    add_140: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_133, slice_170);  add_133 = slice_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_174: "f32[8, 64, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_60, 1, 1024, 9223372036854775807);  convolution_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    cat_36: "f32[8, 576, 14, 14]" = torch.ops.aten.cat.default([cat_34, slice_174], 1);  cat_34 = slice_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    cat_37: "f32[8, 1600, 14, 14]" = torch.ops.aten.cat.default([add_140, cat_36], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_488: "f32[1600, 1]" = torch.ops.aten.unsqueeze.default(arg456_1, -1);  arg456_1 = None
    unsqueeze_489: "f32[1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, -1);  unsqueeze_488 = None
    sub_61: "f32[8, 1600, 14, 14]" = torch.ops.aten.sub.Tensor(cat_37, unsqueeze_489);  cat_37 = unsqueeze_489 = None
    add_141: "f32[1600]" = torch.ops.aten.add.Tensor(arg457_1, 0.001);  arg457_1 = None
    sqrt_61: "f32[1600]" = torch.ops.aten.sqrt.default(add_141);  add_141 = None
    reciprocal_61: "f32[1600]" = torch.ops.aten.reciprocal.default(sqrt_61);  sqrt_61 = None
    mul_183: "f32[1600]" = torch.ops.aten.mul.Tensor(reciprocal_61, 1);  reciprocal_61 = None
    unsqueeze_490: "f32[1600, 1]" = torch.ops.aten.unsqueeze.default(mul_183, -1);  mul_183 = None
    unsqueeze_491: "f32[1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_490, -1);  unsqueeze_490 = None
    mul_184: "f32[8, 1600, 14, 14]" = torch.ops.aten.mul.Tensor(sub_61, unsqueeze_491);  sub_61 = unsqueeze_491 = None
    unsqueeze_492: "f32[1600, 1]" = torch.ops.aten.unsqueeze.default(arg122_1, -1);  arg122_1 = None
    unsqueeze_493: "f32[1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_492, -1);  unsqueeze_492 = None
    mul_185: "f32[8, 1600, 14, 14]" = torch.ops.aten.mul.Tensor(mul_184, unsqueeze_493);  mul_184 = unsqueeze_493 = None
    unsqueeze_494: "f32[1600, 1]" = torch.ops.aten.unsqueeze.default(arg123_1, -1);  arg123_1 = None
    unsqueeze_495: "f32[1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, -1);  unsqueeze_494 = None
    add_142: "f32[8, 1600, 14, 14]" = torch.ops.aten.add.Tensor(mul_185, unsqueeze_495);  mul_185 = unsqueeze_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_61: "f32[8, 1600, 14, 14]" = torch.ops.aten.relu.default(add_142);  add_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_61: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_61, arg283_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_61 = arg283_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_496: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg458_1, -1);  arg458_1 = None
    unsqueeze_497: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_496, -1);  unsqueeze_496 = None
    sub_62: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_497);  convolution_61 = unsqueeze_497 = None
    add_143: "f32[800]" = torch.ops.aten.add.Tensor(arg459_1, 0.001);  arg459_1 = None
    sqrt_62: "f32[800]" = torch.ops.aten.sqrt.default(add_143);  add_143 = None
    reciprocal_62: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_62);  sqrt_62 = None
    mul_186: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_62, 1);  reciprocal_62 = None
    unsqueeze_498: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_186, -1);  mul_186 = None
    unsqueeze_499: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_498, -1);  unsqueeze_498 = None
    mul_187: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_62, unsqueeze_499);  sub_62 = unsqueeze_499 = None
    unsqueeze_500: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg124_1, -1);  arg124_1 = None
    unsqueeze_501: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_500, -1);  unsqueeze_500 = None
    mul_188: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_187, unsqueeze_501);  mul_187 = unsqueeze_501 = None
    unsqueeze_502: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg125_1, -1);  arg125_1 = None
    unsqueeze_503: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_502, -1);  unsqueeze_502 = None
    add_144: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_188, unsqueeze_503);  mul_188 = unsqueeze_503 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_62: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_144);  add_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_62: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_62, arg284_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_62 = arg284_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_504: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg460_1, -1);  arg460_1 = None
    unsqueeze_505: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_504, -1);  unsqueeze_504 = None
    sub_63: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_62, unsqueeze_505);  convolution_62 = unsqueeze_505 = None
    add_145: "f32[800]" = torch.ops.aten.add.Tensor(arg461_1, 0.001);  arg461_1 = None
    sqrt_63: "f32[800]" = torch.ops.aten.sqrt.default(add_145);  add_145 = None
    reciprocal_63: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_63);  sqrt_63 = None
    mul_189: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_63, 1);  reciprocal_63 = None
    unsqueeze_506: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_189, -1);  mul_189 = None
    unsqueeze_507: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_506, -1);  unsqueeze_506 = None
    mul_190: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_63, unsqueeze_507);  sub_63 = unsqueeze_507 = None
    unsqueeze_508: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg126_1, -1);  arg126_1 = None
    unsqueeze_509: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_508, -1);  unsqueeze_508 = None
    mul_191: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_190, unsqueeze_509);  mul_190 = unsqueeze_509 = None
    unsqueeze_510: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg127_1, -1);  arg127_1 = None
    unsqueeze_511: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_510, -1);  unsqueeze_510 = None
    add_146: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_191, unsqueeze_511);  mul_191 = unsqueeze_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_63: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_146);  add_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_63: "f32[8, 1088, 14, 14]" = torch.ops.aten.convolution.default(relu_63, arg285_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_63 = arg285_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_178: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_63, 1, 0, 1024)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    add_147: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_140, slice_178);  add_140 = slice_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_182: "f32[8, 64, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_63, 1, 1024, 9223372036854775807);  convolution_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    cat_38: "f32[8, 640, 14, 14]" = torch.ops.aten.cat.default([cat_36, slice_182], 1);  cat_36 = slice_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    cat_39: "f32[8, 1664, 14, 14]" = torch.ops.aten.cat.default([add_147, cat_38], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_512: "f32[1664, 1]" = torch.ops.aten.unsqueeze.default(arg462_1, -1);  arg462_1 = None
    unsqueeze_513: "f32[1664, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_512, -1);  unsqueeze_512 = None
    sub_64: "f32[8, 1664, 14, 14]" = torch.ops.aten.sub.Tensor(cat_39, unsqueeze_513);  cat_39 = unsqueeze_513 = None
    add_148: "f32[1664]" = torch.ops.aten.add.Tensor(arg463_1, 0.001);  arg463_1 = None
    sqrt_64: "f32[1664]" = torch.ops.aten.sqrt.default(add_148);  add_148 = None
    reciprocal_64: "f32[1664]" = torch.ops.aten.reciprocal.default(sqrt_64);  sqrt_64 = None
    mul_192: "f32[1664]" = torch.ops.aten.mul.Tensor(reciprocal_64, 1);  reciprocal_64 = None
    unsqueeze_514: "f32[1664, 1]" = torch.ops.aten.unsqueeze.default(mul_192, -1);  mul_192 = None
    unsqueeze_515: "f32[1664, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_514, -1);  unsqueeze_514 = None
    mul_193: "f32[8, 1664, 14, 14]" = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_515);  sub_64 = unsqueeze_515 = None
    unsqueeze_516: "f32[1664, 1]" = torch.ops.aten.unsqueeze.default(arg128_1, -1);  arg128_1 = None
    unsqueeze_517: "f32[1664, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_516, -1);  unsqueeze_516 = None
    mul_194: "f32[8, 1664, 14, 14]" = torch.ops.aten.mul.Tensor(mul_193, unsqueeze_517);  mul_193 = unsqueeze_517 = None
    unsqueeze_518: "f32[1664, 1]" = torch.ops.aten.unsqueeze.default(arg129_1, -1);  arg129_1 = None
    unsqueeze_519: "f32[1664, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_518, -1);  unsqueeze_518 = None
    add_149: "f32[8, 1664, 14, 14]" = torch.ops.aten.add.Tensor(mul_194, unsqueeze_519);  mul_194 = unsqueeze_519 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_64: "f32[8, 1664, 14, 14]" = torch.ops.aten.relu.default(add_149);  add_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_64: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_64, arg286_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_64 = arg286_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_520: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg464_1, -1);  arg464_1 = None
    unsqueeze_521: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_520, -1);  unsqueeze_520 = None
    sub_65: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_64, unsqueeze_521);  convolution_64 = unsqueeze_521 = None
    add_150: "f32[800]" = torch.ops.aten.add.Tensor(arg465_1, 0.001);  arg465_1 = None
    sqrt_65: "f32[800]" = torch.ops.aten.sqrt.default(add_150);  add_150 = None
    reciprocal_65: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_65);  sqrt_65 = None
    mul_195: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_65, 1);  reciprocal_65 = None
    unsqueeze_522: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_195, -1);  mul_195 = None
    unsqueeze_523: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_522, -1);  unsqueeze_522 = None
    mul_196: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_65, unsqueeze_523);  sub_65 = unsqueeze_523 = None
    unsqueeze_524: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg130_1, -1);  arg130_1 = None
    unsqueeze_525: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_524, -1);  unsqueeze_524 = None
    mul_197: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_196, unsqueeze_525);  mul_196 = unsqueeze_525 = None
    unsqueeze_526: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg131_1, -1);  arg131_1 = None
    unsqueeze_527: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_526, -1);  unsqueeze_526 = None
    add_151: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_197, unsqueeze_527);  mul_197 = unsqueeze_527 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_65: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_151);  add_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_65: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_65, arg287_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_65 = arg287_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_528: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg466_1, -1);  arg466_1 = None
    unsqueeze_529: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_528, -1);  unsqueeze_528 = None
    sub_66: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_65, unsqueeze_529);  convolution_65 = unsqueeze_529 = None
    add_152: "f32[800]" = torch.ops.aten.add.Tensor(arg467_1, 0.001);  arg467_1 = None
    sqrt_66: "f32[800]" = torch.ops.aten.sqrt.default(add_152);  add_152 = None
    reciprocal_66: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_66);  sqrt_66 = None
    mul_198: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_66, 1);  reciprocal_66 = None
    unsqueeze_530: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_198, -1);  mul_198 = None
    unsqueeze_531: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_530, -1);  unsqueeze_530 = None
    mul_199: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_66, unsqueeze_531);  sub_66 = unsqueeze_531 = None
    unsqueeze_532: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg132_1, -1);  arg132_1 = None
    unsqueeze_533: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_532, -1);  unsqueeze_532 = None
    mul_200: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_199, unsqueeze_533);  mul_199 = unsqueeze_533 = None
    unsqueeze_534: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg133_1, -1);  arg133_1 = None
    unsqueeze_535: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_534, -1);  unsqueeze_534 = None
    add_153: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_200, unsqueeze_535);  mul_200 = unsqueeze_535 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_66: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_153);  add_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_66: "f32[8, 1088, 14, 14]" = torch.ops.aten.convolution.default(relu_66, arg288_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_66 = arg288_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_186: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_66, 1, 0, 1024)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    add_154: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_147, slice_186);  add_147 = slice_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_190: "f32[8, 64, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_66, 1, 1024, 9223372036854775807);  convolution_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    cat_40: "f32[8, 704, 14, 14]" = torch.ops.aten.cat.default([cat_38, slice_190], 1);  cat_38 = slice_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    cat_41: "f32[8, 1728, 14, 14]" = torch.ops.aten.cat.default([add_154, cat_40], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_536: "f32[1728, 1]" = torch.ops.aten.unsqueeze.default(arg468_1, -1);  arg468_1 = None
    unsqueeze_537: "f32[1728, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_536, -1);  unsqueeze_536 = None
    sub_67: "f32[8, 1728, 14, 14]" = torch.ops.aten.sub.Tensor(cat_41, unsqueeze_537);  cat_41 = unsqueeze_537 = None
    add_155: "f32[1728]" = torch.ops.aten.add.Tensor(arg469_1, 0.001);  arg469_1 = None
    sqrt_67: "f32[1728]" = torch.ops.aten.sqrt.default(add_155);  add_155 = None
    reciprocal_67: "f32[1728]" = torch.ops.aten.reciprocal.default(sqrt_67);  sqrt_67 = None
    mul_201: "f32[1728]" = torch.ops.aten.mul.Tensor(reciprocal_67, 1);  reciprocal_67 = None
    unsqueeze_538: "f32[1728, 1]" = torch.ops.aten.unsqueeze.default(mul_201, -1);  mul_201 = None
    unsqueeze_539: "f32[1728, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_538, -1);  unsqueeze_538 = None
    mul_202: "f32[8, 1728, 14, 14]" = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_539);  sub_67 = unsqueeze_539 = None
    unsqueeze_540: "f32[1728, 1]" = torch.ops.aten.unsqueeze.default(arg134_1, -1);  arg134_1 = None
    unsqueeze_541: "f32[1728, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_540, -1);  unsqueeze_540 = None
    mul_203: "f32[8, 1728, 14, 14]" = torch.ops.aten.mul.Tensor(mul_202, unsqueeze_541);  mul_202 = unsqueeze_541 = None
    unsqueeze_542: "f32[1728, 1]" = torch.ops.aten.unsqueeze.default(arg135_1, -1);  arg135_1 = None
    unsqueeze_543: "f32[1728, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_542, -1);  unsqueeze_542 = None
    add_156: "f32[8, 1728, 14, 14]" = torch.ops.aten.add.Tensor(mul_203, unsqueeze_543);  mul_203 = unsqueeze_543 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_67: "f32[8, 1728, 14, 14]" = torch.ops.aten.relu.default(add_156);  add_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_67: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_67, arg289_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_67 = arg289_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_544: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg470_1, -1);  arg470_1 = None
    unsqueeze_545: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_544, -1);  unsqueeze_544 = None
    sub_68: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_67, unsqueeze_545);  convolution_67 = unsqueeze_545 = None
    add_157: "f32[800]" = torch.ops.aten.add.Tensor(arg471_1, 0.001);  arg471_1 = None
    sqrt_68: "f32[800]" = torch.ops.aten.sqrt.default(add_157);  add_157 = None
    reciprocal_68: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_68);  sqrt_68 = None
    mul_204: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_68, 1);  reciprocal_68 = None
    unsqueeze_546: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_204, -1);  mul_204 = None
    unsqueeze_547: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_546, -1);  unsqueeze_546 = None
    mul_205: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_68, unsqueeze_547);  sub_68 = unsqueeze_547 = None
    unsqueeze_548: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg136_1, -1);  arg136_1 = None
    unsqueeze_549: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_548, -1);  unsqueeze_548 = None
    mul_206: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_205, unsqueeze_549);  mul_205 = unsqueeze_549 = None
    unsqueeze_550: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg137_1, -1);  arg137_1 = None
    unsqueeze_551: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_550, -1);  unsqueeze_550 = None
    add_158: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_206, unsqueeze_551);  mul_206 = unsqueeze_551 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_68: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_158);  add_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_68: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_68, arg290_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_68 = arg290_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_552: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg472_1, -1);  arg472_1 = None
    unsqueeze_553: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_552, -1);  unsqueeze_552 = None
    sub_69: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_68, unsqueeze_553);  convolution_68 = unsqueeze_553 = None
    add_159: "f32[800]" = torch.ops.aten.add.Tensor(arg473_1, 0.001);  arg473_1 = None
    sqrt_69: "f32[800]" = torch.ops.aten.sqrt.default(add_159);  add_159 = None
    reciprocal_69: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_69);  sqrt_69 = None
    mul_207: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_69, 1);  reciprocal_69 = None
    unsqueeze_554: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_207, -1);  mul_207 = None
    unsqueeze_555: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_554, -1);  unsqueeze_554 = None
    mul_208: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_69, unsqueeze_555);  sub_69 = unsqueeze_555 = None
    unsqueeze_556: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg138_1, -1);  arg138_1 = None
    unsqueeze_557: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_556, -1);  unsqueeze_556 = None
    mul_209: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_208, unsqueeze_557);  mul_208 = unsqueeze_557 = None
    unsqueeze_558: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg139_1, -1);  arg139_1 = None
    unsqueeze_559: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_558, -1);  unsqueeze_558 = None
    add_160: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_209, unsqueeze_559);  mul_209 = unsqueeze_559 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_69: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_160);  add_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_69: "f32[8, 1088, 14, 14]" = torch.ops.aten.convolution.default(relu_69, arg291_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_69 = arg291_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_194: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_69, 1, 0, 1024)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    add_161: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_154, slice_194);  add_154 = slice_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_198: "f32[8, 64, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_69, 1, 1024, 9223372036854775807);  convolution_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    cat_42: "f32[8, 768, 14, 14]" = torch.ops.aten.cat.default([cat_40, slice_198], 1);  cat_40 = slice_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    cat_43: "f32[8, 1792, 14, 14]" = torch.ops.aten.cat.default([add_161, cat_42], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_560: "f32[1792, 1]" = torch.ops.aten.unsqueeze.default(arg474_1, -1);  arg474_1 = None
    unsqueeze_561: "f32[1792, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_560, -1);  unsqueeze_560 = None
    sub_70: "f32[8, 1792, 14, 14]" = torch.ops.aten.sub.Tensor(cat_43, unsqueeze_561);  cat_43 = unsqueeze_561 = None
    add_162: "f32[1792]" = torch.ops.aten.add.Tensor(arg475_1, 0.001);  arg475_1 = None
    sqrt_70: "f32[1792]" = torch.ops.aten.sqrt.default(add_162);  add_162 = None
    reciprocal_70: "f32[1792]" = torch.ops.aten.reciprocal.default(sqrt_70);  sqrt_70 = None
    mul_210: "f32[1792]" = torch.ops.aten.mul.Tensor(reciprocal_70, 1);  reciprocal_70 = None
    unsqueeze_562: "f32[1792, 1]" = torch.ops.aten.unsqueeze.default(mul_210, -1);  mul_210 = None
    unsqueeze_563: "f32[1792, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_562, -1);  unsqueeze_562 = None
    mul_211: "f32[8, 1792, 14, 14]" = torch.ops.aten.mul.Tensor(sub_70, unsqueeze_563);  sub_70 = unsqueeze_563 = None
    unsqueeze_564: "f32[1792, 1]" = torch.ops.aten.unsqueeze.default(arg140_1, -1);  arg140_1 = None
    unsqueeze_565: "f32[1792, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_564, -1);  unsqueeze_564 = None
    mul_212: "f32[8, 1792, 14, 14]" = torch.ops.aten.mul.Tensor(mul_211, unsqueeze_565);  mul_211 = unsqueeze_565 = None
    unsqueeze_566: "f32[1792, 1]" = torch.ops.aten.unsqueeze.default(arg141_1, -1);  arg141_1 = None
    unsqueeze_567: "f32[1792, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_566, -1);  unsqueeze_566 = None
    add_163: "f32[8, 1792, 14, 14]" = torch.ops.aten.add.Tensor(mul_212, unsqueeze_567);  mul_212 = unsqueeze_567 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_70: "f32[8, 1792, 14, 14]" = torch.ops.aten.relu.default(add_163);  add_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_70: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_70, arg292_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_70 = arg292_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_568: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg476_1, -1);  arg476_1 = None
    unsqueeze_569: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_568, -1);  unsqueeze_568 = None
    sub_71: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_70, unsqueeze_569);  convolution_70 = unsqueeze_569 = None
    add_164: "f32[800]" = torch.ops.aten.add.Tensor(arg477_1, 0.001);  arg477_1 = None
    sqrt_71: "f32[800]" = torch.ops.aten.sqrt.default(add_164);  add_164 = None
    reciprocal_71: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_71);  sqrt_71 = None
    mul_213: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_71, 1);  reciprocal_71 = None
    unsqueeze_570: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_213, -1);  mul_213 = None
    unsqueeze_571: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_570, -1);  unsqueeze_570 = None
    mul_214: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_71, unsqueeze_571);  sub_71 = unsqueeze_571 = None
    unsqueeze_572: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg142_1, -1);  arg142_1 = None
    unsqueeze_573: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_572, -1);  unsqueeze_572 = None
    mul_215: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_214, unsqueeze_573);  mul_214 = unsqueeze_573 = None
    unsqueeze_574: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg143_1, -1);  arg143_1 = None
    unsqueeze_575: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_574, -1);  unsqueeze_574 = None
    add_165: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_215, unsqueeze_575);  mul_215 = unsqueeze_575 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_71: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_165);  add_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_71: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_71, arg293_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_71 = arg293_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_576: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg478_1, -1);  arg478_1 = None
    unsqueeze_577: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_576, -1);  unsqueeze_576 = None
    sub_72: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_71, unsqueeze_577);  convolution_71 = unsqueeze_577 = None
    add_166: "f32[800]" = torch.ops.aten.add.Tensor(arg479_1, 0.001);  arg479_1 = None
    sqrt_72: "f32[800]" = torch.ops.aten.sqrt.default(add_166);  add_166 = None
    reciprocal_72: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_72);  sqrt_72 = None
    mul_216: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_72, 1);  reciprocal_72 = None
    unsqueeze_578: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_216, -1);  mul_216 = None
    unsqueeze_579: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_578, -1);  unsqueeze_578 = None
    mul_217: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_72, unsqueeze_579);  sub_72 = unsqueeze_579 = None
    unsqueeze_580: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg144_1, -1);  arg144_1 = None
    unsqueeze_581: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_580, -1);  unsqueeze_580 = None
    mul_218: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_217, unsqueeze_581);  mul_217 = unsqueeze_581 = None
    unsqueeze_582: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg145_1, -1);  arg145_1 = None
    unsqueeze_583: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_582, -1);  unsqueeze_582 = None
    add_167: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_218, unsqueeze_583);  mul_218 = unsqueeze_583 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_72: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_167);  add_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_72: "f32[8, 1088, 14, 14]" = torch.ops.aten.convolution.default(relu_72, arg294_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_72 = arg294_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_202: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_72, 1, 0, 1024)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    add_168: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_161, slice_202);  add_161 = slice_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_206: "f32[8, 64, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_72, 1, 1024, 9223372036854775807);  convolution_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    cat_44: "f32[8, 832, 14, 14]" = torch.ops.aten.cat.default([cat_42, slice_206], 1);  cat_42 = slice_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    cat_45: "f32[8, 1856, 14, 14]" = torch.ops.aten.cat.default([add_168, cat_44], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_584: "f32[1856, 1]" = torch.ops.aten.unsqueeze.default(arg480_1, -1);  arg480_1 = None
    unsqueeze_585: "f32[1856, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_584, -1);  unsqueeze_584 = None
    sub_73: "f32[8, 1856, 14, 14]" = torch.ops.aten.sub.Tensor(cat_45, unsqueeze_585);  cat_45 = unsqueeze_585 = None
    add_169: "f32[1856]" = torch.ops.aten.add.Tensor(arg481_1, 0.001);  arg481_1 = None
    sqrt_73: "f32[1856]" = torch.ops.aten.sqrt.default(add_169);  add_169 = None
    reciprocal_73: "f32[1856]" = torch.ops.aten.reciprocal.default(sqrt_73);  sqrt_73 = None
    mul_219: "f32[1856]" = torch.ops.aten.mul.Tensor(reciprocal_73, 1);  reciprocal_73 = None
    unsqueeze_586: "f32[1856, 1]" = torch.ops.aten.unsqueeze.default(mul_219, -1);  mul_219 = None
    unsqueeze_587: "f32[1856, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_586, -1);  unsqueeze_586 = None
    mul_220: "f32[8, 1856, 14, 14]" = torch.ops.aten.mul.Tensor(sub_73, unsqueeze_587);  sub_73 = unsqueeze_587 = None
    unsqueeze_588: "f32[1856, 1]" = torch.ops.aten.unsqueeze.default(arg146_1, -1);  arg146_1 = None
    unsqueeze_589: "f32[1856, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_588, -1);  unsqueeze_588 = None
    mul_221: "f32[8, 1856, 14, 14]" = torch.ops.aten.mul.Tensor(mul_220, unsqueeze_589);  mul_220 = unsqueeze_589 = None
    unsqueeze_590: "f32[1856, 1]" = torch.ops.aten.unsqueeze.default(arg147_1, -1);  arg147_1 = None
    unsqueeze_591: "f32[1856, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_590, -1);  unsqueeze_590 = None
    add_170: "f32[8, 1856, 14, 14]" = torch.ops.aten.add.Tensor(mul_221, unsqueeze_591);  mul_221 = unsqueeze_591 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_73: "f32[8, 1856, 14, 14]" = torch.ops.aten.relu.default(add_170);  add_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_73: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_73, arg295_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_73 = arg295_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_592: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg482_1, -1);  arg482_1 = None
    unsqueeze_593: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_592, -1);  unsqueeze_592 = None
    sub_74: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_73, unsqueeze_593);  convolution_73 = unsqueeze_593 = None
    add_171: "f32[800]" = torch.ops.aten.add.Tensor(arg483_1, 0.001);  arg483_1 = None
    sqrt_74: "f32[800]" = torch.ops.aten.sqrt.default(add_171);  add_171 = None
    reciprocal_74: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_74);  sqrt_74 = None
    mul_222: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_74, 1);  reciprocal_74 = None
    unsqueeze_594: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_222, -1);  mul_222 = None
    unsqueeze_595: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_594, -1);  unsqueeze_594 = None
    mul_223: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_74, unsqueeze_595);  sub_74 = unsqueeze_595 = None
    unsqueeze_596: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg148_1, -1);  arg148_1 = None
    unsqueeze_597: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_596, -1);  unsqueeze_596 = None
    mul_224: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_223, unsqueeze_597);  mul_223 = unsqueeze_597 = None
    unsqueeze_598: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg149_1, -1);  arg149_1 = None
    unsqueeze_599: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_598, -1);  unsqueeze_598 = None
    add_172: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_224, unsqueeze_599);  mul_224 = unsqueeze_599 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_74: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_172);  add_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_74: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_74, arg296_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_74 = arg296_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_600: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg484_1, -1);  arg484_1 = None
    unsqueeze_601: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_600, -1);  unsqueeze_600 = None
    sub_75: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_74, unsqueeze_601);  convolution_74 = unsqueeze_601 = None
    add_173: "f32[800]" = torch.ops.aten.add.Tensor(arg485_1, 0.001);  arg485_1 = None
    sqrt_75: "f32[800]" = torch.ops.aten.sqrt.default(add_173);  add_173 = None
    reciprocal_75: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_75);  sqrt_75 = None
    mul_225: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_75, 1);  reciprocal_75 = None
    unsqueeze_602: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_225, -1);  mul_225 = None
    unsqueeze_603: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_602, -1);  unsqueeze_602 = None
    mul_226: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_75, unsqueeze_603);  sub_75 = unsqueeze_603 = None
    unsqueeze_604: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg150_1, -1);  arg150_1 = None
    unsqueeze_605: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_604, -1);  unsqueeze_604 = None
    mul_227: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_226, unsqueeze_605);  mul_226 = unsqueeze_605 = None
    unsqueeze_606: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg151_1, -1);  arg151_1 = None
    unsqueeze_607: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_606, -1);  unsqueeze_606 = None
    add_174: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_227, unsqueeze_607);  mul_227 = unsqueeze_607 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_75: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_174);  add_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_75: "f32[8, 1088, 14, 14]" = torch.ops.aten.convolution.default(relu_75, arg297_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_75 = arg297_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_210: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_75, 1, 0, 1024)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    add_175: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_168, slice_210);  add_168 = slice_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_214: "f32[8, 64, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_75, 1, 1024, 9223372036854775807);  convolution_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    cat_46: "f32[8, 896, 14, 14]" = torch.ops.aten.cat.default([cat_44, slice_214], 1);  cat_44 = slice_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    cat_47: "f32[8, 1920, 14, 14]" = torch.ops.aten.cat.default([add_175, cat_46], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_608: "f32[1920, 1]" = torch.ops.aten.unsqueeze.default(arg486_1, -1);  arg486_1 = None
    unsqueeze_609: "f32[1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_608, -1);  unsqueeze_608 = None
    sub_76: "f32[8, 1920, 14, 14]" = torch.ops.aten.sub.Tensor(cat_47, unsqueeze_609);  cat_47 = unsqueeze_609 = None
    add_176: "f32[1920]" = torch.ops.aten.add.Tensor(arg487_1, 0.001);  arg487_1 = None
    sqrt_76: "f32[1920]" = torch.ops.aten.sqrt.default(add_176);  add_176 = None
    reciprocal_76: "f32[1920]" = torch.ops.aten.reciprocal.default(sqrt_76);  sqrt_76 = None
    mul_228: "f32[1920]" = torch.ops.aten.mul.Tensor(reciprocal_76, 1);  reciprocal_76 = None
    unsqueeze_610: "f32[1920, 1]" = torch.ops.aten.unsqueeze.default(mul_228, -1);  mul_228 = None
    unsqueeze_611: "f32[1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_610, -1);  unsqueeze_610 = None
    mul_229: "f32[8, 1920, 14, 14]" = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_611);  sub_76 = unsqueeze_611 = None
    unsqueeze_612: "f32[1920, 1]" = torch.ops.aten.unsqueeze.default(arg152_1, -1);  arg152_1 = None
    unsqueeze_613: "f32[1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_612, -1);  unsqueeze_612 = None
    mul_230: "f32[8, 1920, 14, 14]" = torch.ops.aten.mul.Tensor(mul_229, unsqueeze_613);  mul_229 = unsqueeze_613 = None
    unsqueeze_614: "f32[1920, 1]" = torch.ops.aten.unsqueeze.default(arg153_1, -1);  arg153_1 = None
    unsqueeze_615: "f32[1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_614, -1);  unsqueeze_614 = None
    add_177: "f32[8, 1920, 14, 14]" = torch.ops.aten.add.Tensor(mul_230, unsqueeze_615);  mul_230 = unsqueeze_615 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_76: "f32[8, 1920, 14, 14]" = torch.ops.aten.relu.default(add_177);  add_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_76: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_76, arg298_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_76 = arg298_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_616: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg488_1, -1);  arg488_1 = None
    unsqueeze_617: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_616, -1);  unsqueeze_616 = None
    sub_77: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_76, unsqueeze_617);  convolution_76 = unsqueeze_617 = None
    add_178: "f32[800]" = torch.ops.aten.add.Tensor(arg489_1, 0.001);  arg489_1 = None
    sqrt_77: "f32[800]" = torch.ops.aten.sqrt.default(add_178);  add_178 = None
    reciprocal_77: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_77);  sqrt_77 = None
    mul_231: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_77, 1);  reciprocal_77 = None
    unsqueeze_618: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_231, -1);  mul_231 = None
    unsqueeze_619: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_618, -1);  unsqueeze_618 = None
    mul_232: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_77, unsqueeze_619);  sub_77 = unsqueeze_619 = None
    unsqueeze_620: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg154_1, -1);  arg154_1 = None
    unsqueeze_621: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_620, -1);  unsqueeze_620 = None
    mul_233: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_232, unsqueeze_621);  mul_232 = unsqueeze_621 = None
    unsqueeze_622: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg155_1, -1);  arg155_1 = None
    unsqueeze_623: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_622, -1);  unsqueeze_622 = None
    add_179: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_233, unsqueeze_623);  mul_233 = unsqueeze_623 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_77: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_179);  add_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_77: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_77, arg299_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_77 = arg299_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_624: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg490_1, -1);  arg490_1 = None
    unsqueeze_625: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_624, -1);  unsqueeze_624 = None
    sub_78: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_77, unsqueeze_625);  convolution_77 = unsqueeze_625 = None
    add_180: "f32[800]" = torch.ops.aten.add.Tensor(arg491_1, 0.001);  arg491_1 = None
    sqrt_78: "f32[800]" = torch.ops.aten.sqrt.default(add_180);  add_180 = None
    reciprocal_78: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_78);  sqrt_78 = None
    mul_234: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_78, 1);  reciprocal_78 = None
    unsqueeze_626: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_234, -1);  mul_234 = None
    unsqueeze_627: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_626, -1);  unsqueeze_626 = None
    mul_235: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_78, unsqueeze_627);  sub_78 = unsqueeze_627 = None
    unsqueeze_628: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg156_1, -1);  arg156_1 = None
    unsqueeze_629: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_628, -1);  unsqueeze_628 = None
    mul_236: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_235, unsqueeze_629);  mul_235 = unsqueeze_629 = None
    unsqueeze_630: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg157_1, -1);  arg157_1 = None
    unsqueeze_631: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_630, -1);  unsqueeze_630 = None
    add_181: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_236, unsqueeze_631);  mul_236 = unsqueeze_631 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_78: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_181);  add_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_78: "f32[8, 1088, 14, 14]" = torch.ops.aten.convolution.default(relu_78, arg300_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_78 = arg300_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_218: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_78, 1, 0, 1024)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    add_182: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_175, slice_218);  add_175 = slice_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_222: "f32[8, 64, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_78, 1, 1024, 9223372036854775807);  convolution_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    cat_48: "f32[8, 960, 14, 14]" = torch.ops.aten.cat.default([cat_46, slice_222], 1);  cat_46 = slice_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    cat_49: "f32[8, 1984, 14, 14]" = torch.ops.aten.cat.default([add_182, cat_48], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_632: "f32[1984, 1]" = torch.ops.aten.unsqueeze.default(arg492_1, -1);  arg492_1 = None
    unsqueeze_633: "f32[1984, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_632, -1);  unsqueeze_632 = None
    sub_79: "f32[8, 1984, 14, 14]" = torch.ops.aten.sub.Tensor(cat_49, unsqueeze_633);  cat_49 = unsqueeze_633 = None
    add_183: "f32[1984]" = torch.ops.aten.add.Tensor(arg493_1, 0.001);  arg493_1 = None
    sqrt_79: "f32[1984]" = torch.ops.aten.sqrt.default(add_183);  add_183 = None
    reciprocal_79: "f32[1984]" = torch.ops.aten.reciprocal.default(sqrt_79);  sqrt_79 = None
    mul_237: "f32[1984]" = torch.ops.aten.mul.Tensor(reciprocal_79, 1);  reciprocal_79 = None
    unsqueeze_634: "f32[1984, 1]" = torch.ops.aten.unsqueeze.default(mul_237, -1);  mul_237 = None
    unsqueeze_635: "f32[1984, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_634, -1);  unsqueeze_634 = None
    mul_238: "f32[8, 1984, 14, 14]" = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_635);  sub_79 = unsqueeze_635 = None
    unsqueeze_636: "f32[1984, 1]" = torch.ops.aten.unsqueeze.default(arg158_1, -1);  arg158_1 = None
    unsqueeze_637: "f32[1984, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_636, -1);  unsqueeze_636 = None
    mul_239: "f32[8, 1984, 14, 14]" = torch.ops.aten.mul.Tensor(mul_238, unsqueeze_637);  mul_238 = unsqueeze_637 = None
    unsqueeze_638: "f32[1984, 1]" = torch.ops.aten.unsqueeze.default(arg159_1, -1);  arg159_1 = None
    unsqueeze_639: "f32[1984, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_638, -1);  unsqueeze_638 = None
    add_184: "f32[8, 1984, 14, 14]" = torch.ops.aten.add.Tensor(mul_239, unsqueeze_639);  mul_239 = unsqueeze_639 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_79: "f32[8, 1984, 14, 14]" = torch.ops.aten.relu.default(add_184);  add_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_79: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_79, arg301_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_79 = arg301_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_640: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg494_1, -1);  arg494_1 = None
    unsqueeze_641: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_640, -1);  unsqueeze_640 = None
    sub_80: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_79, unsqueeze_641);  convolution_79 = unsqueeze_641 = None
    add_185: "f32[800]" = torch.ops.aten.add.Tensor(arg495_1, 0.001);  arg495_1 = None
    sqrt_80: "f32[800]" = torch.ops.aten.sqrt.default(add_185);  add_185 = None
    reciprocal_80: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_80);  sqrt_80 = None
    mul_240: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_80, 1);  reciprocal_80 = None
    unsqueeze_642: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_240, -1);  mul_240 = None
    unsqueeze_643: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_642, -1);  unsqueeze_642 = None
    mul_241: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_80, unsqueeze_643);  sub_80 = unsqueeze_643 = None
    unsqueeze_644: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg160_1, -1);  arg160_1 = None
    unsqueeze_645: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_644, -1);  unsqueeze_644 = None
    mul_242: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_241, unsqueeze_645);  mul_241 = unsqueeze_645 = None
    unsqueeze_646: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg161_1, -1);  arg161_1 = None
    unsqueeze_647: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_646, -1);  unsqueeze_646 = None
    add_186: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_242, unsqueeze_647);  mul_242 = unsqueeze_647 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_80: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_186);  add_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_80: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_80, arg302_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_80 = arg302_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_648: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg496_1, -1);  arg496_1 = None
    unsqueeze_649: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_648, -1);  unsqueeze_648 = None
    sub_81: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_80, unsqueeze_649);  convolution_80 = unsqueeze_649 = None
    add_187: "f32[800]" = torch.ops.aten.add.Tensor(arg497_1, 0.001);  arg497_1 = None
    sqrt_81: "f32[800]" = torch.ops.aten.sqrt.default(add_187);  add_187 = None
    reciprocal_81: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_81);  sqrt_81 = None
    mul_243: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_81, 1);  reciprocal_81 = None
    unsqueeze_650: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_243, -1);  mul_243 = None
    unsqueeze_651: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_650, -1);  unsqueeze_650 = None
    mul_244: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_81, unsqueeze_651);  sub_81 = unsqueeze_651 = None
    unsqueeze_652: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg162_1, -1);  arg162_1 = None
    unsqueeze_653: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_652, -1);  unsqueeze_652 = None
    mul_245: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_244, unsqueeze_653);  mul_244 = unsqueeze_653 = None
    unsqueeze_654: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg163_1, -1);  arg163_1 = None
    unsqueeze_655: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_654, -1);  unsqueeze_654 = None
    add_188: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_245, unsqueeze_655);  mul_245 = unsqueeze_655 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_81: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_188);  add_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_81: "f32[8, 1088, 14, 14]" = torch.ops.aten.convolution.default(relu_81, arg303_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_81 = arg303_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_226: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_81, 1, 0, 1024)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    add_189: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_182, slice_226);  add_182 = slice_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_230: "f32[8, 64, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_81, 1, 1024, 9223372036854775807);  convolution_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    cat_50: "f32[8, 1024, 14, 14]" = torch.ops.aten.cat.default([cat_48, slice_230], 1);  cat_48 = slice_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    cat_51: "f32[8, 2048, 14, 14]" = torch.ops.aten.cat.default([add_189, cat_50], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_656: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg498_1, -1);  arg498_1 = None
    unsqueeze_657: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_656, -1);  unsqueeze_656 = None
    sub_82: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(cat_51, unsqueeze_657);  cat_51 = unsqueeze_657 = None
    add_190: "f32[2048]" = torch.ops.aten.add.Tensor(arg499_1, 0.001);  arg499_1 = None
    sqrt_82: "f32[2048]" = torch.ops.aten.sqrt.default(add_190);  add_190 = None
    reciprocal_82: "f32[2048]" = torch.ops.aten.reciprocal.default(sqrt_82);  sqrt_82 = None
    mul_246: "f32[2048]" = torch.ops.aten.mul.Tensor(reciprocal_82, 1);  reciprocal_82 = None
    unsqueeze_658: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(mul_246, -1);  mul_246 = None
    unsqueeze_659: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_658, -1);  unsqueeze_658 = None
    mul_247: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_82, unsqueeze_659);  sub_82 = unsqueeze_659 = None
    unsqueeze_660: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg164_1, -1);  arg164_1 = None
    unsqueeze_661: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_660, -1);  unsqueeze_660 = None
    mul_248: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(mul_247, unsqueeze_661);  mul_247 = unsqueeze_661 = None
    unsqueeze_662: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg165_1, -1);  arg165_1 = None
    unsqueeze_663: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_662, -1);  unsqueeze_662 = None
    add_191: "f32[8, 2048, 14, 14]" = torch.ops.aten.add.Tensor(mul_248, unsqueeze_663);  mul_248 = unsqueeze_663 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_82: "f32[8, 2048, 14, 14]" = torch.ops.aten.relu.default(add_191);  add_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_82: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_82, arg304_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_82 = arg304_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_664: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg500_1, -1);  arg500_1 = None
    unsqueeze_665: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_664, -1);  unsqueeze_664 = None
    sub_83: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_82, unsqueeze_665);  convolution_82 = unsqueeze_665 = None
    add_192: "f32[800]" = torch.ops.aten.add.Tensor(arg501_1, 0.001);  arg501_1 = None
    sqrt_83: "f32[800]" = torch.ops.aten.sqrt.default(add_192);  add_192 = None
    reciprocal_83: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_83);  sqrt_83 = None
    mul_249: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_83, 1);  reciprocal_83 = None
    unsqueeze_666: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_249, -1);  mul_249 = None
    unsqueeze_667: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_666, -1);  unsqueeze_666 = None
    mul_250: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_83, unsqueeze_667);  sub_83 = unsqueeze_667 = None
    unsqueeze_668: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg166_1, -1);  arg166_1 = None
    unsqueeze_669: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_668, -1);  unsqueeze_668 = None
    mul_251: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_250, unsqueeze_669);  mul_250 = unsqueeze_669 = None
    unsqueeze_670: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg167_1, -1);  arg167_1 = None
    unsqueeze_671: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_670, -1);  unsqueeze_670 = None
    add_193: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_251, unsqueeze_671);  mul_251 = unsqueeze_671 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_83: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_193);  add_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_83: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_83, arg305_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_83 = arg305_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_672: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg502_1, -1);  arg502_1 = None
    unsqueeze_673: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_672, -1);  unsqueeze_672 = None
    sub_84: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_83, unsqueeze_673);  convolution_83 = unsqueeze_673 = None
    add_194: "f32[800]" = torch.ops.aten.add.Tensor(arg503_1, 0.001);  arg503_1 = None
    sqrt_84: "f32[800]" = torch.ops.aten.sqrt.default(add_194);  add_194 = None
    reciprocal_84: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_84);  sqrt_84 = None
    mul_252: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_84, 1);  reciprocal_84 = None
    unsqueeze_674: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_252, -1);  mul_252 = None
    unsqueeze_675: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_674, -1);  unsqueeze_674 = None
    mul_253: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_84, unsqueeze_675);  sub_84 = unsqueeze_675 = None
    unsqueeze_676: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg168_1, -1);  arg168_1 = None
    unsqueeze_677: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_676, -1);  unsqueeze_676 = None
    mul_254: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_253, unsqueeze_677);  mul_253 = unsqueeze_677 = None
    unsqueeze_678: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg169_1, -1);  arg169_1 = None
    unsqueeze_679: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_678, -1);  unsqueeze_678 = None
    add_195: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_254, unsqueeze_679);  mul_254 = unsqueeze_679 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_84: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_195);  add_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_84: "f32[8, 1088, 14, 14]" = torch.ops.aten.convolution.default(relu_84, arg306_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_84 = arg306_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_234: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_84, 1, 0, 1024)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    add_196: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_189, slice_234);  add_189 = slice_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_238: "f32[8, 64, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_84, 1, 1024, 9223372036854775807);  convolution_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    cat_52: "f32[8, 1088, 14, 14]" = torch.ops.aten.cat.default([cat_50, slice_238], 1);  cat_50 = slice_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    cat_53: "f32[8, 2112, 14, 14]" = torch.ops.aten.cat.default([add_196, cat_52], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_680: "f32[2112, 1]" = torch.ops.aten.unsqueeze.default(arg504_1, -1);  arg504_1 = None
    unsqueeze_681: "f32[2112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_680, -1);  unsqueeze_680 = None
    sub_85: "f32[8, 2112, 14, 14]" = torch.ops.aten.sub.Tensor(cat_53, unsqueeze_681);  cat_53 = unsqueeze_681 = None
    add_197: "f32[2112]" = torch.ops.aten.add.Tensor(arg505_1, 0.001);  arg505_1 = None
    sqrt_85: "f32[2112]" = torch.ops.aten.sqrt.default(add_197);  add_197 = None
    reciprocal_85: "f32[2112]" = torch.ops.aten.reciprocal.default(sqrt_85);  sqrt_85 = None
    mul_255: "f32[2112]" = torch.ops.aten.mul.Tensor(reciprocal_85, 1);  reciprocal_85 = None
    unsqueeze_682: "f32[2112, 1]" = torch.ops.aten.unsqueeze.default(mul_255, -1);  mul_255 = None
    unsqueeze_683: "f32[2112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_682, -1);  unsqueeze_682 = None
    mul_256: "f32[8, 2112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_85, unsqueeze_683);  sub_85 = unsqueeze_683 = None
    unsqueeze_684: "f32[2112, 1]" = torch.ops.aten.unsqueeze.default(arg170_1, -1);  arg170_1 = None
    unsqueeze_685: "f32[2112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_684, -1);  unsqueeze_684 = None
    mul_257: "f32[8, 2112, 14, 14]" = torch.ops.aten.mul.Tensor(mul_256, unsqueeze_685);  mul_256 = unsqueeze_685 = None
    unsqueeze_686: "f32[2112, 1]" = torch.ops.aten.unsqueeze.default(arg171_1, -1);  arg171_1 = None
    unsqueeze_687: "f32[2112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_686, -1);  unsqueeze_686 = None
    add_198: "f32[8, 2112, 14, 14]" = torch.ops.aten.add.Tensor(mul_257, unsqueeze_687);  mul_257 = unsqueeze_687 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_85: "f32[8, 2112, 14, 14]" = torch.ops.aten.relu.default(add_198);  add_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_85: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_85, arg307_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_85 = arg307_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_688: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg506_1, -1);  arg506_1 = None
    unsqueeze_689: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_688, -1);  unsqueeze_688 = None
    sub_86: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_85, unsqueeze_689);  convolution_85 = unsqueeze_689 = None
    add_199: "f32[800]" = torch.ops.aten.add.Tensor(arg507_1, 0.001);  arg507_1 = None
    sqrt_86: "f32[800]" = torch.ops.aten.sqrt.default(add_199);  add_199 = None
    reciprocal_86: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_86);  sqrt_86 = None
    mul_258: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_86, 1);  reciprocal_86 = None
    unsqueeze_690: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_258, -1);  mul_258 = None
    unsqueeze_691: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_690, -1);  unsqueeze_690 = None
    mul_259: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_86, unsqueeze_691);  sub_86 = unsqueeze_691 = None
    unsqueeze_692: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg172_1, -1);  arg172_1 = None
    unsqueeze_693: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_692, -1);  unsqueeze_692 = None
    mul_260: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_259, unsqueeze_693);  mul_259 = unsqueeze_693 = None
    unsqueeze_694: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg173_1, -1);  arg173_1 = None
    unsqueeze_695: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_694, -1);  unsqueeze_694 = None
    add_200: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_260, unsqueeze_695);  mul_260 = unsqueeze_695 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_86: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_200);  add_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_86: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_86, arg308_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_86 = arg308_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_696: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg508_1, -1);  arg508_1 = None
    unsqueeze_697: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_696, -1);  unsqueeze_696 = None
    sub_87: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_86, unsqueeze_697);  convolution_86 = unsqueeze_697 = None
    add_201: "f32[800]" = torch.ops.aten.add.Tensor(arg509_1, 0.001);  arg509_1 = None
    sqrt_87: "f32[800]" = torch.ops.aten.sqrt.default(add_201);  add_201 = None
    reciprocal_87: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_87);  sqrt_87 = None
    mul_261: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_87, 1);  reciprocal_87 = None
    unsqueeze_698: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_261, -1);  mul_261 = None
    unsqueeze_699: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_698, -1);  unsqueeze_698 = None
    mul_262: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_87, unsqueeze_699);  sub_87 = unsqueeze_699 = None
    unsqueeze_700: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg174_1, -1);  arg174_1 = None
    unsqueeze_701: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_700, -1);  unsqueeze_700 = None
    mul_263: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_262, unsqueeze_701);  mul_262 = unsqueeze_701 = None
    unsqueeze_702: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg175_1, -1);  arg175_1 = None
    unsqueeze_703: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_702, -1);  unsqueeze_702 = None
    add_202: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_263, unsqueeze_703);  mul_263 = unsqueeze_703 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_87: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_202);  add_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_87: "f32[8, 1088, 14, 14]" = torch.ops.aten.convolution.default(relu_87, arg309_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_87 = arg309_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_242: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_87, 1, 0, 1024)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    add_203: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_196, slice_242);  add_196 = slice_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_246: "f32[8, 64, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_87, 1, 1024, 9223372036854775807);  convolution_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    cat_54: "f32[8, 1152, 14, 14]" = torch.ops.aten.cat.default([cat_52, slice_246], 1);  cat_52 = slice_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    cat_55: "f32[8, 2176, 14, 14]" = torch.ops.aten.cat.default([add_203, cat_54], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_704: "f32[2176, 1]" = torch.ops.aten.unsqueeze.default(arg510_1, -1);  arg510_1 = None
    unsqueeze_705: "f32[2176, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_704, -1);  unsqueeze_704 = None
    sub_88: "f32[8, 2176, 14, 14]" = torch.ops.aten.sub.Tensor(cat_55, unsqueeze_705);  cat_55 = unsqueeze_705 = None
    add_204: "f32[2176]" = torch.ops.aten.add.Tensor(arg511_1, 0.001);  arg511_1 = None
    sqrt_88: "f32[2176]" = torch.ops.aten.sqrt.default(add_204);  add_204 = None
    reciprocal_88: "f32[2176]" = torch.ops.aten.reciprocal.default(sqrt_88);  sqrt_88 = None
    mul_264: "f32[2176]" = torch.ops.aten.mul.Tensor(reciprocal_88, 1);  reciprocal_88 = None
    unsqueeze_706: "f32[2176, 1]" = torch.ops.aten.unsqueeze.default(mul_264, -1);  mul_264 = None
    unsqueeze_707: "f32[2176, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_706, -1);  unsqueeze_706 = None
    mul_265: "f32[8, 2176, 14, 14]" = torch.ops.aten.mul.Tensor(sub_88, unsqueeze_707);  sub_88 = unsqueeze_707 = None
    unsqueeze_708: "f32[2176, 1]" = torch.ops.aten.unsqueeze.default(arg176_1, -1);  arg176_1 = None
    unsqueeze_709: "f32[2176, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_708, -1);  unsqueeze_708 = None
    mul_266: "f32[8, 2176, 14, 14]" = torch.ops.aten.mul.Tensor(mul_265, unsqueeze_709);  mul_265 = unsqueeze_709 = None
    unsqueeze_710: "f32[2176, 1]" = torch.ops.aten.unsqueeze.default(arg177_1, -1);  arg177_1 = None
    unsqueeze_711: "f32[2176, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_710, -1);  unsqueeze_710 = None
    add_205: "f32[8, 2176, 14, 14]" = torch.ops.aten.add.Tensor(mul_266, unsqueeze_711);  mul_266 = unsqueeze_711 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_88: "f32[8, 2176, 14, 14]" = torch.ops.aten.relu.default(add_205);  add_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_88: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_88, arg310_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_88 = arg310_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_712: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg512_1, -1);  arg512_1 = None
    unsqueeze_713: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_712, -1);  unsqueeze_712 = None
    sub_89: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_88, unsqueeze_713);  convolution_88 = unsqueeze_713 = None
    add_206: "f32[800]" = torch.ops.aten.add.Tensor(arg513_1, 0.001);  arg513_1 = None
    sqrt_89: "f32[800]" = torch.ops.aten.sqrt.default(add_206);  add_206 = None
    reciprocal_89: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_89);  sqrt_89 = None
    mul_267: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_89, 1);  reciprocal_89 = None
    unsqueeze_714: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_267, -1);  mul_267 = None
    unsqueeze_715: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_714, -1);  unsqueeze_714 = None
    mul_268: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_89, unsqueeze_715);  sub_89 = unsqueeze_715 = None
    unsqueeze_716: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg178_1, -1);  arg178_1 = None
    unsqueeze_717: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_716, -1);  unsqueeze_716 = None
    mul_269: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_268, unsqueeze_717);  mul_268 = unsqueeze_717 = None
    unsqueeze_718: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg179_1, -1);  arg179_1 = None
    unsqueeze_719: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_718, -1);  unsqueeze_718 = None
    add_207: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_269, unsqueeze_719);  mul_269 = unsqueeze_719 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_89: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_207);  add_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_89: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_89, arg311_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_89 = arg311_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_720: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg514_1, -1);  arg514_1 = None
    unsqueeze_721: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_720, -1);  unsqueeze_720 = None
    sub_90: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_89, unsqueeze_721);  convolution_89 = unsqueeze_721 = None
    add_208: "f32[800]" = torch.ops.aten.add.Tensor(arg515_1, 0.001);  arg515_1 = None
    sqrt_90: "f32[800]" = torch.ops.aten.sqrt.default(add_208);  add_208 = None
    reciprocal_90: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_90);  sqrt_90 = None
    mul_270: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_90, 1);  reciprocal_90 = None
    unsqueeze_722: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_270, -1);  mul_270 = None
    unsqueeze_723: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_722, -1);  unsqueeze_722 = None
    mul_271: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_90, unsqueeze_723);  sub_90 = unsqueeze_723 = None
    unsqueeze_724: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg180_1, -1);  arg180_1 = None
    unsqueeze_725: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_724, -1);  unsqueeze_724 = None
    mul_272: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_271, unsqueeze_725);  mul_271 = unsqueeze_725 = None
    unsqueeze_726: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg181_1, -1);  arg181_1 = None
    unsqueeze_727: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_726, -1);  unsqueeze_726 = None
    add_209: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_272, unsqueeze_727);  mul_272 = unsqueeze_727 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_90: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_209);  add_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_90: "f32[8, 1088, 14, 14]" = torch.ops.aten.convolution.default(relu_90, arg312_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_90 = arg312_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_250: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_90, 1, 0, 1024)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    add_210: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_203, slice_250);  add_203 = slice_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_254: "f32[8, 64, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_90, 1, 1024, 9223372036854775807);  convolution_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    cat_56: "f32[8, 1216, 14, 14]" = torch.ops.aten.cat.default([cat_54, slice_254], 1);  cat_54 = slice_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    cat_57: "f32[8, 2240, 14, 14]" = torch.ops.aten.cat.default([add_210, cat_56], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_728: "f32[2240, 1]" = torch.ops.aten.unsqueeze.default(arg516_1, -1);  arg516_1 = None
    unsqueeze_729: "f32[2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_728, -1);  unsqueeze_728 = None
    sub_91: "f32[8, 2240, 14, 14]" = torch.ops.aten.sub.Tensor(cat_57, unsqueeze_729);  cat_57 = unsqueeze_729 = None
    add_211: "f32[2240]" = torch.ops.aten.add.Tensor(arg517_1, 0.001);  arg517_1 = None
    sqrt_91: "f32[2240]" = torch.ops.aten.sqrt.default(add_211);  add_211 = None
    reciprocal_91: "f32[2240]" = torch.ops.aten.reciprocal.default(sqrt_91);  sqrt_91 = None
    mul_273: "f32[2240]" = torch.ops.aten.mul.Tensor(reciprocal_91, 1);  reciprocal_91 = None
    unsqueeze_730: "f32[2240, 1]" = torch.ops.aten.unsqueeze.default(mul_273, -1);  mul_273 = None
    unsqueeze_731: "f32[2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_730, -1);  unsqueeze_730 = None
    mul_274: "f32[8, 2240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_91, unsqueeze_731);  sub_91 = unsqueeze_731 = None
    unsqueeze_732: "f32[2240, 1]" = torch.ops.aten.unsqueeze.default(arg182_1, -1);  arg182_1 = None
    unsqueeze_733: "f32[2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_732, -1);  unsqueeze_732 = None
    mul_275: "f32[8, 2240, 14, 14]" = torch.ops.aten.mul.Tensor(mul_274, unsqueeze_733);  mul_274 = unsqueeze_733 = None
    unsqueeze_734: "f32[2240, 1]" = torch.ops.aten.unsqueeze.default(arg183_1, -1);  arg183_1 = None
    unsqueeze_735: "f32[2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_734, -1);  unsqueeze_734 = None
    add_212: "f32[8, 2240, 14, 14]" = torch.ops.aten.add.Tensor(mul_275, unsqueeze_735);  mul_275 = unsqueeze_735 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_91: "f32[8, 2240, 14, 14]" = torch.ops.aten.relu.default(add_212);  add_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_91: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_91, arg313_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_91 = arg313_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_736: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg518_1, -1);  arg518_1 = None
    unsqueeze_737: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_736, -1);  unsqueeze_736 = None
    sub_92: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_91, unsqueeze_737);  convolution_91 = unsqueeze_737 = None
    add_213: "f32[800]" = torch.ops.aten.add.Tensor(arg519_1, 0.001);  arg519_1 = None
    sqrt_92: "f32[800]" = torch.ops.aten.sqrt.default(add_213);  add_213 = None
    reciprocal_92: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_92);  sqrt_92 = None
    mul_276: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_92, 1);  reciprocal_92 = None
    unsqueeze_738: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_276, -1);  mul_276 = None
    unsqueeze_739: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_738, -1);  unsqueeze_738 = None
    mul_277: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_739);  sub_92 = unsqueeze_739 = None
    unsqueeze_740: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg184_1, -1);  arg184_1 = None
    unsqueeze_741: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_740, -1);  unsqueeze_740 = None
    mul_278: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_277, unsqueeze_741);  mul_277 = unsqueeze_741 = None
    unsqueeze_742: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg185_1, -1);  arg185_1 = None
    unsqueeze_743: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_742, -1);  unsqueeze_742 = None
    add_214: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_278, unsqueeze_743);  mul_278 = unsqueeze_743 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_92: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_214);  add_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_92: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_92, arg314_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_92 = arg314_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_744: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg520_1, -1);  arg520_1 = None
    unsqueeze_745: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_744, -1);  unsqueeze_744 = None
    sub_93: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_92, unsqueeze_745);  convolution_92 = unsqueeze_745 = None
    add_215: "f32[800]" = torch.ops.aten.add.Tensor(arg521_1, 0.001);  arg521_1 = None
    sqrt_93: "f32[800]" = torch.ops.aten.sqrt.default(add_215);  add_215 = None
    reciprocal_93: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_93);  sqrt_93 = None
    mul_279: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_93, 1);  reciprocal_93 = None
    unsqueeze_746: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_279, -1);  mul_279 = None
    unsqueeze_747: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_746, -1);  unsqueeze_746 = None
    mul_280: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_93, unsqueeze_747);  sub_93 = unsqueeze_747 = None
    unsqueeze_748: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg186_1, -1);  arg186_1 = None
    unsqueeze_749: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_748, -1);  unsqueeze_748 = None
    mul_281: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_280, unsqueeze_749);  mul_280 = unsqueeze_749 = None
    unsqueeze_750: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg187_1, -1);  arg187_1 = None
    unsqueeze_751: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_750, -1);  unsqueeze_750 = None
    add_216: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_281, unsqueeze_751);  mul_281 = unsqueeze_751 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_93: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_216);  add_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_93: "f32[8, 1088, 14, 14]" = torch.ops.aten.convolution.default(relu_93, arg315_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_93 = arg315_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_258: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_93, 1, 0, 1024)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    add_217: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_210, slice_258);  add_210 = slice_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_262: "f32[8, 64, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_93, 1, 1024, 9223372036854775807);  convolution_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    cat_58: "f32[8, 1280, 14, 14]" = torch.ops.aten.cat.default([cat_56, slice_262], 1);  cat_56 = slice_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    cat_59: "f32[8, 2304, 14, 14]" = torch.ops.aten.cat.default([add_217, cat_58], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_752: "f32[2304, 1]" = torch.ops.aten.unsqueeze.default(arg522_1, -1);  arg522_1 = None
    unsqueeze_753: "f32[2304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_752, -1);  unsqueeze_752 = None
    sub_94: "f32[8, 2304, 14, 14]" = torch.ops.aten.sub.Tensor(cat_59, unsqueeze_753);  cat_59 = unsqueeze_753 = None
    add_218: "f32[2304]" = torch.ops.aten.add.Tensor(arg523_1, 0.001);  arg523_1 = None
    sqrt_94: "f32[2304]" = torch.ops.aten.sqrt.default(add_218);  add_218 = None
    reciprocal_94: "f32[2304]" = torch.ops.aten.reciprocal.default(sqrt_94);  sqrt_94 = None
    mul_282: "f32[2304]" = torch.ops.aten.mul.Tensor(reciprocal_94, 1);  reciprocal_94 = None
    unsqueeze_754: "f32[2304, 1]" = torch.ops.aten.unsqueeze.default(mul_282, -1);  mul_282 = None
    unsqueeze_755: "f32[2304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_754, -1);  unsqueeze_754 = None
    mul_283: "f32[8, 2304, 14, 14]" = torch.ops.aten.mul.Tensor(sub_94, unsqueeze_755);  sub_94 = unsqueeze_755 = None
    unsqueeze_756: "f32[2304, 1]" = torch.ops.aten.unsqueeze.default(arg188_1, -1);  arg188_1 = None
    unsqueeze_757: "f32[2304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_756, -1);  unsqueeze_756 = None
    mul_284: "f32[8, 2304, 14, 14]" = torch.ops.aten.mul.Tensor(mul_283, unsqueeze_757);  mul_283 = unsqueeze_757 = None
    unsqueeze_758: "f32[2304, 1]" = torch.ops.aten.unsqueeze.default(arg189_1, -1);  arg189_1 = None
    unsqueeze_759: "f32[2304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_758, -1);  unsqueeze_758 = None
    add_219: "f32[8, 2304, 14, 14]" = torch.ops.aten.add.Tensor(mul_284, unsqueeze_759);  mul_284 = unsqueeze_759 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_94: "f32[8, 2304, 14, 14]" = torch.ops.aten.relu.default(add_219);  add_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_94: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_94, arg316_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_94 = arg316_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_760: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg524_1, -1);  arg524_1 = None
    unsqueeze_761: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_760, -1);  unsqueeze_760 = None
    sub_95: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_94, unsqueeze_761);  convolution_94 = unsqueeze_761 = None
    add_220: "f32[800]" = torch.ops.aten.add.Tensor(arg525_1, 0.001);  arg525_1 = None
    sqrt_95: "f32[800]" = torch.ops.aten.sqrt.default(add_220);  add_220 = None
    reciprocal_95: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_95);  sqrt_95 = None
    mul_285: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_95, 1);  reciprocal_95 = None
    unsqueeze_762: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_285, -1);  mul_285 = None
    unsqueeze_763: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_762, -1);  unsqueeze_762 = None
    mul_286: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_95, unsqueeze_763);  sub_95 = unsqueeze_763 = None
    unsqueeze_764: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg190_1, -1);  arg190_1 = None
    unsqueeze_765: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_764, -1);  unsqueeze_764 = None
    mul_287: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_286, unsqueeze_765);  mul_286 = unsqueeze_765 = None
    unsqueeze_766: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg191_1, -1);  arg191_1 = None
    unsqueeze_767: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_766, -1);  unsqueeze_766 = None
    add_221: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_287, unsqueeze_767);  mul_287 = unsqueeze_767 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_95: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_221);  add_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_95: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_95, arg317_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_95 = arg317_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_768: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg526_1, -1);  arg526_1 = None
    unsqueeze_769: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_768, -1);  unsqueeze_768 = None
    sub_96: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_95, unsqueeze_769);  convolution_95 = unsqueeze_769 = None
    add_222: "f32[800]" = torch.ops.aten.add.Tensor(arg527_1, 0.001);  arg527_1 = None
    sqrt_96: "f32[800]" = torch.ops.aten.sqrt.default(add_222);  add_222 = None
    reciprocal_96: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_96);  sqrt_96 = None
    mul_288: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_96, 1);  reciprocal_96 = None
    unsqueeze_770: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_288, -1);  mul_288 = None
    unsqueeze_771: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_770, -1);  unsqueeze_770 = None
    mul_289: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_771);  sub_96 = unsqueeze_771 = None
    unsqueeze_772: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg192_1, -1);  arg192_1 = None
    unsqueeze_773: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_772, -1);  unsqueeze_772 = None
    mul_290: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_289, unsqueeze_773);  mul_289 = unsqueeze_773 = None
    unsqueeze_774: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg193_1, -1);  arg193_1 = None
    unsqueeze_775: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_774, -1);  unsqueeze_774 = None
    add_223: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_290, unsqueeze_775);  mul_290 = unsqueeze_775 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_96: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_223);  add_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_96: "f32[8, 1088, 14, 14]" = torch.ops.aten.convolution.default(relu_96, arg318_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_96 = arg318_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_266: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_96, 1, 0, 1024)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    add_224: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_217, slice_266);  add_217 = slice_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_270: "f32[8, 64, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_96, 1, 1024, 9223372036854775807);  convolution_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    cat_60: "f32[8, 1344, 14, 14]" = torch.ops.aten.cat.default([cat_58, slice_270], 1);  cat_58 = slice_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    cat_61: "f32[8, 2368, 14, 14]" = torch.ops.aten.cat.default([add_224, cat_60], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_776: "f32[2368, 1]" = torch.ops.aten.unsqueeze.default(arg528_1, -1);  arg528_1 = None
    unsqueeze_777: "f32[2368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_776, -1);  unsqueeze_776 = None
    sub_97: "f32[8, 2368, 14, 14]" = torch.ops.aten.sub.Tensor(cat_61, unsqueeze_777);  cat_61 = unsqueeze_777 = None
    add_225: "f32[2368]" = torch.ops.aten.add.Tensor(arg529_1, 0.001);  arg529_1 = None
    sqrt_97: "f32[2368]" = torch.ops.aten.sqrt.default(add_225);  add_225 = None
    reciprocal_97: "f32[2368]" = torch.ops.aten.reciprocal.default(sqrt_97);  sqrt_97 = None
    mul_291: "f32[2368]" = torch.ops.aten.mul.Tensor(reciprocal_97, 1);  reciprocal_97 = None
    unsqueeze_778: "f32[2368, 1]" = torch.ops.aten.unsqueeze.default(mul_291, -1);  mul_291 = None
    unsqueeze_779: "f32[2368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_778, -1);  unsqueeze_778 = None
    mul_292: "f32[8, 2368, 14, 14]" = torch.ops.aten.mul.Tensor(sub_97, unsqueeze_779);  sub_97 = unsqueeze_779 = None
    unsqueeze_780: "f32[2368, 1]" = torch.ops.aten.unsqueeze.default(arg194_1, -1);  arg194_1 = None
    unsqueeze_781: "f32[2368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_780, -1);  unsqueeze_780 = None
    mul_293: "f32[8, 2368, 14, 14]" = torch.ops.aten.mul.Tensor(mul_292, unsqueeze_781);  mul_292 = unsqueeze_781 = None
    unsqueeze_782: "f32[2368, 1]" = torch.ops.aten.unsqueeze.default(arg195_1, -1);  arg195_1 = None
    unsqueeze_783: "f32[2368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_782, -1);  unsqueeze_782 = None
    add_226: "f32[8, 2368, 14, 14]" = torch.ops.aten.add.Tensor(mul_293, unsqueeze_783);  mul_293 = unsqueeze_783 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_97: "f32[8, 2368, 14, 14]" = torch.ops.aten.relu.default(add_226);  add_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_97: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_97, arg319_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_97 = arg319_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_784: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg530_1, -1);  arg530_1 = None
    unsqueeze_785: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_784, -1);  unsqueeze_784 = None
    sub_98: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_97, unsqueeze_785);  convolution_97 = unsqueeze_785 = None
    add_227: "f32[800]" = torch.ops.aten.add.Tensor(arg531_1, 0.001);  arg531_1 = None
    sqrt_98: "f32[800]" = torch.ops.aten.sqrt.default(add_227);  add_227 = None
    reciprocal_98: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_98);  sqrt_98 = None
    mul_294: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_98, 1);  reciprocal_98 = None
    unsqueeze_786: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_294, -1);  mul_294 = None
    unsqueeze_787: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_786, -1);  unsqueeze_786 = None
    mul_295: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_98, unsqueeze_787);  sub_98 = unsqueeze_787 = None
    unsqueeze_788: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg196_1, -1);  arg196_1 = None
    unsqueeze_789: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_788, -1);  unsqueeze_788 = None
    mul_296: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_295, unsqueeze_789);  mul_295 = unsqueeze_789 = None
    unsqueeze_790: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg197_1, -1);  arg197_1 = None
    unsqueeze_791: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_790, -1);  unsqueeze_790 = None
    add_228: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_296, unsqueeze_791);  mul_296 = unsqueeze_791 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_98: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_228);  add_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_98: "f32[8, 800, 14, 14]" = torch.ops.aten.convolution.default(relu_98, arg320_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_98 = arg320_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_792: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg532_1, -1);  arg532_1 = None
    unsqueeze_793: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_792, -1);  unsqueeze_792 = None
    sub_99: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_98, unsqueeze_793);  convolution_98 = unsqueeze_793 = None
    add_229: "f32[800]" = torch.ops.aten.add.Tensor(arg533_1, 0.001);  arg533_1 = None
    sqrt_99: "f32[800]" = torch.ops.aten.sqrt.default(add_229);  add_229 = None
    reciprocal_99: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_99);  sqrt_99 = None
    mul_297: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_99, 1);  reciprocal_99 = None
    unsqueeze_794: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_297, -1);  mul_297 = None
    unsqueeze_795: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_794, -1);  unsqueeze_794 = None
    mul_298: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_99, unsqueeze_795);  sub_99 = unsqueeze_795 = None
    unsqueeze_796: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg198_1, -1);  arg198_1 = None
    unsqueeze_797: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_796, -1);  unsqueeze_796 = None
    mul_299: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_298, unsqueeze_797);  mul_298 = unsqueeze_797 = None
    unsqueeze_798: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(arg199_1, -1);  arg199_1 = None
    unsqueeze_799: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_798, -1);  unsqueeze_798 = None
    add_230: "f32[8, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_299, unsqueeze_799);  mul_299 = unsqueeze_799 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_99: "f32[8, 800, 14, 14]" = torch.ops.aten.relu.default(add_230);  add_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_99: "f32[8, 1088, 14, 14]" = torch.ops.aten.convolution.default(relu_99, arg321_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_99 = arg321_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_274: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_99, 1, 0, 1024)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    add_231: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_224, slice_274);  add_224 = slice_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_278: "f32[8, 64, 14, 14]" = torch.ops.aten.slice.Tensor(convolution_99, 1, 1024, 9223372036854775807);  convolution_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    cat_62: "f32[8, 1408, 14, 14]" = torch.ops.aten.cat.default([cat_60, slice_278], 1);  cat_60 = slice_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    cat_63: "f32[8, 2432, 14, 14]" = torch.ops.aten.cat.default([add_231, cat_62], 1);  add_231 = cat_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_800: "f32[2432, 1]" = torch.ops.aten.unsqueeze.default(arg534_1, -1);  arg534_1 = None
    unsqueeze_801: "f32[2432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_800, -1);  unsqueeze_800 = None
    sub_100: "f32[8, 2432, 14, 14]" = torch.ops.aten.sub.Tensor(cat_63, unsqueeze_801);  unsqueeze_801 = None
    add_232: "f32[2432]" = torch.ops.aten.add.Tensor(arg535_1, 0.001);  arg535_1 = None
    sqrt_100: "f32[2432]" = torch.ops.aten.sqrt.default(add_232);  add_232 = None
    reciprocal_100: "f32[2432]" = torch.ops.aten.reciprocal.default(sqrt_100);  sqrt_100 = None
    mul_300: "f32[2432]" = torch.ops.aten.mul.Tensor(reciprocal_100, 1);  reciprocal_100 = None
    unsqueeze_802: "f32[2432, 1]" = torch.ops.aten.unsqueeze.default(mul_300, -1);  mul_300 = None
    unsqueeze_803: "f32[2432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_802, -1);  unsqueeze_802 = None
    mul_301: "f32[8, 2432, 14, 14]" = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_803);  sub_100 = unsqueeze_803 = None
    unsqueeze_804: "f32[2432, 1]" = torch.ops.aten.unsqueeze.default(arg200_1, -1);  arg200_1 = None
    unsqueeze_805: "f32[2432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_804, -1);  unsqueeze_804 = None
    mul_302: "f32[8, 2432, 14, 14]" = torch.ops.aten.mul.Tensor(mul_301, unsqueeze_805);  mul_301 = unsqueeze_805 = None
    unsqueeze_806: "f32[2432, 1]" = torch.ops.aten.unsqueeze.default(arg201_1, -1);  arg201_1 = None
    unsqueeze_807: "f32[2432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_806, -1);  unsqueeze_806 = None
    add_233: "f32[8, 2432, 14, 14]" = torch.ops.aten.add.Tensor(mul_302, unsqueeze_807);  mul_302 = unsqueeze_807 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_100: "f32[8, 2432, 14, 14]" = torch.ops.aten.relu.default(add_233);  add_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_100: "f32[8, 2304, 7, 7]" = torch.ops.aten.convolution.default(relu_100, arg322_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  relu_100 = arg322_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:133, code: x_s1 = x_s[:, :self.num_1x1_c, :, :]
    slice_282: "f32[8, 2048, 7, 7]" = torch.ops.aten.slice.Tensor(convolution_100, 1, 0, 2048)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_808: "f32[2432, 1]" = torch.ops.aten.unsqueeze.default(arg536_1, -1);  arg536_1 = None
    unsqueeze_809: "f32[2432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_808, -1);  unsqueeze_808 = None
    sub_101: "f32[8, 2432, 14, 14]" = torch.ops.aten.sub.Tensor(cat_63, unsqueeze_809);  cat_63 = unsqueeze_809 = None
    add_234: "f32[2432]" = torch.ops.aten.add.Tensor(arg537_1, 0.001);  arg537_1 = None
    sqrt_101: "f32[2432]" = torch.ops.aten.sqrt.default(add_234);  add_234 = None
    reciprocal_101: "f32[2432]" = torch.ops.aten.reciprocal.default(sqrt_101);  sqrt_101 = None
    mul_303: "f32[2432]" = torch.ops.aten.mul.Tensor(reciprocal_101, 1);  reciprocal_101 = None
    unsqueeze_810: "f32[2432, 1]" = torch.ops.aten.unsqueeze.default(mul_303, -1);  mul_303 = None
    unsqueeze_811: "f32[2432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_810, -1);  unsqueeze_810 = None
    mul_304: "f32[8, 2432, 14, 14]" = torch.ops.aten.mul.Tensor(sub_101, unsqueeze_811);  sub_101 = unsqueeze_811 = None
    unsqueeze_812: "f32[2432, 1]" = torch.ops.aten.unsqueeze.default(arg202_1, -1);  arg202_1 = None
    unsqueeze_813: "f32[2432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_812, -1);  unsqueeze_812 = None
    mul_305: "f32[8, 2432, 14, 14]" = torch.ops.aten.mul.Tensor(mul_304, unsqueeze_813);  mul_304 = unsqueeze_813 = None
    unsqueeze_814: "f32[2432, 1]" = torch.ops.aten.unsqueeze.default(arg203_1, -1);  arg203_1 = None
    unsqueeze_815: "f32[2432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_814, -1);  unsqueeze_814 = None
    add_235: "f32[8, 2432, 14, 14]" = torch.ops.aten.add.Tensor(mul_305, unsqueeze_815);  mul_305 = unsqueeze_815 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_101: "f32[8, 2432, 14, 14]" = torch.ops.aten.relu.default(add_235);  add_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_101: "f32[8, 1600, 14, 14]" = torch.ops.aten.convolution.default(relu_101, arg323_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_101 = arg323_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_816: "f32[1600, 1]" = torch.ops.aten.unsqueeze.default(arg538_1, -1);  arg538_1 = None
    unsqueeze_817: "f32[1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_816, -1);  unsqueeze_816 = None
    sub_102: "f32[8, 1600, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_101, unsqueeze_817);  convolution_101 = unsqueeze_817 = None
    add_236: "f32[1600]" = torch.ops.aten.add.Tensor(arg539_1, 0.001);  arg539_1 = None
    sqrt_102: "f32[1600]" = torch.ops.aten.sqrt.default(add_236);  add_236 = None
    reciprocal_102: "f32[1600]" = torch.ops.aten.reciprocal.default(sqrt_102);  sqrt_102 = None
    mul_306: "f32[1600]" = torch.ops.aten.mul.Tensor(reciprocal_102, 1);  reciprocal_102 = None
    unsqueeze_818: "f32[1600, 1]" = torch.ops.aten.unsqueeze.default(mul_306, -1);  mul_306 = None
    unsqueeze_819: "f32[1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_818, -1);  unsqueeze_818 = None
    mul_307: "f32[8, 1600, 14, 14]" = torch.ops.aten.mul.Tensor(sub_102, unsqueeze_819);  sub_102 = unsqueeze_819 = None
    unsqueeze_820: "f32[1600, 1]" = torch.ops.aten.unsqueeze.default(arg204_1, -1);  arg204_1 = None
    unsqueeze_821: "f32[1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_820, -1);  unsqueeze_820 = None
    mul_308: "f32[8, 1600, 14, 14]" = torch.ops.aten.mul.Tensor(mul_307, unsqueeze_821);  mul_307 = unsqueeze_821 = None
    unsqueeze_822: "f32[1600, 1]" = torch.ops.aten.unsqueeze.default(arg205_1, -1);  arg205_1 = None
    unsqueeze_823: "f32[1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_822, -1);  unsqueeze_822 = None
    add_237: "f32[8, 1600, 14, 14]" = torch.ops.aten.add.Tensor(mul_308, unsqueeze_823);  mul_308 = unsqueeze_823 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_102: "f32[8, 1600, 14, 14]" = torch.ops.aten.relu.default(add_237);  add_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_102: "f32[8, 1600, 7, 7]" = torch.ops.aten.convolution.default(relu_102, arg324_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 50);  relu_102 = arg324_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_824: "f32[1600, 1]" = torch.ops.aten.unsqueeze.default(arg540_1, -1);  arg540_1 = None
    unsqueeze_825: "f32[1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_824, -1);  unsqueeze_824 = None
    sub_103: "f32[8, 1600, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_102, unsqueeze_825);  convolution_102 = unsqueeze_825 = None
    add_238: "f32[1600]" = torch.ops.aten.add.Tensor(arg541_1, 0.001);  arg541_1 = None
    sqrt_103: "f32[1600]" = torch.ops.aten.sqrt.default(add_238);  add_238 = None
    reciprocal_103: "f32[1600]" = torch.ops.aten.reciprocal.default(sqrt_103);  sqrt_103 = None
    mul_309: "f32[1600]" = torch.ops.aten.mul.Tensor(reciprocal_103, 1);  reciprocal_103 = None
    unsqueeze_826: "f32[1600, 1]" = torch.ops.aten.unsqueeze.default(mul_309, -1);  mul_309 = None
    unsqueeze_827: "f32[1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_826, -1);  unsqueeze_826 = None
    mul_310: "f32[8, 1600, 7, 7]" = torch.ops.aten.mul.Tensor(sub_103, unsqueeze_827);  sub_103 = unsqueeze_827 = None
    unsqueeze_828: "f32[1600, 1]" = torch.ops.aten.unsqueeze.default(arg206_1, -1);  arg206_1 = None
    unsqueeze_829: "f32[1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_828, -1);  unsqueeze_828 = None
    mul_311: "f32[8, 1600, 7, 7]" = torch.ops.aten.mul.Tensor(mul_310, unsqueeze_829);  mul_310 = unsqueeze_829 = None
    unsqueeze_830: "f32[1600, 1]" = torch.ops.aten.unsqueeze.default(arg207_1, -1);  arg207_1 = None
    unsqueeze_831: "f32[1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_830, -1);  unsqueeze_830 = None
    add_239: "f32[8, 1600, 7, 7]" = torch.ops.aten.add.Tensor(mul_311, unsqueeze_831);  mul_311 = unsqueeze_831 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_103: "f32[8, 1600, 7, 7]" = torch.ops.aten.relu.default(add_239);  add_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_103: "f32[8, 2176, 7, 7]" = torch.ops.aten.convolution.default(relu_103, arg325_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_103 = arg325_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_290: "f32[8, 2048, 7, 7]" = torch.ops.aten.slice.Tensor(convolution_103, 1, 0, 2048)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    add_240: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(slice_282, slice_290);  slice_282 = slice_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:134, code: x_s2 = x_s[:, self.num_1x1_c:, :, :]
    slice_286: "f32[8, 256, 7, 7]" = torch.ops.aten.slice.Tensor(convolution_100, 1, 2048, 9223372036854775807);  convolution_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_294: "f32[8, 128, 7, 7]" = torch.ops.aten.slice.Tensor(convolution_103, 1, 2048, 9223372036854775807);  convolution_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    cat_64: "f32[8, 384, 7, 7]" = torch.ops.aten.cat.default([slice_286, slice_294], 1);  slice_286 = slice_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    cat_65: "f32[8, 2432, 7, 7]" = torch.ops.aten.cat.default([add_240, cat_64], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_832: "f32[2432, 1]" = torch.ops.aten.unsqueeze.default(arg542_1, -1);  arg542_1 = None
    unsqueeze_833: "f32[2432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_832, -1);  unsqueeze_832 = None
    sub_104: "f32[8, 2432, 7, 7]" = torch.ops.aten.sub.Tensor(cat_65, unsqueeze_833);  cat_65 = unsqueeze_833 = None
    add_241: "f32[2432]" = torch.ops.aten.add.Tensor(arg543_1, 0.001);  arg543_1 = None
    sqrt_104: "f32[2432]" = torch.ops.aten.sqrt.default(add_241);  add_241 = None
    reciprocal_104: "f32[2432]" = torch.ops.aten.reciprocal.default(sqrt_104);  sqrt_104 = None
    mul_312: "f32[2432]" = torch.ops.aten.mul.Tensor(reciprocal_104, 1);  reciprocal_104 = None
    unsqueeze_834: "f32[2432, 1]" = torch.ops.aten.unsqueeze.default(mul_312, -1);  mul_312 = None
    unsqueeze_835: "f32[2432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_834, -1);  unsqueeze_834 = None
    mul_313: "f32[8, 2432, 7, 7]" = torch.ops.aten.mul.Tensor(sub_104, unsqueeze_835);  sub_104 = unsqueeze_835 = None
    unsqueeze_836: "f32[2432, 1]" = torch.ops.aten.unsqueeze.default(arg208_1, -1);  arg208_1 = None
    unsqueeze_837: "f32[2432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_836, -1);  unsqueeze_836 = None
    mul_314: "f32[8, 2432, 7, 7]" = torch.ops.aten.mul.Tensor(mul_313, unsqueeze_837);  mul_313 = unsqueeze_837 = None
    unsqueeze_838: "f32[2432, 1]" = torch.ops.aten.unsqueeze.default(arg209_1, -1);  arg209_1 = None
    unsqueeze_839: "f32[2432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_838, -1);  unsqueeze_838 = None
    add_242: "f32[8, 2432, 7, 7]" = torch.ops.aten.add.Tensor(mul_314, unsqueeze_839);  mul_314 = unsqueeze_839 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_104: "f32[8, 2432, 7, 7]" = torch.ops.aten.relu.default(add_242);  add_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_104: "f32[8, 1600, 7, 7]" = torch.ops.aten.convolution.default(relu_104, arg326_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_104 = arg326_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_840: "f32[1600, 1]" = torch.ops.aten.unsqueeze.default(arg544_1, -1);  arg544_1 = None
    unsqueeze_841: "f32[1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_840, -1);  unsqueeze_840 = None
    sub_105: "f32[8, 1600, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_104, unsqueeze_841);  convolution_104 = unsqueeze_841 = None
    add_243: "f32[1600]" = torch.ops.aten.add.Tensor(arg545_1, 0.001);  arg545_1 = None
    sqrt_105: "f32[1600]" = torch.ops.aten.sqrt.default(add_243);  add_243 = None
    reciprocal_105: "f32[1600]" = torch.ops.aten.reciprocal.default(sqrt_105);  sqrt_105 = None
    mul_315: "f32[1600]" = torch.ops.aten.mul.Tensor(reciprocal_105, 1);  reciprocal_105 = None
    unsqueeze_842: "f32[1600, 1]" = torch.ops.aten.unsqueeze.default(mul_315, -1);  mul_315 = None
    unsqueeze_843: "f32[1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_842, -1);  unsqueeze_842 = None
    mul_316: "f32[8, 1600, 7, 7]" = torch.ops.aten.mul.Tensor(sub_105, unsqueeze_843);  sub_105 = unsqueeze_843 = None
    unsqueeze_844: "f32[1600, 1]" = torch.ops.aten.unsqueeze.default(arg210_1, -1);  arg210_1 = None
    unsqueeze_845: "f32[1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_844, -1);  unsqueeze_844 = None
    mul_317: "f32[8, 1600, 7, 7]" = torch.ops.aten.mul.Tensor(mul_316, unsqueeze_845);  mul_316 = unsqueeze_845 = None
    unsqueeze_846: "f32[1600, 1]" = torch.ops.aten.unsqueeze.default(arg211_1, -1);  arg211_1 = None
    unsqueeze_847: "f32[1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_846, -1);  unsqueeze_846 = None
    add_244: "f32[8, 1600, 7, 7]" = torch.ops.aten.add.Tensor(mul_317, unsqueeze_847);  mul_317 = unsqueeze_847 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_105: "f32[8, 1600, 7, 7]" = torch.ops.aten.relu.default(add_244);  add_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_105: "f32[8, 1600, 7, 7]" = torch.ops.aten.convolution.default(relu_105, arg327_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_105 = arg327_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_848: "f32[1600, 1]" = torch.ops.aten.unsqueeze.default(arg546_1, -1);  arg546_1 = None
    unsqueeze_849: "f32[1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_848, -1);  unsqueeze_848 = None
    sub_106: "f32[8, 1600, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_105, unsqueeze_849);  convolution_105 = unsqueeze_849 = None
    add_245: "f32[1600]" = torch.ops.aten.add.Tensor(arg547_1, 0.001);  arg547_1 = None
    sqrt_106: "f32[1600]" = torch.ops.aten.sqrt.default(add_245);  add_245 = None
    reciprocal_106: "f32[1600]" = torch.ops.aten.reciprocal.default(sqrt_106);  sqrt_106 = None
    mul_318: "f32[1600]" = torch.ops.aten.mul.Tensor(reciprocal_106, 1);  reciprocal_106 = None
    unsqueeze_850: "f32[1600, 1]" = torch.ops.aten.unsqueeze.default(mul_318, -1);  mul_318 = None
    unsqueeze_851: "f32[1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_850, -1);  unsqueeze_850 = None
    mul_319: "f32[8, 1600, 7, 7]" = torch.ops.aten.mul.Tensor(sub_106, unsqueeze_851);  sub_106 = unsqueeze_851 = None
    unsqueeze_852: "f32[1600, 1]" = torch.ops.aten.unsqueeze.default(arg212_1, -1);  arg212_1 = None
    unsqueeze_853: "f32[1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_852, -1);  unsqueeze_852 = None
    mul_320: "f32[8, 1600, 7, 7]" = torch.ops.aten.mul.Tensor(mul_319, unsqueeze_853);  mul_319 = unsqueeze_853 = None
    unsqueeze_854: "f32[1600, 1]" = torch.ops.aten.unsqueeze.default(arg213_1, -1);  arg213_1 = None
    unsqueeze_855: "f32[1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_854, -1);  unsqueeze_854 = None
    add_246: "f32[8, 1600, 7, 7]" = torch.ops.aten.add.Tensor(mul_320, unsqueeze_855);  mul_320 = unsqueeze_855 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_106: "f32[8, 1600, 7, 7]" = torch.ops.aten.relu.default(add_246);  add_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_106: "f32[8, 2176, 7, 7]" = torch.ops.aten.convolution.default(relu_106, arg328_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_106 = arg328_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_298: "f32[8, 2048, 7, 7]" = torch.ops.aten.slice.Tensor(convolution_106, 1, 0, 2048)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    add_247: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(add_240, slice_298);  add_240 = slice_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_302: "f32[8, 128, 7, 7]" = torch.ops.aten.slice.Tensor(convolution_106, 1, 2048, 9223372036854775807);  convolution_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    cat_66: "f32[8, 512, 7, 7]" = torch.ops.aten.cat.default([cat_64, slice_302], 1);  cat_64 = slice_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    cat_67: "f32[8, 2560, 7, 7]" = torch.ops.aten.cat.default([add_247, cat_66], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_856: "f32[2560, 1]" = torch.ops.aten.unsqueeze.default(arg548_1, -1);  arg548_1 = None
    unsqueeze_857: "f32[2560, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_856, -1);  unsqueeze_856 = None
    sub_107: "f32[8, 2560, 7, 7]" = torch.ops.aten.sub.Tensor(cat_67, unsqueeze_857);  cat_67 = unsqueeze_857 = None
    add_248: "f32[2560]" = torch.ops.aten.add.Tensor(arg549_1, 0.001);  arg549_1 = None
    sqrt_107: "f32[2560]" = torch.ops.aten.sqrt.default(add_248);  add_248 = None
    reciprocal_107: "f32[2560]" = torch.ops.aten.reciprocal.default(sqrt_107);  sqrt_107 = None
    mul_321: "f32[2560]" = torch.ops.aten.mul.Tensor(reciprocal_107, 1);  reciprocal_107 = None
    unsqueeze_858: "f32[2560, 1]" = torch.ops.aten.unsqueeze.default(mul_321, -1);  mul_321 = None
    unsqueeze_859: "f32[2560, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_858, -1);  unsqueeze_858 = None
    mul_322: "f32[8, 2560, 7, 7]" = torch.ops.aten.mul.Tensor(sub_107, unsqueeze_859);  sub_107 = unsqueeze_859 = None
    unsqueeze_860: "f32[2560, 1]" = torch.ops.aten.unsqueeze.default(arg214_1, -1);  arg214_1 = None
    unsqueeze_861: "f32[2560, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_860, -1);  unsqueeze_860 = None
    mul_323: "f32[8, 2560, 7, 7]" = torch.ops.aten.mul.Tensor(mul_322, unsqueeze_861);  mul_322 = unsqueeze_861 = None
    unsqueeze_862: "f32[2560, 1]" = torch.ops.aten.unsqueeze.default(arg215_1, -1);  arg215_1 = None
    unsqueeze_863: "f32[2560, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_862, -1);  unsqueeze_862 = None
    add_249: "f32[8, 2560, 7, 7]" = torch.ops.aten.add.Tensor(mul_323, unsqueeze_863);  mul_323 = unsqueeze_863 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_107: "f32[8, 2560, 7, 7]" = torch.ops.aten.relu.default(add_249);  add_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_107: "f32[8, 1600, 7, 7]" = torch.ops.aten.convolution.default(relu_107, arg329_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_107 = arg329_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_864: "f32[1600, 1]" = torch.ops.aten.unsqueeze.default(arg550_1, -1);  arg550_1 = None
    unsqueeze_865: "f32[1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_864, -1);  unsqueeze_864 = None
    sub_108: "f32[8, 1600, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_107, unsqueeze_865);  convolution_107 = unsqueeze_865 = None
    add_250: "f32[1600]" = torch.ops.aten.add.Tensor(arg551_1, 0.001);  arg551_1 = None
    sqrt_108: "f32[1600]" = torch.ops.aten.sqrt.default(add_250);  add_250 = None
    reciprocal_108: "f32[1600]" = torch.ops.aten.reciprocal.default(sqrt_108);  sqrt_108 = None
    mul_324: "f32[1600]" = torch.ops.aten.mul.Tensor(reciprocal_108, 1);  reciprocal_108 = None
    unsqueeze_866: "f32[1600, 1]" = torch.ops.aten.unsqueeze.default(mul_324, -1);  mul_324 = None
    unsqueeze_867: "f32[1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_866, -1);  unsqueeze_866 = None
    mul_325: "f32[8, 1600, 7, 7]" = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_867);  sub_108 = unsqueeze_867 = None
    unsqueeze_868: "f32[1600, 1]" = torch.ops.aten.unsqueeze.default(arg216_1, -1);  arg216_1 = None
    unsqueeze_869: "f32[1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_868, -1);  unsqueeze_868 = None
    mul_326: "f32[8, 1600, 7, 7]" = torch.ops.aten.mul.Tensor(mul_325, unsqueeze_869);  mul_325 = unsqueeze_869 = None
    unsqueeze_870: "f32[1600, 1]" = torch.ops.aten.unsqueeze.default(arg217_1, -1);  arg217_1 = None
    unsqueeze_871: "f32[1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_870, -1);  unsqueeze_870 = None
    add_251: "f32[8, 1600, 7, 7]" = torch.ops.aten.add.Tensor(mul_326, unsqueeze_871);  mul_326 = unsqueeze_871 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_108: "f32[8, 1600, 7, 7]" = torch.ops.aten.relu.default(add_251);  add_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_108: "f32[8, 1600, 7, 7]" = torch.ops.aten.convolution.default(relu_108, arg330_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 50);  relu_108 = arg330_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_872: "f32[1600, 1]" = torch.ops.aten.unsqueeze.default(arg552_1, -1);  arg552_1 = None
    unsqueeze_873: "f32[1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_872, -1);  unsqueeze_872 = None
    sub_109: "f32[8, 1600, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_108, unsqueeze_873);  convolution_108 = unsqueeze_873 = None
    add_252: "f32[1600]" = torch.ops.aten.add.Tensor(arg553_1, 0.001);  arg553_1 = None
    sqrt_109: "f32[1600]" = torch.ops.aten.sqrt.default(add_252);  add_252 = None
    reciprocal_109: "f32[1600]" = torch.ops.aten.reciprocal.default(sqrt_109);  sqrt_109 = None
    mul_327: "f32[1600]" = torch.ops.aten.mul.Tensor(reciprocal_109, 1);  reciprocal_109 = None
    unsqueeze_874: "f32[1600, 1]" = torch.ops.aten.unsqueeze.default(mul_327, -1);  mul_327 = None
    unsqueeze_875: "f32[1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_874, -1);  unsqueeze_874 = None
    mul_328: "f32[8, 1600, 7, 7]" = torch.ops.aten.mul.Tensor(sub_109, unsqueeze_875);  sub_109 = unsqueeze_875 = None
    unsqueeze_876: "f32[1600, 1]" = torch.ops.aten.unsqueeze.default(arg218_1, -1);  arg218_1 = None
    unsqueeze_877: "f32[1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_876, -1);  unsqueeze_876 = None
    mul_329: "f32[8, 1600, 7, 7]" = torch.ops.aten.mul.Tensor(mul_328, unsqueeze_877);  mul_328 = unsqueeze_877 = None
    unsqueeze_878: "f32[1600, 1]" = torch.ops.aten.unsqueeze.default(arg219_1, -1);  arg219_1 = None
    unsqueeze_879: "f32[1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_878, -1);  unsqueeze_878 = None
    add_253: "f32[8, 1600, 7, 7]" = torch.ops.aten.add.Tensor(mul_329, unsqueeze_879);  mul_329 = unsqueeze_879 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_109: "f32[8, 1600, 7, 7]" = torch.ops.aten.relu.default(add_253);  add_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_109: "f32[8, 2176, 7, 7]" = torch.ops.aten.convolution.default(relu_109, arg331_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_109 = arg331_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_306: "f32[8, 2048, 7, 7]" = torch.ops.aten.slice.Tensor(convolution_109, 1, 0, 2048)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:145, code: resid = x_s1 + out1
    add_254: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(add_247, slice_306);  add_247 = slice_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_310: "f32[8, 128, 7, 7]" = torch.ops.aten.slice.Tensor(convolution_109, 1, 2048, 9223372036854775807);  convolution_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    cat_68: "f32[8, 640, 7, 7]" = torch.ops.aten.cat.default([cat_66, slice_310], 1);  cat_66 = slice_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:42, code: x = torch.cat(x, dim=1)
    cat_69: "f32[8, 2688, 7, 7]" = torch.ops.aten.cat.default([add_254, cat_68], 1);  add_254 = cat_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_880: "f32[2688, 1]" = torch.ops.aten.unsqueeze.default(arg554_1, -1);  arg554_1 = None
    unsqueeze_881: "f32[2688, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_880, -1);  unsqueeze_880 = None
    sub_110: "f32[8, 2688, 7, 7]" = torch.ops.aten.sub.Tensor(cat_69, unsqueeze_881);  cat_69 = unsqueeze_881 = None
    add_255: "f32[2688]" = torch.ops.aten.add.Tensor(arg555_1, 0.001);  arg555_1 = None
    sqrt_110: "f32[2688]" = torch.ops.aten.sqrt.default(add_255);  add_255 = None
    reciprocal_110: "f32[2688]" = torch.ops.aten.reciprocal.default(sqrt_110);  sqrt_110 = None
    mul_330: "f32[2688]" = torch.ops.aten.mul.Tensor(reciprocal_110, 1);  reciprocal_110 = None
    unsqueeze_882: "f32[2688, 1]" = torch.ops.aten.unsqueeze.default(mul_330, -1);  mul_330 = None
    unsqueeze_883: "f32[2688, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_882, -1);  unsqueeze_882 = None
    mul_331: "f32[8, 2688, 7, 7]" = torch.ops.aten.mul.Tensor(sub_110, unsqueeze_883);  sub_110 = unsqueeze_883 = None
    unsqueeze_884: "f32[2688, 1]" = torch.ops.aten.unsqueeze.default(arg220_1, -1);  arg220_1 = None
    unsqueeze_885: "f32[2688, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_884, -1);  unsqueeze_884 = None
    mul_332: "f32[8, 2688, 7, 7]" = torch.ops.aten.mul.Tensor(mul_331, unsqueeze_885);  mul_331 = unsqueeze_885 = None
    unsqueeze_886: "f32[2688, 1]" = torch.ops.aten.unsqueeze.default(arg221_1, -1);  arg221_1 = None
    unsqueeze_887: "f32[2688, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_886, -1);  unsqueeze_886 = None
    add_256: "f32[8, 2688, 7, 7]" = torch.ops.aten.add.Tensor(mul_332, unsqueeze_887);  mul_332 = unsqueeze_887 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_110: "f32[8, 2688, 7, 7]" = torch.ops.aten.relu.default(add_256);  add_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean: "f32[8, 2688, 1, 1]" = torch.ops.aten.mean.dim(relu_110, [-1, -2], True);  relu_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:274, code: x = self.classifier(x)
    convolution_110: "f32[8, 1000, 1, 1]" = torch.ops.aten.convolution.default(mean, arg332_1, arg333_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean = arg332_1 = arg333_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:275, code: return self.flatten(x)
    view: "f32[8, 1000]" = torch.ops.aten.reshape.default(convolution_110, [8, 1000]);  convolution_110 = None
    return (view,)
    